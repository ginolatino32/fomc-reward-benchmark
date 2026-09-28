#!/usr/bin/env python3
"""Offline audit of the uploaded adaptive-reader pilot; never invokes an LLM.

Usage: python audit_pilot.py --source /path/to/extracted/upload --output ./audit
The source root contains adaptive_reader_run_20260920_actual and source_context.
Original artifacts are not modified. Correctly nested TF-IDF is a diagnostic fix,
not an adaptation result or independent holdout.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,io,json,unittest
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import jsonschema

ALPHAS=np.logspace(-4,4,9)
CATS=['positive','negative','neutral','hypothetical','unclear']

def norm(train,test,numeric):
    a,b=np.array(train,dtype=float,copy=True),np.array(test,dtype=float,copy=True)
    if numeric:
        sd=a.std(axis=0);sd=np.where(sd>1e-12,sd,1.)
        a/=sd;b/=sd
    mu=a.mean(axis=0);a-=mu;b-=mu
    scale=np.sqrt(max(float((a*a).sum()/len(a)),1e-15))
    return a/scale,b/scale

def load_module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def run(src:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    root=src/'adaptive_reader_run_20260920_actual';data=src/'source_context/focused_data';pkg=src/'source_context/FOMC_Agents_Workshop_Package'
    panel=pd.read_csv(data/'analysis_panel.csv');reps=np.load(data/'representations.npz',allow_pickle=False)
    roles=pd.read_csv(root/'manifests/controller_only/fold_roles.csv');mapping=pd.read_csv(root/'manifests/controller_only/passage_mapping.csv')
    raw_input=[json.loads(x) for x in (root/'manifests/agent_inputs/passages.jsonl').read_text().splitlines()]
    texts={x['passage_id']:x['text'] for x in raw_input}
    dates=panel.date.to_numpy();y=panel.state_return.to_numpy(float);length=panel[['document_sentence_count']].to_numpy(float)
    mt=pd.read_csv(data/'meeting_text.csv')
    assert np.array_equal(dates,reps['dates']) and np.array_equal(dates,mt.date.to_numpy())
    assert np.all(roles.meeting_date.to_numpy()==dates[roles.panel_row.to_numpy()])
    schema=json.loads((pkg/'schemas/reader_response.schema.json').read_text())
    rows=[];issues=[];all_hashes=[]
    for path in sorted((root/'reader_responses').rglob('*.json')):
        v=json.loads(path.read_text());pid=v['passage_id'];resp=v['response'];errs=list(jsonschema.Draft202012Validator(schema).iter_errors(resp))
        validq=all(ev.get('quote') and ev['quote'] in texts[pid] for ev in resp.get('evidence',[]))
        expectedh=hashlib.sha256(texts[pid].encode()).hexdigest()
        row={'passage_id':pid,'tone':resp['tone'],'phase':v.get('phase','smoke'),'production':'static_fresh' in str(path),'schema_valid':not errs,'quotes_valid':bool(validq),'input_hash_valid':v.get('input_hash')==expectedh,'prompt_hash':v.get('prompt_hash'),'candidate_hash':v.get('candidate_hash'),'path':str(path.relative_to(root))}
        rows.append(row)
        for e in errs: issues.append({'path':row['path'],'issue':e.message})
    res=pd.DataFrame(rows);prod=res[res.production].copy();res.to_csv(out/'response_audit.csv',index=False)
    assert not prod.passage_id.duplicated().any()
    df=prod.merge(mapping,on='passage_id',validate='one_to_one')
    counts=pd.crosstab(df.meeting_date,df.tone).reindex(columns=CATS,fill_value=0)
    joined=panel[['date','state_return','document_sentence_count']].merge(counts,left_on='date',right_index=True,how='left')
    features=joined[CATS].to_numpy(float)
    # Independent primal ridge implementation with training-only block transforms.
    def fit(name,train,test,feat=None,tfidf_nested=True):
        # Legacy nonnested TF-IDF only for reproducing the supplied control.
        tf_cache=None
        if name=='tfidf' and not tfidf_nested:
            vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=10000,sublinear_tf=True,lowercase=True,dtype=np.float64)
            tf_cache=vec.fit_transform(mt.original.iloc[train]).toarray()
            fixed_e=vec.transform(mt.original.iloc[test]).toarray()
        def matrices(tr,te,inner_tr=None,inner_te=None):
            if name=='tfidf':
                if not tfidf_nested:
                    if inner_tr is None:f,g=tf_cache,fixed_e
                    else:f,g=tf_cache[inner_tr],tf_cache[inner_te]
                else:
                    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=10000,sublinear_tf=True,lowercase=True,dtype=np.float64)
                    f=vec.fit_transform(mt.original.iloc[tr]).toarray();g=vec.transform(mt.original.iloc[te]).toarray()
                numeric=False
            else:f,g=feat[tr],feat[te];numeric=(name!='embedding')
            a,b=norm(length[tr],length[te],True);c,d=norm(f,g,numeric)
            return np.column_stack((a,c))/np.sqrt(2),np.column_stack((b,d))/np.sqrt(2)
        losses=np.zeros(len(ALPHAS));nval=0
        for ti,vi in TimeSeriesSplit(n_splits=3).split(train):
            u,v=train[ti],train[vi];x,z=matrices(u,v,ti,vi)
            for j,alpha in enumerate(ALPHAS):
                yp=Ridge(alpha=alpha,solver='cholesky').fit(x,y[u]).predict(z)
                losses[j]+=np.square(y[v]-yp).sum()
            nval+=len(v)
        alpha=ALPHAS[int(np.argmin(losses))];x,z=matrices(train,test)
        model=Ridge(alpha=alpha,solver='cholesky').fit(x,y[train]);pred=model.predict(z)
        return pred,alpha,losses/nval,float(np.mean((y[train]-model.predict(x))**2))
    ci=roles[(roles.fold==0)&(roles.role=='calibration')].panel_row.to_numpy()
    fi=roles[(roles.fold==0)&(roles.role=='feedback')].panel_row.to_numpy()
    fp,fa,fcs,trainmse=fit('fresh_flash',ci,fi,features)
    f_mse=float(np.mean((fp-y[fi])**2));fb=json.loads((root/'proposals/feedback_fold0_round1.json').read_text())
    pd.DataFrame({'meeting_date':dates[fi],'y':y[fi],'prediction':fp,'error':y[fi]-fp}).to_csv(out/'independent_feedback_predictions.csv',index=False)
    pd.DataFrame({'alpha':ALPHAS,'inner_validation_mse':fcs,'selected':ALPHAS==fa}).to_csv(out/'correct_fresh_reader_alpha_selection.csv',index=False)
    counts.to_csv(out/'production_meeting_category_counts.csv')
    # Full reference-control refits, then corrected TF-IDF.
    feats={'embedding':reps['gemini'],'rules_linear':reps['cvj'],'rules_quadratic':np.column_stack((reps['cvj'],reps['cvj'][:,0]**2,reps['cvj'][:,1]**2,reps['cvj'][:,0]*reps['cvj'][:,1]))}
    p_rows=[];t_rows=[]
    for name in ['embedding','rules_linear','rules_quadratic','tfidf','tfidf_nested_corrected']:
        print('Fitting',name,flush=True)
        for fold in sorted(roles.fold.unique()):
            tr=roles[(roles.fold==fold)&(roles.role=='calibration')].panel_row.to_numpy()
            te=roles[(roles.fold==fold)&(roles.role=='evaluation')].panel_row.to_numpy()
            base='tfidf' if name.startswith('tfidf') else name
            yp,a,cv,_=fit(base,tr,te,feats.get(base),tfidf_nested=(name=='tfidf_nested_corrected'))
            for idx,pr in zip(te,yp):p_rows.append({'arm_id':name,'fold':fold,'meeting_date':dates[idx],'y':y[idx],'prediction':pr,'calibration_mean':y[tr].mean()})
            for aa,ss in zip(ALPHAS,cv):t_rows.append({'arm_id':name,'fold':fold,'alpha':aa,'cv_mse':ss,'selected':aa==a})
    pp=pd.DataFrame(p_rows);pp.to_csv(out/'independent_control_predictions.csv',index=False);pd.DataFrame(t_rows).to_csv(out/'independent_control_tuning.csv',index=False)
    orig=pd.read_csv(root/'results/agent_predictions.csv')
    comparison=pp.merge(orig[['arm_id','fold','meeting_date','prediction']],on=['arm_id','fold','meeting_date'],suffixes=('_refit','_submitted'))
    comparison['absolute_prediction_difference']=abs(comparison.prediction_refit-comparison.prediction_submitted)
    comparison.to_csv(out/'control_prediction_comparison.csv',index=False)
    metrics=[]
    for name,g in pp.groupby('arm_id'):
        err=g.y-g.prediction;mse=np.mean(err**2)
        metrics.append({'arm_id':name,'n_eval':len(g),'mse':mse,'rmse':np.sqrt(mse),'mae':np.mean(abs(err)),'oos_r2':1-mse/np.mean((g.y-g.calibration_mean)**2)})
    pd.DataFrame(metrics).to_csv(out/'independent_control_performance.csv',index=False)
    # Rerun supplied tests by changing only runtime file locations.
    mod=load_module(root/'tests/test_actual_run.py','submitted_tests');mod.ACTUAL_DIR=root;mod.PKG=pkg
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(mod))
    (out/'supplied_test_results.txt').write_text(stream.getvalue())
    # Compare hashes and chronology without treating self-authored metadata as provider authentication.
    lock=json.loads((root/'execution_lock.json').read_text());h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    hash_checks={k: h(pkg/path)==lock['prompt_hashes'][k] for k,path in {'frozen_reader':'prompts/frozen_reader.txt','blind_proposer':'prompts/blind_proposer.txt','reflector':'prompts/reflector.txt','proposed_protocol':'protocol/proposed_protocol.json'}.items()}
    proposals=[]
    for path in sorted((root/'proposals').glob('*proposals/*.json')):
        x=json.loads(path.read_text());actualhash=hashlib.sha256(x['memory_text'].encode()).hexdigest()
        proposals.append({'file':str(path.relative_to(root)),'candidate_hash':actualhash,'stored_hash_matches':actualhash==x.get('candidate_hash'),'words':len(x['memory_text'].split()),'stored_word_count':x.get('word_count')})
    pd.DataFrame(proposals).to_csv(out/'proposal_checks.csv',index=False)
    curve=pd.read_csv(root/'results/learning_curve.csv');chashes={x['candidate_hash'] for x in proposals}
    unmatched=[x for x in curve.candidate_sha256 if x not in chashes]
    mrole=mapping.merge(roles[roles.fold==0],on='meeting_date')
    role_counts=mrole.groupby('role').size().to_dict()
    smoke=res[~res.production].merge(mapping,on='passage_id',validate='one_to_one').merge(roles[roles.fold==0],on='meeting_date',how='left')
    smoke['role']=smoke.role.fillna('unlabeled')
    smoke.groupby('role').size().rename('n_passages').to_csv(out/'smoke_roles.csv')
    keymiss={};prompt_hash=hash_checks['frozen_reader']
    # Check strongest numeric source claims and risk notes.
    summary={
      'source_root':str(src),'production_records':len(prod),'production_unique_passages':prod.passage_id.nunique(),'production_meetings':df.meeting_date.nunique(),
      'phase_counts':prod.phase.value_counts().to_dict(),'all_saved_responses':len(res),'all_unique_passages':res.passage_id.nunique(),
      'schema_valid_production':int(prod.schema_valid.sum()),'quote_valid_production':int(prod.quotes_valid.sum()),'input_hash_valid_production':int(prod.input_hash_valid.sum()),
      'fold0_role_passages':{k:int(v) for k,v in role_counts.items()},'smoke_roles':{k:int(v) for k,v in smoke.groupby('role').size().to_dict().items()},
      'calibration_alpha':float(fa),'calibration_training_mse':trainmse,'calibration_inner_validation_mse_selected':float(fcs[np.argmin(fcs)]),'feedback_mse':f_mse,
      'feedback_mse_difference_from_report':f_mse-fb['f_overall_mse'],'feedback_top_six_dates':list(dates[fi][np.argsort(-(y[fi]-fp)**2)[:6]]),
      'max_control_prediction_difference':float(comparison.absolute_prediction_difference.max()),'supplied_tests_run':tests.testsRun,'supplied_tests_passed':tests.wasSuccessful(),
      'lock_file_hash':h(root/'execution_lock.json'),'prompt_file_hash_checks':hash_checks,'unmatched_learning_curve_candidate_hashes':unmatched,
      'adaptive_reader_predictions':int(orig.arm_id.isin(['static_fresh','blind_search','reflection_ungated','reflection_gated']).sum()),
      'candidate_response_files':int((prod.candidate_hash!=hashlib.sha256(b'').hexdigest()).sum()),
      'provider_invocation_independently_authenticated':False,'provider_transcripts_present':any('transcript' in p.name for p in src.rglob('*') if p.is_file()),
      'auditor_llm_inference_calls':0
    }
    (out/'audit_summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:int(x)))
    (out/'schema_issues.json').write_text(json.dumps(issues,indent=2))
    print(json.dumps(summary,indent=2,default=lambda x:int(x)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.source.resolve(),a.output.resolve())
