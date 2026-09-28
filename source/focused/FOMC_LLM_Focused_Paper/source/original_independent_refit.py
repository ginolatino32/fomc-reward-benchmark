"""Independent offline refit of delivered FOMC representation experiments.

No provider calls. Uses sklearn Ridge with explicitly transformed feature blocks,
not the submitted eigensolver. Corrects the PCA nesting for a separate sensitivity.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 for predictable runtime.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit

ALPHAS=np.logspace(-4,4,9)
LABELS=['positive','negative','neutral','hypothetical','unclear']

def l2rows(x:np.ndarray)->np.ndarray:
    norm=np.linalg.norm(x,axis=1)
    if not np.isfinite(x).all() or np.any(norm<=0):raise ValueError('Invalid vectors')
    return x/norm[:,None]

def block(x:np.ndarray,z:np.ndarray,numeric:bool=False):
    if numeric:
        sd=x.std(axis=0);sd=np.where(sd>1e-12,sd,1.)
        x=x/sd;z=z/sd
    mu=x.mean(axis=0);x=x-mu;z=z-mu
    scale=max(float(np.sum(x*x)/len(x)),1e-15)**.5
    return x/scale,z/scale

def prepare(root:Path):
    c=root/'inputs/canonical_minimal';h=root/'inputs/handoff'
    panel=pd.read_csv(c/'data/analysis_panel.csv')
    mask=panel.state_return.notna().to_numpy();d=panel.loc[mask].reset_index(drop=True)
    meta=pd.DataFrame([json.loads(s) for s in (h/'inputs/metadata_DO_NOT_SEND_TO_MODELS.jsonl').read_text().splitlines()])
    text=pd.DataFrame([json.loads(s) for s in (h/'inputs/passages_original_and_masked.jsonl').read_text().splitlines()])
    records=meta.merge(text,on='record_id',validate='one_to_one')
    # Dictionary keys are IDs rather than row positions throughout.
    arrays={}
    z=np.load(c/'data/cached_representation_matrices.npz',allow_pickle=False)
    idx={str(date):i for i,date in enumerate(z['date'])}
    arrays[('legacy_gemini','original')]=z['gemini'][[idx[x] for x in d.date]].astype(float)
    if 'e5' in z:
        arrays[('legacy_e5','original')]=z['e5'][[idx[x] for x in d.date]].astype(float)
    pool_rows=[]
    for arm in ['gemini001_refresh','gemini2','voyage_isolated','voyage_context']:
        for var in ['original','numbers_masked']:
            p=root/f'models/{arm}/{var}/embeddings.npz';a=np.load(p,allow_pickle=False)
            ids=a['record_id'].astype(str);v=l2rows(a['embeddings'].astype(float))
            rows=records[records.variant==var].set_index('record_id').loc[ids]
            grouped={date:np.mean(v[np.flatnonzero(rows.meeting_date.to_numpy()==date)],axis=0) for date in d.date}
            arrays[(arm,var)]=np.vstack([grouped[date] for date in d.date])
            pool_rows.append({'arm':arm,'variant':var,'shape':list(v.shape),'max_float32_pool_rounding':float(np.max(np.abs(arrays[(arm,var)]-arrays[(arm,var)].astype('float32'))))})
    texts={}
    for var in ['original','numbers_masked']:
        selected=records[records.variant==var].sort_values(['meeting_date','within_meeting_order'])
        mt=selected.groupby('meeting_date',sort=False).text.agg(' '.join)
        texts[var]=d.date.map(mt)
    # Preserve original lexical input EXACTLY for the historical benchmark.
    texts['original']=d.passage_text
    counts=pd.read_csv(c/'results/audited_reconstruction/meeting_counts.csv').set_index('meeting_date')
    arrays[('audited_counts','original')]=counts.loc[d.date,['negative_count','positive_count']].to_numpy(float)
    labels=pd.read_csv(root/'models/flash/validated_labels.csv')
    for var in ['original','numbers_masked']:
        valid=labels[(labels.variant==var)&(labels.status=='VALID')].merge(records[['record_id','meeting_date']],on='record_id',validate='one_to_one')
        assert len(valid)==1197
        feat=pd.crosstab(valid.meeting_date,valid.tone).reindex(columns=LABELS,fill_value=0).loc[d.date]
        arrays[('flash',var)]=feat.to_numpy(float)
    if all(x in d.columns for x in ['stock_mentions_total','negative_stock_mentions','positive_stock_mentions']):
        arrays[('initial_signed','original')]=d[['stock_mentions_total','negative_stock_mentions','positive_stock_mentions']].to_numpy(float)
    folds=pd.read_csv(c/'results/common_task_folds.csv').to_dict('records')
    return d,arrays,texts,folds,pool_rows

def fit(root:Path,out:Path):
    d,arrays,texts,folds,pool_rows=prepare(root)
    y=d.state_return.to_numpy(float);length=d[['document_sentence_count']].to_numpy(float)
    keys=[]
    for arm,var in arrays:
        if arm=='audited_counts':
            keys += [(arm,var,'split'),(arm,var,'joint'),(arm,var,'ols')]
        else: keys.append((arm,var,'full'))
    keys += [('tfidf',v,'full') for v in texts]+[('length_only','original','full')]
    keys += [(a,'original','pca3_correct_nested') for a in ['legacy_gemini','gemini2','voyage_isolated','voyage_context']]
    pred_rows=[];tune_rows=[]
    for arm,var,mode in keys:
        start=time.time()
        for f in folds:
            tr=np.flatnonzero(d.date.le(f['train_last']).to_numpy())
            te=np.flatnonzero(d.date.between(f['test_first'],f['test_last']).to_numpy())
            assert len(tr)==f['train_n'] and len(te)==f['test_n'] and tr.max()<te.min()
            def features(a,b):
                if arm=='length_only':return block(length[a],length[b],True)
                if (arm=='audited_counts' and mode in ['joint','ols']) or arm=='initial_signed':
                    v=np.column_stack([length,arrays[(arm,var)]])
                    return block(v[a],v[b],True)
                lx,lz=block(length[a],length[b],True)
                if arm=='tfidf':
                    vect=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=10000,sublinear_tf=True,lowercase=True,stop_words=None,dtype=np.float64)
                    x=vect.fit_transform(texts[var].iloc[a]).toarray();z=vect.transform(texts[var].iloc[b]).toarray()
                    ex,ez=block(x,z)
                else:
                    v=arrays[(arm,var)]
                    if mode=='pca3_correct_nested':
                        # The PCA is re-fitted on each inner training subset too.
                        pca=PCA(n_components=3,svd_solver='full',random_state=20260919)
                        x=pca.fit_transform(v[a]);z=pca.transform(v[b]);ex,ez=block(x,z,True)
                    else:ex,ez=block(v[a],v[b],arm in ['audited_counts','flash','initial_signed'])
                return np.column_stack([lx,ex])/np.sqrt(2),np.column_stack([lz,ez])/np.sqrt(2)
            if mode=='ols':
                alpha=0.;x,z=features(tr,te);model=LinearRegression().fit(x,y[tr]);yp=model.predict(z)
            else:
                losses=np.zeros(len(ALPHAS));ni=0
                for ta,va in TimeSeriesSplit(n_splits=3).split(tr):
                    a=tr[ta];b=tr[va];x,z=features(a,b)
                    for j,alpha in enumerate(ALPHAS):
                        model=Ridge(alpha=float(alpha),fit_intercept=True,solver='cholesky').fit(x,y[a]);losses[j]+=np.sum((y[b]-model.predict(z))**2)
                    ni+=len(b)
                selected=int(np.argmin(losses));alpha=float(ALPHAS[selected]);x,z=features(tr,te)
                model=Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(x,y[tr]);yp=model.predict(z)
                for al,loss in zip(ALPHAS,losses):tune_rows.append({'arm':arm,'variant':var,'mode':mode,'fold':f['fold'],'alpha':al,'inner_mse':loss/ni,'selected':al==alpha})
            for i,p in zip(te,yp):
                pred_rows.append({'arm':arm,'variant':var,'mode':mode,'fold':f['fold'],'meeting_date':d.date.iloc[i],'y':float(y[i]),'prediction':float(p),'training_mean':float(y[tr].mean()),'alpha':alpha})
        print(arm,var,mode,'seconds',round(time.time()-start,2),flush=True)
    pred=pd.DataFrame(pred_rows);pred.to_csv(out/'independent_predictions.csv',index=False)
    pd.DataFrame(tune_rows).to_csv(out/'independent_tuning.csv',index=False)
    summary=[]
    for k,g in pred.groupby(['arm','variant','mode']):
        se=(g.y-g.prediction)**2;den=np.mean((g.y-g.training_mean)**2)
        summary.append(dict(zip(['arm','variant','mode'],k),n=len(g),mse=se.mean(),rmse=np.sqrt(se.mean()),mae=np.mean(abs(g.y-g.prediction)),oos_r2=1-se.mean()/den))
    s=pd.DataFrame(summary);s.to_csv(out/'independent_performance.csv',index=False)
    pd.DataFrame(pool_rows).to_json(out/'pooling_precision.json',orient='records',indent=2)
    saved=pd.read_csv(root/'tables/predictions.csv');comp=[]
    for (arm,var,mode),g in pred.groupby(['arm','variant','mode']):
        names={'audited_counts':{'split':('audited_reconstruction_split','full_count_length'),'joint':('audited_reconstruction_joint','full_count_length_joint'),'ols':('audited_reconstruction_ols','full_count_length_ols')},'flash':{'full':('flash','five_class_counts_length')},'tfidf':{'full':('tfidf_bigram_only','tfidf_bigram_length')}}
        if mode=='pca3_correct_nested':name,readout=arm,'three_pc_length'
        elif arm in names:name,readout=names[arm][mode]
        else:name,readout=arm,'full_vector_length'
        q=saved[(saved.arm_id==name)&(saved.variant==var)&(saved.readout==readout)]
        if q.empty:continue
        z=g.merge(q[['meeting_date','y_pred','selected_alpha']],on='meeting_date',validate='one_to_one')
        comp.append({'arm':arm,'variant':var,'mode':mode,'n':len(z),'max_abs_prediction_difference':float(abs(z.prediction-z.y_pred).max()),'alpha_differences':int((z.alpha!=z.selected_alpha).sum()),'interpretation':'PCA corrected inside inner folds' if mode=='pca3_correct_nested' else 'independent original-specification refit'})
    pd.DataFrame(comp).to_csv(out/'refit_vs_submitted.csv',index=False)
    print(s.to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);fit(a.input,a.output)
