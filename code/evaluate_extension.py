"""Frozen-feature comparisons with training-only transformations and nested tuning."""
from __future__ import annotations
import os
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ.setdefault(v,'1')
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge,LinearRegression
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'source/focused/FOMC_LLM_Focused_Paper'
ALPHAS=np.logspace(-4,4,9)
def norm(x,z,numeric=False):
    if numeric:
        sd=x.std(0);sd=np.where(sd>1e-12,sd,1.);x=x/sd;z=z/sd
    mu=x.mean(0);x=x-mu;z=z-mu
    s=max(float(np.square(x).sum()/len(x)),1e-15)**.5
    return x/s,z/s

def load():
    d=pd.read_csv(F/'data/analysis_panel.csv');a=np.load(F/'data/representations.npz',allow_pickle=False)
    q=pd.read_csv(ROOT/'data/additional_meeting_features.csv').set_index('meeting_date').loc[d.date]
    assert np.array_equal(d.date.to_numpy(),a['dates'])
    controls=d[['document_sentence_count']].to_numpy(float);volume=q[['passage_count']].to_numpy(float)
    negpos=a['cvj'].astype(float);ph=q[['negative_phrase_hits','positive_phrase_hits']].to_numpy(float)
    votes=q[['rule_passages_negative','rule_passages_positive']].to_numpy(float)
    fl=q[['flash_negative','flash_positive']].to_numpy(float)
    def quad(b):return np.column_stack([b,b[:,0]**2,b[:,1]**2,b[:,0]*b[:,1]])
    models={
    'rules_whole_linear':(controls,negpos,True),
    'rules_whole_quadratic':(controls,quad(negpos),True),
    'rules_matched_phrase':(controls,ph,True),
    'rules_matched_quadratic':(controls,quad(ph),True),
    'rules_matched_two_votes':(controls,votes,True),
    'rules_matched_four_votes':(controls,q[['rule_passages_negative','rule_passages_positive','rule_passages_mixed','rule_passages_unclassified']].to_numpy(float),True),
    'rules_matched_plus_volume':(np.column_stack([controls,volume]),ph,True),
    'rules_two_votes_plus_volume':(np.column_stack([controls,volume]),votes,True),
    'flash_two_counts':(controls,fl,True),
    'flash_two_plus_volume':(np.column_stack([controls,volume]),fl,True),
    'flash_five':(controls,a['flash'],True),
    'gemini':(controls,a['gemini'],False),
    'gemini_plus_volume':(np.column_stack([controls,volume]),a['gemini'],False),
    'length_volume_only':(np.column_stack([controls,volume]),np.zeros((len(d),0)),True),
    'tfidf':(controls,None,False),
    }
    return d,a,q,models

def predict_fixed(y,models,d,folds,out_prefix):
    texts=pd.read_csv(F/'data/meeting_text.csv').original
    pp=[];tt=[]
    for name,(control,rep,numeric) in models.items():
        for f in folds.to_dict('records'):
            tr=np.flatnonzero(d.date.le(f['train_last'])&np.isfinite(y));te=np.flatnonzero(d.date.between(f['test_first'],f['test_last'])&np.isfinite(y))
            if len(tr)<20 or not len(te):continue
            assert tr.max()<te.min()
            def features(u,v):
                cx,cz=norm(control[u],control[v],True)
                if name=='length_volume_only':return cx,cz
                if rep is None:
                    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=10000,sublinear_tf=True,lowercase=True,stop_words=None,dtype=np.float64)
                    x=vec.fit_transform(texts.iloc[u]).toarray();z=vec.transform(texts.iloc[v]).toarray()
                else:x,z=rep[u],rep[v]
                ex,ez=norm(x,z,numeric)
                return np.column_stack([cx,ex])/np.sqrt(2),np.column_stack([cz,ez])/np.sqrt(2)
            scores=np.zeros(len(ALPHAS));ni=0
            for tu,vu in TimeSeriesSplit(n_splits=3).split(tr):
                u,v=tr[tu],tr[vu];x,z=features(u,v);yc=y[u]-y[u].mean()
                # Evaluate all penalties with one spectral decomposition (linear ridge).
                vals,basis=eigh(x@x.T,check_finite=False);vals=np.maximum(vals,0.)
                proj=basis.T@yc;testbasis=(z@x.T)@basis
                pred=testbasis@(proj[:,None]/(vals[:,None]+ALPHAS))+y[u].mean()
                scores+=np.square(y[v,None]-pred).sum(0);ni+=len(v)
            al=float(ALPHAS[np.argmin(scores)]);x,z=features(tr,te)
            pred=Ridge(alpha=al,fit_intercept=True,solver='cholesky').fit(x,y[tr]).predict(z)
            pp.extend({'model':name,'fold':int(f['fold']),'date':d.date.iloc[i],'y':float(y[i]),'prediction':float(p),'training_mean':float(y[tr].mean()),'alpha':al,'train_n':len(tr)} for i,p in zip(te,pred))
            tt.extend({'model':name,'fold':f['fold'],'alpha':float(aa),'validation_mse':ss/ni,'selected':aa==al} for aa,ss in zip(ALPHAS,scores))
        print(out_prefix,name,'done',flush=True)
    pred=pd.DataFrame(pp);pred.to_csv(ROOT/f'results/{out_prefix}_predictions.csv',index=False);pd.DataFrame(tt).to_csv(ROOT/f'results/{out_prefix}_tuning.csv',index=False)
    report=[]
    for m,g in pred.groupby('model'):
        mse=float(np.square(g.y-g.prediction).mean());den=float(np.square(g.y-g.training_mean).mean())
        report.append({'model':m,'n':len(g),'mse':mse,'rmse':mse**.5,'mae':float(abs(g.y-g.prediction).mean()),'oos_r2':1-mse/den})
    report=pd.DataFrame(report).set_index('model')
    if 'rules_whole_linear' in report.index:report['gain_vs_whole_rules_pct']=100*(1-report.mse/report.loc['rules_whole_linear','mse'])
    report.to_csv(ROOT/f'results/{out_prefix}_performance.csv')
    return pred,report

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--target',default='state_return');pa.add_argument('--core-only',action='store_true');args=pa.parse_args()
    d,a,q,models=load();folds=pd.read_csv(F/'data/folds.csv')
    if args.target=='state_return':y=d.state_return.to_numpy(float)
    else:
        add=pd.read_csv(ROOT/'data/alternative_return_targets.csv').set_index('date').reindex(d.date)
        y=add[args.target].to_numpy(float)
    if args.core_only:models={k:v for k,v in models.items() if k in ['rules_whole_linear','rules_whole_quadratic','rules_matched_phrase','flash_five','gemini','tfidf']}
    predict_fixed(y,models,d,folds,args.target)
if __name__=='__main__':main()
