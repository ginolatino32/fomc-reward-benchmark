"""Frozen common-outcome comparison using the existing chronological ridge code.
Runs only after all count files exist; no outcomes used to repair matching rules.
"""
from __future__ import annotations
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import json, hashlib, importlib.util, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits
threadpool_limits(1)
import original_benchmark as b
R=Path(__file__).resolve().parents[1]


def numeric_kernel(d,tr,te,columns):
    x=d.iloc[tr][columns].to_numpy(float);z=d.iloc[te][columns].to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError('Count benchmark requires complete, aligned observations')
    sd=x.std(0);sd=np.where(sd>1e-12,sd,1.)
    return b.centered_kernel(x/sd,z/sd)


def evaluate():
    full=pd.read_csv(R/'data/analysis_panel.csv')
    vectors=np.load(R/'data/cached_representation_matrices.npz',allow_pickle=False)
    assert np.array_equal(full.date.astype(str).to_numpy(),vectors['date'].astype(str))
    variants=['uploaded_as_received','dictionary_only_correction','audited_reconstruction']
    for version in variants:
        fname='meeting_counts.csv' if version=='audited_reconstruction' else 'author_method_minutes_meeting_counts.csv'
        counts=pd.read_csv(R/'results'/version/fname)
        assert counts.meeting_date.is_unique and len(counts)==len(full)==200
        if version != 'audited_reconstruction':
            counts=counts.rename(columns={'author_method_negative_count':'negative_count','author_method_positive_count':'positive_count'})
        counts=counts.set_index('meeting_date').reindex(full.date)
        assert counts.negative_count.notna().all()
        for sign in ['negative','positive']:
            full[version+'_'+sign]=counts[sign+'_count'].to_numpy(float)
    ix=np.flatnonzero(full.state_return.notna().to_numpy());d=full.iloc[ix].reset_index(drop=True)
    arr={key:vectors[key][ix].astype(float) for key in ['gemini','e5']}
    y=d.state_return.to_numpy(float)
    assert len(d)==199
    specifications={
        'gemini_vector':['length','gemini'],
        'historical_counts':['counts'],
        'attention_counts':['attention'],
        'tfidf_bigram_only':['length','bi'],
        'length_only':['length'],
    }
    cols={}
    for v in variants:
        cols[v]=[v+'_negative',v+'_positive']
        cols[v+'_joint']=['document_sentence_count']+cols[v]
        specifications[v+'_split']=['length',v]
        specifications[v+'_joint']=[v+'_joint']
    specifications['audited_counts_plus_gemini']=['length','audited_reconstruction','gemini']
    def kernel(tr,te,name):
        if name in cols: return numeric_kernel(d,tr,te,cols[name])
        return b.block_kernels(d,arr,tr,te,name)
    rows=[];tuning=[];folds=[]
    for fold,te in enumerate(np.array_split(np.arange(100,len(d)),5)):
        tr=np.arange(te[0]);assert d.date.iloc[tr].max()<d.date.iloc[te].min()
        keys=sorted({x for bs in specifications.values() for x in bs})
        losses={m:np.zeros(len(b.ALPHAS)) for m in specifications};nval=0
        for it,iv in TimeSeriesSplit(n_splits=3).split(tr):
            kernels={name:kernel(it,iv,name) for name in keys}
            for m,bs in specifications.items():
                k=sum(kernels[name][0] for name in bs)/len(bs)
                q=sum(kernels[name][1] for name in bs)/len(bs)
                pr,_=b.predict_grid(k,q,y[it]);losses[m]+=((y[iv,None]-pr)**2).sum(0)
            nval+=len(iv)
        kernels={name:kernel(tr,te,name) for name in keys}
        preds={};parameters={}
        for m,bs in specifications.items():
            best=int(np.argmin(losses[m]));alpha=float(b.ALPHAS[best])
            k=sum(kernels[name][0] for name in bs)/len(bs)
            q=sum(kernels[name][1] for name in bs)/len(bs)
            pr,edf=b.predict_grid(k,q,y[tr]);preds[m]=pr[:,best]
            parameters[m]=(alpha,float(edf[best]))
            for a,l in zip(b.ALPHAS,losses[m]/nval):
                tuning.append(dict(model=m,fold=fold,alpha=a,inner_mse=l))
        for v in variants:
            columns=['document_sentence_count']+cols[v]
            x=d.iloc[tr][columns].to_numpy(float);z=d.iloc[te][columns].to_numpy(float)
            mu=x.mean(0);sd=x.std(0);sd=np.where(sd>1e-12,sd,1.)
            x=np.column_stack([np.ones(len(tr)),(x-mu)/sd]);z=np.column_stack([np.ones(len(te)),(z-mu)/sd])
            coef=np.linalg.lstsq(x,y[tr],rcond=None)[0]
            preds[v+'_ols']=z@coef;parameters[v+'_ols']=(0.,float(np.linalg.matrix_rank(x)))
        preds['historical_mean']=np.repeat(y[tr].mean(),len(te));parameters['historical_mean']=(np.nan,1.)
        for m,pr in preds.items():
            for i,p in zip(te,pr):
                rows.append(dict(model=m,fold=fold,date=d.date.iloc[i],y=y[i],prediction=p,
                                 train_n=len(tr),train_last=d.date.iloc[tr].max(),alpha=parameters[m][0],edf=parameters[m][1]))
        folds.append(dict(fold=fold,train_n=len(tr),test_n=len(te),train_first=d.date.iloc[tr].min(),train_last=d.date.iloc[tr].max(),test_first=d.date.iloc[te].min(),test_last=d.date.iloc[te].max()))
        print('evaluation fold',fold,'complete',flush=True)
    p=pd.DataFrame(rows);p.to_csv(R/'results/common_task_predictions.csv',index=False)
    pd.DataFrame(tuning).to_csv(R/'results/common_task_tuning.csv',index=False)
    pd.DataFrame(folds).to_csv(R/'results/common_task_folds.csv',index=False)
    d.to_csv(R/'data/aligned_count_analysis_panel.csv',index=False)
    p['loss']=(p.y-p.prediction)**2;p['absolute_error']=abs(p.y-p.prediction)
    summaries=[]
    for period,sub in [('All evaluation',p),('Through 2016',p[p.date<'2017-01-01']),('From 2017',p[p.date>='2017-01-01'])]:
        mean_mse=sub[sub.model=='historical_mean'].loss.mean()
        gmse=sub[sub.model=='gemini_vector'].loss.mean()
        for m,g in sub.groupby('model'):
            mse=g.loss.mean()
            summaries.append(dict(period=period,model=m,n=len(g),mse=mse,rmse=np.sqrt(mse),mae=g.absolute_error.mean(),oos_r2=1-mse/mean_mse,gemini_mse_reduction=1-gmse/mse))
    summary=pd.DataFrame(summaries);summary.to_csv(R/'results/common_task_performance.csv',index=False)
    byfold=[]
    for f,sub in p.groupby('fold'):
        gmse=sub[sub.model=='gemini_vector'].loss.mean()
        for m,g in sub.groupby('model'):
            byfold.append(dict(fold=f,model=m,n=len(g),mse=g.loss.mean(),rmse=np.sqrt(g.loss.mean()),gemini_mse_reduction=1-gmse/g.loss.mean()))
    pd.DataFrame(byfold).to_csv(R/'results/common_task_by_fold.csv',index=False)
    # Verify that the comparison did not move the embedding goalposts.
    ref=pd.read_csv(R/'manuscript/results/predictions_state_return.csv')
    checks={}
    for current,previous in [('gemini_vector','gemini_vector'),('historical_counts','signed_counts'),('attention_counts','attention_counts'),('historical_mean','historical_mean')]:
        got=p[p.model==current].sort_values('date')
        old=ref[ref.model==previous].sort_values('date')
        assert got.date.tolist()==old.date.tolist()
        diff=float(np.max(np.abs(got.prediction.to_numpy()-old.prediction.to_numpy())))
        tolerance=1e-10 if current=='gemini_vector' else 1e-8
        assert diff<tolerance,(current,diff)
        # Tiny cross-platform eigen-solver variation in near-unpenalized count fits.
        checks[current]={'max_absolute_prediction_difference':diff,'n':len(got)}
    # Conditional fixed-loss intervals, no selection-adjusted p-values.
    wide=p.pivot(index='date',columns='model',values='loss').sort_index();n=len(wide)
    rng=np.random.default_rng(20260919)
    starts=rng.integers(0,n-4+1,size=(9999,int(np.ceil(n/4))))
    boot=(starts[:,:,None]+np.arange(4)).reshape(9999,-1)[:,:n]
    intervals=[]
    for v in variants:
        for suffix in ['split','joint','ols']:
            m=v+'_'+suffix
            gm=wide.gemini_vector.to_numpy()[boot].mean(1)
            cm=wide[m].to_numpy()[boot].mean(1)
            gain=1-gm/cm;lo,hi=np.quantile(gain,[.025,.975])
            intervals.append(dict(comparator=m,gain=1-wide.gemini_vector.mean()/wide[m].mean(),conditional_lower=lo,conditional_upper=hi,draws=9999,block_length=4))
    pd.DataFrame(intervals).to_csv(R/'results/conditional_loss_intervals.csv',index=False)
    (R/'audit/prediction_reproduction_checks.json').write_text(json.dumps(checks,indent=2))
    print(summary[summary.period=='All evaluation'].to_string(index=False),flush=True)

if __name__=='__main__':evaluate()
