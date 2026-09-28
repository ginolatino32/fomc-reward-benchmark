"""Numerical validation of recovered published data and one earlier-start sensitivity.
No new model generation. Every variation is reported; none replaces the legacy
100-observation initialization or selects a preferred reported result.
"""
from pathlib import Path
import json, os
from datetime import datetime, timezone
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit
from scipy.linalg import eigh
from evaluate_extension import ROOT,F,load,norm,predict_fixed,ALPHAS

def inputs():
    d,a,q,models=load()
    ct=pd.read_csv(ROOT/'data/author_published_all_counts.csv').set_index('meeting_date').reindex(d.date)
    rt=pd.read_csv(ROOT/'data/author_published_return_figure2.csv').set_index('date').reindex(d.date)
    control=d[['document_sentence_count']].to_numpy(float)
    h=ct[['author_negative_all','author_positive_all']].to_numpy(float)
    hx=ct[['author_negative_excl_financial','author_positive_excl_financial']].to_numpy(float)
    use={k:v for k,v in models.items() if k in ['gemini','flash_five','rules_whole_linear','rules_whole_quadratic']}
    use.update({'published_attention_counts':(control,ct[['author_total_attention']].to_numpy(float),True),
                'published_human_counts':(control,h,True),
                'published_human_counts_quadratic':(control,np.column_stack([h,h[:,0]**2,h[:,1]**2,h[:,0]*h[:,1]]),True),
                'published_human_counts_excl_financial':(control,hx,True)})
    return d,rt,use

def graph_precision():
    d,rt,use=inputs();base=rt.author_published_return_pct.to_numpy(float)
    rng=np.random.default_rng(20260919)
    ys=np.column_stack([base,rt.author_return_negative_panel_pct.to_numpy(float),rt.author_return_positive_panel_pct.to_numpy(float),base[:,None]+rng.uniform(-.01,.01,size=(len(d),100))])
    names=['two_panel_mean','negative_panel','positive_panel']+[f'jitter_{i:03d}' for i in range(100)]
    folds=pd.read_csv(F/'data/folds.csv');records=[];prediction_checks=[]
    available=np.isfinite(base)
    for name,(c,rep,num) in use.items():
        pp=[];yy=[];means=[];alls=[];dates=[]
        for f in folds.to_dict('records'):
            tr=np.flatnonzero(d.date.le(f['train_last'])&available);te=np.flatnonzero(d.date.between(f['test_first'],f['test_last'])&available)
            if not len(te):continue
            def fe(u,v):
                x,z=norm(c[u],c[v],True);ex,ez=norm(rep[u],rep[v],num)
                return np.column_stack([x,ex])/np.sqrt(2),np.column_stack([z,ez])/np.sqrt(2)
            scores=np.zeros((len(ALPHAS),ys.shape[1]))
            for tu,vu in TimeSeriesSplit(n_splits=3).split(tr):
                u,v=tr[tu],tr[vu];x,z=fe(u,v);val,V=eigh(x@x.T);val=np.maximum(val,0);mu=np.mean(ys[u],axis=0)
                w=V.T@(ys[u]-mu);cross=(z@x.T)@V
                for ia,alpha in enumerate(ALPHAS):
                    pred=cross@(w/(val[:,None]+alpha))+mu
                    scores[ia]+=np.square(ys[v]-pred).sum(0)
            al=ALPHAS[np.argmin(scores,axis=0)];x,z=fe(tr,te);val,V=eigh(x@x.T);val=np.maximum(val,0)
            mu=ys[tr].mean(0);pred=((z@x.T)@V)@((V.T@(ys[tr]-mu))/(val[:,None]+al[None,:]))+mu
            pp.append(pred);yy.append(ys[te]);means.append(np.tile(mu,(len(te),1)));alls.append(al);dates.extend(d.date.iloc[te])
        pp=np.vstack(pp);yy=np.vstack(yy);means=np.vstack(means)
        for j,nm in enumerate(names):
            mse=np.mean((yy[:,j]-pp[:,j])**2)
            records.append({'model':name,'target_variant':nm,'n':len(pp),'mse':mse,'rmse':np.sqrt(mse),'mae':np.mean(abs(yy[:,j]-pp[:,j])),'oos_r2':1-mse/np.mean((yy[:,j]-means[:,j])**2),'selected_alphas':json.dumps([float(x[j]) for x in alls])})
        ref=pd.read_csv(ROOT/'results/author_original_target_predictions.csv');ref=ref[ref.model.eq(name)].set_index('date').loc[dates]
        prediction_checks.append({'model':name,'max_abs_prediction_difference':float(max(abs(pp[:,0]-ref.prediction.to_numpy())))})
    records=pd.DataFrame(records);records.to_csv(ROOT/'results/graph_precision_performance.csv',index=False)
    wide=records.pivot(index='target_variant',columns='model',values='mse');gains=[]
    for m in ['gemini','flash_five']:
        for b in ['published_attention_counts','published_human_counts','published_human_counts_quadratic','published_human_counts_excl_financial']:
            v=100*(1-wide[m]/wide[b]);gains.append({'model':m,'comparator':b,'original_gain_pct':v.loc['two_panel_mean'],'minimum_gain_across_103_variants_pct':v.min(),'maximum_gain_across_103_variants_pct':v.max(),'max_absolute_change_pp':max(abs(v-v.loc['two_panel_mean']))})
    pd.DataFrame(gains).to_csv(ROOT/'results/graph_precision_gain_ranges.csv',index=False)
    pd.DataFrame(prediction_checks).to_csv(ROOT/'results/graph_precision_reference_checks.csv',index=False)
    assert max(x['max_abs_prediction_difference'] for x in prediction_checks)<1e-7
    print(pd.DataFrame(gains).to_string(index=False))

def earlier_start():
    d,rt,use=inputs();y=rt.author_published_return_pct.to_numpy(float);valid=np.flatnonzero(np.isfinite(y));fold=[]
    for k,start in enumerate(range(60,len(valid),20)):
        tr=valid[:start];te=valid[start:start+20]
        fold.append({'fold':k,'train_last':d.date.iloc[tr[-1]],'test_first':d.date.iloc[te[0]],'test_last':d.date.iloc[te[-1]],'train_n':len(tr),'test_n':len(te)})
    folds=pd.DataFrame(fold);folds.to_csv(ROOT/'data/author_earlier_start_folds.csv',index=False)
    _,r=predict_fixed(y,use,d,folds,'author_earlier_start')
    print(r.round(5).to_string())

def original_regression_sanity():
    r=pd.read_csv(ROOT/'data/author_published_return_figure2.csv').merge(pd.read_csv(ROOT/'data/author_published_all_counts.csv'),left_on='date',right_on='meeting_date')
    x=pd.DataFrame({f'rx{i}':r.author_published_return_pct.shift(i)/100 for i in range(3)});good=x.notna().all(axis=1);out=[]
    for label in ['negative','positive']:
        fit=sm.OLS(r.loc[good,f'author_{label}_all'],sm.add_constant(x[good])).fit(cov_type='HAC',cov_kwds={'maxlags':4})
        row={'series':label,'n':int(fit.nobs),'r2':fit.rsquared,'scope':'182 observations; first two pre-1994 return lags unavailable; not exact 184-observation replication'}
        for key in fit.params.index:row[key]=float(fit.params[key]);row[f't_{key}']=float(fit.tvalues[key])
        out.append(row)
    pd.DataFrame(out).to_csv(ROOT/'results/published_regression_sanity.csv',index=False)

def main():
    plan=ROOT/'docs/ADDITIONAL_SOURCE_DIAGNOSTICS.json'
    if not plan.exists():plan.write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Post-discovery diagnostics; source-overlap results with 100 initial meetings already observed. No preferred result is selected.','graph_precision':'Refit tuning for two individual source panels plus 100 fixed uniform +/-0.01 percentage-point perturbations of all recovered returns; retain original mean as reference.','earlier_start':'One fixed initial training length of60 and blocks of20 to use more of original source overlap. This is a sensitivity, not a new holdout. Original100 initialization remains primary.','original_regression':'Sanity-check published count-on-return coefficients with two lags; unavailable pre-1994 lags reduce N to182.'},indent=2))
    graph_precision();earlier_start();original_regression_sanity()
if __name__=='__main__':main()
