from pathlib import Path
import numpy as np,pandas as pd,json
from scipy import stats
ROOT=Path(__file__).resolve().parents[1];SEED=20260919
p=pd.read_csv(ROOT/'results/state_return_predictions.csv');wide=p.pivot(index='date',columns='model',values='prediction').sort_index();obs=p[p.model.eq('gemini')].set_index('date').loc[wide.index];y=obs.y;loss=wide.subtract(y,axis=0)**2
pairs=[('gemini','rules_whole_linear'),('gemini','rules_whole_quadratic'),('gemini','rules_matched_phrase'),('flash_five','rules_whole_quadratic'),('flash_two_plus_volume','rules_two_votes_plus_volume'),('gemini_plus_volume','rules_matched_plus_volume')]
rows=[]
for a,b in pairs:
    la,lb=loss[a].to_numpy(),loss[b].to_numpy();n=len(la)
    for L in [4,2,8]:
        rng=np.random.default_rng(SEED);starts=rng.integers(0,n-L+1,size=(9999,int(np.ceil(n/L))));ind=(starts[:,:,None]+np.arange(L)).reshape(9999,-1)[:,:n]
        ratio=100*(1-la[ind].mean(1)/lb[ind].mean(1));d=lb-la;centered=(d-d.mean())[ind].mean(1)
        tail=(1+np.sum(abs(centered)>=abs(d.mean())))/10000
        lo,hi=np.quantile(ratio,[.025,.975]);rows.append(dict(model=a,benchmark=b,n=n,block_length=L,mse_gain_pct=100*(1-la.mean()/lb.mean()),ci_low=lo,ci_high=hi,conditional_two_sided_p=tail,scope='post-discovery fixed loss comparison; not research-selection adjusted'))
t=pd.DataFrame(rows);mask=t.block_length.eq(4);pv=t.loc[mask,'conditional_two_sided_p'].to_numpy();ix=np.argsort(pv);adj=np.minimum(1,np.maximum.accumulate(pv[ix]*np.arange(len(ix),0,-1)));res=np.empty(len(ix));res[ix]=adj;t.loc[mask,'holm_six']=res
t.to_csv(ROOT/'results/new_paired_contrasts.csv',index=False)
# A training-selected representation-family comparison; no use of outer losses.
tune=pd.read_csv(ROOT/'results/state_return_tuning.csv');families={'lexical_family':['rules_whole_linear','rules_whole_quadratic','rules_matched_phrase','rules_matched_quadratic','rules_matched_four_votes','tfidf'], 'llm_family':['gemini','flash_five']}
sel=[];selp=[]
for fam,models in families.items():
 for f in sorted(tune.fold.unique()):
    cand=tune[tune.model.isin(models)&tune.fold.eq(f)&tune.selected].sort_values(['validation_mse','model']);best=cand.iloc[0]
    sel.append({'family':fam,'fold':f,'chosen_model':best.model,'validation_mse':best.validation_mse,'alpha':best.alpha,'scope':'training-only selection; family defined after historical research'})
    q=p[p.model.eq(best.model)&p.fold.eq(f)].copy();q['selected_from']=q['model'];q['model']=fam;selp.append(q)
sp=pd.concat(selp);sp.to_csv(ROOT/'results/family_selected_predictions.csv',index=False);pd.DataFrame(sel).to_csv(ROOT/'results/family_selections.csv',index=False)
rr=[]
for fam,g in sp.groupby('model'):
 mse=float(((g.y-g.prediction)**2).mean());rr.append({'family':fam,'n':len(g),'mse':mse,'rmse':mse**.5,'mae':abs(g.y-g.prediction).mean(),'oos_r2':1-mse/((g.y-g.training_mean)**2).mean()})
pd.DataFrame(rr).to_csv(ROOT/'results/family_selected_performance.csv',index=False)
# Direct reproducibility checks of the independent spectrum tuning implementation.
ref=pd.read_csv(ROOT/'source/focused/FOMC_LLM_Focused_Paper/results/predictions.csv');names={'gemini':'gemini','flash_five':'flash','rules_whole_linear':'cvj','tfidf':'tfidf'};checks=[]
for k,v in names.items():
 a=p[p.model.eq(k)].merge(ref[ref.model.eq(v)],on='date',suffixes=('_new','_old'),validate='one_to_one');err=float(abs(a.prediction_new-a.prediction_old).max());checks.append({'model':k,'n':len(a),'max_prediction_delta':err,'passed':bool(err<1e-8)})
(ROOT/'results/numerical_verification.json').write_text(json.dumps(checks,indent=2));assert all(x['passed'] for x in checks)
print(t[t.block_length.eq(4)].round(4).to_string(index=False));print(pd.DataFrame(rr).round(4).to_string(index=False));print(pd.DataFrame(sel).to_string(index=False))
