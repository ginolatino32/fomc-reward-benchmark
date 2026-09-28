"""Nested chronological ridge benchmarks, with training-only preprocessing.
Parallel jobs are Python workers, not autonomous language-model agents.
All predictions are exploratory retrospective evaluations, not pristine holdouts.
"""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='1'
from pathlib import Path
import json,time,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit
BASE=Path(__file__).resolve().parents[1]
ALPHAS=np.logspace(-4,4,9)
C=['document_sentence_count','stock_mentions_total','negative_stock_mentions','positive_stock_mentions']
A=['document_sentence_count','stock_mentions_total']
F=C+['log_sentence_count','log_stock_count','negative_share','positive_share','stock_mentions_total_per_sentence','negative_stock_mentions_per_sentence','positive_stock_mentions_per_sentence']
MODEL_BLOCKS={
 'attention_counts':['attention'], 'signed_counts':['counts'], 'flexible_counts':['flex'],
 'original_3anchors':['length','anchors3'], 'counts_3anchors':['counts','anchors3'],
 'counts_allanchors':['counts','anchorsall'], 'gemini_vector':['length','gemini'],
 'counts_gemini':['counts','gemini'], 'e5_vector':['length','e5'], 'counts_e5':['counts','e5'],
 'counts_tfidf_unigram':['counts','uni'], 'counts_tfidf_bigram':['counts','bi'],
 'counts_tfidf_gemini':['counts','bi','gemini']}
SELECTIONS={
 'selected_count_family':['attention_counts','signed_counts','flexible_counts'],
 'selected_lexical_family':['counts_tfidf_unigram','counts_tfidf_bigram'],
 'selected_embedding_family':['counts_3anchors','counts_allanchors','counts_gemini','counts_e5'],
 'selected_all_text_family':['counts_tfidf_unigram','counts_tfidf_bigram','counts_3anchors','counts_allanchors','counts_gemini','counts_e5','counts_tfidf_gemini']}

def centered_kernel(x,z):
 """Center on TRAINING mean; normalize each block to avg training norm^2=1."""
 k=np.asarray(x@x.T);q=np.asarray(z@x.T)
 col=k.mean(0);mu=float(k.mean())
 kc=k-col[None,:]-col[:,None]+mu
 qc=q-col[None,:]-q.mean(1)[:,None]+mu
 scale=max(float(np.trace(kc)/len(kc)),1e-15)
 return kc/scale,qc/scale

def block_kernels(d,arr,train,test,block):
 if block in ['uni','bi']:
  v=TfidfVectorizer(ngram_range=(1,1 if block=='uni' else 2),min_df=2,max_features=10000,
       sublinear_tf=True,lowercase=True,stop_words=None,dtype=np.float64)
  x=v.fit_transform(d.passage_text.iloc[train]);z=v.transform(d.passage_text.iloc[test])
  k=(x@x.T).toarray();q=(z@x.T).toarray();col=k.mean(0);mu=k.mean()
  kc=k-col[None,:]-col[:,None]+mu;qc=q-col[None,:]-q.mean(1)[:,None]+mu
  sc=max(float(np.trace(kc)/len(kc)),1e-15)
  return kc/sc,qc/sc
 if block in ['gemini','e5']:return centered_kernel(arr[block][train],arr[block][test])
 cols={'length':['document_sentence_count'],'attention':A,'counts':C,'flex':F,
  'anchors3':['smart_sim_policy_reaction','smart_sim_financial_conditions','smart_sim_wealth_effect_consumption'],
  'anchorsall':[c for c in d if c.startswith('smart_sim_')]}[block]
 x=d.iloc[train][cols].to_numpy(float);z=d.iloc[test][cols].to_numpy(float)
 med=np.nanmedian(x,axis=0);med=np.where(np.isfinite(med),med,0.)
 x=np.where(np.isfinite(x),x,med);z=np.where(np.isfinite(z),z,med)
 sd=x.std(0);sd=np.where(sd>1e-12,sd,1.)
 return centered_kernel(x/sd,z/sd)

def predict_grid(k,q,y):
 ev,U=eigh((k+k.T)/2);ev=np.maximum(ev,0.)
 yt=y-y.mean();coef=U.T@yt
 pred=y.mean()+(q@U)@(coef[:,None]/(ev[:,None]+ALPHAS[None,:]))
 edf=1+(ev[:,None]/(ev[:,None]+ALPHAS[None,:])).sum(0)
 return pred,edf

def worker(target):
 start=time.time();full=pd.read_csv(BASE/'data/analysis_panel.csv');v=np.load(BASE/'data/cached_representation_matrices.npz',allow_pickle=False)
 ix=np.flatnonzero(full[target].notna().to_numpy());d=full.iloc[ix].reset_index(drop=True)
 arr={k:v[k][ix].astype(float) for k in ['gemini','e5']}
 n=len(d);initial=25 if n<100 else 100;initial=min(initial,n//2+1)
 blocks=np.array_split(np.arange(initial,n),5)
 y=d[target].to_numpy(float);rows=[];tuning=[];audit=[]
 for fold,te in enumerate(blocks):
  tr=np.arange(te[0]);assert d.date.iloc[tr].max()<d.date.iloc[te].min()
  specs=list(TimeSeriesSplit(n_splits=3).split(tr))
  cache={};inner_losses={m:np.zeros(len(ALPHAS)) for m in MODEL_BLOCKS};denom=0
  for j,(it,iv) in enumerate(specs):
   for b in sorted({b for bs in MODEL_BLOCKS.values() for b in bs}):cache[(j,b)]=block_kernels(d,arr,it,iv,b)
   for m,bs in MODEL_BLOCKS.items():
    k=sum(cache[(j,b)][0] for b in bs)/len(bs);q=sum(cache[(j,b)][1] for b in bs)/len(bs)
    pr,_=predict_grid(k,q,y[it]);inner_losses[m]+=((y[iv,None]-pr)**2).sum(0)
   denom+=len(iv)
  for b in sorted({b for bs in MODEL_BLOCKS.values() for b in bs}):cache[('outer',b)]=block_kernels(d,arr,tr,te,b)
  preds={};metadata={}
  for m,bs in MODEL_BLOCKS.items():
   losses=inner_losses[m]/denom;best=int(np.argmin(losses));alpha=ALPHAS[best]
   k=sum(cache[('outer',b)][0] for b in bs)/len(bs);q=sum(cache[('outer',b)][1] for b in bs)/len(bs)
   pr,edf=predict_grid(k,q,y[tr]);preds[m]=pr[:,best]
   metadata[m]={'selected_model':m,'alpha':alpha,'edf':float(edf[best]),'inner_mse':float(losses[best])}
   for a,l in zip(ALPHAS,losses):tuning.append({'outcome':target,'fold':fold,'model':m,'alpha':a,'inner_mse':l})
  preds['historical_mean']=np.repeat(y[tr].mean(),len(te));metadata['historical_mean']={'selected_model':'historical_mean','alpha':np.nan,'edf':1,'inner_mse':np.nan}
  for sel,candidates in SELECTIONS.items():
   chosen=min(candidates,key=lambda m:metadata[m]['inner_mse']);preds[sel]=preds[chosen];metadata[sel]=metadata[chosen].copy()
  for m,pr in preds.items():
   for idx,pred in zip(te,pr):rows.append({'outcome':target,'fold':fold,'date':d.date.iloc[idx],'y':y[idx],
      'model':m,'prediction':float(pred),'train_n':len(tr),'train_max_date':d.date.iloc[tr].max(),**metadata[m]})
  audit.append({'outcome':target,'fold':fold,'training_n':len(tr),'test_n':len(te),'train_first':d.date.iloc[tr].min(),
       'train_last':d.date.iloc[tr].max(),'test_first':d.date.iloc[te].min(),'test_last':d.date.iloc[te].max()})
  print(target,'fold',fold,'finished',round(time.time()-start,1),flush=True)
 pd.DataFrame(rows).to_csv(BASE/f'results/predictions_{target}.csv',index=False)
 pd.DataFrame(tuning).to_csv(BASE/f'results/tuning_{target}.csv',index=False)
 pd.DataFrame(audit).to_csv(BASE/f'results/folds_{target}.csv',index=False)
 return {'outcome':target,'n':n,'evaluation_n':n-initial,'elapsed_seconds':time.time()-start,'worker_pid':os.getpid()}

if __name__=='__main__':
 tasks=json.loads((BASE/'docs/EXPLORATORY_RUN_PLAN.json').read_text())['objectives']
 if len(sys.argv)>1:tasks=sys.argv[1:]
 spec={'models':MODEL_BLOCKS,'selection_families':SELECTIONS,'alpha_grid':ALPHAS.tolist(),'created_before_outcomes_review':True}
 (BASE/'docs/BENCHMARK_SPECIFICATION.json').write_text(json.dumps(spec,indent=2))
 done=[]
 with ProcessPoolExecutor(max_workers=4) as ex:
  fs={ex.submit(worker,t):t for t in tasks}
  for f in as_completed(fs):
   result=f.result();done.append(result);print('COMPLETED',result,flush=True)
   (BASE/'logs/benchmark_workers.json').write_text(json.dumps(done,indent=2))
