"""Integrity checks; passing is not a claim of semantic accuracy or acceptance."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'source/focused/FOMC_LLM_Focused_Paper'
def main():
 checks=[]
 def check(name,condition,detail=''):
  checks.append({'check':name,'passed':bool(condition),'detail':str(detail)})
 d=pd.read_csv(F/'data/analysis_panel.csv');a=np.load(F/'data/representations.npz',allow_pickle=False)
 check('199 labels, exact vector-date alignment',len(d)==199 and np.array_equal(d.date.to_numpy(),a['dates']))
 check('full 3072-dimensional retained representation',a['gemini'].shape==(199,3072) and np.isfinite(a['gemini']).all())
 m=pd.read_csv(ROOT/'data/matched_passage_counts.csv');check('1197 unique matched passages',len(m)==1197 and m.record_id.is_unique)
 q=pd.read_csv(ROOT/'data/additional_meeting_features.csv');check('200 meetings and all five label counts sum to passage volume',len(q)==200 and q.passage_count.sum()==1197 and np.array_equal(q[['flash_negative','flash_positive','flash_neutral','flash_hypothetical','flash_unclear']].sum(axis=1),q.passage_count))
 c=pd.read_csv(ROOT/'data/author_published_all_counts.csv');r=pd.read_csv(ROOT/'data/author_published_return_figure2.csv');j=c.merge(r,left_on='meeting_date',right_on='date',validate='one_to_one')
 check('184 unique source observations',len(j)==184 and j.date.is_unique)
 check('original totals independently match published exhibit',c.author_total_attention.sum()==975 and c.author_negative_all.sum()==322 and c.author_positive_all.sum()==414)
 check('all negative counts match independently dated scatter',np.array_equal(j.author_negative_all,j.author_negative_scatter))
 check('all positive counts match independently dated scatter',np.array_equal(j.author_positive_all,j.author_positive_scatter))
 check('directional counts bounded by attention',((c.author_negative_all+c.author_positive_all)<=c.author_total_attention).all())
 check('restricted tone counts bounded by full counts',(c.author_negative_excl_financial<=c.author_negative_all).all() and (c.author_positive_excl_financial<=c.author_positive_all).all())
 check('return panels agree within 0.01 percentage point',abs(r.author_return_negative_panel_pct-r.author_return_positive_panel_pct).max()<.01)
 check('source-return mean is the two-panel average',np.max(abs(r.author_published_return_pct-(r.author_return_negative_panel_pct+r.author_return_positive_panel_pct)/2))<1e-12)
 for prefix,n in [('state_return',99),('author_original_target',35),('author_overlap',35),('author_earlier_start',75),('price_log_inclusive',99),('price_log_excl_effective_policy',99)]:
  p=pd.read_csv(ROOT/f'results/{prefix}_predictions.csv');s=pd.read_csv(ROOT/f'results/{prefix}_performance.csv').set_index('model');ok=True;err=0
  for model,g in p.groupby('model'):
   mse=float(np.square(g.y-g.prediction).mean());den=float(np.square(g.y-g.training_mean).mean());err=max(err,abs(mse-s.loc[model,'mse']),abs(np.sqrt(mse)-s.loc[model,'rmse']),abs(1-mse/den-s.loc[model,'oos_r2']))
   ok=ok and len(g)==n and g.date.is_unique and np.isfinite(g[['y','prediction']]).all().all()
  check(prefix+' coverage and finite predictions',ok)
  check(prefix+' metrics independently recomputed',err<1e-9,err)
 for row in json.loads((ROOT/'results/numerical_verification.json').read_text()):check('baseline refit: '+row['model'],row['passed'],row['max_prediction_delta'])
 z=pd.read_csv(ROOT/'results/graph_precision_reference_checks.csv');check('vectorized diagnostic refit matches reference',z.max_abs_prediction_difference.max()<1e-7,z.max_abs_prediction_difference.max())
 z=pd.read_csv(ROOT/'results/graph_precision_performance.csv');check('all 103 declared source-resolution variants retained',z.groupby('model').target_variant.nunique().eq(103).all())
 for name in ['EXTENSION_PLAN.json','AUTHOR_COMPARATOR_PLAN.json','ADDITIONAL_SOURCE_DIAGNOSTICS.json']:check('retained extension record: '+name,(ROOT/'docs'/name).exists())
 result={'checks':checks,'passed':sum(x['passed'] for x in checks),'total':len(checks),'scope':'computational and source-integrity checks only; not semantic validation, no independent holdout created'}
 (ROOT/'results/final_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 if result['passed']!=result['total']:raise SystemExit(1)
if __name__=='__main__':main()
