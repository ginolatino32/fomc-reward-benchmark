"""Assemble manuscript tables without altering observations or fitted predictions."""
from pathlib import Path
import json
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'source/focused/FOMC_LLM_Focused_Paper'
OUT=ROOT/'results';T=ROOT/'manuscript/tables';T.mkdir(parents=True,exist_ok=True)
NAMES={'rules_whole_linear':'Reconstructed algorithm: linear','rules_whole_quadratic':'Reconstructed algorithm: quadratic','tfidf':'TF-IDF unigrams/bigrams','gemini':'Language-model embedding','flash_five':'LLM-classified tone counts','published_attention_counts':'Original mention frequencies','published_human_counts':'Original human tone: linear','published_human_counts_quadratic':'Original human tone: quadratic','published_human_counts_excl_financial':'Original human tone: excluding staff financial review','published_human_counts_ols':'Original human tone: OLS','rules_matched_phrase':'Rules: matched-passage phrase counts','rules_matched_plus_volume':'Rules: matched phrases + passage volume','rules_two_votes_plus_volume':'Rules: two passage votes + volume','flash_two_plus_volume':'LLM: two passage categories + volume','gemini_plus_volume':'Embedding + passage volume','rules_matched_four_votes':'Rules: four passage vote categories'}
def get(stem):return pd.read_csv(OUT/f'{stem}_performance.csv').set_index('model')
def rows(tab,keys):
    r=tab.loc[keys].copy().reset_index();r.insert(1,'representation',r.model.map(NAMES));return r

def paired(file,prefix):
    pred=pd.read_csv(OUT/file);w=pred.pivot(index='date',columns='model',values='prediction');y=pred[pred.model.eq('gemini')].set_index('date').y.reindex(w.index)
    loss=w.subtract(y,axis=0)**2;n=len(w);rng=np.random.default_rng(20260919);L=4
    idx=(rng.integers(0,n-L+1,size=(9999,int(np.ceil(n/L))))[:,:,None]+np.arange(L)).reshape(9999,-1)[:,:n]
    result=[]
    for a in ['gemini','flash_five']:
        for b in ['published_attention_counts','published_human_counts','published_human_counts_quadratic','published_human_counts_excl_financial']:
            la,lb=loss[a].to_numpy(),loss[b].to_numpy();v=100*(1-la[idx].mean(1)/lb[idx].mean(1));diff=lb-la
            result.append({'model':a,'comparator':b,'n':n,'mse_gain_pct':100*(1-la.mean()/lb.mean()),'ci_low':np.quantile(v,.025),'ci_high':np.quantile(v,.975),'raw_conditional_two_sided_p':(1+(np.abs((diff-diff.mean())[idx].mean(1))>=abs(diff.mean())).sum())/10000,'scope':'post-discovery comparison, conditional on saved losses; no independent confirmatory interpretation'})
    r=pd.DataFrame(result);r.to_csv(OUT/f'{prefix}_all_contrasts.csv',index=False);return r

def main():
    full=get('state_return');src=get('author_original_target');old=get('author_overlap');early=get('author_earlier_start')
    r=rows(full,['tfidf','rules_whole_linear','rules_whole_quadratic','gemini','flash_five']);r['gain_vs_linear_rules_pct']=100*(1-r.mse/full.loc['rules_whole_linear','mse']);r['gain_vs_quadratic_rules_pct']=100*(1-r.mse/full.loc['rules_whole_quadratic','mse']);r.to_csv(T/'table1_full_period.csv',index=False)
    r=rows(src,['published_attention_counts','published_human_counts','published_human_counts_ols','published_human_counts_quadratic','published_human_counts_excl_financial','gemini','flash_five']);r['gain_vs_human_linear_pct']=100*(1-r.mse/src.loc['published_human_counts','mse']);r.to_csv(T/'table2_original_source.csv',index=False)
    r=rows(full,['rules_matched_phrase','rules_matched_plus_volume','gemini_plus_volume','rules_two_votes_plus_volume','flash_two_plus_volume','rules_matched_four_votes']);r.to_csv(T/'table3_matched_inputs.csv',index=False)
    out=[]
    for key,label in [('state_return','Inherited excess price-return proxy'),('price_log_inclusive','Inclusive daily log price return'),('price_log_excl_effective_policy','Price return excluding effective-change days')]:
        z=get(key);out.append({'target':key,'description':label,'n':int(z.loc['gemini','n']),'rules_linear_rmse':z.loc['rules_whole_linear','rmse'],'rules_quadratic_rmse':z.loc['rules_whole_quadratic','rmse'],'embedding_rmse':z.loc['gemini','rmse'],'tone_rmse':z.loc['flash_five','rmse'],'embedding_gain_vs_quadratic_pct':100*(1-z.loc['gemini','mse']/z.loc['rules_whole_quadratic','mse']),'tone_gain_vs_quadratic_pct':100*(1-z.loc['flash_five','mse']/z.loc['rules_whole_quadratic','mse'])})
    pd.DataFrame(out).to_csv(T/'table4_target_sensitivity.csv',index=False)
    out=[]
    for key in ['published_attention_counts','published_human_counts','published_human_counts_quadratic','published_human_counts_excl_financial','gemini','flash_five']:
        out.append({'model':key,'representation':NAMES[key],'primary_eval_n':35,'primary_rmse':src.loc[key,'rmse'],'primary_r2':src.loc[key,'oos_r2'],'earlier_eval_n':75,'earlier_rmse':early.loc[key,'rmse'],'earlier_r2':early.loc[key,'oos_r2']})
    pd.DataFrame(out).to_csv(T/'table5_initialization_sensitivity.csv',index=False)
    paired('author_original_target_predictions.csv','author_original_target');paired('author_earlier_start_predictions.csv','author_earlier_start')
    r=rows(old,['published_human_counts','published_human_counts_quadratic','published_human_counts_excl_financial','gemini','flash_five']);r.to_csv(T/'tableS_original_counts_old_proxy.csv',index=False)
    z=pd.read_csv(OUT/'author_vs_reconstruction_counts.csv');out=[]
    for s in ['negative','positive']:
        x,y=z['author_'+s],z['reconstructed_'+s];out.append({'tone':s,'n':len(z),'original_mean':x.mean(),'reconstruction_mean':y.mean(),'correlation':x.corr(y),'mean_absolute_count_difference':abs(x-y).mean(),'exact_meeting_count_agreements':int((x==y).sum()),'scope':'Different coding and document scope; not a classification-accuracy estimate.'})
    pd.DataFrame(out).to_csv(OUT/'reconstruction_vs_original_summary.csv',index=False)
    print('Assembled five main tables and source comparison diagnostics.')
if __name__=='__main__':main()
