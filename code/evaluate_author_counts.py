"""Direct comparison with graph-recovered published human-coded counts.
The calendar overlap, initial 100 training labels and legacy fold boundaries are
fixed before looking at any losses against the newly recovered author counts.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np,pandas as pd
from evaluate_extension import ROOT,F,load,predict_fixed,norm
from sklearn.linear_model import LinearRegression

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--target",choices=["state_return","author_published_return_pct"],default="state_return");args=ap.parse_args()
    d,a,q,models=load();author=pd.read_csv(ROOT/'data/author_published_all_counts.csv').set_index('meeting_date').reindex(d.date)
    available=author.author_negative_all.notna().to_numpy()
    if args.target=='state_return':y=d.state_return.to_numpy(float)
    else:
        source=pd.read_csv(ROOT/'data/author_published_return_figure2.csv').set_index('date').reindex(d.date)
        y=source[args.target].to_numpy(float)
    y[~available]=np.nan
    prefix='author_overlap' if args.target=='state_return' else 'author_original_target'
    h=author[['author_negative_all','author_positive_all']].to_numpy(float)
    hx=author[['author_negative_excl_financial','author_positive_excl_financial']].to_numpy(float)
    quad=lambda z:np.column_stack([z,z[:,0]**2,z[:,1]**2,z[:,0]*z[:,1]])
    c=d[['document_sentence_count']].to_numpy(float)
    use={k:v for k,v in models.items() if k in ['gemini','flash_five','rules_whole_linear','rules_whole_quadratic','tfidf']}
    use['published_attention_counts']=(c,author[['author_total_attention']].to_numpy(float),True)
    use['published_human_counts']=(c,h,True)
    use['published_human_counts_quadratic']=(c,quad(h),True)
    # Excluding staff-financial discussion is a scope sensitivity, not a main comparator.
    use['published_human_counts_excl_financial']=(c,hx,True)
    folds=pd.read_csv(F/'data/folds.csv')
    pred,rep=predict_fixed(y,use,d,folds,prefix)
    ols=[]
    for f in folds.to_dict('records'):
        tr=np.flatnonzero(d.date.le(f['train_last'])&np.isfinite(y));te=np.flatnonzero(d.date.between(f['test_first'],f['test_last'])&np.isfinite(y))
        if not len(te):continue
        xx=np.column_stack([c,h]);mdl=LinearRegression().fit(xx[tr],y[tr]);pp=mdl.predict(xx[te])
        ols.extend({'model':'published_human_counts_ols','fold':f['fold'],'date':d.date.iloc[i],'y':y[i],'prediction':p,'training_mean':y[tr].mean(),'alpha':0.,'train_n':len(tr)} for i,p in zip(te,pp))
    pred=pd.concat([pred,pd.DataFrame(ols)],ignore_index=True);pred.to_csv(ROOT/f'results/{prefix}_predictions.csv',index=False)
    rows=[]
    for m,g in pred.groupby('model'):
        mse=np.mean((g.y-g.prediction)**2);rows.append({'model':m,'n':len(g),'mse':mse,'rmse':mse**.5,'mae':abs(g.y-g.prediction).mean(),'oos_r2':1-mse/np.mean((g.y-g.training_mean)**2)})
    r=pd.DataFrame(rows);den=r.set_index('model').loc['published_human_counts','mse'];r['gain_vs_published_human_pct']=100*(1-r.mse/den);r.to_csv(ROOT/f'results/{prefix}_performance.csv',index=False)
    w=pred.pivot(index='date',columns='model',values='prediction');yy=pred[pred.model.eq('gemini')].set_index('date').y.reindex(w.index);loss=w.subtract(yy,axis=0)**2
    cr=[]
    for aa,bb in [('gemini','published_human_counts'),('flash_five','published_human_counts'),('gemini','published_human_counts_quadratic'),('gemini','published_human_counts_ols'),('rules_whole_linear','published_human_counts')]:
        la,lb=loss[aa].to_numpy(),loss[bb].to_numpy();n=len(la);L=4;rng=np.random.default_rng(20260919);ind=(rng.integers(0,n-L+1,size=(9999,int(np.ceil(n/L))))[:,:,None]+np.arange(L)).reshape(9999,-1)[:,:n];gain=100*(1-la[ind].mean(1)/lb[ind].mean(1));lo,hi=np.quantile(gain,[.025,.975]);diff=lb-la;p=(1+np.sum(abs((diff-diff.mean())[ind].mean(1))>=abs(diff.mean())))/10000
        cr.append({'model':aa,'benchmark':bb,'n':n,'mse_gain_pct':100*(1-la.mean()/lb.mean()),'ci_low':lo,'ci_high':hi,'conditional_two_sided_p':p,'block_length':4,'scope':'new comparator on already explored overlap; not independent validation'})
    pd.DataFrame(cr).to_csv(ROOT/f'results/{prefix}_contrasts.csv',index=False)
    # Agreement diagnoses reconstruction fidelity, not human-label accuracy.
    ag=pd.DataFrame({'date':d.date,'author_negative':h[:,0],'author_positive':h[:,1],'reconstructed_negative':a['cvj'][:,0],'reconstructed_positive':a['cvj'][:,1]})[available]
    ag.to_csv(ROOT/'results/author_vs_reconstruction_counts.csv',index=False)
    print(r.round(5).to_string(index=False));print(pd.DataFrame(cr).round(5).to_string(index=False))
if __name__=='__main__':main()
