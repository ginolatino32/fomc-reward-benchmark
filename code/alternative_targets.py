"""Trading-day price-return sensitivities from the retained Yahoo and FRED caches.
These are NOT newly downloaded total-return data. The separately downloaded
original-author graph provides an independent-source criterion for 1994-2016.
"""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'source/focused/FOMC_LLM_Focused_Paper';M=ROOT/'source/market'
def main():
 p=pd.read_csv(F/'data/analysis_panel.csv');meetings=pd.DatetimeIndex(['2000-02-02']+p.date.tolist())
 raw=pd.read_csv(M/'yahoo_gspc_daily.csv',parse_dates=['date']).set_index('date').sort_index();assert raw.index.is_unique
 raw['daily_log_pct']=100*np.log(raw.sp500/raw.sp500.shift(1));raw['previous_date']=pd.Series(raw.index,index=raw.index).shift(1)
 ff=pd.read_csv(M/'fred_federal_funds_target_range.csv',parse_dates=['date']);moves=ff.loc[abs(ff.ffr_daily_change_bp.fillna(0))>1e-6,'date'];nonmeeting=set(moves)-set(meetings)
 # These are effective-date removals, not verified announcement-date exclusions.
 # Preserve the paper's exceptional inclusion of September 17, 2001.
 nonmeeting.discard(pd.Timestamp('2001-09-17'))
 records=[];ledger=[]
 for i,date in enumerate(meetings[1:],start=1):
  start=meetings[i-1]+pd.Timedelta(days=1);end=date-pd.Timedelta(days=2);g=raw.loc[start:end].dropna(subset=['daily_log_pct']);assert len(g)>0
  clean=g.loc[~g.index.isin(nonmeeting)]
  records.append({'date':date.strftime('%Y-%m-%d'),'price_log_inclusive':g.daily_log_pct.sum(),'price_log_excl_effective_policy':clean.daily_log_pct.sum(),'n_trading_returns':len(g),'n_effective_policy_days_removed':len(g)-len(clean),'first_return_day':str(g.index.min().date()),'last_return_day':str(g.index.max().date()),'prior_close_day':str(g.previous_date.iloc[0].date())})
  for day,r in g.iterrows():ledger.append({'meeting_date':date.strftime('%Y-%m-%d'),'return_date':str(day.date()),'log_price_return_pct':r.daily_log_pct,'excluded_by_effective_date_sensitivity':day in nonmeeting})
 out=pd.DataFrame(records);au=pd.read_csv(ROOT/'data/author_published_return_figure2.csv');out=out.merge(au[['date','author_published_return_pct']],on='date',how='left',validate='one_to_one');out.to_csv(ROOT/'data/alternative_return_targets.csv',index=False);pd.DataFrame(ledger).to_csv(ROOT/'data/alternative_return_day_ledger.csv',index=False)
 # Independent-vendor-cache agreement on an overlapping interval, not a new download.
 fr=pd.read_csv(M/'SP500.csv');fr['date']=pd.to_datetime(fr.observation_date);fr['sp500_fred']=pd.to_numeric(fr.SP500,errors='coerce');j=raw.reset_index().merge(fr[['date','sp500_fred']],on='date').dropna(subset=['sp500_fred']);delta=j.sp500-j.sp500_fred
 j['close_difference']=delta;j.to_csv(ROOT/'results/yahoo_fred_overlap_check.csv',index=False)
 corr=p[['date','state_return']].merge(out,on='date').set_index('date')[['state_return','price_log_inclusive','price_log_excl_effective_policy','author_published_return_pct']].corr();corr.to_csv(ROOT/'results/return_definition_correlations.csv')
 audit={'sources':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in M.glob('*.csv')},'status':'retained source caches; no new daily French download claimed','daily_price_source':'Yahoo S&P 500 close','gross_log_return':'sum of observed close-to-close log returns whose return dates lie in [previous_meeting+1 calendar day, current_meeting-2 calendar days]','dividends_included':False,'risk_free_adjustment_in_gross_targets':False,'policy_exclusion':'secondary sensitivity removes non-meeting dates with a recorded target change; effective dates may differ from announcement dates; retains2001-09-17','n_targets':len(out),'yahoo_fred_overlap':{'n':len(j),'first':str(j.date.min().date()),'last':str(j.date.max().date()),'max_abs_close_difference':float(abs(delta).max()),'median_abs_close_difference':float(abs(delta).median()),'n_differences_over_one_cent':int((abs(delta)>.01).sum())},'author_graph_target_n':int(out.author_published_return_pct.notna().sum())}
 (ROOT/'results/alternative_target_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2));print(corr)
if __name__=='__main__':main()
