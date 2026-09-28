"""Recover date-labelled returns and counts from the authors' vector scatterplot.
Independent source: author-hosted June 2020 appendix, Figure 2, PDF page 3.
No OCR, imputed historical returns, manual sentiment coding, or API inference.
"""
from pathlib import Path
import re,json,hashlib
import numpy as np,pandas as pd,fitz
ROOT=Path(__file__).resolve().parents[1]
PDF=ROOT/'external/FedPut_onlineappendix_rfs2.pdf'
def main():
 assert hashlib.sha256(PDF.read_bytes()).hexdigest() == '025cb24a1fe23d36b3b618e1ced859596f5951324f63d5205bf8c763a126db46', 'Unexpected source PDF version; review calibration before proceeding'
 p=fitz.open(PDF)[2];dd=p.get_drawings();tr=p.get_texttrace()
 markers={d['seqno']:d for d in dd if d['color']==(0.,0.,1.) and len(d['items'])==4}
 assert len(markers)==368
 dates=[]
 for t in tr:
  label=''.join(chr(c[0]) for c in t['chars'])
  if not re.fullmatch(r'\d{2}[a-z]{3}\d{4}',label):continue
  d=markers[t['seqno']-1];r=d['rect'];x=(r.x0+r.x1)/2;y=(r.y0+r.y1)/2
  dates.append({'date':pd.to_datetime(label,format='%d%b%Y').strftime('%Y-%m-%d'),'panel':'negative' if x<330 else 'positive','x':x,'y':y,'drawing_seqno':d['seqno'],'label_seqno':t['seqno'],'label':label,'label_gap':t['bbox'][0]-x})
 df=pd.DataFrame(dates);assert len(df)==368
 audit={}
 for panel,left,right in [('negative',100,320),('positive',340,565)]:
  ticks=[]
  for d in dd:
   if d['color']!=(0.,0.,0.) or len(d['items'])!=1:continue
   c=d['items'][0]
   if c[0]!='l':continue
   a,b=c[1:]
   if abs(a.x-b.x)<1e-5 and 413<a.y<414 and 416<b.y<417 and left<a.x<right:ticks.append(a.x)
  ticks=np.array(sorted(ticks));assert len(ticks)==5
  xx=np.array([-30,-20,-10,0,10],float);slope,intercept=np.polyfit(xx,ticks,1)
  grid=[]
  for d in dd:
   if d['color'] is None or d['color'][0]<.9 or len(d['items'])!=1:continue
   c=d['items'][0]
   if c[0]=='l':
    a,b=c[1:]
    if abs(a.y-b.y)<1e-5 and b.x-a.x>200 and left<a.x<right and 300<a.y<410:grid.append(a.y)
  grid=np.array(sorted(grid));assert len(grid)==3
  zero=grid[-1];unit=(grid[-1]-grid[0])/10
  ix=df.panel.eq(panel);df.loc[ix,'return_recovered_pct']=(df.loc[ix,'x']-intercept)/slope;ct=(zero-df.loc[ix,'y'])/unit
  df.loc[ix,'count_recovered']=np.rint(ct).astype(int)
  assert max(abs(ct-np.rint(ct)))<.005
  audit[panel]={'axis_ticks_pdf':ticks.tolist(),'slope_pdf_points_per_return_percentage':slope,'intercept':intercept,'tick_fit_max_residual':float(max(abs(ticks-(slope*xx+intercept)))),'count_rounding_max_residual':float(max(abs(ct-np.rint(ct))))}
 a=df[df.panel.eq('negative')].set_index('date').sort_index();b=df[df.panel.eq('positive')].set_index('date').sort_index();assert a.index.equals(b.index)
 out=pd.DataFrame({'date':a.index,'author_return_negative_panel_pct':a.return_recovered_pct.to_numpy(),'author_return_positive_panel_pct':b.return_recovered_pct.to_numpy(),'author_negative_scatter':a.count_recovered.to_numpy(int),'author_positive_scatter':b.count_recovered.to_numpy(int)})
 out['author_published_return_pct']=(out.author_return_negative_panel_pct+out.author_return_positive_panel_pct)/2
 diff=abs(out.author_return_negative_panel_pct-out.author_return_positive_panel_pct);assert diff.max()<.01
 orig=pd.read_csv(ROOT/'data/author_published_counts_figure5.csv');join=out.merge(orig,left_on='date',right_on='meeting_date',validate='one_to_one');assert len(join)==184
 assert np.array_equal(join.author_negative_scatter,join.author_negative_all) and np.array_equal(join.author_positive_scatter,join.author_positive_all)
 df.to_csv(ROOT/'data/appendix_scatter_marker_audit.csv',index=False);out.to_csv(ROOT/'data/author_published_return_figure2.csv',index=False)
 audit.update({'records':184,'max_two_panel_return_difference_pct':float(diff.max()),'negative_counts_equal_independent_figure5':True,'positive_counts_equal_independent_figure5':True,'source':{'file':PDF.name,'sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),'source_url':'https://drive.google.com/file/d/1YA1jNNnWax_E8-QHpZwse7wOYjadf0JN/view','figure':2,'pdf_page':3,'version':'June 2020 author-hosted appendix','method':'read named-date text operators and their immediately preceding circle vectors; calibrate tick locations; average two independently recovered panel returns','precision_note':'graph-recovered approximate percentages, not exact author raw returns; use conservative 0.01 percentage-point perturbation diagnostic'},'descriptive':{'mean':float(out.author_published_return_pct.mean()),'sd':float(out.author_published_return_pct.std()),'min':float(out.author_published_return_pct.min()),'max':float(out.author_published_return_pct.max())}})
 (ROOT/'results/return_recovery_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2));print(out.head().to_string(index=False))
if __name__=='__main__':main()
