"""Recover the actual published unclassified stock-phrase frequency, Figure4A.
The time axis uses year+month/12. Only the 184 post-1994 vertices with known
scheduled meeting months are used. No human labels are created.
"""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,fitz
from recover_published_counts import points
ROOT=Path(__file__).resolve().parents[1];PDF=ROOT/'source/w26894.pdf'
def main():
 assert hashlib.sha256(PDF.read_bytes()).hexdigest() == 'dbc3380b1463b0e72b6360a4e46cdc39658cfdf83606b9d010ebde0377d0bb88', 'Unexpected source PDF version; review calibration before proceeding'
 p=fitz.open(PDF)[48];draw=p.get_drawings();cand=[(i,d) for i,d in enumerate(draw) if len(d['items'])>300 and d['rect'].y1<300];assert len(cand)==1
 idx,d=cand[0];xy=points(d)[-184:];out=pd.read_csv(ROOT/'data/author_published_counts_figure5.csv');date=pd.to_datetime(out.meeting_date);months=date.dt.year+date.dt.month/12
 slope,intercept=np.polyfit(months,xy[:,0],1);res=abs(xy[:,0]-(slope*months+intercept));assert max(res)<.02
 # y ticks 0,5,...25 independently calibrate frequency.
 yticks=[]
 for dd in draw:
  if dd['color']!=(0.,0.,0.) or len(dd['items'])!=1:continue
  c=dd['items'][0]
  if c[0]=='l':
   a,b=c[1:]
   if abs(a.y-b.y)<1e-6 and 3<abs(a.x-b.x)<4 and 80<a.y<270:yticks.append(a.y)
 yticks=np.array(sorted(yticks,reverse=True));assert len(yticks)==6
 ys,yi=np.polyfit(np.arange(0,26,5),yticks,1);val=(xy[:,1]-yi)/ys;n=np.rint(val).astype(int);assert max(abs(val-n))<.005
 assert np.all(n>=out.author_negative_all+out.author_positive_all)
 out['author_total_attention']=n;out['author_attention_fractional']=val;out.to_csv(ROOT/'data/author_published_all_counts.csv',index=False)
 audit={'figure':'4A','pdf_page':49,'draw_index':idx,'n':184,'x_convention':'year+month/12; one observation per scheduled meeting month in1994-2016','max_x_residual_pdf_points':float(max(res)),'max_rounding_residual_count':float(max(abs(val-n))),'sum_counts':int(sum(n)),'published_table6_total':975,'aggregate_matches_table6':bool(sum(n)==975),'all_direction_counts_bounded_by_total':True,'y_ticks_pdf':yticks.tolist(),'sha256':hashlib.sha256(PDF.read_bytes()).hexdigest()}
 (ROOT/'results/attention_recovery_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
