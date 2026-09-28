"""Recover author-published integer counts from vector paths, not OCR or recoding.

Figure 5, NBER w26894, PDF p.50 (printed p.48). Gray paths are all minutes;
black paths exclude Staff Review of Financial Situation. Preserve source geometry.
"""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd,fitz
from calendar_anchors import OFFICIAL,SOURCES
ROOT=Path(__file__).resolve().parents[1]
PDF=ROOT/'source/w26894.pdf'
F=ROOT/'source/focused/FOMC_LLM_Focused_Paper'
def points(d):
    out=[]
    for cmd in d['items']:
        if cmd[0]!='l':raise ValueError('Nonlinear plot segment')
        for v in cmd[1:]:
            pp=(v.x,v.y)
            if not out or out[-1]!=pp:out.append(pp)
    return np.array(out,float)
def main():
    assert hashlib.sha256(PDF.read_bytes()).hexdigest() == 'dbc3380b1463b0e72b6360a4e46cdc39658cfdf83606b9d010ebde0377d0bb88', 'Unexpected source PDF version; review calibration before proceeding'
    doc=fitz.open(PDF);page=doc[49];draw=page.get_drawings()
    curves=[(i,d) for i,d in enumerate(draw) if len(d['items'])>150]
    assert len(curves)==4
    curves.sort(key=lambda v:(v[1]['rect'].y0, sum(v[1]['color'])))
    negall=[(i,d) for i,d in curves if d['rect'].y1<300 and d['color'][0]>.4][0]
    negex=[(i,d) for i,d in curves if d['rect'].y1<300 and d['color'][0]<.1][0]
    posall=[(i,d) for i,d in curves if d['rect'].y0>300 and d['color'][0]>.4][0]
    posex=[(i,d) for i,d in curves if d['rect'].y0>300 and d['color'][0]<.1][0]
    x=points(negall[1])[:,0];assert len(x)==184 and np.all(np.diff(x)>0)
    known=pd.read_csv(F/'data/analysis_panel.csv');known=known[known.date.le('2016-12-31')]
    # Include first unlabeled meeting to obtain the existing 136-calendar-date anchor set.
    dates=pd.DatetimeIndex(['2000-02-02']+known.date.tolist());assert len(dates)==136
    epoch=pd.Timestamp('1970-01-01');days=np.array((dates-epoch).days,float)
    slope,intercept=np.polyfit(days,x[-136:],1)
    recovered_days=(x-intercept)/slope;rounded=np.rint(recovered_days).astype(int)
    date_resid=np.max(np.abs(recovered_days-rounded));known_resid=np.max(np.abs(recovered_days[-136:]-days))
    # Do not infer exact calendar dates by rounding the graph's fractional-year axis.
    # Assign known scheduled meeting dates in order, and verify geometric alignment.
    prior=[f'{year}-{day}' for year,days_ in OFFICIAL.items() for day in days_]
    datesall=pd.DatetimeIndex(prior+dates.strftime('%Y-%m-%d').tolist())
    assert len(datesall)==184 and datesall.is_monotonic_increasing
    alldays=np.asarray((datesall-epoch).days,float)
    calendar_resid=float(np.max(np.abs(recovered_days-alldays)))
    assert known_resid<1.0 and calendar_resid<1.0, (known_resid,calendar_resid)
    # A fractional-year check is reported separately; it is not used to alter dates.
    fractional_year=datesall.year+(datesall.dayofyear-1)/np.where(datesall.is_leap_year,366,365)
    fracfit=np.polyfit(fractional_year,x,1)
    frac_resid=float(np.max(abs(x-np.polyval(fracfit,fractional_year))))
    pd.DataFrame({'meeting_date':datesall.strftime('%Y-%m-%d'),
                  'source':[SOURCES.get(y,'frozen canonical 2000-2016 panel') for y in datesall.year],
                  'calendar_day_residual':recovered_days-alldays}).to_csv(ROOT/'data/figure5_calendar_anchors.csv',index=False)
    # Use horizontal grid positions, not the observed count totals, to calibrate Y.
    grid=[]
    for i,d in enumerate(draw):
      for cmd in d['items']:
        if cmd[0]=='l':
          aa,bb=cmd[1:]
          if abs(aa.y-bb.y)<1e-4 and abs(aa.x-bb.x)>300 and 80<aa.y<520:grid.append((i,aa.y))
    grids=sorted(set(round(y,4) for _,y in grid));print('grid y',grids)
    out=pd.DataFrame({'meeting_date':datesall.strftime('%Y-%m-%d'),'x_pdf':x})
    audit={}
    for name,(idx,dd),bounds in [('author_negative_all',negall,(80,270)),('author_negative_excl_financial',negex,(80,270)),('author_positive_all',posall,(330,510)),('author_positive_excl_financial',posex,(330,510))]:
        coords=points(dd);ys=sorted(set(y for y in grids if bounds[0]<y<bounds[1]))
        # Identify the three drawn 0,5,10 gridlines; the 15 tick lacks a full gridline.
        best=None
        import itertools
        for ys4 in itertools.combinations(ys,3):
            dif=np.diff(ys4)
            if max(dif)-min(dif)<.03 and 50<dif.mean()<60:
                best=ys4
        if best is None:raise ValueError((name,ys))
        y0=best[-1];unit=(best[-1]-best[0])/10
        yy=np.interp(x,coords[:,0],coords[:,1]);values=(y0-yy)/unit;counts=np.rint(values).astype(int)
        err=float(abs(values-counts).max());assert err<.005 and min(counts)>=0
        out[name]=counts;out[name+'_fractional']=values
        audit[name]={'drawing_index':idx,'path_vertices':len(coords),'integer_max_residual':err,'total':int(sum(counts)),'zero_grid':y0,'points_per_count':unit,'count_support':sorted(set(map(int,counts))),'interpolated_x_count':int(sum(np.min(abs(x[:,None]-coords[:,0]),axis=1)>1e-5))}
        pd.DataFrame(coords,columns=['x_pdf','y_pdf']).to_csv(ROOT/f'data/figure5_{name}_vertices.csv',index=False)
    # External validation, not constraints used to construct the counts.
    totals={'published_table6_negative':322,'published_table6_positive':414}
    audit['aggregate_checks']={'negative_equal_table6':int(out.author_negative_all.sum())==322,'positive_equal_table6':int(out.author_positive_all.sum())==414}
    audit['alignment']={'observations':len(x),'existing_date_anchors':136,'max_calendar_day_residual':calendar_resid,'max_fractional_year_axis_residual_pdf_points':frac_resid,'max_known_date_residual':known_resid,'x_slope_per_day':slope,'x_intercept_epoch1970':intercept,'first_meeting':str(datesall[0].date()),'last_meeting':str(datesall[-1].date())}
    audit['source']={'file':'w26894.pdf','sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),'figure':'5','pdf_page':50,'printed_page':48,'type':'author-published human-coded aggregate counts recovered from vector lines; NOT author-supplied raw data or new annotation'}
    out.to_csv(ROOT/'data/author_published_counts_figure5.csv',index=False);(ROOT/'results/figure_recovery_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2));print(out.head(8).to_string(index=False))
if __name__=='__main__':main()
