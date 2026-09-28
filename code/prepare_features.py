"""Additional outcome-blind matched-input measures. No model API calls."""
from __future__ import annotations
import importlib.util,sys,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'source/focused/FOMC_LLM_Focused_Paper'
C=ROOT/'source/cvj/FOMC_CVJ_Benchmark_Update_20260919'
spec=importlib.util.spec_from_file_location('audited_matcher',C/'code/audited_counts.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
def main():
    p=pd.read_json(F/'data/passages_original_and_masked.jsonl',lines=True)
    m=pd.read_json(F/'data/passage_metadata.jsonl',lines=True)
    d=m.merge(p,on='record_id',validate='one_to_one');d=d[d.variant.eq('original')].sort_values(['meeting_date','within_meeting_order'])
    rows=[];audit=[]
    for i,r in enumerate(d.to_dict('records')):
        clean=mod.clean(r['text']);hits=mod.extract(clean);s=Counter(x['sign'] for x in hits)
        # Each selected passage is processed separately: no cross-passage matches.
        # A literal same-input comparator counts phrase hits in every presented passage.
        # Alternative votes count at most one signed observation per passage.
        neg,pos=s['negative'],s['positive'];vote='negative' if neg>0 and pos==0 else 'positive' if pos>0 and neg==0 else 'mixed' if neg>0 and pos>0 else 'unclassified'
        rows.append({'record_id':r['record_id'],'meeting_date':str(r['meeting_date'])[:10],'order':r['within_meeting_order'],
                     'negative_phrase_hits':neg,'positive_phrase_hits':pos,'unclassified_phrase_hits':s['unclassified'],'ambiguous_phrase_hits':s['ambiguous'],
                     'noun_hits':len(hits),'direction_vote':vote,'text_sha256':hashlib.sha256(r['text'].encode()).hexdigest()})
        audit.extend({'record_id':r['record_id'],'meeting_date':str(r['meeting_date'])[:10],**h} for h in hits)
        if (i+1)%200==0:print('matched passages',i+1,flush=True)
    r=pd.DataFrame(rows);r.to_csv(ROOT/'data/matched_passage_counts.csv',index=False);pd.DataFrame(audit).to_csv(ROOT/'data/matched_passage_hit_audit.csv',index=False)
    names=['negative_phrase_hits','positive_phrase_hits','unclassified_phrase_hits','ambiguous_phrase_hits','noun_hits']
    agg=r.groupby('meeting_date')[names].sum();agg['passage_count']=r.groupby('meeting_date').size()
    for v in ['negative','positive','mixed','unclassified']:
        agg['rule_passages_'+v]=r.assign(z=r.direction_vote.eq(v).astype(int)).groupby('meeting_date').z.sum()
    flash=pd.read_csv(F/'data/flash_validated_labels.csv');fl=flash[flash.variant.eq('original')][['record_id','tone','status']]
    ff=r.merge(fl,on='record_id',validate='one_to_one');assert ff.status.eq('VALID').all() or ff.status.eq('valid').all()
    for label in ['negative','positive','neutral','hypothetical','unclear']:
        agg['flash_'+label]=ff.assign(z=ff.tone.eq(label).astype(int)).groupby('meeting_date').z.sum()
    agg.reset_index().to_csv(ROOT/'data/additional_meeting_features.csv',index=False)
    print('Completed',len(r),'passages;',len(agg),'meetings')
if __name__=='__main__':main()
