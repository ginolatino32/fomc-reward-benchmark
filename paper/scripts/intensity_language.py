"""Descriptive (post hoc) check used in Section 5: how widely do the returns vary across meetings
whose equity passages use strong-decline wording? Run from the archive root."""
import json, re, pandas as pd
U = 'source/focused/FOMC_LLM_Focused_Paper/data/'
meta = {json.loads(l)['record_id']: json.loads(l)['meeting_date'] for l in open(U + 'passage_metadata.jsonl')}
rows = [(meta[r['record_id']], r['text']) for r in map(json.loads, open(U + 'passages_original_and_masked.jsonl')) if r['record_id'].endswith('__original')]
df = pd.DataFrame(rows, columns=['date', 'text'])
pat = re.compile(r'(?:equity|stock)[^.]{0,80}\b(?:declin\w*|fell|drop\w*|lower)\b[^.]{0,20}\b(?:sharply|steep\w*|substantially|considerably|significantly)'
                 r'|(?:sharp|steep|substantial|considerable)\w*\s+(?:declin\w*|drop\w*)[^.]{0,40}(?:equity|stock)', re.I)
m = df.assign(hit=df.text.str.contains(pat)).groupby('date').hit.any()
p = pd.read_csv(U + 'analysis_panel.csv').set_index('date').join(m.rename('strong'))
g = p[p.strong.fillna(False).astype(bool)].state_return
print(len(g), 'meetings; preceding return min/median/max:', round(g.min(), 2), round(g.median(), 2), round(g.max(), 2))
