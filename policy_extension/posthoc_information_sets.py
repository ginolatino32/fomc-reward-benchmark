"""POST HOC diagnostic (not in the locked protocol; labeled as such wherever reported).
Asks whether text substitutes for returns: controls = length + policy state only;
compare adding the return block versus adding each text block. Same estimator and folds."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import run_policy_extension as R
d, feats = R.build(*R.load(Path(sys.argv[1])))
keep = d.y_cycle.notna().to_numpy(); P = d.loc[keep].reset_index(drop=True); F = {k: v[keep] for k, v in feats.items()}
F['returns'] = P[['r_neg', 'r_pos', 'rnext_neg', 'rnext_pos']].to_numpy(float)
folds = []
for i, tl in enumerate(R.BOUNDS):
    tr = np.flatnonzero(P.date.le(tl).to_numpy()); nxt = R.BOUNDS[i+1] if i+1 < len(R.BOUNDS) else '9999'
    folds.append({'fold': i, 'tr': tr, 'te': np.flatnonzero((P.date.gt(tl) & P.date.le(nxt)).to_numpy())})
pred = R.evaluate(P, F, P.y_cycle.to_numpy(float), ['document_sentence_count', 'L', 'dL_m'], ['controls_only', 'returns', 'rules', 'flash', 'embedding'], folds)
s = R.summarize(pred); rng = np.random.default_rng(R.SEED + 1)
c = pd.DataFrame([R.bootstrap(pred, a, 'controls_only', rng) for a in ['returns', 'rules', 'flash', 'embedding']])
s['spec'] = c['spec'] = 'posthoc_state_controls_only'
s.to_csv('results/posthoc_performance.csv', index=False); c.to_csv('results/posthoc_contrasts.csv', index=False)
print(s.round(3).to_string(index=False)); print(c.round(4).to_string(index=False))
