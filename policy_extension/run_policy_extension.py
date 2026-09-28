"""Locked policy-reaction extension (see protocol_locked.json; SHA-256 checked at start).

Question: does FOMC equity-market discussion in the minutes of meeting m predict the
target-rate change over the next policy cycle beyond the returns themselves?
Estimator, blocks, tuning and folds follow the main paper exactly.
Run: python run_policy_extension.py --archive <FOMC_Revised_Exposition_and_Data> --out <dir>
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

HERE = Path(__file__).resolve().parent
PROTOCOL_SHA = '1adb261e1d9cdef148f9d85d1ddc01a84f6aff7a00e47549bc86d019b39ec53e'
ALPHAS = np.logspace(-4, 4, 9)
SEED, DRAWS, BLOCK = 20260926, 9999, 4
BOUNDS = ['2012-08-01', '2015-01-28', '2017-07-26', '2020-01-29', '2022-07-27']


def check_protocol():
    s = (HERE / 'protocol_locked.json').read_bytes()
    assert hashlib.sha256(s).hexdigest() == PROTOCOL_SHA, 'protocol changed after lock'


def block(x, z, numeric):
    if numeric:
        sd = x.std(axis=0); sd = np.where(sd > 1e-12, sd, 1.)
        x = x / sd; z = z / sd
    mu = x.mean(axis=0); x = x - mu; z = z - mu
    scale = max(float(np.sum(x * x) / len(x)), 1e-15) ** .5
    return x / scale, z / scale


def load(archive: Path):
    foc = archive / 'source/focused/FOMC_LLM_Focused_Paper/data'
    panel = pd.read_csv(foc / 'analysis_panel.csv')
    rep = np.load(foc / 'representations.npz', allow_pickle=False)
    assert list(rep['dates'].astype(str)) == list(panel.date)
    counts = pd.read_csv(archive / 'source/cvj/FOMC_CVJ_Benchmark_Update_20260919/results/audited_reconstruction/meeting_counts.csv').set_index('meeting_date')
    extra = pd.read_csv(archive / 'data/additional_meeting_features.csv').set_index('meeting_date')
    human = pd.read_csv(archive / 'data/author_published_all_counts.csv').set_index('meeting_date')
    ffr = pd.read_csv(archive / 'source/market/fred_federal_funds_target_range.csv', parse_dates=['date']).set_index('date').ffr_target_midpoint
    return panel, rep, counts, extra, human, ffr


def level(ffr, day):
    return float(ffr.loc[:day].iloc[-1])


def build(panel, rep, counts, extra, human, ffr):
    d = panel.copy()
    dates = pd.to_datetime(d.date)
    one = pd.Timedelta(days=1)
    L = np.array([level(ffr, t + one) for t in dates]) * 100          # bp
    L_prev0 = level(ffr, pd.Timestamp('2000-02-02') + one) * 100        # meeting before first panel row
    d['L'] = L
    d['dL_m'] = np.diff(np.r_[L_prev0, L])
    d['y_cycle'] = pd.Series(L).shift(-1) - L
    pre = np.array([level(ffr, t - one) for t in dates]) * 100
    d['y_decision'] = pd.Series(L - pre).shift(-1)                     # change around meeting m+1 only
    r = d.state_return.to_numpy(float)
    d['r_neg'] = np.minimum(r, 0); d['r_pos'] = np.maximum(r, 0)
    rn = pd.Series(r).shift(-1)
    d['rnext_neg'] = np.minimum(rn, 0); d['rnext_pos'] = np.maximum(rn, 0)
    # text representations (row order = panel.date)
    feats = {
        'mention_frequency': d[['stock_mentions_total']].to_numpy(float),
        'rules': counts.loc[d.date, ['negative_count', 'positive_count']].to_numpy(float),
        'flash': rep['flash'].astype(float),
        'embedding': rep['gemini'].astype(float),
    }
    fl = extra.loc[d.date, ['flash_positive', 'flash_negative', 'flash_neutral', 'flash_hypothetical', 'flash_unclear']].to_numpy(float)
    assert np.array_equal(fl, feats['flash']), 'flash column order differs from positive,negative,neutral,hypothetical,unclear'
    assert np.array_equal(rep['cvj'].astype(float)[:, [0, 1]], feats['rules']) or np.array_equal(rep['cvj'].astype(float)[:, [1, 0]], feats['rules'])
    hum = human.reindex(d.date)[['author_negative_all', 'author_positive_all']]
    feats['human'] = hum.to_numpy(float)
    return d, feats


def evaluate(d, feats, y, ctrl_cols, reps, folds, embed_names=('embedding',)):
    ctrl = d[ctrl_cols].to_numpy(float)
    preds = []
    for name in reps:
        for f in folds:
            tr, te = f['tr'], f['te']
            def features(a, b):
                cx, cz = block(ctrl[a], ctrl[b], True)
                if name == 'controls_only':
                    return cx, cz
                v = feats[name]
                ex, ez = block(v[a], v[b], name not in embed_names)
                return np.column_stack([cx, ex]) / np.sqrt(2), np.column_stack([cz, ez]) / np.sqrt(2)
            losses = np.zeros(len(ALPHAS)); ni = 0
            for ta, va in TimeSeriesSplit(n_splits=3).split(tr):
                a, b = tr[ta], tr[va]; x, z = features(a, b)
                for j, al in enumerate(ALPHAS):
                    m = Ridge(alpha=float(al), fit_intercept=True, solver='cholesky').fit(x, y[a])
                    losses[j] += np.sum((y[b] - m.predict(z)) ** 2)
                ni += len(b)
            al = float(ALPHAS[int(np.argmin(losses))])
            x, z = features(tr, te)
            p = Ridge(alpha=al, fit_intercept=True, solver='cholesky').fit(x, y[tr]).predict(z)
            for i, pi in zip(te, p):
                preds.append({'model': name, 'fold': f['fold'], 'date': d.date.iloc[i], 'y': y[i], 'prediction': pi, 'training_mean': y[tr].mean(), 'alpha': al})
    return pd.DataFrame(preds)


def summarize(pred):
    rows = []
    for k, g in pred.groupby('model', sort=False):
        se = (g.y - g.prediction) ** 2
        rows.append({'model': k, 'n': len(g), 'mse': se.mean(), 'rmse': np.sqrt(se.mean()), 'mae': np.mean(abs(g.y - g.prediction)),
                     'oos_r2': 1 - se.mean() / np.mean((g.y - g.training_mean) ** 2)})
    return pd.DataFrame(rows)


def bootstrap(pred, a, b, rng):
    ga = pred[pred.model == a].set_index('date'); gb = pred[pred.model == b].set_index('date')
    la = ((ga.y - ga.prediction) ** 2).to_numpy(); lb = ((gb.loc[ga.index].y - gb.loc[ga.index].prediction) ** 2).to_numpy()
    n = len(la); dd = la - lb; dbar = dd.mean()
    starts = np.arange(n - BLOCK + 1); nb = int(np.ceil(n / BLOCK))
    G = np.empty(DRAWS); cen = np.empty(DRAWS)
    for k in range(DRAWS):
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, nb)])[:n]
        G[k] = 100 * (1 - la[idx].mean() / lb[idx].mean())
        cen[k] = dd[idx].mean() - dbar
    p = float(np.mean(np.abs(cen) >= abs(dbar)))
    return {'model': a, 'comparator': b, 'n': n, 'gain_pct': 100 * (1 - la.mean() / lb.mean()),
            'ci_low': float(np.percentile(G, 2.5)), 'ci_high': float(np.percentile(G, 97.5)), 'p_two_sided': p}


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


def hac_ols(y, X, lags=4):
    X = np.column_stack([np.ones(len(X)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ b
    n, k = X.shape; XtXi = np.linalg.inv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1); G = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None]); S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    return b, V


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--archive', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    check_protocol()
    d, feats = build(*load(a.archive))
    log = {}
    # ---- sanity reproduction of the main paper before any policy estimate ----
    mfolds = []
    for i, tl in enumerate(BOUNDS):
        tr = np.flatnonzero(d.date.le(tl).to_numpy())
        nxt = BOUNDS[i + 1] if i + 1 < len(BOUNDS) else '9999'
        te = np.flatnonzero((d.date.gt(tl) & d.date.le(nxt)).to_numpy())
        mfolds.append({'fold': i, 'tr': tr, 'te': te})
    rep_pred = evaluate(d, feats, d.state_return.to_numpy(float), ['document_sentence_count'], ['rules', 'embedding'], mfolds)
    rep_sum = summarize(rep_pred).set_index('model')
    log['main_paper_reproduction_rmse'] = rep_sum.rmse.round(3).to_dict()
    assert abs(rep_sum.loc['embedding', 'rmse'] - 3.522) < 5e-4 and abs(rep_sum.loc['rules', 'rmse'] - 4.100) < 5e-4, rep_sum
    # ---- policy panel ----
    keep = d.y_cycle.notna().to_numpy()
    P = d.loc[keep].reset_index(drop=True); F = {k: v[keep] for k, v in feats.items()}
    log['n_policy_obs'] = int(len(P)); log['policy_first'] = P.date.iloc[0]; log['policy_last'] = P.date.iloc[-1]
    folds = []
    for i, tl in enumerate(BOUNDS):
        tr = np.flatnonzero(P.date.le(tl).to_numpy())
        nxt = BOUNDS[i + 1] if i + 1 < len(BOUNDS) else '9999'
        te = np.flatnonzero((P.date.gt(tl) & P.date.le(nxt)).to_numpy())
        folds.append({'fold': i, 'tr': tr, 'te': te})
    log['test_sizes'] = [len(f['te']) for f in folds]
    assert log['test_sizes'] == [20, 20, 20, 20, 18]
    reps = ['controls_only', 'mention_frequency', 'rules', 'flash', 'embedding']
    full_ctrl = ['document_sentence_count', 'L', 'dL_m', 'r_neg', 'r_pos', 'rnext_neg', 'rnext_pos']
    rng = np.random.default_rng(SEED)
    out = {}
    specs = {'primary': (P.y_cycle.to_numpy(float), full_ctrl),
             'S1_forecast': (P.y_cycle.to_numpy(float), full_ctrl[:5]),
             'S2_decision_day': (P.y_decision.to_numpy(float), full_ctrl)}
    contrasts = [('embedding', 'controls_only'), ('flash', 'controls_only'), ('rules', 'controls_only'), ('embedding', 'rules'), ('flash', 'rules')]
    for name, (y, cc) in specs.items():
        pred = evaluate(P, F, y, cc, reps, folds); pred['spec'] = name
        summ = summarize(pred); summ['spec'] = name
        cons = pd.DataFrame([bootstrap(pred, x, z, rng) for x, z in contrasts]); cons['spec'] = name
        if name == 'primary': cons['holm_p'] = holm(cons.p_two_sided.to_numpy())
        out[name] = (pred, summ, cons)
    # ---- S3: CVJ overlap, earlier start ----
    Q = P[P.date.le('2016-12-14')].reset_index(drop=True); FQ = {k: v[:len(Q)] for k, v in F.items()}
    assert not np.isnan(FQ['human']).any()
    ends = [Q.date.iloc[59], Q.date.iloc[79], Q.date.iloc[99], Q.date.iloc[119]]
    qf = []
    for i, tl in enumerate(ends):
        tr = np.flatnonzero(Q.date.le(tl).to_numpy()); nxt = ends[i + 1] if i + 1 < len(ends) else '2016-12-14'
        te = np.flatnonzero((Q.date.gt(tl) & Q.date.le(nxt)).to_numpy()); qf.append({'fold': i, 'tr': tr, 'te': te})
    log['S3_test_sizes'] = [len(f['te']) for f in qf]
    y3 = Q.y_cycle.to_numpy(float)
    pred3 = evaluate(Q, FQ, y3, full_ctrl, ['controls_only', 'human', 'rules', 'flash', 'embedding'], qf); pred3['spec'] = 'S3_human_overlap'
    summ3 = summarize(pred3); summ3['spec'] = 'S3_human_overlap'
    cons3 = pd.DataFrame([bootstrap(pred3, x, z, rng) for x, z in [('human', 'controls_only'), ('embedding', 'controls_only'), ('flash', 'controls_only'), ('rules', 'controls_only'), ('embedding', 'human'), ('flash', 'human')]]); cons3['spec'] = 'S3_human_overlap'
    out['S3_human_overlap'] = (pred3, summ3, cons3)
    # ---- asymmetry (in-sample, HAC) ----
    y = P.y_cycle.to_numpy(float); C = P[full_ctrl].to_numpy(float)
    Cz = (C - C.mean(0)) / C.std(0)
    asym = []
    for name, cols in [('rules', (0, 1)), ('flash', (1, 0))]:   # (negative, positive) column positions
        v = F[name]; neg = v[:, cols[0]]; pos = v[:, cols[1]]
        z = lambda s: (s - s.mean()) / s.std()
        X = np.column_stack([Cz, z(neg), z(pos)]); b, V = hac_ols(y, X)
        bn, bp = b[-2], b[-1]; se_n, se_p = np.sqrt(V[-2, -2]), np.sqrt(V[-1, -1])
        g = np.zeros(len(b)); g[-2] = 1; g[-1] = 1; wstat = (g @ b) ** 2 / (g @ V @ g)
        from scipy.stats import chi2, norm
        asym.append({'counts': name, 'beta_negative_bp': bn, 'se_negative': se_n, 'p_negative': 2 * norm.sf(abs(bn / se_n)),
                     'beta_positive_bp': bp, 'se_positive': se_p, 'p_positive': 2 * norm.sf(abs(bp / se_p)),
                     'wald_sum_zero': wstat, 'p_symmetry': chi2.sf(wstat, 1), 'n': len(y)})
    # ---- save ----
    pd.concat([o[0] for o in out.values()]).to_csv(a.out / 'predictions.csv', index=False)
    pd.concat([o[1] for o in out.values()]).to_csv(a.out / 'performance.csv', index=False)
    pd.concat([o[2] for o in out.values()]).to_csv(a.out / 'contrasts.csv', index=False)
    pd.DataFrame(asym).to_csv(a.out / 'asymmetry.csv', index=False)
    desc = {'y_cycle_mean': float(P.y_cycle.mean()), 'y_cycle_sd': float(P.y_cycle.std()), 'share_nonzero': float((P.y_cycle != 0).mean()),
            'eval_share_nonzero': float((P.y_cycle[P.date.gt(BOUNDS[0])] != 0).mean()), 'eval_n_cuts': int((P.y_cycle[P.date.gt(BOUNDS[0])] < 0).sum()),
            'eval_n_hikes': int((P.y_cycle[P.date.gt(BOUNDS[0])] > 0).sum())}
    log['target_descriptives'] = desc
    P[['date', 'L', 'dL_m', 'y_cycle', 'y_decision', 'r_neg', 'r_pos', 'rnext_neg', 'rnext_pos', 'document_sentence_count']].to_csv(a.out / 'policy_panel.csv', index=False)
    (a.out / 'run_log.json').write_text(json.dumps(log, indent=2, default=str))
    for k, o in out.items():
        print('==', k); print(o[1].round(3).to_string(index=False)); print(o[2].round(4).to_string(index=False))
    print(pd.DataFrame(asym).round(4).to_string(index=False)); print(json.dumps(log, indent=1, default=str))


if __name__ == '__main__':
    main()
