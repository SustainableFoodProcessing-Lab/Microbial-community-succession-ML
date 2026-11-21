"""
Holdout validation of the final model (Optuna × FC Top10 × RandomForest) with
each seed's own thresholds, compared with the seed-42 fixed thresholds
(Supplementary Tables S12 and S13, holdout rows).

Run from the repository root after notebooks 1–9:
    python per_seed_threshold_holdout.py
Reads output/pkl/ and output/validation_performance.xlsx; writes
output/per_seed_threshold_holdout.xlsx.

The per-seed models are refitted from the features and best parameters saved by
9. per_seed_threshold_test.ipynb (no new search), and evaluated end to end on
each seed's holdout set (n = 27) as in 6. validation.ipynb: the classifier
predicts the stage, and TPC is predicted only for samples classified as Early
or Intermediate. Holdout labels use the same per-seed thresholds.
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
warnings.filterwarnings('ignore')

# ── LSTM stub (to allow loading regression_results.pkl) ──────────
class LSTMRegressor: pass
import __main__; __main__.LSTMRegressor = LSTMRegressor

OUTPUT_DIR = "output"
PKL        = os.path.join(OUTPUT_DIR, "pkl")
OUT        = os.path.join(OUTPUT_DIR, "per_seed_threshold_holdout.xlsx")
SEEDS      = list(range(38, 48))
FINAL_COND, FINAL_MODEL = 'cond1', 'RandomForest'

def load(name):
    path = os.path.join(PKL, name)
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path} — run notebooks 1–9 first.")
    with open(path, 'rb') as f:
        return pickle.load(f)

labeled   = load('labeled_splits.pkl')
thr_log   = load('threshold_log.pkl')
fixed_clf = load('classification_results.pkl')[FINAL_COND][FINAL_MODEL]
fixed_reg = load('regression_results.pkl')[FINAL_COND][FINAL_MODEL]
perseed   = load('per_seed_threshold_results.pkl')
per_clf   = perseed['classification'][FINAL_COND][FINAL_MODEL]
per_reg   = perseed['regression'][FINAL_COND][FINAL_MODEL]
nb6       = pd.read_excel(os.path.join(OUTPUT_DIR, 'validation_performance.xlsx'),
                          sheet_name='Per-seed').set_index('seed')
tpc_col   = labeled['tpc_col']


def label_by_tpc(df, lo, hi):
    """Early=0, Late=1, Intermediate=2 (as in 2. labeling.ipynb)"""
    return np.where(df[tpc_col] < lo, 0, np.where(df[tpc_col] >= hi, 1, 2))


def relabel(seed):
    """Labels from the seed's own Optuna thresholds (as in 9. per_seed_threshold_test.ipynb)."""
    lo, hi = thr_log[seed]['tpc_thr_low'], thr_log[seed]['tpc_thr_high']
    out = {}
    for part in ('train', 'test', 'val'):
        d = labeled['labeled_splits'][seed][part].copy()
        d['Label'] = label_by_tpc(d, lo, hi)
        out[part] = d
    return out


def holdout_e2e(clf, feats_c, reg, feats_r, va):
    y_true = va['Label'].values
    y_pred = clf.predict(va[feats_c].values)
    va_reg = va[pd.Series(y_pred).isin([0, 2]).values]
    pr     = reg.predict(va_reg[feats_r].values)
    return {'acc': accuracy_score(y_true, y_pred),
            'f1':  f1_score(y_true, y_pred, average='macro', zero_division=0),
            'r2':  r2_score(va_reg[tpc_col].values, pr),
            'mae': mean_absolute_error(va_reg[tpc_col].values, pr),
            'reg_n': len(va_reg)}


rows = []
for seed in SEEDS:
    sp = relabel(seed)
    tr, te = sp['train'], sp['test']
    tr_fm, te_fm = tr[tr['Label'] != 1], te[te['Label'] != 1]

    # per-seed models: refit with the saved features and best parameters
    pc, pr = per_clf[seed], per_reg[seed]
    clf = RandomForestClassifier(random_state=42, **pc['best_params'])
    clf.fit(tr[pc['features']].values, tr['Label'].values)
    reg = RandomForestRegressor(random_state=42, **pr['best_params'])
    reg.fit(tr_fm[pr['features']].values, tr_fm[tpc_col].values)

    # check: the refitted models reproduce the saved per-seed CV test results
    y_c = clf.predict(te[pc['features']].values)
    y_r = reg.predict(te_fm[pr['features']].values)
    for got, ref in ((accuracy_score(te['Label'], y_c), pc['test_acc']),
                     (f1_score(te['Label'], y_c, average='macro'), pc['test_f1']),
                     (r2_score(te_fm[tpc_col], y_r), pr['test_r2']),
                     (mean_absolute_error(te_fm[tpc_col], y_r), pr['test_mae'])):
        assert abs(got - ref) < 1e-12, f"seed {seed}: refit does not reproduce notebook 9"

    h_fix = holdout_e2e(fixed_clf[seed]['best_model'], fixed_clf[seed]['features'],
                        fixed_reg[seed]['best_model'], fixed_reg[seed]['features'],
                        labeled['labeled_splits'][seed]['val'])
    h_per = holdout_e2e(clf, pc['features'], reg, pr['features'], sp['val'])

    # check: the fixed-threshold holdout equals 6. validation.ipynb
    for k, col in (('acc', 'clf_acc'), ('f1', 'clf_f1'), ('r2', 'reg_r2'), ('mae', 'reg_mae')):
        assert abs(h_fix[k] - nb6.loc[seed, col]) < 1e-12, f"seed {seed}: fixed holdout differs from notebook 6"

    row = {'seed': seed}
    for k in ('acc', 'f1', 'r2', 'mae', 'reg_n'):
        row[f'{k}_seed42_fixed'], row[f'{k}_per_seed'] = h_fix[k], h_per[k]
    rows.append(row)
df_seed = pd.DataFrame(rows)
print("Checks passed: refit = notebook 9 (CV test), fixed holdout = notebook 6")

# ── paired t-test (pairs = seeds); SD over the 10 seeds with ddof=0 (n), as in all other tables ──
cmp_rows = []
for task, metric, k in (('Classification', 'Accuracy', 'acc'), ('Classification', 'Macro-F1', 'f1'),
                        ('Regression', 'R2', 'r2'), ('Regression', 'MAE (log CFU/g)', 'mae')):
    a, b = df_seed[f'{k}_seed42_fixed'], df_seed[f'{k}_per_seed']
    t, p = stats.ttest_rel(a, b)
    cmp_rows.append({'task': task, 'model': FINAL_MODEL, 'metric': metric,
                     # ddof=0: SD unified to the n-based definition used for every 10-seed mean ± SD
                     'seed42_fixed_mean': a.mean(), 'seed42_fixed_sd': a.std(ddof=0),
                     'per_seed_mean':     b.mean(), 'per_seed_sd':     b.std(ddof=0),
                     'delta': b.mean() - a.mean(), 't': t, 'p_paired': p,
                     'significant_a0.05': 'yes' if p < 0.05 else 'no'})
df_cmp = pd.DataFrame(cmp_rows)

pd.set_option('display.width', 200)
print("\n=== Holdout (n = 27), Optuna × FC Top10 × RandomForest, 10 seeds ===")
print(df_cmp.round(4).to_string(index=False))
print()
print(df_seed.round(4).to_string(index=False))

with pd.ExcelWriter(OUT, engine='openpyxl') as w:
    df_cmp.to_excel(w,  sheet_name='Holdout_final_model', index=False)
    df_seed.to_excel(w, sheet_name='Holdout_per_seed',    index=False)
print(f"\nSaved → {OUT}")
