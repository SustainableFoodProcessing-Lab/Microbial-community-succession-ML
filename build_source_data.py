"""
Build the Source Data workbook for the machine-learning figures and tables
(Figs. 6–9 including the seed-42 holdout panels 9G and 9I–9K, Table 1, Supplementary Figs. S1–S6,
Supplementary Tables S1–S6 and S9–S14).

Run from the repository root after notebooks 1–9 and per_seed_threshold_holdout.py:
    python build_source_data.py
Reads output/pkl/ and the result workbooks in output/; writes output/Source_Data.xlsx.
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error   # holdout values in the Fig. 9G·J notes
from statsmodels.stats.multitest import multipletests   # BH correction of the Fig. 6A·B genus selection (as in 1. preprocessing.ipynb)
warnings.filterwarnings('ignore')

# ── LSTM stub (to allow loading regression_results.pkl) ──────────
class LSTMRegressor: pass
import __main__; __main__.LSTMRegressor = LSTMRegressor

OUTPUT_DIR = "output"
PKL        = os.path.join(OUTPUT_DIR, "pkl")
OUT        = os.path.join(OUTPUT_DIR, "Source_Data.xlsx")
SEEDS      = list(range(38, 48))
LABEL_SEED = 42
LABEL_MAP  = {0: 'early', 1: 'late', 2: 'intermediate'}
CONDS = {
    'cond1': 'Optuna × FC Top10',
    'cond2': 'Optuna × VIP≥1.0',
    'cond3': 'K-means × FC Top10',
    'cond4': 'K-means × VIP≥1.0',
}

def load(name):
    path = os.path.join(PKL, name)
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path} — run notebooks 1–9 first.")
    with open(path, 'rb') as f:
        return pickle.load(f)

def read_sheet(xlsx, sheet):
    path = os.path.join(OUTPUT_DIR, xlsx)
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"Cannot read sheet '{sheet}' of {path} ({e}) — run the notebook that writes it first.")

# ── Load ──────────────────────────────────────────────────────────
print("Loading pkl files...")
genus_pkg    = load('genus_preprocessed.pkl')
labeled_pkg  = load('labeled_splits.pkl')
kmeans_pkg   = load('kmeans_labeled_splits.pkl')
thr_opt      = load('threshold_log.pkl')
thr_km       = load('kmeans_threshold_log.pkl')
scr          = load('screening_results.pkl')
clf_res      = load('classification_results.pkl')
reg_res      = load('regression_results.pkl')
best_clf_pkg = load('best_model_pipeline.pkl')
best_reg_pkg = load('best_reg_pipeline.pkl')
val_shap     = load('validation_shap_allseeds.pkl')
perm_res     = load('permutation_results.pkl')
opt_hist     = load('optuna_trial_history_seed42.pkl')

genus_df = genus_pkg['data'].copy()
tpc_col  = genus_pkg['tpc_col']
opt_splits = labeled_pkg['labeled_splits']
km_splits  = kmeans_pkg['labeled_splits']

sheets = {}   # sheet name -> DataFrame
notes  = {}   # sheet name -> description for the Read Me sheet
below  = {}   # sheet name -> (caption, DataFrame) written under the main table (Fig. 8 AUC, Fig. 9B·E sheets)

# ═════════════════════════ Main figures ═════════════════════════
# ── Fig6C_Fig6D: seed-42 train TPC + labels ──────────────────────
tr42_opt = opt_splits[LABEL_SEED]['train'].copy()
tr42_km  = km_splits[LABEL_SEED]['train'].copy()
sheets['Fig6C_Fig6D'] = pd.DataFrame({
    'sample_id':     tr42_opt['Description'].values,
    'TPC_log_CFU_g': tr42_opt[tpc_col].values.round(3),
    'Optuna_label':  tr42_opt['Label'].map(LABEL_MAP).values,
    'Kmeans_label':  tr42_km['Label'].map(LABEL_MAP).values,
})
notes['Fig6C_Fig6D'] = (
    f"Seed-42 training set (n={len(tr42_opt)}). "
    f"Optuna thresholds: low={thr_opt[LABEL_SEED]['tpc_thr_low']:.3f}, high={thr_opt[LABEL_SEED]['tpc_thr_high']:.3f}. "
    f"K-means thresholds: low={thr_km[LABEL_SEED]['tpc_thr_low']:.3f}, high={thr_km[LABEL_SEED]['tpc_thr_high']:.3f}.")

# ── Fig7_A_C_heatmap: Spearman correlation ───────────────────────
fc10_feats    = scr['cond1_optuna_fc10']['per_seed_features'][LABEL_SEED]
vip_opt_feats = scr['cond2_optuna_vip']['per_seed_features'][LABEL_SEED]
vip_km_feats  = scr['cond4_kmeans_vip']['per_seed_features'][LABEL_SEED]

corr_rows = []
for set_name, feats in [('FC_Top10', fc10_feats), ('VIP_Optuna', vip_opt_feats), ('VIP_Kmeans', vip_km_feats)]:
    sub = tr42_opt[feats + [tpc_col]].copy()
    names = [c.replace('g__', '') for c in feats] + ['TPC']
    sub.columns = names
    for a in names:
        for b in names:
            r, p = stats.spearmanr(sub[a], sub[b])
            corr_rows.append({'feature_set': set_name, 'row': a, 'col': b,
                              'spearman_r': round(float(r), 3), 'p_value': round(float(p), 4)})
sheets['Fig7_A_C_heatmap'] = pd.DataFrame(corr_rows)
notes['Fig7_A_C_heatmap'] = ("Spearman correlation (r) among selected genera + TPC, seed-42 train. "
                             "Three feature sets: FC Top10, VIP Optuna, VIP K-means.")

# ── Fig7_D_K_PCA_PLSDA: PCA/PLS-DA scores ────────────────────────
pca_sets = [
    ('Cond1_Optuna_FC10', fc10_feats,    opt_splits[LABEL_SEED]['train'].copy()),
    ('Cond2_Optuna_VIP',  vip_opt_feats, opt_splits[LABEL_SEED]['train'].copy()),
    ('Cond3_Kmeans_FC10', fc10_feats,    km_splits[LABEL_SEED]['train'].copy()),
    ('Cond4_Kmeans_VIP',  vip_km_feats,  km_splits[LABEL_SEED]['train'].copy()),
]
score_rows, loading_rows = [], []
for cond, feats, data_tr in pca_sets:
    feats_used = [f for f in feats if f in data_tr.columns]
    Xs = StandardScaler().fit_transform(data_tr[feats_used].values)
    y  = data_tr['Label'].values
    pca = PCA(n_components=2)
    T_pca = pca.fit_transform(Xs)
    ev = pca.explained_variance_ratio_
    pls = PLSRegression(n_components=2)
    T_pls, _ = pls.fit_transform(Xs, y)
    for i in range(len(y)):
        score_rows.append({
            'condition': cond, 'sample_id': data_tr['Description'].iloc[i], 'label': LABEL_MAP[y[i]],
            'PC1': round(float(T_pca[i, 0]), 4), 'PC2': round(float(T_pca[i, 1]), 4),
            'PC1_exp_var': round(float(ev[0]), 4), 'PC2_exp_var': round(float(ev[1]), 4),
            'PLSDA1': round(float(T_pls[i, 0]), 4), 'PLSDA2': round(float(T_pls[i, 1]), 4),
        })
    for i, feat in enumerate(feats_used):
        loading_rows.append({
            'condition': cond, 'genus': feat.replace('g__', ''),
            'PCA_PC1_loading': round(float(pca.components_[0, i]), 4),
            'PCA_PC2_loading': round(float(pca.components_[1, i]), 4),
            'PLSDA1_loading':  round(float(pls.x_loadings_[i, 0]), 4),
            'PLSDA2_loading':  round(float(pls.x_loadings_[i, 1]), 4),
        })
sheets['Fig7_D_K_PCA_PLSDA'] = pd.DataFrame(score_rows)
notes['Fig7_D_K_PCA_PLSDA'] = "PCA/PLS-DA sample scores, seed-42 train. StandardScaler applied before PCA."

# ── Fig8: confusion matrix + ROC (final classification model) ────
best_model = best_clf_pkg['model']
best_feats = best_clf_pkg['features']
best_seed  = best_clf_pkg['seed']
te42 = opt_splits[best_seed]['test'].copy()
y_te   = te42['Label'].values
y_pred = best_model.predict(te42[best_feats].values)
y_prob = best_model.predict_proba(te42[best_feats].values)
classes = best_model.classes_

cm = confusion_matrix(y_te, y_pred, labels=classes)
names = [LABEL_MAP[c] for c in classes]
sheets['Fig8_confusion_matrix'] = pd.DataFrame(
    [{'true_label': names[i], 'predicted_label': names[j], 'count': int(cm[i, j])}
     for i in range(len(names)) for j in range(len(names))])

roc_rows = []
for idx, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve((y_te == cls).astype(int), y_prob[:, idx])
    auc_val = auc(fpr, tpr)
    for f_, t_ in zip(fpr, tpr):
        roc_rows.append({'class': LABEL_MAP[cls], 'AUC': round(auc_val, 3),
                         'FPR': round(float(f_), 4), 'TPR': round(float(t_), 4)})
sheets['Fig8_ROC'] = pd.DataFrame(roc_rows)

# Fig. 8A–H: best classifier of each condition (4. classification.ipynb); the Condition 1 rows above are kept as they are
FIG8 = [('cond1', 'RandomForest', 'A', 'E'), ('cond2', 'SVC', 'B', 'F'),
        ('cond3', 'RandomForest', 'C', 'G'), ('cond4', 'SVC', 'D', 'H')]
FIG8_SPLITS = {'cond1': opt_splits, 'cond2': opt_splits, 'cond3': km_splits, 'cond4': km_splits}
assert best_seed == LABEL_SEED and best_clf_pkg['condition'] == 'cond1' and best_clf_pkg['model_name'] == 'RandomForest'
for name, p in (('Fig8_confusion_matrix', 'A'), ('Fig8_ROC', 'E')):
    for col, val in (('model', 'RandomForest'), ('condition', CONDS['cond1']), ('panel', p)):
        sheets[name].insert(0, col, val)
cm_add, roc_add, auc_rows = [], [], []
for cond, mname, p_cm, p_roc in FIG8:
    for s in SEEDS:
        r = clf_res[cond][mname][s]
        te_s = FIG8_SPLITS[cond][s]['test']
        X_s, y_s = te_s[r['features']].values, te_s['Label'].values
        prob_s = r['best_model'].predict_proba(X_s)
        cls_s = list(r['best_model'].classes_)
        for lbl in [0, 2, 1]:                                   # Early, Intermediate, Late
            if (y_s == lbl).any() and (y_s != lbl).any():
                fpr, tpr, _ = roc_curve((y_s == lbl).astype(int), prob_s[:, cls_s.index(lbl)])
                auc_rows.append({'panel': p_roc, 'condition': CONDS[cond], 'model': mname, 'seed': s,
                                 'class': LABEL_MAP[lbl], 'AUC': auc(fpr, tpr)})
        if s != LABEL_SEED or cond == 'cond1':
            continue
        cm_s = confusion_matrix(y_s, r['best_model'].predict(X_s), labels=cls_s)
        cm_add += [{'panel': p_cm, 'condition': CONDS[cond], 'model': mname, 'true_label': LABEL_MAP[a],
                    'predicted_label': LABEL_MAP[b], 'count': int(cm_s[i, j])}
                   for i, a in enumerate(cls_s) for j, b in enumerate(cls_s)]
        for idx, cls in enumerate(cls_s):
            fpr, tpr, _ = roc_curve((y_s == cls).astype(int), prob_s[:, idx])
            auc_val = auc(fpr, tpr)
            roc_add += [{'panel': p_roc, 'condition': CONDS[cond], 'model': mname, 'class': LABEL_MAP[cls],
                         'AUC': round(auc_val, 3), 'FPR': round(float(f_), 4), 'TPR': round(float(t_), 4)}
                        for f_, t_ in zip(fpr, tpr)]
sheets['Fig8_confusion_matrix'] = pd.concat([sheets['Fig8_confusion_matrix'], pd.DataFrame(cm_add)], ignore_index=True)
sheets['Fig8_ROC'] = pd.concat([sheets['Fig8_ROC'], pd.DataFrame(roc_add)], ignore_index=True)
sheets['Fig8_AUC_10seeds'] = pd.DataFrame(auc_rows)
_a = sheets['Fig8_AUC_10seeds'].groupby(['panel', 'condition', 'model', 'class'], sort=False)['AUC']
below['Fig8_AUC_10seeds'] = ("Summary: 10-seed mean and SD (ddof = 0).",
                             pd.DataFrame({'AUC_mean': _a.mean(), 'AUC_sd': _a.std(ddof=0)}).reset_index())
notes['Fig8_confusion_matrix'] = (
    "Confusion matrices of Fig. 8A–D: the best-performing classifier of each condition (A, Optuna × FC Top10, "
    "RandomForest; B, Optuna × VIP≥1.0, SVC; C, K-means × FC Top10, RandomForest; D, K-means × VIP≥1.0, SVC), "
    "seed-42 test set.")
notes['Fig8_ROC'] = ("One-vs-rest ROC curves of Fig. 8E–H (same models as Fig. 8A–D), seed-42 test set, "
                     "with the seed-42 AUC of each class.")
notes['Fig8_AUC_10seeds'] = (
    "Per-seed one-vs-rest AUC of each class for the models of Fig. 8E–H (seeds 38–47, each seed's test set). "
    "The 10-seed mean and SD (ddof = 0) shown in Fig. 8E–H are placed below the per-seed table.")

# ── Fig8_scatter_regression ──────────────────────────────────────
sc_rows = []
for cond in CONDS:
    r = reg_res[cond]['RandomForest'][LABEL_SEED]
    for split, ys, ps in [('train', r['y_train'], r['y_train_pred']), ('test', r['y_test'], r['y_test_pred'])]:
        for obs, pred in zip(ys, ps):
            sc_rows.append({'condition': CONDS[cond], 'split': split,
                            'observed_TPC': round(float(obs), 3), 'predicted_TPC': round(float(pred), 3)})
sheets['Fig8_scatter_regression'] = pd.DataFrame(sc_rows)
notes['Fig8_scatter_regression'] = "Seed-42, RandomForest, per condition. Train and test sets."

# ── Fig9: SHAP values (final models, seed-42 test set) ───────────
import shap
sv = shap.TreeExplainer(best_model).shap_values(te42[best_feats].values)
X_te42 = te42[best_feats].values   # CLR values that colour the Fig. 9A beeswarm (new feature_value column)
shap_rows = []
for cls_idx, cls in enumerate(best_model.classes_):
    sv_c = sv[cls_idx] if isinstance(sv, list) else sv[:, :, cls_idx]
    for i in range(sv_c.shape[0]):
        for j, feat in enumerate(best_feats):
            shap_rows.append({'set': 'clf_test_seed42', 'class': LABEL_MAP[cls],
                              'sample_id': te42['Description'].iloc[i],
                              'true_label': LABEL_MAP[te42['Label'].iloc[i]],
                              'genus': feat.replace('g__', ''), 'SHAP_value': round(float(sv_c[i, j]), 5),
                              'feature_value': round(float(X_te42[i, j]), 4)})   # added: beeswarm colour (CLR)
sheets['Fig9_SHAP_clf'] = pd.DataFrame(shap_rows)
notes['Fig9_SHAP_clf'] = (f"SHAP values (TreeExplainer) of the final classification model "
                          f"({best_clf_pkg['cond_label']}, {best_clf_pkg['model_name']}), seed-{best_seed} test set. "
                          "feature_value = CLR value of the genus (beeswarm colour).")

reg_model, reg_feats, reg_seed = best_reg_pkg['model'], best_reg_pkg['features'], best_reg_pkg['seed']
te_reg = opt_splits[reg_seed]['test'].copy()
te_reg = te_reg[te_reg['Label'].isin([0, 2])].copy()          # Early + Intermediate
sv_r = shap.TreeExplainer(reg_model).shap_values(te_reg[reg_feats].values)
X_te_reg = te_reg[reg_feats].values   # CLR values that colour the Fig. 9D beeswarm (new feature_value column)
sheets['Fig9_SHAP_reg'] = pd.DataFrame(
    [{'set': 'reg_test_seed42', 'sample_id': te_reg['Description'].iloc[i],
      'true_label': LABEL_MAP[te_reg['Label'].iloc[i]], 'genus': feat.replace('g__', ''),
      'SHAP_value': round(float(sv_r[i, j]), 5),
      'feature_value': round(float(X_te_reg[i, j]), 4)}   # added: beeswarm colour (CLR)
     for i in range(sv_r.shape[0]) for j, feat in enumerate(reg_feats)])
notes['Fig9_SHAP_reg'] = (f"SHAP values (TreeExplainer) of the final regression model "
                          f"({best_reg_pkg['cond_label']}, {best_reg_pkg['model_name']}), "
                          f"seed-{reg_seed} test set, Early + Intermediate. "
                          "feature_value = CLR value of the genus (beeswarm colour of Fig. 9D).")

# ── Fig9B / Fig9E: per-seed values behind the 10-seed bars (added: Fig. 9B·E aggregate 10 seeds, sheets above are seed 42 only) ──
# Same calculation as 4. classification.ipynb (SHAP bar cell) and 5. regression.ipynb (SHAP bar cell): mean |SHAP| on each
# seed's test set; a genus not selected in a seed counts as 0; bars show the 10 genera with the lowest mean rank over the
# seeds in which they were selected; error bars are the SD (ddof = 0) across the 10 seeds.
STAGE_ORDER = [0, 2, 1]   # Early, Intermediate, Late (LABEL_ORDER of 4. classification.ipynb)

def shap_display_order(imp):
    """imp: {seed: Series(total mean |SHAP|, index=genus)} → (union genera, mean-rank order, n seeds selected), as in notebooks 4/5."""
    genera = sorted({g for s in SEEDS for g in imp[s].index})
    disp = pd.DataFrame(index=genera, columns=SEEDS, dtype=float)
    for s in SEEDS:
        disp[s] = imp[s].rank(ascending=False, method='min').astype(int).reindex(genera)   # not selected → NaN
    disp['_sort'] = disp[SEEDS].mean(axis=1)
    disp = disp.sort_values('_sort')
    return genera, disp['_sort'], disp[SEEDS].notna().sum(axis=1)

# classification (Fig. 9B): per-stage mean |SHAP| of each stage's predicted-probability output, all test samples
clf_splits_9b = opt_splits if best_clf_pkg['splits_pkl'] == 'labeled_splits.pkl' else km_splits
clf_by_stage, clf_total = {}, {}
for s in SEEDS:
    r = clf_res[best_clf_pkg['condition']][best_clf_pkg['model_name']][s]
    short = [f.replace('g__', '') for f in r['features']]
    sv_s = shap.TreeExplainer(r['best_model']).shap_values(clf_splits_9b[s]['test'][r['features']].values)
    cls_list = list(r['best_model'].classes_)
    clf_by_stage[s] = {}
    for lbl in STAGE_ORDER:
        k = cls_list.index(lbl)
        sv_c = sv_s[k] if isinstance(sv_s, list) else sv_s[:, :, k]
        clf_by_stage[s][lbl] = pd.Series(np.mean(np.abs(sv_c), axis=0), index=short)
    clf_total[s] = sum(clf_by_stage[s][lbl] for lbl in STAGE_ORDER)

genera_b, order_b, nsel_b = shap_display_order(clf_total)
sheets['Fig9B_SHAP_clf_10seeds'] = pd.DataFrame(
    [{'seed': s, 'genus': g, 'selected_in_seed': int(g in clf_total[s].index),
      'mean_abs_shap_early':        clf_by_stage[s][0].get(g, 0.0),
      'mean_abs_shap_intermediate': clf_by_stage[s][2].get(g, 0.0),
      'mean_abs_shap_late':         clf_by_stage[s][1].get(g, 0.0),
      'mean_abs_shap_total':        clf_total[s].get(g, 0.0)}
     for s in SEEDS for g in genera_b])
_b = sheets['Fig9B_SHAP_clf_10seeds']
top_b = list(order_b.index[:10])
_bt = _b[_b['genus'].isin(top_b)].groupby('genus')
below['Fig9B_SHAP_clf_10seeds'] = (
    "Summary: the 10 genera shown in Fig. 9B (display order = top to bottom). Means over 10 seeds with unselected seeds = 0; "
    "SD of the total with ddof = 0 (error bars).",
    pd.DataFrame({'display_order': range(1, len(top_b) + 1), 'genus': top_b,
                  'n_seeds_selected': nsel_b[top_b].values,
                  'mean_rank_selected_seeds': order_b[top_b].values,
                  'mean_abs_shap_early_mean':        _bt['mean_abs_shap_early'].mean()[top_b].values,
                  'mean_abs_shap_intermediate_mean': _bt['mean_abs_shap_intermediate'].mean()[top_b].values,
                  'mean_abs_shap_late_mean':         _bt['mean_abs_shap_late'].mean()[top_b].values,
                  'mean_abs_shap_total_mean':        _bt['mean_abs_shap_total'].mean()[top_b].values,
                  'mean_abs_shap_total_sd':          _bt['mean_abs_shap_total'].std(ddof=0)[top_b].values}))
notes['Fig9B_SHAP_clf_10seeds'] = (
    f"Per-seed values behind Fig. 9B ({best_clf_pkg['cond_label']}, {best_clf_pkg['model_name']}, seeds 38–47, each seed's "
    "test set): mean |SHAP| for each stage output (Early, Intermediate, Late; averaged over all test samples) and their sum. "
    "Rows = seed × every genus selected in at least one seed; selected_in_seed = 0 rows are set to 0, as in the figure. "
    "A summary of the 10 displayed genera (mean rank over the seeds in which they were selected, 10-seed means, SD with "
    "ddof = 0) is placed below the per-seed table.")

# regression (Fig. 9E): mean |SHAP| on the test samples with true label Early or Intermediate
reg_total = {}
for s in SEEDS:
    feats = scr['cond1_optuna_fc10']['per_seed_features'][s]
    te_s  = opt_splits[s]['test']
    te_s  = te_s[te_s['Label'] != 1]
    sv_s  = shap.TreeExplainer(reg_res['cond1']['RandomForest'][s]['best_model']).shap_values(te_s[feats].values)
    reg_total[s] = pd.Series(np.mean(np.abs(sv_s), axis=0), index=[f.replace('g__', '') for f in feats])

genera_e, order_e, nsel_e = shap_display_order(reg_total)
sheets['Fig9E_SHAP_reg_10seeds'] = pd.DataFrame(
    [{'seed': s, 'genus': g, 'selected_in_seed': int(g in reg_total[s].index),
      'mean_abs_shap': reg_total[s].get(g, 0.0)}
     for s in SEEDS for g in genera_e])
_e = sheets['Fig9E_SHAP_reg_10seeds']
top_e = list(order_e.index[:10])
_et = _e[_e['genus'].isin(top_e)].groupby('genus')['mean_abs_shap']
below['Fig9E_SHAP_reg_10seeds'] = (
    "Summary: the 10 genera shown in Fig. 9E (display order = top to bottom). Mean over 10 seeds with unselected seeds = 0; "
    "SD with ddof = 0 (error bars).",
    pd.DataFrame({'display_order': range(1, len(top_e) + 1), 'genus': top_e,
                  'n_seeds_selected': nsel_e[top_e].values,
                  'mean_rank_selected_seeds': order_e[top_e].values,
                  'mean_abs_shap_mean': _et.mean()[top_e].values,
                  'mean_abs_shap_sd':   _et.std(ddof=0)[top_e].values}))
notes['Fig9E_SHAP_reg_10seeds'] = (
    "Per-seed values behind Fig. 9E (Optuna × FC Top10, RandomForest regression, seeds 38–47, each seed's test samples "
    "with true label Early or Intermediate): mean |SHAP|. Rows = seed × every genus selected in at least one seed; "
    "selected_in_seed = 0 rows are set to 0, as in the figure. A summary of the 10 displayed genera is placed below.")

# ═════════════════════════ Table 1 ═════════════════════════
clf_models = ['RandomForest', 'XGBoost', 'SVC', 'LogisticRegression', 'KNN']
reg_models = ['LightGBM', 'RandomForest', 'XGBoost', 'SVR', 'LSTM']

def mean_sd(vals):
    return round(np.mean(vals), 3), round(np.std(vals), 3)

clf_rows = []
for cond in CONDS:
    for m in clf_models:
        a_m, a_s = mean_sd([clf_res[cond][m][s]['test_acc'] for s in SEEDS])
        f_m, f_s = mean_sd([clf_res[cond][m][s]['test_f1'] for s in SEEDS])
        clf_rows.append({'condition': CONDS[cond], 'model': m, 'test_acc_mean': a_m, 'test_acc_sd': a_s,
                         'test_f1_mean': f_m, 'test_f1_sd': f_s})
sheets['Table1_clf'] = pd.DataFrame(clf_rows)
notes['Table1_clf'] = "Classification: mean±SD across 10 seeds (38–47), test set."

reg_rows = []
for cond in CONDS:
    for m in reg_models:
        r_m, r_s = mean_sd([reg_res[cond][m][s]['test_r2'] for s in SEEDS])
        e_m, e_s = mean_sd([reg_res[cond][m][s]['test_mae'] for s in SEEDS])
        reg_rows.append({'condition': CONDS[cond], 'model': m, 'test_r2_mean': r_m, 'test_r2_sd': r_s,
                         'test_mae_mean': e_m, 'test_mae_sd': e_s})
sheets['Table1_reg'] = pd.DataFrame(reg_rows)
notes['Table1_reg'] = "Regression (Early+Intermediate only): mean±SD across 10 seeds (38–47), test set."

# ── Table1_validation_10seed (written by 6. validation.ipynb) ────
per_seed = read_sheet('validation_performance.xlsx', 'Per-seed')
sheets['Table1_validation_10seed'] = pd.DataFrame({
    'Seed':                 per_seed['seed'],
    'CV Test Accuracy':     per_seed['test_acc'],
    'CV Test F1-macro':     per_seed['test_f1'],
    'CV Test R²':           per_seed['test_r2'],
    'CV Test MAE':          per_seed['test_mae'],
    'Holdout Val Accuracy': per_seed['clf_acc'],
    'Holdout Val F1-macro': per_seed['clf_f1'],
    'Holdout Val R²':       per_seed['reg_r2'],
    'Holdout Val MAE':      per_seed['reg_mae'],
})
notes['Table1_validation_10seed'] = ("Per-seed CV test and holdout validation performance of the final model "
                                     "(Optuna × FC Top10 × RF), seeds 38–47.")

# ═════════════════════════ Supplementary tables ═════════════════════════
# ── TableS1_thresholds (same calculation as 2. labeling.ipynb) ────
thr_cols = ['Optuna_low', 'Optuna_high', 'KMeans_low', 'KMeans_high']
df_thr = pd.DataFrame([{'Seed': s,
                        'Optuna_low':  round(thr_opt[s]['tpc_thr_low'], 4),
                        'Optuna_high': round(thr_opt[s]['tpc_thr_high'], 4),
                        'KMeans_low':  round(float(thr_km[s]['tpc_thr_low']), 4),
                        'KMeans_high': round(float(thr_km[s]['tpc_thr_high']), 4)} for s in SEEDS])
s42 = df_thr[df_thr['Seed'] == LABEL_SEED].iloc[0]
stat_rows = [
    {'Seed': 'Mean',      **{c: round(df_thr[c].mean(), 4) for c in thr_cols}},
    {'Seed': 'SD',        **{c: round(df_thr[c].std(ddof=0), 4) for c in thr_cols}},
    {'Seed': 'CV (%)',    **{c: round(df_thr[c].std(ddof=0) / df_thr[c].mean() * 100, 2) for c in thr_cols}},
    {'Seed': 'Δ seed-42', **{c: round(abs(s42[c] - df_thr[c].mean()), 4) for c in thr_cols}},
]
sheets['TableS1_thresholds'] = pd.concat([df_thr, pd.DataFrame(stat_rows)], ignore_index=True)
notes['TableS1_thresholds'] = ("Optuna and K-means TPC thresholds fitted on each seed's training set; mean, "
                               "SD (population, n = 10), CV (%) and deviation of seed 42 from the mean. "
                               "No significant outlier detected by iterative Grubbs test (α = 0.05, n = 10).")

# ── TableS2_FC_features ──────────────────────────────────────────
rows = []
for s in SEEDS:
    feats = scr['cond1_optuna_fc10']['per_seed_features'][s]
    fc_df = scr['cond1_optuna_fc10']['per_seed_fc_df'][s].set_index('feature')
    for rank, feat in enumerate(feats, 1):
        direction = fc_df.loc[feat, 'direction'] if feat in fc_df.index else np.nan
        rows.append({'seed': s, 'rank': rank, 'genus': feat.replace('g__', ''),
                     'abs_CLR_change': round(fc_df.loc[feat, 'abs_change'], 3) if feat in fc_df.index else np.nan,
                     'direction_pos_late': int(direction >= 0) if not np.isnan(direction) else np.nan})
sheets['TableS2_FC_features'] = pd.DataFrame(rows)
notes['TableS2_FC_features'] = ("FC Top10 genera per seed (38–47), Optuna condition. "
                                "direction_pos_late=1 means CLR increases from early to late storage days.")

# ── TableS3_VIP_Optuna / TableS4_VIP_Kmeans ─────────────────────
for sheet, cond_key, lab in [('TableS3_VIP_Optuna', 'cond2_optuna_vip', 'Optuna'),
                             ('TableS4_VIP_Kmeans', 'cond4_kmeans_vip', 'K-means')]:
    sheets[sheet] = pd.DataFrame(
        [{'seed': s, 'genus': row['feature'].replace('g__', ''), 'VIP_score': round(row['vip_score'], 3)}
         for s in SEEDS for _, row in scr[cond_key]['per_seed_vip_df'][s].iterrows()
         if row['feature'] in scr[cond_key]['per_seed_features'][s]])           # selected genera only
    notes[sheet] = f"VIP≥1.0 genera per seed, PLS-DA on {lab} labels."

# ── TableS5_classification_full / TableS6_regression_full ────────
rows = []
for cond in CONDS:
    for m in clf_models:
        d = {k: [clf_res[cond][m][s][k] for s in SEEDS] for k in ['train_acc', 'test_acc', 'train_f1', 'test_f1']}
        row = {'condition': CONDS[cond], 'model': m}
        for k, v in d.items():
            row[f'{k}_mean'], row[f'{k}_sd'] = mean_sd(v)
        rows.append(row)
sheets['TableS5_classification_full'] = pd.DataFrame(rows)
notes['TableS5_classification_full'] = "All 20 combinations (4 conditions × 5 models). Mean±SD across 10 seeds."

rows = []
for cond in CONDS:
    for m in reg_models:
        d = {k: [reg_res[cond][m][s][k] for s in SEEDS] for k in ['train_r2', 'test_r2', 'train_mae', 'test_mae']}
        row = {'condition': CONDS[cond], 'model': m}
        for k, v in d.items():
            row[f'{k}_mean'], row[f'{k}_sd'] = mean_sd(v)
        rows.append(row)
sheets['TableS6_regression_full'] = pd.DataFrame(rows)
notes['TableS6_regression_full'] = "All 20 combinations (4 conditions × 5 models). Mean±SD across 10 seeds."

# ── TableS9–S11 (written by 8. zero_replacement_test.ipynb) ──────
ZR = 'zero_replacement_test.xlsx'
sheets['TableS9_FC_top10'] = read_sheet(ZR, 'FC_top10_all_seeds')
notes['TableS9_FC_top10'] = ("Fold-change Top 10 genera per seed under the current (1 × 10⁻⁶) and alternative "
                             "(half of the smallest non-zero value in each sample) zero replacements.")
sheets['TableS10_final_model'] = read_sheet(ZR, 'Final_model_performance')
notes['TableS10_final_model'] = ("Final model (Optuna × FC Top10 × RF) under the two zero replacements: CV test and "
                                 "holdout performance, mean and SD across 10 seeds, paired t-test (pairs = seeds).")
sheets['TableS10_per_seed_values'] = read_sheet(ZR, 'Final_model_per_seed')
notes['TableS10_per_seed_values'] = ("Per-seed CV test and holdout performance of the final model under the current "
                                     "and alternative zero replacements, used in Table S10.")
sheets['TableS11_SHAP_top10'] = read_sheet(ZR, 'SHAP_top10')
notes['TableS11_SHAP_top10'] = ("SHAP top 10 of the final model under the two zero replacements: genera selected in "
                                "at least 5 of the 10 seeds, ordered by mean rank over the seeds in which they were selected.")
sheets['TableS11_SHAP_excluded'] = read_sheet(ZR, 'SHAP_excluded')
notes['TableS11_SHAP_excluded'] = "Genera in the SHAP top 10 in fewer than 5 of the 10 seeds (not listed in Table S11)."
sheets['TableS11_SHAP_rank_per_seed'] = read_sheet(ZR, 'SHAP_rank_per_seed')
notes['TableS11_SHAP_rank_per_seed'] = "SHAP rank of each genus in each seed, used for the mean ranks in Table S11."

# ── TableS12–S13 (written by 9. per_seed_threshold_test.ipynb) ──
PS = 'per_seed_threshold_test.xlsx'
sheets['TableS12_per_seed_clf'] = read_sheet(PS, 'Perf_classification')
notes['TableS12_per_seed_clf'] = ("Classification with the seed-42 fixed thresholds vs each seed's own thresholds: "
                                  "4 conditions × 5 models, mean and SD across 10 seeds, paired t-test (pairs = seeds).")
sheets['TableS13_per_seed_reg'] = read_sheet(PS, 'Perf_regression')
notes['TableS13_per_seed_reg'] = ("Regression (Early + Intermediate) with the seed-42 fixed thresholds vs each seed's own "
                                  "thresholds: 4 conditions × 5 models, mean and SD across 10 seeds, paired t-test.")
sheets['TableS12_S13_per_seed_values'] = read_sheet(PS, 'Perf_per_seed_raw')
notes['TableS12_S13_per_seed_values'] = "Per-seed values used in Tables S12 and S13."

# ── TableS12–S13 holdout rows (written by per_seed_threshold_holdout.py) ──
PH = 'per_seed_threshold_holdout.xlsx'
sheets['TableS12_S13_holdout'] = read_sheet(PH, 'Holdout_final_model')
notes['TableS12_S13_holdout'] = ("Holdout validation (n = 27) of the final model (Optuna × FC Top10 × RF) with the seed-42 "
                                 "fixed thresholds vs each seed's own thresholds: mean and SD across 10 seeds, paired t-test.")
sheets['TableS12_S13_holdout_values'] = read_sheet(PH, 'Holdout_per_seed')
notes['TableS12_S13_holdout_values'] = "Per-seed holdout values used in Tables S12 and S13."

# ── TableS14_kmeans_reversed: K-means with the storage-day axis reversed ──
def extract_day(desc):
    m = re.match(r'A(\d+)-\d+', str(desc))
    return int(m.group(1)) if m else np.nan

def kmeans_thresholds(df, day, seed):
    """K-means procedure of 2. labeling.ipynb: standardized (day, TPC) → k = 3 → midpoints of sorted TPC centres."""
    km = KMeans(n_clusters=3, random_state=seed, n_init=20).fit(StandardScaler().fit_transform(np.c_[day, df[tpc_col]]))
    c = km.cluster_centers_[:, 1] * df[tpc_col].std(ddof=0) + df[tpc_col].mean()
    order = np.argsort(c)
    rank = np.argsort(order)[km.labels_]          # cluster id → TPC rank
    c = c[order]
    return rank, (c[0] + c[1]) / 2, (c[1] + c[2]) / 2

rows = []
for seed in SEEDS:
    tr = km_splits[seed]['train']
    day = tr['Description'].apply(extract_day).values
    cl_o, lo_o, hi_o = kmeans_thresholds(tr, day, seed)
    cl_r, lo_r, hi_r = kmeans_thresholds(tr, day.max() + day.min() - day, seed)   # day 0 ↔ day 18
    assert np.isclose(lo_o, thr_km[seed]['tpc_thr_low']) and np.isclose(hi_o, thr_km[seed]['tpc_thr_high'])
    rows.append({'seed': seed,
                 'thr_low_original': round(lo_o, 4), 'thr_low_reversed': round(lo_r, 4),
                 'thr_high_original': round(hi_o, 4), 'thr_high_reversed': round(hi_r, 4),
                 'n_mismatched_samples': int((cl_o != cl_r).sum()),
                 'identical': 'Yes' if (cl_o == cl_r).all() and np.isclose(lo_o, lo_r) and np.isclose(hi_o, hi_r) else 'No'})
sheets['TableS14_kmeans_reversed'] = pd.DataFrame(rows)
notes['TableS14_kmeans_reversed'] = ("K-means thresholds (log CFU/g) on each seed's training set with the original and "
                                     "reversed storage-day axis (day 0 ↔ day 18), and the number of training samples "
                                     "assigned to a different stage cluster.")

# ═════════════════════════ Supplementary figures ═════════════════════════
# ── FigS1_Optuna_Kmeans_paramspace ───────────────────────────────
tdf = opt_hist['trials_dataframe'].copy()
best_so_far, best_val = [], float('-inf')
for v in tdf['value']:
    if v is not None and not np.isnan(v) and v > best_val:
        best_val = v
    best_so_far.append(round(best_val, 4))
tdf['best_so_far'] = best_so_far
tdf = tdf[['number', 'value', 'best_so_far', 'params_tpc_thr_low', 'params_tpc_thr_high', 'state']]
tdf.columns = [c.replace('params_', '') for c in tdf.columns]
sheets['FigS1_Optuna_Kmeans_paramspace'] = tdf
km42 = thr_km[LABEL_SEED]
notes['FigS1_Optuna_Kmeans_paramspace'] = (
    f"Optuna seed-42 trial history ({len(tdf)} trials). Best: thr_low={opt_hist['best_params']['tpc_thr_low']:.4f}, "
    f"thr_high={opt_hist['best_params']['tpc_thr_high']:.4f}, score={opt_hist['best_value']:.2f}. "
    f"K-means seed-42 cluster centres (TPC): {', '.join(f'{float(c):.3f}' for c in km42['centers'])}; "
    f"thresholds: low={float(km42['tpc_thr_low']):.3f}, high={float(km42['tpc_thr_high']):.3f}.")

# ── FigS2_seed_feature_trends: CLR by day ────────────────────────
trend_df = genus_df.copy()
trend_df['Day'] = trend_df['Description'].apply(extract_day)
trend_rows = []
for feat in list(set(fc10_feats + vip_opt_feats + vip_km_feats)):
    if feat not in trend_df.columns:
        continue
    for day, grp in trend_df.groupby('Day'):
        trend_rows.append({
            'genus': feat.replace('g__', ''),
            'feature_set': ('FC_Top10' if feat in fc10_feats else
                            'VIP_Optuna' if feat in vip_opt_feats else 'VIP_Kmeans'),
            'storage_day': int(day),
            'mean_CLR': round(float(grp[feat].mean()), 4),
            'sd_CLR':   round(float(grp[feat].std(ddof=0)), 4),  # ddof=0: SD unified to the n-based definition
            'n': int(grp[feat].notna().sum()),
        })
sheets['FigS2_seed_feature_trends'] = (pd.DataFrame(trend_rows)
                                       .sort_values(['feature_set', 'genus', 'storage_day'])
                                       .reset_index(drop=True))
notes['FigS2_seed_feature_trends'] = ("Day-wise mean±SD CLR abundance for selected genera (seed-42 feature sets). "
                                      "All 270 samples used for trend.")

# ── FigS3_PCA_PLSDA_loadings ─────────────────────────────────────
sheets['FigS3_PCA_PLSDA_loadings'] = pd.DataFrame(loading_rows)
notes['FigS3_PCA_PLSDA_loadings'] = ("PCA (components_) and PLS-DA (x_loadings_) taxon loadings, seed-42 train, "
                                     "all 4 conditions.")

# ── FigS4_seed_model_performance ─────────────────────────────────
sheets['FigS4_seed_model_performance'] = pd.DataFrame([{
    'seed': s,
    'clf_train_acc': round(clf_res['cond1']['RandomForest'][s]['train_acc'], 4),
    'clf_test_acc':  round(clf_res['cond1']['RandomForest'][s]['test_acc'], 4),
    'clf_train_f1':  round(clf_res['cond1']['RandomForest'][s]['train_f1'], 4),
    'clf_test_f1':   round(clf_res['cond1']['RandomForest'][s]['test_f1'], 4),
    'reg_train_r2':  round(reg_res['cond1']['RandomForest'][s]['train_r2'], 4),
    'reg_test_r2':   round(reg_res['cond1']['RandomForest'][s]['test_r2'], 4),
    'reg_train_mae': round(reg_res['cond1']['RandomForest'][s]['train_mae'], 4),
    'reg_test_mae':  round(reg_res['cond1']['RandomForest'][s]['test_mae'], 4),
} for s in SEEDS])
notes['FigS4_seed_model_performance'] = ("Per-seed classification and regression performance across 10 seeds "
                                         "(38–47), plotted in Fig. S4.")

# ── FigS5_permutation_test + FigS5_ttest_results ─────────────────
metrics = ['acc', 'f1', 'r2', 'mae']
df_perm = pd.DataFrame(
    [{'permutation': i + 1,
      **{f'test1_{m}': round(float(perm_res['null_test1'][m][i]), 4) for m in metrics},
      **{f'test2_{m}': round(float(perm_res['null_test2'][m][i]), 4) for m in metrics}}
     for i in range(perm_res['n_perm'])])
df_perm.loc[len(df_perm)] = ['real_value'] + [round(float(perm_res['real'][f'{t}_{m}']), 4)
                                               for t in ['test1', 'test2'] for m in metrics]
df_perm.loc[len(df_perm)] = ['p_value'] + [float(perm_res['pval'][f'{t}_{m}'])        # (k + 1) / (N + 1)
                                           for t in ['test1', 'test2'] for m in metrics]
sheets['FigS5_permutation_test'] = df_perm
notes['FigS5_permutation_test'] = ("1,000 permutation null distributions. Test1=fixed features, Test2=full-pipeline. "
                                   "Real values and p-values in last two rows.")

tt_rows = []
for name, col_t, col_v in [('Accuracy', 'test_acc', 'clf_acc'), ('F1-macro', 'test_f1', 'clf_f1'),
                           ('R²', 'test_r2', 'reg_r2'), ('MAE', 'test_mae', 'reg_mae')]:
    t, p = stats.ttest_rel(per_seed[col_t], per_seed[col_v])
    tt_rows.append({'Metric': name, 't-statistic': round(float(t), 4), 'p-value': round(float(p), 4),
                    'Significant (α=0.05)': 'Yes' if p < 0.05 else 'No'})
sheets['FigS5_ttest_results'] = pd.DataFrame(tt_rows)
notes['FigS5_ttest_results'] = ("Paired t-test of CV test vs holdout validation performance across 10 seeds "
                                "(values in Table1_validation_10seed).")

# ── FigS6_SHAP_rank_stability (written by 6. validation.ipynb) ───
sheets['FigS6_SHAP_rank_stability'] = pd.concat(
    [val_shap['clf_shap_df'].assign(task='Classification'),
     val_shap['reg_shap_df'].assign(task='Regression (Early + Intermediate)')],
    ignore_index=True)[['task', 'seed', 'rank', 'genus', 'mean_abs_shap']]
notes['FigS6_SHAP_rank_stability'] = ("Top-10 SHAP rank and mean |SHAP| of each genus on each seed's holdout "
                                      "validation set, final model (Optuna × FC Top10 × RF).")

# ═════════════════════════ Main-figure panels added for the 2nd revision ═════════════════════════
# ── Fig6A / Fig6B: day-mean heatmaps before / after CLR (same calculation as 1. preprocessing.ipynb, heatmap cell) ──
rare_df    = load('genus_rare_filtered.pkl')['data']                      # relative abundance (%), rare-filtered, before CLR
train_desc = set(load('splits.pkl')['splits'][LABEL_SEED]['train']['Description'])   # seed-42 training set (ANOVA only)
clr_train  = genus_df[genus_df['Description'].isin(train_desc)].copy()
clr_train['Day'] = clr_train['Description'].apply(extract_day)
p_values, feats_valid = [], []
for feat in [c for c in genus_df.columns if c.startswith('g__')]:
    groups = [g[feat].dropna().values for _, g in clr_train.groupby('Day')]
    valid  = [g for g in groups if len(g) > 1]
    if len(valid) >= 2:
        _, p = stats.f_oneway(*valid)
        p_values.append(float(p) if not np.isnan(p) else 1.0)
        feats_valid.append(feat)
_, q_values, _, _ = multipletests(p_values, method='fdr_bh')
sig_features = [f for f, q in zip(feats_valid, q_values) if q < 0.05]

clr_plot = genus_df.copy()
clr_plot['Day'] = clr_plot['Description'].apply(extract_day)
clr_day_mean = clr_plot.groupby('Day')[sig_features].mean()               # all samples, after CLR
raw_plot = rare_df.copy()
raw_plot['Day'] = raw_plot['Description'].apply(extract_day)
raw_day_mean = raw_plot.groupby('Day')[[f for f in sig_features if f in raw_plot.columns]].mean()
sorted_features = (clr_day_mean.max() - clr_day_mean.min()).abs().sort_values(ascending=False).index.tolist()
sorted_in_raw   = [f for f in sorted_features if f in raw_day_mean.columns]

def day_matrix(day_mean, cols):
    """Heatmap matrix: rows = storage day, columns = genera in x-axis order (left → right)."""
    m = day_mean[cols].round(4)
    m.columns = [c.replace('g__', '') for c in cols]
    return m.reset_index().rename(columns={'Day': 'storage_day'})

sheets['Fig6A_heatmap_before_CLR'] = day_matrix(raw_day_mean, sorted_in_raw)
sheets['Fig6B_heatmap_after_CLR']  = day_matrix(clr_day_mean, sorted_features)
_sel6 = (f"the {len(sig_features)} genera with a significant storage-day effect (one-way ANOVA on CLR values of the "
         "seed-42 training set, Benjamini–Hochberg q < 0.05); columns in x-axis order (largest change of the day-mean CLR first)")
notes['Fig6A_heatmap_before_CLR'] = (f"Day-wise mean relative abundance (%) before CLR (rare-taxon filtered), all "
                                     f"{len(raw_plot)} samples, for {_sel6}. Rows = storage day.")
notes['Fig6B_heatmap_after_CLR']  = (f"Day-wise mean CLR value, all {len(clr_plot)} samples, for {_sel6}. Rows = storage day.")

# ── Fig9G, 9I–9K: seed-42 holdout set (same calculations as 6. validation.ipynb) ──
assert best_seed == reg_seed == LABEL_SEED          # the holdout panels use the seed-42 classification and regression models
va42   = clf_splits_9b[LABEL_SEED]['val'].copy()    # split file of the classification pipeline, as in 6. validation.ipynb
Xv     = va42[best_feats].values
yv     = va42['Label'].values
yv_hat = best_model.predict(Xv)

# Fig. 9G: confusion matrix (rows/columns in the order Early, Intermediate, Late)
cm_v = confusion_matrix(yv, yv_hat, labels=STAGE_ORDER)
sheets['Fig9G_holdout_confusion'] = pd.DataFrame(
    [{'true_label': LABEL_MAP[a], 'predicted_label': LABEL_MAP[b], 'count': int(cm_v[i, j])}
     for i, a in enumerate(STAGE_ORDER) for j, b in enumerate(STAGE_ORDER)])
notes['Fig9G_holdout_confusion'] = (
    f"Confusion matrix of the final classification model on the seed-{LABEL_SEED} holdout set (n = {len(va42)}): "
    f"accuracy = {accuracy_score(yv, yv_hat):.3f}, macro-F1 = {f1_score(yv, yv_hat, average='macro', zero_division=0):.3f} "
    "for seed 42.")

# Fig. 9I: SHAP of the classification model on the holdout set
sv_v  = shap.TreeExplainer(best_model).shap_values(Xv)
cls_v = list(best_model.classes_)
def sv_class(k):
    return sv_v[k] if isinstance(sv_v, list) else sv_v[:, :, k]
sv_stack = np.stack([sv_class(k) for k in range(len(cls_v))], axis=0)            # (classes, samples, genera)
top_v    = np.argsort(np.mean(np.abs(sv_stack), axis=(0, 1)))[::-1][:10]        # display order of Fig. 9I
stage_v  = {lbl: np.mean(np.abs(sv_class(cls_v.index(lbl))), axis=0) for lbl in STAGE_ORDER}
sheets['Fig9I_SHAP_clf_holdout_bar'] = pd.DataFrame(
    [{'display_order': r + 1, 'genus': best_feats[i].replace('g__', ''),
      'mean_abs_shap_early':        stage_v[0][i],
      'mean_abs_shap_intermediate': stage_v[2][i],
      'mean_abs_shap_late':         stage_v[1][i],
      'mean_abs_shap_total':        stage_v[0][i] + stage_v[2][i] + stage_v[1][i]}   # bar length (stacked as in the figure)
     for r, i in enumerate(top_v)])
notes['Fig9I_SHAP_clf_holdout_bar'] = (
    f"Mean |SHAP| of each stage output over the seed-{LABEL_SEED} holdout samples (n = {len(va42)}) for the 10 genera of "
    "Fig. 9I, their sum and the display order (top to bottom).")

# Fig. 9J: end-to-end scatter — TPC is predicted only for samples the classifier assigned to Early or Intermediate
fm_v    = np.isin(yv_hat, [0, 2])
va_reg  = va42[fm_v]
tpc_hat = reg_model.predict(va_reg[reg_feats].values)
sheets['Fig9J_holdout_scatter'] = pd.DataFrame({
    'sample_id':       va_reg['Description'].values,
    'true_label':      va_reg['Label'].map(LABEL_MAP).values,
    'predicted_label': [LABEL_MAP[c] for c in yv_hat[fm_v]],
    'observed_TPC':    va_reg[tpc_col].values.round(3),
    'predicted_TPC':   tpc_hat.round(3)})
notes['Fig9J_holdout_scatter'] = (
    f"Observed vs predicted TPC (log CFU/g) on the seed-{LABEL_SEED} holdout set for the {len(va_reg)} samples classified "
    f"as Early or Intermediate: R² = "
    f"{r2_score(va_reg[tpc_col], tpc_hat):.3f}, MAE = {mean_absolute_error(va_reg[tpc_col], tpc_hat):.3f} for seed 42.")

# Fig. 9K: SHAP of the regression model on the holdout samples with true label Early or Intermediate
va_fm = va42[va42['Label'] != 1]
X_fm  = va_fm[reg_feats].values
sv_k  = np.array(shap.TreeExplainer(reg_model).shap_values(X_fm))
sheets['Fig9K_SHAP_reg_holdout'] = pd.DataFrame(
    [{'sample_id': va_fm['Description'].iloc[i], 'true_label': LABEL_MAP[va_fm['Label'].iloc[i]],
      'genus': feat.replace('g__', ''), 'SHAP_value': round(float(sv_k[i, j]), 5),
      'feature_value': round(float(X_fm[i, j]), 4)}
     for i in range(len(va_fm)) for j, feat in enumerate(reg_feats)])
notes['Fig9K_SHAP_reg_holdout'] = (
    f"SHAP values (TreeExplainer) of the final regression model on the seed-{LABEL_SEED} holdout samples with true label "
    f"Early or Intermediate (n = {len(va_fm)}); feature_value = CLR value of the genus (beeswarm colour).")

# ═════════════════════════ Write ═════════════════════════
sheet_order = [
    'Fig6A_heatmap_before_CLR', 'Fig6B_heatmap_after_CLR',                       # added (2nd revision)
    'Fig6C_Fig6D', 'Fig7_A_C_heatmap', 'Fig7_D_K_PCA_PLSDA',
    'Fig8_confusion_matrix', 'Fig8_ROC', 'Fig8_AUC_10seeds', 'Fig8_scatter_regression',
    'Fig9_SHAP_clf', 'Fig9_SHAP_reg', 'Fig9B_SHAP_clf_10seeds', 'Fig9E_SHAP_reg_10seeds',
    'Fig9G_holdout_confusion', 'Fig9I_SHAP_clf_holdout_bar',                             # added (2nd revision)
    'Fig9J_holdout_scatter', 'Fig9K_SHAP_reg_holdout',                                    # added (2nd revision)
    'Table1_clf', 'Table1_reg', 'Table1_validation_10seed',
    'TableS1_thresholds', 'TableS2_FC_features', 'TableS3_VIP_Optuna', 'TableS4_VIP_Kmeans',
    'TableS5_classification_full', 'TableS6_regression_full',
    'TableS9_FC_top10', 'TableS10_final_model', 'TableS10_per_seed_values',
    'TableS11_SHAP_top10', 'TableS11_SHAP_excluded', 'TableS11_SHAP_rank_per_seed',
    'TableS12_per_seed_clf', 'TableS13_per_seed_reg', 'TableS12_S13_per_seed_values',
    'TableS12_S13_holdout', 'TableS12_S13_holdout_values', 'TableS14_kmeans_reversed',
    'FigS1_Optuna_Kmeans_paramspace', 'FigS2_seed_feature_trends', 'FigS3_PCA_PLSDA_loadings',
    'FigS4_seed_model_performance', 'FigS5_permutation_test', 'FigS5_ttest_results',
    'FigS6_SHAP_rank_stability',
]
assert set(sheet_order) == set(sheets), set(sheets) ^ set(sheet_order)
# added checks for the new sheets: no duplicates, every sheet described in Read Me, Excel's 31-character sheet-name limit
assert len(sheet_order) == len(set(sheet_order)) == 45, len(sheet_order)
assert all(name in notes for name in sheet_order), [n for n in sheet_order if n not in notes]
assert all(len(name) <= 31 for name in sheet_order), [n for n in sheet_order if len(n) > 31]

read_me = pd.DataFrame([{'Sheet': name, 'Description': notes[name]} for name in sheet_order])
with pd.ExcelWriter(OUT, engine='openpyxl') as writer:
    read_me.to_excel(writer, sheet_name='Read Me', index=False)
    for name in sheet_order:
        sheets[name].to_excel(writer, sheet_name=name, index=False)
        if name in below:   # summary table under the per-seed table (Fig. 9B·E sheets)
            caption, df_below = below[name]
            r0 = len(sheets[name]) + 2                        # header + rows, then one blank row
            writer.sheets[name].cell(row=r0 + 1, column=1, value=caption)
            df_below.to_excel(writer, sheet_name=name, index=False, startrow=r0 + 1)

print(f"Saved → {OUT}  ({len(sheet_order)} sheets + Read Me)")
