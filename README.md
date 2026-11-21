# Microbial community succession and machine learning-based estimation of microbial load during refrigerated storage

> Code repository for the manuscript submitted to *Nature Communications*
> Manuscript ID: NCOMMS-26-034432

**Authors:** Eun Seo Lee<sup>a</sup>, Tae Ho Kim<sup>b</sup>, Hansol Doh<sup>a</sup>, Young Suk Kim<sup>a</sup>, Woo-Ju Kim<sup>b,c</sup>, Sun Ae Kim<sup>a</sup>

<sup>a</sup> Department of Food Science and Biotechnology, Ewha Womans University, Seoul, South Korea\
<sup>b</sup> Department of Food Science and Biotechnology, Seoul National University of Science and Technology (SeoulTech), Seoul, South Korea\
<sup>c</sup> Research Institute of Food and Biotechnology, Seoul National University of Science and Technology (SeoulTech), Seoul, South Korea

**Corresponding authors:** Sun Ae Kim (<sunaekim@ewha.ac.kr>), Woo-Ju Kim (<wjk700@seoultech.ac.kr>)

---

## Overview

This repository contains all analysis code used to estimate the microbial load stage and total plate count (TPC) of refrigerated chicken breast from 16S rRNA amplicon sequencing data (genus-level CLR-transformed microbiome profiles).
The pipeline implements a two-stage approach: **(1) microbial load stage classification → (2) TPC regression**, with systematic controls for data leakage and label sensitivity.

### Microbial Load Stage Definition

Labels are defined on a continuous TPC (log CFU/g) scale using threshold-based assignment:

| Label                  | TPC (log CFU/g)                      | Code value |
| ---------------------- | ------------------------------------ | ---------- |
| **Early**        | < threshold_low                      | 0          |
| **Intermediate** | threshold_low ≤ · < threshold_high | 2          |
| **Late**         | ≥ threshold_high                    | 1          |

Thresholds are optimized independently on each seed's training data using Optuna (TPE sampler) or K-means clustering. The threshold derived from seed-42 is selected as the representative value (inter-seed std ≤ 0.06 log CFU/g) and applied uniformly across all 10 seeds to ensure biologically consistent label assignment. Because the seed-42 training set contains samples from the test and validation sets of other seeds, `9. per_seed_threshold_test.ipynb` repeats the analysis with each seed's own thresholds, with `per_seed_threshold_holdout.py` for the holdout validation of the final model.

---

## Repository Structure

```text
.
├── 1. preprocessing.ipynb        # CLR transformation, quality filtering, train/test/val split
├── 2. labeling.ipynb             # TPC-based label assignment (Optuna / K-means), threshold stability test
├── 3. screening.ipynb            # Feature selection — 4 conditions × 10 seeds
├── 4. classification.ipynb       # 5-model classification + SHAP + sensitivity analysis
├── 5. regression.ipynb           # 5-model regression (Early + Intermediate only) + SHAP
├── 6. validation.ipynb           # Independent holdout validation (n = 27); performance reported as mean ± SD across 10 seeds (38–47) with paired t-test vs. CV test
├── 7. leakage_test.ipynb         # Permutation-based leakage verification (1,000 permutations); CV test vs. holdout validation comparison across 10 seeds
├── 8. zero_replacement_test.ipynb    # Feature selection, performance and SHAP under an alternative zero replacement
├── 9. per_seed_threshold_test.ipynb  # Classification and regression with each seed's own thresholds
├── per_seed_threshold_holdout.py # Holdout validation of the final model with each seed's own thresholds
├── build_source_data.py          # Builds output/Source_Data.xlsx from the saved outputs
├── requirements.txt              # Pinned package versions (Python 3.13.9)
├── LICENSE                       # MIT License
│
├── data/
│   └── Sanitation.xlsx           # Relative abundance (%, phylum to genus level) and TPC (log CFU/g) of each sample (input)
└── output/                       # All generated files
    ├── pkl/                      # Serialized intermediate results (see PKL File Descriptions)
    ├── figures_*/, class_*/, reg_*/, val_figures/   # Figures
    └── *.xlsx                    # Result tables (classification_performance.xlsx, regression_performance.xlsx, ...)
```

---

## Data Availability

Raw 16S rRNA amplicon sequencing data (FASTQ files) have been deposited in the NCBI Sequence Read Archive (SRA) under BioProject accession PRJNA1377274 (https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1377274).

The relative abundance tables (%, phylum to genus level) and TPC (log CFU/g) of each sample are available in this repository as data/Sanitation.xlsx, and the CLR-transformed genus table used in the analysis as output/pkl/genus_preprocessed.pkl. All derived intermediate outputs (labeled splits, screening results, trained models) are also provided as serialized .pkl files to allow reproduction of downstream analyses without re-running computationally intensive steps.

> The datasets generated and/or analysed during the current study are available in this GitHub repository and the associated SRA entry (PRJNA1377274). Correspondence and requests for materials should be addressed to the corresponding authors.

---

## PKL File Descriptions (`output/pkl/`)

PKL files allow starting the pipeline from any intermediate stage without re-running earlier steps.

| File | Generated by | Contents |
| ------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------- |
| `genus_preprocessed.pkl` | `1. preprocessing.ipynb` | CLR-transformed genus table, sample metadata (n = 270, prior to splitting) |
| `genus_rare_filtered.pkl` | `1. preprocessing.ipynb` | Relative abundance table before zero replacement and CLR (used in notebook 8 and `build_source_data.py`) |
| `splits.pkl` | `1. preprocessing.ipynb` | Base train/test/val indices for each of the 10 seeds (before labeling) |
| `labeled_splits.pkl` | `2. labeling.ipynb` | Optuna-based labels; threshold derived from seed-42 and applied uniformly across 10 seeds |
| `kmeans_labeled_splits.pkl` | `2. labeling.ipynb` | K-means-based labels; same fixed-threshold approach |
| `threshold_log.pkl` | `2. labeling.ipynb` | Optuna TPC thresholds independently fit for all 10 label seeds |
| `kmeans_threshold_log.pkl` | `2. labeling.ipynb` | K-means TPC thresholds independently fit for all 10 label seeds |
| `avg_thresholds.pkl` | `2. labeling.ipynb` | Seed-42 thresholds applied to all 10 seeds (Optuna and K-means) |
| `optuna_trial_history_seed42.pkl` | `2. labeling.ipynb` | Optuna trial history for the seed-42 training set |
| `screening_results.pkl` | `3. screening.ipynb` | Selected features per condition per seed (4 conditions × 10 seeds) |
| `classification_results.pkl` | `4. classification.ipynb` | Per-seed classification metrics (5 models × 4 conditions × 10 seeds) |
| `best_models.pkl` | `4. classification.ipynb` | Top-5 by Accuracy / Top-5 by F1 — summary DataFrames |
| `sensitivity_results.pkl` | `4. classification.ipynb` | F1 / Accuracy for 10 label seeds × 10 train seeds |
| `best_model_pipeline.pkl` | `4. classification.ipynb` | Final classification pipeline (Optuna × FC Top10 × RF, seed-42); used in `6. validation.ipynb`, `7. leakage_test.ipynb` and `build_source_data.py` |
| `regression_results.pkl` | `5. regression.ipynb` | Per-seed regression metrics (5 models × 4 conditions × 10 seeds) |
| `reg_sensitivity_results.pkl` | `5. regression.ipynb` | R² / MAE for 10 label seeds × 10 train seeds |
| `best_reg_pipeline.pkl` | `5. regression.ipynb` | Final regression pipeline (Optuna × FC Top10 × RF, seed-42) |
| `validation_results.pkl` | `6. validation.ipynb` | Holdout validation metrics (n = 27) |
| `validation_shap_allseeds.pkl` | `6. validation.ipynb` | SHAP values and ranks on the validation sets of all 10 seeds |
| `permutation_results.pkl` | `7. leakage_test.ipynb` | Null distributions, observed values and p values of the two permutation tests |
| `zero_replacement_results.pkl` | `8. zero_replacement_test.ipynb` | Results under the current and alternative zero replacements |
| `zero_replacement_grid.pkl` | `8. zero_replacement_test.ipynb` | 4 conditions × 5 models under the alternative zero replacement |
| `per_seed_threshold_results.pkl` | `9. per_seed_threshold_test.ipynb` | Results with each seed's own thresholds |

---

## Execution Order

All paths are relative to the repository root, so the notebooks run without editing any path. Open them from the repository root (e.g., start `jupyter lab` in this folder) and run them in the order below, each with *Restart Kernel and Run All Cells*. All outputs are written to `output/`.

```mermaid
flowchart LR
    N1[1. preprocessing] --> N2[2. labeling] --> N3[3. screening] --> N4[4. classification] --> N5[5. regression]
    N5 --> N6[6. validation] --> N7[7. leakage_test] --> N8[8. zero_replacement_test] --> N9[9. per_seed_threshold_test]
    N9 --> H[per_seed_threshold_holdout.py] --> S[build_source_data.py]
```

1. preprocessing.ipynb ↓ genus_preprocessed.pkl, genus_rare_filtered.pkl, splits.pkl
2. labeling.ipynb ↓ labeled_splits.pkl, kmeans_labeled_splits.pkl, threshold_log.pkl, kmeans_threshold_log.pkl, avg_thresholds.pkl, optuna_trial_history_seed42.pkl
3. screening.ipynb ↓ screening_results.pkl
4. classification.ipynb ↓ classification_results.pkl, best_models.pkl, sensitivity_results.pkl, best_model_pipeline.pkl
5. regression.ipynb ↓ regression_results.pkl, reg_sensitivity_results.pkl, best_reg_pipeline.pkl
6. validation.ipynb ↓ validation_results.pkl, validation_shap_allseeds.pkl
7. leakage_test.ipynb ↓ permutation_results.pkl
8. zero_replacement_test.ipynb ↓ zero_replacement_results.pkl, zero_replacement_grid.pkl
9. per_seed_threshold_test.ipynb ↓ per_seed_threshold_results.pkl
10. `python per_seed_threshold_holdout.py` ↓ output/per_seed_threshold_holdout.xlsx
11. `python build_source_data.py` ↓ output/Source_Data.xlsx

---

## Analysis Design

### 4 Conditions

All downstream analyses (classification and regression) are evaluated under four combinations of labeling method and feature selection method:

| Condition | Labeling threshold | Feature selection                         |
| --------- | ------------------ | ----------------------------------------- |
| Cond 1    | Optuna (TPE)       | Fold-change Top 10                        |
| Cond 2    | Optuna (TPE)       | Day-ANOVA (q < 0.05) → PLS-DA VIP ≥ 1.0 |
| Cond 3    | K-means            | Fold-change Top 10                        |
| Cond 4    | K-means            | Day-ANOVA (q < 0.05) → PLS-DA VIP ≥ 1.0 |

Feature selection is performed on **training data only** in each seed to prevent data leakage.

### Seed Strategy

- **label_seed**: Controls Optuna/K-means threshold optimization. Thresholds are fit independently for all 10 label seeds (38–47); the seed-42 threshold is selected as representative (inter-seed std ≤ 0.06 log CFU/g) and applied uniformly to all train/test/val splits.
- **train_seed**: Controls random 70/20/10 train/test/val splits (`train_test_split`). 10 seeds (38–47) used for all performance reporting (mean ± SD).
- **Sensitivity analysis**: 10 label seeds × 10 train seeds = 100 combinations tested in `4. classification.ipynb` and `5. regression.ipynb`.

### Leakage Controls

1. **Label generation**: Labels are assigned by TPC thresholds. The thresholds are derived from the seed-42 training set (Optuna: TPC with a storage-day-based score; K-means: standardized storage day and TPC) and applied to all seeds.
2. **Feature selection**: FC Top10 and VIP computed from training data only in each seed.
3. **Permutation test** (`7. leakage_test.ipynb`): Two-test design with the seed-42 classification and regression models — (i) fixed-feature permutation with the seed-42 FC Top10 (the same 10 genera for classification and regression) to confirm model performance exceeds chance, (ii) full-pipeline permutation, in which 10 genera are re-selected in the training part of every fold of every permutation with a label-based stage-mean FC, to confirm feature selection itself does not introduce leakage. The main FC (storage-day means) does not use the labels, so its selection would not change when the labels are permuted. Both tests share the same 1,000-permutation sequence (rng seed = 42).

---

## Software Requirements

All results were produced with Python 3.13.9 (CPU only). Install the pinned dependencies via:

```bash
pip install -r requirements.txt
```

### Core Dependencies

| Package      | Version | Purpose                               |
| ------------ | ------- | ------------------------------------- |
| numpy        | 2.3.5   | Array operations                      |
| pandas       | 2.3.3   | Data manipulation                     |
| scikit-learn | 1.7.2   | ML models, CV, metrics                |
| xgboost      | 3.2.0   | XGBoost classifier/regressor          |
| lightgbm     | 4.6.0   | LightGBM regressor                    |
| optuna       | 4.9.0   | Bayesian threshold optimization       |
| shap         | 0.52.0  | SHAP feature importance               |
| scipy        | 1.16.3  | Statistics (ANOVA, chi2, paired t-test) |
| statsmodels  | 0.14.5  | BH FDR correction                     |
| matplotlib   | 3.10.6  | Visualization                         |
| seaborn      | 0.13.2  | Heatmaps                              |
| tensorflow   | 2.21.0  | LSTM regressor                        |
| openpyxl     | 3.1.5   | Reading Sanitation.xlsx, writing Excel tables |
| ipython      | 9.7.0   | Notebook display                      |

The full list is in `requirements.txt`.

---

## Reproducibility Notes

- All random operations use explicit `random_state` / `numpy.random.default_rng(42)` where applicable.
- Model internal random states are fixed at 42 across all train seeds; this isolates data-partitioning variability as the sole source of performance variance across the 10 seeds.
- `labeled_splits.pkl` and `kmeans_labeled_splits.pkl` store the exact train/test/val indices for each seed; re-loading these files guarantees identical splits across environments.
- The permutation test in `7. leakage_test.ipynb` uses a fixed `rng = np.random.default_rng(42)` shared between Test 1 and Test 2, enabling direct null-distribution comparison.
- SHAP values in `4. classification.ipynb` and `5. regression.ipynb` are computed on the test set of each seed with that seed's model (seed 42 for the beeswarm plots, all 10 seeds for the aggregated rankings); results are deterministic given the fixed seeds.
- LSTM results vary slightly between runs because of non-deterministic operations in TensorFlow. All reported LSTM values are those of the stored outputs (`output/pkl/regression_results.pkl` and, for the per-seed-threshold analysis, `output/pkl/per_seed_threshold_results.pkl`).

---

## License

The code in this repository is released under the MIT License; see [LICENSE](LICENSE).

---

## Citation

If you use this code or data, please cite:

> Lee, E.S., Kim, T.H., Doh, H., Kim, Y.S., Kim, W.J., and Kim, S.A. Microbial community succession and machine learning-based estimation of microbial load during refrigerated storage. *Nature Communications* (under review, 2026). [DOI — to be updated upon acceptance]
