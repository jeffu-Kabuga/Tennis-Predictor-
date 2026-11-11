# Phase 5 Implementation Log — Model Interpretation

**Date:** 2026-11-11
**Status:** COMPLETE

---

## Objective

Explain which features drive the Random Forest's predictions using two
complementary importance methods — built-in (impurity-based) and permutation
importances — and save a horizontal bar chart summarising the top-15
permutation importances.

---

## Steps Executed

### 1. Load model and validation data
`models/rf_model.pkl` loaded with `joblib.load`.
`data/val.csv` loaded with `pandas.read_csv`.

### 2. Build feature matrix
`build_features(val_df)` applied (transform-only, encoder already fitted in Phase 4).
Result: 10,361 rows × 23 features.

### 3. Built-in (impurity-based) importances
`model.feature_importances_` paired with feature names, sorted descending.

### 4. Permutation importances
`sklearn.inspection.permutation_importance` run on the full validation set with
`n_repeats=10, random_state=42, n_jobs=-1`.

### 5. Comparison note printed
Both methods agree that `Rank_diff` is the single most important feature.

### 6. Chart saved
Top-15 permutation importances rendered as a horizontal bar chart and saved to
`models/permutation_importance.png` (62 KB, 150 dpi).

---

## Results

### Top-10 Built-in (Impurity-Based) Importances

| Rank | Feature          | Importance |
|------|------------------|-----------|
| 1    | Rank_diff        | 0.193056  |
| 2    | Rank_2           | 0.157629  |
| 3    | Rank_1           | 0.156120  |
| 4    | Pts_diff         | 0.138796  |
| 5    | Pts_1            | 0.114170  |
| 6    | Pts_2            | 0.113453  |
| 7    | Round            | 0.044864  |
| 8    | Surface_Hard     | 0.010213  |
| 9    | Surface_Clay     | 0.009458  |
| 10   | Series_ATP250    | 0.006761  |

### Top-10 Permutation Importances

| Rank | Feature                  | Mean Δ Accuracy |
|------|--------------------------|----------------|
| 1    | Rank_diff                | +0.026638      |
| 2    | Rank_2                   | +0.002278      |
| 3    | Rank_1                   | +0.000984      |
| 4    | Series_Masters Cup       | +0.000068      |
| 5–10 | (near-zero / negative)   | ≤ 0.000        |

### Comparison Note

> `Rank_diff` ranks **#1 in both methods**.
> `Rank_1` and `Rank_2` appear in the top-3 of both methods (3 of 10 shared features).
>
> **Observation — collinearity:** Built-in importances spread importance across
> `Rank_diff`, `Rank_1`, `Rank_2`, `Pts_diff`, `Pts_1`, `Pts_2` roughly evenly.
> Permutation importances collapse most weight onto `Rank_diff` alone.  This is
> the expected behaviour when highly correlated features are present: the
> impurity method over-counts importance by splitting it across correlated
> siblings, while permutation correctly identifies `Rank_diff` as the one
> sufficient predictor.  Features ranked 5–10 by permutation show near-zero or
> marginally negative importance, indicating the one-hot-encoded context
> columns (surface, series, court) add minimal signal once ranking is accounted
> for.

---

## Output File Summary

| File                              | Description                                    |
|-----------------------------------|------------------------------------------------|
| `src/04_interpret_model.py`       | Interpretation script                          |
| `models/permutation_importance.png` | Horizontal bar chart — top-15 perm. importances |

---

## Deliverable Checklist

- [x] Top-10 built-in importances printed to console
- [x] Top-10 permutation importances printed to console
- [x] `models/permutation_importance.png` saved (62 KB)
- [x] Comparison note printed — `Rank_diff` ranks #1 in both methods;
      collinearity between `Rank_1/2` and `Rank_diff` noted.
