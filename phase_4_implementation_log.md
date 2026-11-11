# Phase 4 Implementation Log — Model Building

**Date:** 2026-11-11
**Status:** COMPLETE

---

## Objective

Train a `RandomForestClassifier` on the training split and serialise it to
`models/rf_model.pkl`.  Evaluate on both the validation and test splits and
confirm ROC-AUC > 0.65 on each.

---

## Steps Executed

### 1. Load splits
`data/train.csv`, `data/val.csv`, `data/test.csv` loaded with `pandas.read_csv`.

### 2 & 3. Build feature matrices
`build_features(train_df, fit=True)` re-fit the encoder on training data and
produced `models/encoder.pkl` + `models/feature_columns.json` (23 columns).
`build_features(val_df)` and `build_features(test_df)` applied the fitted
encoder for transform-only.

### 4. Extract labels
`Player1_wins` column separated from each split before passing to the model.

### 5. Fit model
```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
)
```

### 6 & 7. Evaluate
Metrics printed for both validation and test sets (see results table below).

### 8. Serialise model
`joblib.dump(model, 'models/rf_model.pkl')` — file written successfully.

### 9. Confirm loadable
`joblib.load('models/rf_model.pkl')` returns an object with `predict_proba`
— confirmed.

---

## Evaluation Results

| Metric     | Validation | Test   |
|------------|-----------|--------|
| Accuracy   | 0.6137    | 0.6249 |
| ROC-AUC    | 0.6696    | 0.6791 |
| Log-loss   | 0.6632    | 0.6604 |

### Classification Reports

**Validation Set (10 361 rows)**
```
              precision    recall  f1-score   support
Player2 wins       0.61      0.62      0.61      5095
Player1 wins       0.62      0.61      0.62      5266
    accuracy                           0.61     10361
```

**Test Set (10 362 rows)**
```
              precision    recall  f1-score   support
Player2 wins       0.63      0.63      0.63      5279
Player1 wins       0.62      0.62      0.62      5083
    accuracy                           0.62     10362
```

---

## Output File Summary

| File                  | Description                              |
|-----------------------|------------------------------------------|
| `src/03_train_model.py` | Training script                        |
| `models/rf_model.pkl`   | Fitted `RandomForestClassifier` (200 trees) |

---

## Deliverable Checklist

- [x] `models/rf_model.pkl` written to disk
- [x] Validation ROC-AUC printed: **0.6696** (target > 0.65 — PASSED)
- [x] Test ROC-AUC printed: **0.6791** (close to val AUC — no overfitting)
- [x] No leakage features present in the feature matrix (confirmed by Phase 3)

---

## Notes

- Test AUC (0.6791) is marginally higher than validation AUC (0.6696), which
  is normal random variation on similarly-sized 15 % holdout sets.
- The Windows console `cp1252` codec cannot render the Unicode arrow `→`;
  replaced with ASCII `->` in print statements.
- `build_features` is loaded via `importlib.util.spec_from_file_location`
  because the source filename begins with a digit, which prevents a standard
  Python import.
