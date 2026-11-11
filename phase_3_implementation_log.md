# Phase 3 Implementation Log — Feature Engineering

**Date:** 2026-11-11
**Status:** COMPLETE

---

## Objective

Provide a reusable `build_features(df, fit=False, encoder=None)` function in
`src/02_feature_engineering.py` that transforms any split (or a single-row
DataFrame) into a numeric feature matrix ready for the model.

---

## Steps Executed

### 1. ROUND_ORDER dict
Defined a dict mapping the seven canonical round strings to integers 1–7.
Unknown round values (e.g. `Round Robin`) map to `-1` via `.fillna(-1)`.

### 2. Round ordinal encoding
`df['Round'] = df['Round'].map(ROUND_ORDER).fillna(-1).astype(int)`
Applied before assembling the feature matrix.

### 3. Derived numeric features
- `Rank_diff = Rank_1 - Rank_2`
- `Pts_diff  = Pts_1  - Pts_2`

### 4. One-hot encoding
`OneHotEncoder(handle_unknown='ignore', sparse_output=False)` fitted on
`Surface`, `Court`, `Series`.
- `fit=True`  → fits + saves to `models/encoder.pkl`
- `fit=False` → loads `models/encoder.pkl` (or accepts pre-loaded encoder)

### 5. Assembled feature matrix (23 columns)
Numeric base (9): `Rank_1`, `Rank_2`, `Rank_diff`, `Pts_1`, `Pts_2`,
`Pts_diff`, `pts_available`, `Round`, `Best of`
One-hot (14): 4 Surface, 2 Court, 8 Series dummies

### 6. Saved canonical column list
`models/feature_columns.json` — 23 column names in exact model-input order.

---

## Output File Summary

| File | Description |
|------|-------------|
| `src/02_feature_engineering.py` | Module with `build_features()` |
| `models/encoder.pkl` | Fitted `OneHotEncoder` (Surface / Court / Series) |
| `models/feature_columns.json` | 23 feature column names in canonical order |

---

## Verification Results

| Check | Result |
|-------|--------|
| Training matrix shape | 48,351 x 23 |
| Validation matrix shape | 10,361 x 23 |
| Test matrix shape | 10,362 x 23 |
| Single-row shape | 1 x 23 |
| `encoder.pkl` present | Yes |
| `feature_columns.json` present | Yes |
| No leakage columns (`Date`, `Player1_wins`) | Confirmed dropped |
| Import without side-effects | Confirmed (guard: `if __name__ == "__main__"`) |

---

## Deviations from Plan

- `Round Robin` exists in the training data but is not listed in the plan's
  `ROUND_ORDER`.  The plan explicitly states "Unknown values map to -1",
  so `Round Robin` → -1. No change needed.
- `Date` column was retained in the splits by Phase 2 (noted in that log).
  It is silently dropped inside `build_features` alongside `Player1_wins`.
- `sparse_output=False` is the sklearn >= 1.2 replacement for the deprecated
  `sparse=False`; used to silence the FutureWarning on the installed version.

---

## Deliverable Checklist

- [x] `build_features(df, fit=False)` importable from `src/02_feature_engineering.py`
- [x] `models/encoder.pkl` produced when run on training data
- [x] `models/feature_columns.json` listing all 23 feature column names in order
- [x] Function works on a single-row DataFrame (verified: shape (1, 23))
