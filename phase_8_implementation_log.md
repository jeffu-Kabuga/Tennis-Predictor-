# Phase 8 Implementation Log — Testing & Validation

**Date:** 2026-11-11  
**Status:** COMPLETE

---

## Objective

Write and pass a full suite of smoke tests that verify the end-to-end pipeline
without errors and confirm that all key artefacts exist and behave correctly.

---

## File Produced

| File | Description |
|---|---|
| `tests/test_smoke.py` | 10 pytest test cases across 4 test classes (113 lines) |

---

## Test Suite Structure

### Class `TestDataPreparation` (5 tests)
- Runs `src/01_prepare_data.py` as a subprocess and asserts **exit code 0**.
- Asserts existence of `data/train.csv`, `data/val.csv`, `data/test.csv`, and `data/player_ranks.csv`.

### Class `TestFeatureEngineering` (1 test)
- Imports `build_features` from `src/02_feature_engineering.py` via `importlib.util`
  (required because the filename starts with a digit).
- Calls `build_features(train_df, fit=False)` and asserts:
  - Return type is `pd.DataFrame`
  - Row count > 0
  - `feature_names` list is non-empty
  - Column count matches `feature_names`

### Class `TestModel` (2 tests)
- Loads `models/rf_model.pkl` with `joblib`.
- Asserts the loaded object has a `predict_proba` attribute.
- Feeds a single row (first row of `data/train.csv`) through `build_features`
  then `model.predict_proba` and asserts:
  - Output shape is `(1, 2)`
  - Win probability is in `[0, 1]`

### Class `TestLiveChallenge` (2 tests)
- Imports `get_holdout_matches` and `predict_match` from `src/05_live_challenge.py`.
- Calls `get_holdout_matches()` and asserts type is `pd.DataFrame` with ≥ 1 row.
- Passes the first row to `predict_match` and asserts the return is a `float` in `[0, 1]`.

---

## Implementation Notes

- `PROJECT_ROOT` is resolved from `__file__` so `pytest` can be invoked from
  any working directory.
- All modules with digit-prefixed filenames are loaded via a shared
  `_import_module(filename, module_name)` helper that uses
  `importlib.util.spec_from_file_location`.
- The data-prep smoke test re-runs the full script (not just asserts file
  existence) to validate the script itself, not just cached artefacts.
- `fit=False` is used in feature engineering and model tests to rely on the
  already-fitted `models/encoder.pkl` — no side effects on training artefacts.

---

## Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.14.5, pytest-9.2.0
collected 10 items

tests/test_smoke.py::TestDataPreparation::test_script_exits_cleanly          PASSED
tests/test_smoke.py::TestDataPreparation::test_train_csv_exists              PASSED
tests/test_smoke.py::TestDataPreparation::test_val_csv_exists                PASSED
tests/test_smoke.py::TestDataPreparation::test_test_csv_exists               PASSED
tests/test_smoke.py::TestDataPreparation::test_player_ranks_csv_exists       PASSED
tests/test_smoke.py::TestFeatureEngineering::test_build_features_returns_nonempty_dataframe PASSED
tests/test_smoke.py::TestModel::test_model_has_predict_proba                 PASSED
tests/test_smoke.py::TestModel::test_model_predicts_on_dummy_input           PASSED
tests/test_smoke.py::TestLiveChallenge::test_get_holdout_matches_returns_dataframe PASSED
tests/test_smoke.py::TestLiveChallenge::test_predict_match_returns_valid_probability PASSED

============================= 10 passed in 19.09s =============================
```

---

## Deliverable Checklist

- [x] `pytest tests/ -v` passes all smoke tests (10/10 passed)
- [x] `streamlit run app.py` runs clean in browser (validated in Phase 7)
- [x] No Python warnings or errors in the terminal during test run
