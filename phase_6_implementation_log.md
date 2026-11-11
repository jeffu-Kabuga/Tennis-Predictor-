# Phase 6 Implementation Log — Live Challenge Helpers

**Date:** 2026-11-11
**Status:** COMPLETE

---

## Objective

Implement the three helper functions that the Streamlit "Live Challenge" tab
will consume:

| Function | Purpose |
|---|---|
| `get_holdout_matches()` | Returns a 50-row test-set sample with player names and tournament info re-attached |
| `predict_match(row_df)` | Applies feature engineering and the Random Forest to return P(Player 1 wins) |
| `format_match_label(row)` | Builds the human-readable "P1 vs P2 — Tournament (Date)" string |

---

## Implementation Notes

### File: `src/05_live_challenge.py`

#### `get_holdout_matches()`

The test split (`data/test.csv`) was built **without** player names — they
were dropped in Phase 2 to prevent leakage into the model.  To display
real match context in the UI the original `atp_tennis.csv` must be re-joined.

**Join strategy:**  
A 7-column key `Date + Series + Court + Surface + Round + Rank_1 + Rank_2` was
verified to be **collision-free** across all 69,074 pre-filtered rows:

```
3-col key (Date, Rank_1, Rank_2):            4 duplicates
7-col key (adds Series, Court, Surface, Round): 0 duplicates
```

The raw CSV is preprocessed identically to `01_prepare_data.py` (sentinel
replacement, rank filtering) before the join so that Pts values and the
filtered row set match exactly.  A reproducible 50-row sample is drawn with
`random_state=42`.

**Result:** 50 rows returned, 0 rows with a missing player name.

#### `predict_match(row_df)`

`02_feature_engineering.py` starts with a digit, so it cannot be imported
with a plain `import` statement.  It is loaded once via
`importlib.util.spec_from_file_location` and cached with
`@functools.lru_cache(maxsize=1)`.  `_FE_PATH` is constructed relative to
`__file__` (an absolute path) so the function works correctly regardless of
the caller's working directory (important for `app.py` in the project root).

The model is similarly cached with `@functools.lru_cache(maxsize=1)`.

#### `format_match_label(row)`

Simple f-string; slices `str(row["Date"])[:10]` to normalise both
`datetime64` and plain string date representations to `YYYY-MM-DD`.

---

## Smoke Test Results

All tests run from the project root (`c:\Users\ADMIN\Desktop\tennis-Predictor`):

| Test | Result |
|---|---|
| Import without side-effects | ✅ PASS |
| `get_holdout_matches()` returns 50 rows with correct columns | ✅ PASS |
| 0 rows with missing `Player_1` name | ✅ PASS |
| `predict_match()` returns `float` in [0, 1] | ✅ PASS (0.415 on sample row) |
| `format_match_label()` returns non-empty string | ✅ PASS (`'Youzhny M. vs Levy H. — Kroger St. Jude (2001-02-19)'`) |

---

## Output File Summary

| File | Description |
|---|---|
| `src/05_live_challenge.py` | Module with all three public helper functions |

---

## Deliverable Checklist

- [x] `get_holdout_matches()` returns a DataFrame with ≥ 1 row and correct columns
- [x] `predict_match(row_df)` returns a float between 0 and 1
- [x] `format_match_label(row)` returns a non-empty string
- [x] Functions importable without side-effects (no top-level code that runs on import)
