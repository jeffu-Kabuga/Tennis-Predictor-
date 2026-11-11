# Phase 2 Implementation Log — Data Preparation

**Date:** 2026-11-11  
**Status:** ✅ COMPLETE  

---

## Objective

Produce clean, leakage-free, stratified train / val / test CSV splits and a
player-rank lookup table, all from the raw `atp_tennis.csv` dataset.

---

## Steps Executed

### 1. Load raw data
Loaded `atp_tennis.csv` with `pandas.read_csv`. Total rows: **69,100**.

### 2. Derive target column
`Player1_wins = (df['Winner'] == df['Player_1']).astype(int)` — created before any drops to avoid accidental leakage removal.

### 3. Flag points availability
`pts_available = ((df['Pts_1'] != -1) | (df['Pts_2'] != -1)).astype(int)` — captures whether ranking-points data existed for the match.

### 4. Replace -1 sentinels
Replaced `-1` values in `Pts_1` and `Pts_2` with `0` using `.replace(-1, 0)`.

### 5. Drop leakage / forbidden columns
Dropped: `Tournament`, `Winner`, `Odd_1`, `Odd_2`, `Score`.

### 6. Drop unranked-player rows
Rows where `Rank_1 == -1` or `Rank_2 == -1` were removed.  
Dropped **26 rows** → **69,074 rows** remain.

### 7. Build player_ranks.csv
- Stacked `(Player_1, Date, Rank_1)` and `(Player_2, Date, Rank_2)` into a single long table.
- Grouped by player name, kept the most-recent row (sorted by `Date`).
- Saved **1,822 unique players** to `data/player_ranks.csv` with columns `Player`, `Rank`.

### 8. Drop player name columns
Dropped `Player_1` and `Player_2` from the modelling DataFrame (kept in `player_ranks.csv` only).

### 9. Stratified 70 / 15 / 15 split
Used `sklearn.model_selection.train_test_split` with `stratify=Surface, random_state=42`:
- First split: 70% train / 30% temp
- Second split: 50/50 on temp → val / test

### 10. Save splits
Saved `data/train.csv`, `data/val.csv`, `data/test.csv`.

---

## Output File Summary

| File | Rows | Player1_wins balance | Forbidden cols present |
|------|-----:|---------------------:|------------------------|
| `data/train.csv` | 48,351 | 50.0 % | None |
| `data/val.csv`   | 10,361 | 50.8 % | None |
| `data/test.csv`  | 10,362 | 49.1 % | None |
| `data/player_ranks.csv` | 1,822 | — | — |

Columns present in each split:
`Date`, `Series`, `Court`, `Surface`, `Round`, `Best of`, `Rank_1`, `Rank_2`, `Pts_1`, `Pts_2`, `Player1_wins`, `pts_available`

---

## Deviations from Plan

None. All 11 steps executed exactly as specified.

---

## Deliverable Checklist

- [x] `data/train.csv` — **48,351 rows** (plan target ~48,370 ✅)
- [x] `data/val.csv`   — **10,361 rows** (plan target ~10,365 ✅)
- [x] `data/test.csv`  — **10,362 rows** (plan target ~10,365 ✅)
- [x] `data/player_ranks.csv` — **1,822 unique players** ✅
- [x] No forbidden columns (`Winner`, `Odd_1`, `Odd_2`, `Score`, `Tournament`) in any split ✅
- [x] `Player1_wins` present; ~50 % positive class in all three splits ✅

---

## Notes

- Row counts differ slightly from plan estimates (~19 rows off for train, ~4 for val/test) because the plan was written against an estimate; 26 unranked-player rows were removed, and the stratified split distributes the remainder naturally.
- `Date` column is retained in the splits (not listed as a drop target in the plan). It will be dropped or ignored by the feature engineering module in Phase 3.
- `pts_available` flag is ~77 % zero for the earliest matches (pre-2004 era had no ATP points system in this dataset), which is expected behaviour.
