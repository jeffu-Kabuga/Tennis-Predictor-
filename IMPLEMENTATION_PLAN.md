# ATP Tennis Win-Probability Predictor — Implementation Plan

> Reference design decisions and dataset analysis are in [`tennis-predictor-plan.md`](tennis-predictor-plan.md).
> This document is the step-by-step coding guide. Each phase is self-contained and must be completed and verified before the next begins.

---

## Phase 1 — Project Setup

**Goal:** Create the full folder skeleton and install all dependencies so every subsequent phase has a clean environment to work in.

### Steps
1. Create the following directories (if they do not already exist):
   - `src/`
   - `data/`
   - `models/`
   - `tests/`
2. Create `requirements.txt` with pinned dependencies:
   - `pandas>=2.0`
   - `scikit-learn>=1.3`
   - `eli5>=0.13`
   - `streamlit>=1.35`
   - `joblib>=1.3`
   - `matplotlib>=3.7`
3. Verify the environment by running `pip install -r requirements.txt`.

### Deliverables
- [ ] Folder skeleton exists (`src/`, `data/`, `models/`, `tests/`)
- [ ] `requirements.txt` committed with all dependencies
- [ ] `pip install` completes without errors

---

## Phase 2 — Module 1: Data Preparation

**File:** `src/01_prepare_data.py`

**Goal:** Produce clean, leakage-free, split data files and a player-rank lookup table.

### Steps
1. Load `atp_tennis.csv` with `pandas.read_csv`.
2. **Derive target column** before any drops:
   - `Player1_wins = (df['Winner'] == df['Player_1']).astype(int)`
3. **Flag points availability** before replacing sentinels:
   - `pts_available = ((df['Pts_1'] != -1) | (df['Pts_2'] != -1)).astype(int)`
4. **Replace `-1` sentinels** in `Pts_1` and `Pts_2` with `0`.
5. **Drop leakage and forbidden columns:**
   `Tournament`, `Winner`, `Odd_1`, `Odd_2`, `Score`
6. **Drop rows** where `Rank_1 == -1` or `Rank_2 == -1` (unranked players).
7. **Build `data/player_ranks.csv`** from the full (pre-split) dataset:
   - For each unique player name (from `Player_1` and `Player_2` columns), find the row with the most recent `Date` and record their rank.
   - Save as two columns: `Player`, `Rank`.
8. **Drop player name columns** from the modelling DataFrame: `Player_1`, `Player_2`.
9. **Perform 70/15/15 stratified random split** on `Surface`:
   - First split off 30 % (temp), stratified on `Surface`.
   - Split the 30 % temp into two equal halves (15 % val, 15 % test), again stratified on `Surface`.
   - Use `random_state=42` throughout.
10. Save `data/train.csv`, `data/val.csv`, `data/test.csv`.
11. Print row counts and `Player1_wins` balance for each split.

### Deliverables
- [ ] `data/train.csv` (~48 370 rows)
- [ ] `data/val.csv` (~10 365 rows)
- [ ] `data/test.csv` (~10 365 rows)
- [ ] `data/player_ranks.csv` (one row per unique player name)
- [ ] No columns named `Winner`, `Odd_1`, `Odd_2`, `Score`, `Tournament` in any split
- [ ] `Player1_wins` present; roughly 50 % positive class

---

## Phase 3 — Module 2: Feature Engineering

**File:** `src/02_feature_engineering.py`

**Goal:** Provide a reusable `build_features(df, fit=False, encoder=None)` function that transforms any split into a numeric feature matrix ready for the model.

### Steps
1. Define `ROUND_ORDER` — a dict mapping round name strings to integers:
   - `'1st Round'` → 1, `'2nd Round'` → 2, `'3rd Round'` → 3, `'4th Round'` → 4,
     `'Quarterfinals'` → 5, `'Semifinals'` → 6, `'The Final'` → 7
   - Unknown values map to -1.
2. Apply ordinal encoding: `df['Round'] = df['Round'].map(ROUND_ORDER).fillna(-1).astype(int)`
3. Compute derived numeric features:
   - `Rank_diff = Rank_1 - Rank_2`
   - `Pts_diff = Pts_1 - Pts_2`
4. One-hot encode `Surface`, `Court`, `Series` using `sklearn.preprocessing.OneHotEncoder`:
   - When `fit=True`: fit the encoder on the input DataFrame; save it to `models/encoder.pkl`.
   - When `fit=False`: load `models/encoder.pkl` and transform.
   - Use `handle_unknown='ignore'` to safely handle unseen categories in val/test/live prediction.
5. Assemble the final feature DataFrame from: `Rank_1`, `Rank_2`, `Rank_diff`, `Pts_1`, `Pts_2`, `Pts_diff`, `pts_available`, `Round` (ordinal), `Best of`, and all one-hot columns.
6. Save the canonical column list to `models/feature_columns.json`.
7. Return `(X, feature_names)`.

### Deliverables
- [ ] `build_features(df, fit=False)` function importable from `src/02_feature_engineering.py`
- [ ] `models/encoder.pkl` produced when run on training data
- [ ] `models/feature_columns.json` listing all feature column names in order
- [ ] Function works on a single-row DataFrame (needed for live prediction in the UI)

---

## Phase 4 — Module 3: Model Building

**File:** `src/03_train_model.py`

**Goal:** Train the Random Forest classifier on the training split and report evaluation metrics on val and test sets.

### Steps
1. Load `data/train.csv`, `data/val.csv`, `data/test.csv`.
2. Call `build_features(train_df, fit=True)` → `X_train`.
3. Call `build_features(val_df)` and `build_features(test_df)` → `X_val`, `X_test`.
4. Extract labels: `y_train`, `y_val`, `y_test` from the `Player1_wins` column.
5. Instantiate and fit:
   ```python
   RandomForestClassifier(
       n_estimators=200,
       random_state=42,
       class_weight='balanced',
       n_jobs=-1
   )
   ```
6. Evaluate on val set — print:
   - Accuracy
   - ROC-AUC
   - Log-loss
   - Classification report
7. Evaluate on test set — print the same metrics.
8. Serialise model: `joblib.dump(model, 'models/rf_model.pkl')`.
9. Confirm the model file exists and is loadable.

### Deliverables
- [ ] `models/rf_model.pkl` written to disk
- [ ] Validation ROC-AUC printed (target: > 0.65)
- [ ] Test ROC-AUC printed (should be close to validation AUC — large gap signals overfitting)
- [ ] No leakage features (`Winner`, `Score`, `Odd_*`, player names) present in the feature matrix

---

## Phase 5 — Module 4: Model Interpretation

**File:** `src/04_interpret_model.py`

**Goal:** Understand which features drive the model's predictions using two complementary importance methods.

### Steps
1. Load `models/rf_model.pkl` and `data/val.csv`.
2. Build `X_val` using `build_features(val_df)`.
3. **Built-in (impurity-based) importances:**
   - Extract `model.feature_importances_` paired with feature names.
   - Sort descending and print top 10.
4. **Permutation importances:**
   - Run `sklearn.inspection.permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)`.
   - Sort by mean importance descending and print top 10.
5. **Compare:** note any features ranked very differently between the two methods (may indicate instability or collinearity).
6. **Save chart:** horizontal bar chart of top-15 permutation importances saved to `models/permutation_importance.png` using `matplotlib`.

### Deliverables
- [ ] Top-10 built-in importances printed to console
- [ ] Top-10 permutation importances printed to console
- [ ] `models/permutation_importance.png` saved
- [ ] Brief comparison note printed (e.g. "Rank_diff ranks #1 in both methods")

---

## Phase 6 — Module 5: Live Challenge Helpers

**File:** `src/05_live_challenge.py`

**Goal:** Provide the helper functions consumed by the Streamlit "Live Challenge" tab.

### Steps
1. Implement `get_holdout_matches()`:
   - Load `data/test.csv` — but this split was built **without** player names. Therefore, also load the original `atp_tennis.csv`, filter to the same rows (match on `Date` + `Rank_1` + `Rank_2`), and re-attach `Player_1`, `Player_2`, `Tournament`, `Date` for display purposes only.
   - Return a DataFrame with display columns (`Player_1`, `Player_2`, `Tournament`, `Date`) plus all feature columns needed for prediction.
   - Limit to a random sample of 50 matches for UI usability (use `random_state=42`).
2. Implement `predict_match(row_df)`:
   - Accept a single-row DataFrame containing feature columns.
   - Load `models/rf_model.pkl` (cache with `functools.lru_cache` or Streamlit `@st.cache_resource`).
   - Apply `build_features(row_df)` and call `model.predict_proba`.
   - Return `float` — probability that Player 1 wins.
3. Implement `format_match_label(row)`:
   - Returns a display string: `"Player_1 vs Player_2 — Tournament (Date)"`.

### Deliverables
- [ ] `get_holdout_matches()` returns a DataFrame with ≥ 1 row and correct columns
- [ ] `predict_match(row_df)` returns a float between 0 and 1
- [ ] `format_match_label(row)` returns a non-empty string
- [ ] Functions importable without side-effects (no top-level code that runs on import)

---

## Phase 7 — Module 6: Streamlit Dashboard

**File:** `app.py`

**Goal:** Deliver the two-tab Streamlit web application.

### Steps

#### Shared setup
1. Import all necessary modules; load `models/rf_model.pkl`, `data/player_ranks.csv`, and `models/feature_columns.json` using `@st.cache_resource` / `@st.cache_data`.
2. Add a sidebar with:
   - Project title and one-line description.
   - Dataset provenance note (ATP matches 2000–2026).
   - Warning that predictions are ranking-based only (no betting odds, no player-name bias).

#### Tab 1 — Player vs Player
3. Two `st.selectbox` widgets populated from `player_ranks.csv` (alphabetically sorted player names).
4. Auto-display the last known rank beneath each selector.
5. Match-context widgets:
   - `Surface` — selectbox: Hard, Clay, Grass, Carpet
   - `Series` — selectbox: International, ATP500, Grand Slam, Masters, Masters Cup, International Gold, ATP250
   - `Court` — radio: Outdoor / Indoor
   - `Round` — selectbox: 1st Round … The Final
   - `Best of` — radio: 3 / 5
6. "Predict" button:
   - Assemble a single-row dict using selected ranks and match context.
   - Call `predict_match(row_df)`.
   - Display result as two `st.metric` cards (e.g. "Player A Win Probability: 63 %") and two `st.progress` bars.
   - Show a brief interpretation note.

#### Tab 2 — Live Challenge
7. Call `get_holdout_matches()` to populate a `st.selectbox` with match labels from `format_match_label`.
8. On selection, call `predict_match` and show the predicted probability.
9. "Reveal actual result" `st.button`:
   - When clicked, display the actual winner and whether the model was correct.
   - Colour the result green (correct) or red (incorrect).

### Deliverables
- [ ] `streamlit run app.py` starts without import errors
- [ ] Tab 1: selecting two players and clicking Predict shows win probabilities
- [ ] Tab 2: selecting a match shows prediction; reveal button shows actual winner
- [ ] Sidebar visible with project description

---

## Phase 8 — Testing & Validation

**File:** `tests/test_smoke.py`

**Goal:** Verify that the pipeline runs end-to-end without errors and that key outputs exist.

### Steps
1. **Smoke test — data prep:** run `src/01_prepare_data.py` as a subprocess; assert exit code 0; assert `data/train.csv`, `data/val.csv`, `data/test.csv`, `data/player_ranks.csv` exist.
2. **Smoke test — feature engineering:** import `build_features` from `src/02_feature_engineering.py`; load `data/train.csv`; call `build_features(df, fit=False)`; assert output is a non-empty DataFrame.
3. **Smoke test — model:** load `models/rf_model.pkl` with `joblib`; assert it has a `predict_proba` method; assert it predicts without error on a one-row dummy input.
4. **Smoke test — live challenge:** import `get_holdout_matches` and `predict_match` from `src/05_live_challenge.py`; call both; assert types are correct.
5. Run all tests with `pytest tests/ -v` and confirm all pass.

### Full run validation (manual)
- Run scripts in order: `01` → `02` (via `03`) → `03` → `04` → start Streamlit.
- Confirm Streamlit opens in browser with no Python errors in the terminal.
- Make at least one prediction in Tab 1 and one reveal in Tab 2.

### Deliverables
- [ ] `pytest tests/ -v` passes all smoke tests
- [ ] `streamlit run app.py` runs clean in browser
- [ ] No Python warnings or errors in the terminal during a full walkthrough

---

## Implementation Order Summary

```
Phase 1  →  Phase 2  →  Phase 3  →  Phase 4
  ↓
Phase 5  →  Phase 6  →  Phase 7  →  Phase 8
```

Each phase depends on the outputs of the previous one. Do not begin a phase until the deliverable checklist of the prior phase is fully checked off.
