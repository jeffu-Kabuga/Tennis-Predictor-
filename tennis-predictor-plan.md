# ATP Tennis Win-Probability Predictor — Project Plan

## Top-Level Overview

Build a machine learning web application that predicts the win probability for a match between two ATP players.
The dataset (`atp_tennis.csv`) contains ~69 100 historical ATP matches from 2000-01-03 to 2026-11-01.
The app is delivered as a multi-tab Streamlit dashboard backed by a trained Random Forest classifier.

**Confirmed design decisions**
- Ranking-points columns (`Pts_1`, `Pts_2`) are kept; `-1` sentinels are replaced with `0` and a binary `pts_available` flag is added.
- Split strategy: random stratified split — 70 % train / 15 % val / 15 % test, stratified by `Surface`.
- Player ranking lookup in the UI: each player's most recent rank from the full dataset.
- Live Prediction Challenge: a dedicated Streamlit tab showing held-out matches with a "Reveal result" interaction.
- Betting odds (`Odd_1`, `Odd_2`) and player names (`Player_1`, `Player_2`) are **never** used as model features.
- `Score` and `Tournament` columns are dropped (post-match leakage and high-cardinality noise respectively).

---

## 1. PROJECT SETUP

### 1.1 Dataset Summary

**Columns**

| Column | Type | Meaning | Action |
|--------|------|---------|--------|
| `Tournament` | string (high-cardinality) | Tournament name | Drop — too many levels, adds noise |
| `Date` | date string | Match date | Keep for splitting; not a model feature |
| `Series` | categorical (~8 values) | Tournament tier (Grand Slam, ATP500, etc.) | One-hot encode |
| `Court` | categorical (Indoor/Outdoor) | Venue type | One-hot encode |
| `Surface` | categorical (Hard/Clay/Grass/Carpet) | Playing surface | One-hot encode; stratify split on this |
| `Round` | ordinal string | Round in the draw (1st Round → The Final) | Ordinal-encode (integer mapping) |
| `Best of` | integer (3 or 5) | Match format | Keep as numeric feature |
| `Player_1` | string (very high-cardinality) | First player name | **Drop from features** — UI only |
| `Player_2` | string (very high-cardinality) | Second player name | **Drop from features** — UI only |
| `Winner` | string | Winning player name | **Target derivation only** — derive binary label then drop |
| `Rank_1` | integer | ATP ranking of Player_1 | Core feature |
| `Rank_2` | integer | ATP ranking of Player_2 | Core feature |
| `Pts_1` | integer (-1 = missing) | Ranking points Player_1 | Replace -1 → 0; add `pts_available` flag |
| `Pts_2` | integer (-1 = missing) | Ranking points Player_2 | Replace -1 → 0; add `pts_available` flag |
| `Odd_1` | float (-1 = missing) | Bookmaker odds Player_1 | **Drop** — forbidden by constraint + leakage risk |
| `Odd_2` | float (-1 = missing) | Bookmaker odds Player_2 | **Drop** — forbidden by constraint + leakage risk |
| `Score` | string | Set-by-set score | **Drop** — post-match information, hard leakage |

**Target variable**
Derive a binary column `Player1_wins`:
- `1` if `Winner == Player_1`
- `0` if `Winner == Player_2`

**Data quality issues**
- `-1` sentinels in `Pts_1`, `Pts_2`, `Odd_1`, `Odd_2` (widespread in 2000–2005 data).
- `Round` is ordinal but stored as free-text strings — requires an explicit ordering map.
- No explicit `NaN`s observed; all missingness is encoded as `-1`.
- Dataset includes very recent matches (2026-11-01) so a pure temporal holdout is viable if ever needed.

**Data leakage columns (summary)**

| Column | Risk level | Reason |
|--------|-----------|--------|
| `Score` | Hard leakage | Only known after the match |
| `Odd_1` / `Odd_2` | Soft leakage + forbidden | Encodes post-market information; forbidden by constraint |
| `Player_1` / `Player_2` | Forbidden as features | Name encoding leaks career reputation |
| `Winner` | Target only | Must be converted to label and then removed |

---

### 1.2 Recommended Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python 3.11 | Stable, widely available, excellent ML ecosystem |
| Data handling | pandas 2.x | Standard dataframe library |
| ML | scikit-learn 1.x | Random Forest, preprocessing, evaluation — all in one package |
| Model interpretation | scikit-learn (built-in importance) + `eli5` permutation importance | Beginner-friendly; SHAP is optional stretch goal |
| Web UI | **Streamlit** | Single-file Python apps; no HTML/CSS required; ideal for workshops |
| Alternative considered | Gradio | Simpler for single-prediction demos but lacks multi-tab layout needed here |
| Environment management | `venv` + `requirements.txt` | Minimal, no extra tooling needed |
| Optional testing | `pytest` with two smoke tests | Verify preprocessing and model loading |

---

### 1.3 Project Structure

```
tennis-predictor/
├── atp_tennis.csv              # Raw dataset (provided)
├── requirements.txt            # All pip dependencies
├── tennis-predictor-plan.md    # This plan file
│
├── src/
│   ├── 01_prepare_data.py      # Module 1 — cleaning, leakage removal, splitting
│   ├── 02_feature_engineering.py  # Module 2 — feature creation and encoding
│   ├── 03_train_model.py       # Module 3 — Random Forest training and evaluation
│   ├── 04_interpret_model.py   # Module 4 — feature importance analysis
│   └── 05_live_challenge.py    # Module 5 — holdout prediction helper (used by Streamlit)
│
├── app.py                      # Module 6 — Streamlit dashboard (entry point)
│
├── models/
│   └── rf_model.pkl            # Serialised trained model (produced by Module 3)
│
├── data/
│   ├── train.csv               # 70 % training split
│   ├── val.csv                 # 15 % validation split
│   ├── test.csv                # 15 % test / holdout split
│   └── player_ranks.csv        # Player → most-recent rank lookup table
│
└── tests/
    └── test_smoke.py           # Optional smoke tests
```

**File responsibilities**

- `src/01_prepare_data.py` — loads raw CSV, drops leakage/forbidden columns, replaces `-1` sentinels, derives target label, produces train/val/test CSVs and `player_ranks.csv`.
- `src/02_feature_engineering.py` — defines a reusable `build_features(df)` function: ordinal-encodes `Round`, one-hot encodes `Surface`/`Court`/`Series`, computes `Rank_diff`, `Pts_diff`, `pts_available`.
- `src/03_train_model.py` — applies feature engineering to train split, trains `RandomForestClassifier`, evaluates on val and test, serialises model to `models/rf_model.pkl`.
- `src/04_interpret_model.py` — loads model and val set, prints built-in feature importances, runs permutation importance, saves a bar-chart PNG.
- `src/05_live_challenge.py` — exposes `get_holdout_matches()` and `predict_match(row)` helpers consumed by the Streamlit app.
- `app.py` — multi-tab Streamlit app: Tab 1 player-vs-player prediction, Tab 2 live challenge.

---

## 2. MODULE-BY-MODULE BUILD PLAN

### Module 1 — Data Preparation

**Objective** — Produce clean, leakage-free, split data files ready for feature engineering.

**Status** — [ ] pending

**Key steps**
1. Load `atp_tennis.csv` with `pandas.read_csv`.
2. Derive target column: `Player1_wins = (Winner == Player_1).astype(int)`.
3. Drop columns: `Tournament`, `Winner`, `Player_1`, `Player_2`, `Odd_1`, `Odd_2`, `Score`.
4. Replace `-1` with `0` in `Pts_1`, `Pts_2`; add `pts_available = ((Pts_1 != 0) | (Pts_2 != 0)).astype(int)` **before** the replacement (flag must reflect original state).
5. Drop rows where `Rank_1` or `Rank_2` is `-1` (unranked entries — rare but invalid).
6. Perform an 70/15/15 stratified random split on `Surface`.
7. Save `data/train.csv`, `data/val.csv`, `data/test.csv`.
8. Build and save `data/player_ranks.csv` from the **full** dataset (max `Date` per player → their `Rank`).

**Expected output** — Three CSV files + player rank lookup; no leakage columns present; `Player1_wins` column confirmed balanced (roughly 50 % since player order is arbitrary).

---

### Module 2 — Feature Engineering

**Objective** — Transform the cleaned splits into numeric feature matrices the model can consume.

**Status** — [ ] pending

**Key steps**
1. Define `ROUND_ORDER` dict mapping round strings to integers (`1st Round`→1 … `The Final`→7).
2. Map `Round` to integers using `ROUND_ORDER`; unknown rounds fall back to -1.
3. One-hot encode `Surface`, `Court`, `Series` using `pd.get_dummies` (fit-on-train, apply to val/test to avoid leakage in encoding).
4. Compute derived numeric features:
   - `Rank_diff = Rank_1 - Rank_2` (negative = Player 1 is better-ranked)
   - `Pts_diff = Pts_1 - Pts_2`
5. Drop raw `Player_1`/`Player_2` name columns if still present (should already be gone from Module 1).
6. Return / save the aligned feature matrices and the final feature column list.

**Expected output** — `build_features(df, fit=False)` function that returns a numeric DataFrame; feature column list saved to `models/feature_columns.json`.

---

### Module 3 — Model Building

**Objective** — Train and evaluate a Random Forest classifier; persist the model artefact.

**Status** — [ ] pending

**Key steps**
1. Load `data/train.csv` and `data/val.csv`; apply `build_features`.
2. Instantiate `RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')`.
3. Fit on training features + `Player1_wins` labels.
4. Evaluate on validation set: accuracy, ROC-AUC, log-loss; print a classification report.
5. Evaluate on test set for final reported numbers.
6. Serialise model with `joblib.dump` to `models/rf_model.pkl`.
7. Print and save the feature column order to `models/feature_columns.json`.

**Expected output** — Trained model file; printed evaluation metrics; approximate target: AUC > 0.65 on validation set.

---

### Module 4 — Model Interpretation

**Objective** — Understand which features drive predictions using two complementary methods.

**Status** — [ ] pending

**Key steps**
1. Load model and validation feature matrix.
2. Extract `model.feature_importances_` → sort and print top 10 (built-in, impurity-based).
3. Run `sklearn.inspection.permutation_importance` on the validation set (repeats=10) → print top 10.
4. Compare the two lists — note any features that rank very differently (sign of instability).
5. Save a horizontal bar chart of permutation importances as `models/permutation_importance.png`.

**Expected output** — Printed importance tables + PNG chart; narrative comparison of built-in vs permutation results.

---

### Module 5 — Live Prediction Challenge (Streamlit Tab)

**Objective** — Let the user select a real held-out match, see the model's probability, then reveal the actual result.

**Status** — [ ] pending

**Key steps**
1. In `src/05_live_challenge.py`, expose:
   - `get_holdout_matches()` — loads `data/test.csv` (still has `Player_1`, `Player_2`, `Date`, `Tournament` for display), returns a display-friendly DataFrame.
   - `predict_match(row)` — applies `build_features` to one row, loads model, returns `P(Player_1 wins)`.
2. In Streamlit Tab 2 ("Live Challenge"):
   - Show a selectbox of holdout matches formatted as `"Player_A vs Player_B — Tournament, Date"`.
   - On selection, display model's predicted win probability as a gauge/progress bar.
   - Show a "Reveal actual result" button that unhides the `Winner` column value.

**Expected output** — Functional Streamlit tab; model prediction visible before the result is revealed.

**Note** — The "live" quality is simulated: these are historical matches already in the dataset but held out from training. The label is present in the test split and simply hidden until the user clicks the reveal button.

---

### Module 6 — Streamlit Web UI Dashboard

**Objective** — Deliver the main player-vs-player win probability interface.

**Status** — [ ] pending

**Key steps**
1. Build `app.py` with two tabs: "Player vs Player" and "Live Challenge".
2. **Tab 1 — Player vs Player**
   - Load `data/player_ranks.csv` for player name ↔ rank mapping.
   - Two `st.selectbox` widgets: Player 1 and Player 2 (sorted alphabetically).
   - Additional match-context widgets: `Surface` dropdown, `Round` dropdown, `Series` dropdown, `Court` radio, `Best of` radio.
   - On "Predict" button click: assemble a single-row feature DataFrame, call model, display:
     - Win probability bars for both players (e.g. `st.metric` + `st.progress`).
     - A note that the prediction is ranking-based only.
3. **Tab 2 — Live Challenge** — delegated to Module 5 implementation in `src/05_live_challenge.py`.
4. Add a sidebar with project description and dataset provenance.

**Expected output** — Running Streamlit app (`streamlit run app.py`); both tabs functional; prediction shown within ~1 s.

---

## 3. ASSUMPTIONS & OPEN QUESTIONS

### Assumptions
- Player order within a row (`Player_1` / `Player_2`) is arbitrary (not seeded or ranked order), so the 50 % base rate for `Player1_wins` is expected.
- `-1` in ranking columns (`Rank_1`, `Rank_2`) represents genuinely unranked players; those rows will be dropped.
- `Pts_available` flag will be `1` if **either** player has non-zero points — this captures whether the era had ranking-point data.
- All features available at prediction time in the UI (ranks, surface, round, etc.) are legitimately pre-match information.
- The `Series` column maps cleanly to a small categorical set; a few novel values in test rows that were unseen at training time will be handled by `pd.get_dummies` alignment.

### Open Questions / Confirmed Decisions
| Question | Decision |
|----------|---------|
| Use betting odds as features? | **No** — forbidden by constraint |
| Use player names as features? | **No** — forbidden by constraint |
| How to handle `Pts_*` missingness? | Replace `-1` → `0`; add `pts_available` binary flag |
| Train/test split strategy? | Random stratified 70/15/15 on `Surface` |
| How to look up player rank in UI? | Most recent rank per player from full dataset |
| Live challenge format? | Embedded Streamlit tab with "Reveal result" button |

---

## 4. RISKS & TRADE-OFFS

### Risk 1 — Data Leakage from Random Split
**Problem** — A random split means a player can appear in both train and test sets (with different opponents). The model may implicitly learn player-specific patterns even without using names, because `Rank_1`/`Rank_2` values for a specific player recur across splits.

**Mitigation** — Accept this as a known limitation (the user chose this approach for training size). Document it clearly. If evaluation AUC seems unrealistically high (> 0.80), consider switching to a temporal split. The forbidden-feature rules (no names, no odds) already remove the most direct leakage vectors.

---

### Risk 2 — High-Cardinality Encoding (`Series`, `Round`)
**Problem** — If `pd.get_dummies` is applied independently to train and test, column sets can differ (new tournament series or unusual round names in test).

**Mitigation** — Fit the one-hot encoder on the training set only; align test/val columns to the training set's column list (fill missing dummies with 0, drop unexpected ones). Save the canonical column list to `models/feature_columns.json` and enforce it at prediction time.

---

### Risk 3 — Imbalanced Representation of Surfaces / Eras
**Problem** — Hard-court matches dominate ATP data; early-era matches (2000–2005) have no points data. The stratified split on `Surface` mitigates class imbalance but does not fully address era effects.

**Mitigation** — Use `class_weight='balanced'` in the Random Forest to weight minority surfaces. The `pts_available` flag lets the model learn that predictions in the no-points era are inherently less certain. Note in the Streamlit UI that predictions for less-common surfaces (Carpet) may be less reliable.

---

## Implementation Notes for Agent Mode

When implementing in Agent mode, process one subtask at a time and update the `Status` field above after each is complete. The subtasks should be executed in this order:

1. `src/01_prepare_data.py` + data splits
2. `src/02_feature_engineering.py` + feature column list
3. `src/03_train_model.py` + model artefact
4. `src/04_interpret_model.py` + importance chart
5. `src/05_live_challenge.py` + helper functions
6. `app.py` + Streamlit dashboard
7. `requirements.txt`
8. Optional: `tests/test_smoke.py`
