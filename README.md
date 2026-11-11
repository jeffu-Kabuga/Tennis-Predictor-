# 🎾 Tennis Predictor

A machine-learning web application that predicts ATP tennis match win probabilities using player rankings and match context. Built with a Random Forest classifier trained on ~69,000 historical ATP matches (2000–2026) and served through an interactive Streamlit dashboard.

---

## Overview

Given two ATP players and a match context (surface, tournament series, round, etc.), the model outputs a win probability for each player. The approach is deliberately interpretable — it uses only pre-match, publicly available ranking data with no betting odds, no player-name memorisation, and no head-to-head history.

**Model performance:**
- Validation ROC-AUC: **0.6696**
- Test ROC-AUC: **0.6791** ✅ (target was > 0.65)

---

## Features

- **Player vs Player tab** — select any two of 1,822 ATP players, adjust ranking inputs, set match context and get a win probability instantly
- **Live Challenge tab** — pick from 50 real holdout matches, see the model's prediction, then reveal the actual result
- No betting odds used — pure ranking-based model
- Fully reproducible pipeline from raw CSV to deployed dashboard

---

## Tech Stack

| Category | Tool / Library |
|---|---|
| Language | Python 3.11+ |
| ML | scikit-learn ≥ 1.3 (Random Forest) |
| Data | pandas ≥ 2.0 |
| Dashboard | Streamlit ≥ 1.35 |
| Serialisation | joblib ≥ 1.3 |
| Visualisation | matplotlib ≥ 3.7 |
| Testing | pytest |

---

## Project Structure

```
tennis-Predictor/
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt                # Python dependencies
├── atp_tennis.csv                  # Raw ATP dataset (~69k matches, 2000–2026)  *
│
├── src/
│   ├── 01_prepare_data.py          # Data cleaning & train/val/test split
│   ├── 02_feature_engineering.py   # Feature transforms & one-hot encoding
│   ├── 03_train_model.py           # Random Forest training & evaluation
│   ├── 04_interpret_model.py       # Feature importance analysis & chart
│   └── 05_live_challenge.py        # Helpers for holdout-match predictions
│
├── data/
│   ├── train.csv                   # 70% split — 48,351 rows  *
│   ├── val.csv                     # 15% split — 10,361 rows  *
│   ├── test.csv                    # 15% split — 10,362 rows  *
│   └── player_ranks.csv            # 1,822 player → rank lookup
│
├── models/
│   ├── rf_model.pkl                # Trained RandomForestClassifier  *
│   ├── encoder.pkl                 # Fitted OneHotEncoder
│   ├── feature_columns.json        # Canonical 23-feature list
│   └── permutation_importance.png  # Top-15 feature importance chart
│
└── tests/
    └── test_smoke.py               # 10 pytest smoke tests (all passing)
```

> `*` — excluded from git (see `.gitignore`); regenerate with the pipeline scripts below.

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Regenerate data & model (first run only)

```bash
python src/01_prepare_data.py      # → data/train.csv, val.csv, test.csv, player_ranks.csv
python src/03_train_model.py       # → models/rf_model.pkl, encoder.pkl, feature_columns.json
python src/04_interpret_model.py   # → models/permutation_importance.png
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

### 4. Run tests

```bash
pytest tests/ -v
# Expected: 10 passed
```

---

## Pipeline Walkthrough

```
atp_tennis.csv
     │
     ▼
01_prepare_data.py
  • Remove post-match leakage columns (Score, Odds, Winner)
  • Replace sentinel –1 values
  • Stratified 70/15/15 split on Surface
     │
     ▼
02_feature_engineering.py  (fit on train, transform-only on val/test/live)
  • Ordinal-encode Round (1st Round=1 … The Final=7)
  • Derive Rank_diff = Rank_1 – Rank_2, Pts_diff = Pts_1 – Pts_2
  • One-hot encode Surface (4), Court (2), Series (8)
  • → 23 numeric features total
     │
     ▼
03_train_model.py
  • RandomForestClassifier(n_estimators=200, class_weight='balanced')
  • Evaluate on val & test → ROC-AUC 0.6696 / 0.6791
  • Serialise to models/rf_model.pkl
     │
     ▼
app.py  (Streamlit)
  • Tab 1: Custom Player vs Player prediction
  • Tab 2: Live Challenge — holdout match picker + reveal
```

---

## Features Used by the Model

The model uses **23 features** — all pre-match, ranking-based, no player identity:

| Feature | Type | Description |
|---|---|---|
| `Rank_1`, `Rank_2` | Numeric | ATP world ranking |
| `Rank_diff` | Numeric | Rank_1 − Rank_2 (top predictor) |
| `Pts_1`, `Pts_2` | Numeric | Ranking points |
| `Pts_diff` | Numeric | Pts_1 − Pts_2 |
| `pts_available` | Binary | Whether ranking points data exists (era flag) |
| `Round` | Ordinal | Match round (1–7) |
| `Best of` | Numeric | 3 or 5 sets |
| `Surface_*` | One-hot | Hard / Clay / Grass / Carpet |
| `Court_*` | One-hot | Indoor / Outdoor |
| `Series_*` | One-hot | Grand Slam / Masters 1000 / ATP500 / … |

**Top feature by permutation importance:** `Rank_diff` (0.0266 mean accuracy decrease)

---

## Dataset

- **Source:** `atp_tennis.csv` — ATP tour match results
- **Range:** 2000-01-03 to 2026-11-01
- **Size:** ~69,100 rows before cleaning, 69,074 after
- **Target:** `Player1_wins` — binary label derived from `Winner == Player_1`; class balance is ~50% because player order is arbitrary

---

## Known Limitations

- Predictions are ranking-only — upsets driven by form, injury, or head-to-head matchups are not captured
- Pre-2005 data has no ranking-points records (handled via `pts_available` flag and –1 sentinel replacement)
- Random train/test split means the same player can appear in both sets with different opponents — acceptable because player identity is not used as a feature

---

## Project Phases

The project was built across 8 phases, each with a logged implementation report:

| Phase | Description | Status |
|---|---|---|
| 1 | Project setup (folders, deps) | ✅ Complete |
| 2 | Data preparation | ✅ Complete |
| 3 | Feature engineering | ✅ Complete |
| 4 | Model training & evaluation | ✅ Complete |
| 5 | Model interpretation | ✅ Complete |
| 6 | Live challenge helpers | ✅ Complete |
| 7 | Streamlit dashboard | ✅ Complete |
| 8 | Testing & validation | ✅ Complete |

---

## Disclaimer

This tool is for educational and analytical purposes only. Predictions are based solely on ATP rankings and match context. No betting odds are used or implied. Do not use this application for gambling decisions.
