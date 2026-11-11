# Phase 7 Implementation Log — Streamlit Dashboard

**Date:** 2026-11-11  
**Status:** COMPLETE

---

## Objective

Deliver the two-tab Streamlit web application (`app.py`) that exposes the
trained Random Forest model via a browser UI.

---

## File Produced

| File | Description |
|---|---|
| `app.py` | Self-contained Streamlit application (≈ 225 lines) |

---

## Architecture

### Shared setup
- `@st.cache_resource` caches the RF model and the two helper modules
  (`feature_engineering`, `live_challenge`) so they are loaded once per
  server session.
- `@st.cache_data` caches `player_ranks.csv` and `feature_columns.json`.
- Both modules are loaded via `importlib.util.spec_from_file_location` because
  `02_feature_engineering.py` starts with a digit and cannot be imported with
  a plain `import` statement.
- Sidebar shows: project title, dataset provenance (ATP 2000–2026), and a
  disclaimer that predictions are ranking-only (no betting odds, no
  player-name bias).

### Tab 1 — Player vs Player
- Two `st.selectbox` widgets populated from `player_ranks.csv` (alphabetically
  sorted), each showing the player's last-known ATP rank via `st.metric`.
- Optional points override inputs (`st.number_input`) for both players.
- Match-context controls: Surface (selectbox), Series (selectbox), Court
  (radio), Round (selectbox), Best of (radio).
- "Predict" button:
  - Builds a single-row DataFrame from selected values.
  - Calls `build_features(row)` → `model.predict_proba`.
  - Displays two `st.metric` cards + two `st.progress` bars for both players.
  - Prints an interpretation note (strong / moderate / slight favourite).

### Tab 2 — Live Challenge
- Calls `get_holdout_matches()` (50-row reproducible sample from test set).
- Populates `st.selectbox` with `format_match_label` strings.
- On selection: shows player names, ranks, tournament, surface as metrics;
  displays predicted win probabilities with progress bars.
- "Reveal Actual Result" button:
  - Reads `Player1_wins` from the row.
  - Compares predicted winner with actual winner.
  - Shows green `st.success` if correct, red `st.error` if wrong.

---

## Implementation Notes

- `_predict_pvp` passes `round_name` as a raw string to `build_features`;
  the function maps it via `ROUND_ORDER` internally, so no pre-conversion
  is needed in `app.py`.
- `pts_available` flag is set to `1` only when at least one ranking-points
  override was provided; otherwise `0` (model default).
- The `__file__`-relative path construction (`os.path.join(os.path.dirname(__file__), "src", …)`)
  ensures the app works when launched from any working directory.

---

## Smoke Test Results

| Test | Result |
|---|---|
| Python AST parse — no syntax errors | ✅ PASS |
| `build_features` imports via importlib | ✅ PASS |
| `live_challenge` imports via importlib | ✅ PASS |
| All model / data files present | ✅ PASS |
| Tab 1 logic: Federer vs Nadal, Clay Grand Slam Final | ✅ PASS — P(P1) = 0.650 |
| Tab 2: 50 holdout rows loaded | ✅ PASS |
| Tab 2: format_match_label non-empty | ✅ PASS |
| Tab 2: predict_match returns float in [0,1] | ✅ PASS — 0.415 |
| Tab 2: reveal logic (predicted vs actual) | ✅ PASS — correct on sample row |
| `streamlit run app.py` — server up, no errors | ✅ PASS — listening on :8501 |

---

## Deliverable Checklist

- [x] `streamlit run app.py` starts without import errors
- [x] Tab 1: selecting two players and clicking Predict shows win probabilities
- [x] Tab 2: selecting a match shows prediction; reveal button shows actual winner
- [x] Sidebar visible with project description
