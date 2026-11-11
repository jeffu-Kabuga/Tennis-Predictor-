"""
Phase 7 — Streamlit Dashboard
ATP Tennis Win-Probability Predictor — two-tab web application.

Tabs
----
1. Player vs Player  — assemble a custom match and predict win probability.
2. Live Challenge    — pick a real holdout match, predict, then reveal the result.
"""

import importlib.util
import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODEL_PATH = "models/rf_model.pkl"
_PLAYER_RANKS_PATH = "data/player_ranks.csv"
_FEATURE_COLUMNS_PATH = "models/feature_columns.json"

# 02_feature_engineering starts with a digit → import via importlib
_FE_PATH = os.path.join(os.path.dirname(__file__), "src", "02_feature_engineering.py")
_LC_PATH = os.path.join(os.path.dirname(__file__), "src", "05_live_challenge.py")


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_model():
    return joblib.load(_MODEL_PATH)


@st.cache_data
def _load_player_ranks() -> pd.DataFrame:
    df = pd.read_csv(_PLAYER_RANKS_PATH)
    df["Player"] = df["Player"].str.strip()
    return df.sort_values("Player").reset_index(drop=True)


@st.cache_data
def _load_feature_columns() -> list:
    with open(_FEATURE_COLUMNS_PATH) as fh:
        return json.load(fh)


@st.cache_resource
def _load_fe_module():
    spec = importlib.util.spec_from_file_location("feature_engineering", _FE_PATH)
    fe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe)
    return fe


@st.cache_resource
def _load_lc_module():
    spec = importlib.util.spec_from_file_location("live_challenge", _LC_PATH)
    lc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lc)
    return lc


# ---------------------------------------------------------------------------
# Prediction helper (Tab 1)
# ---------------------------------------------------------------------------
def _predict_pvp(rank1: int, rank2: int, pts1: int, pts2: int,
                 pts_available: int, surface: str, series: str,
                 court: str, round_name: str, best_of: int) -> float:
    """
    Assemble a single-row DataFrame and return P(Player 1 wins).
    """
    fe = _load_fe_module()

    row = pd.DataFrame([{
        "Rank_1": rank1,
        "Rank_2": rank2,
        "Pts_1": pts1,
        "Pts_2": pts2,
        "pts_available": pts_available,
        "Surface": surface,
        "Court": court,
        "Series": series,
        "Round": round_name,   # build_features will re-map via ROUND_ORDER
        "Best of": best_of,
    }])

    model = _load_model()
    X, _ = fe.build_features(row)
    prob = float(model.predict_proba(X)[0, 1])
    return prob


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ATP Tennis Predictor",
    page_icon="🎾",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎾 ATP Tennis Predictor")
    st.markdown(
        "Win-probability predictions for ATP men's tennis matches "
        "based on ranking and match context."
    )
    st.divider()
    st.markdown("**Dataset**")
    st.markdown("ATP Tour matches, 2000 – 2026  \n`atp_tennis.csv` (~69 k rows)")
    st.divider()
    st.markdown("**⚠️ Important notice**")
    st.markdown(
        "Predictions are derived from **ATP world rankings and match context only** "
        "(surface, series, round, points).  "
        "No betting odds, player names, or head-to-head history are used.  "
        "This is a statistical model, not a betting tool."
    )
    st.divider()
    st.caption("Phase 7 — Streamlit Dashboard")

# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------
player_ranks_df = _load_player_ranks()
player_names = player_ranks_df["Player"].tolist()
rank_lookup: dict = dict(zip(player_ranks_df["Player"], player_ranks_df["Rank"]))

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🏟 Player vs Player", "🎯 Live Challenge"])


# ============================================================
# TAB 1 — Player vs Player
# ============================================================
with tab1:
    st.header("Player vs Player Prediction")
    st.markdown(
        "Select two players and set the match context, then click **Predict** "
        "to see win probabilities."
    )

    # --- Player selectors ---------------------------------------------------
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.subheader("Player 1")
        p1_name = st.selectbox("Select Player 1", player_names, key="p1_select")
        p1_rank = rank_lookup.get(p1_name, 500)
        st.metric("Last known ATP rank", p1_rank)
        p1_pts = st.number_input(
            "Override ranking points (0 = use model default)",
            min_value=0, max_value=20000, value=0, step=50, key="p1_pts"
        )

    with col_p2:
        st.subheader("Player 2")
        # Default to a different player so the two selectors don't start identical
        default_p2_idx = min(1, len(player_names) - 1)
        p2_name = st.selectbox("Select Player 2", player_names,
                               index=default_p2_idx, key="p2_select")
        p2_rank = rank_lookup.get(p2_name, 500)
        st.metric("Last known ATP rank", p2_rank)
        p2_pts = st.number_input(
            "Override ranking points (0 = use model default)",
            min_value=0, max_value=20000, value=0, step=50, key="p2_pts"
        )

    st.divider()

    # --- Match context -------------------------------------------------------
    st.subheader("Match Context")
    ctx_col1, ctx_col2, ctx_col3 = st.columns(3)

    with ctx_col1:
        surface = st.selectbox(
            "Surface", ["Hard", "Clay", "Grass", "Carpet"], key="ctx_surface"
        )
        series = st.selectbox(
            "Series",
            ["International", "ATP250", "ATP500", "Grand Slam",
             "Masters", "Masters 1000", "Masters Cup", "International Gold"],
            key="ctx_series",
        )

    with ctx_col2:
        court = st.radio("Court", ["Outdoor", "Indoor"], key="ctx_court")
        round_name = st.selectbox(
            "Round",
            ["1st Round", "2nd Round", "3rd Round", "4th Round",
             "Quarterfinals", "Semifinals", "The Final"],
            key="ctx_round",
        )

    with ctx_col3:
        best_of = st.radio("Best of", [3, 5], key="ctx_bestof")

    st.divider()

    # --- Predict button ------------------------------------------------------
    if st.button("⚡ Predict", use_container_width=True, key="predict_btn"):
        if p1_name == p2_name:
            st.warning("Please select two different players.")
        else:
            pts1 = p1_pts if p1_pts > 0 else 0
            pts2 = p2_pts if p2_pts > 0 else 0
            pts_avail = 1 if (pts1 > 0 or pts2 > 0) else 0

            with st.spinner("Computing…"):
                prob1 = _predict_pvp(
                    rank1=p1_rank, rank2=p2_rank,
                    pts1=pts1, pts2=pts2,
                    pts_available=pts_avail,
                    surface=surface, series=series, court=court,
                    round_name=round_name, best_of=best_of,
                )
            prob2 = 1.0 - prob1

            st.subheader("Prediction Results")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(f"{p1_name}", f"{prob1 * 100:.1f}%", delta="Player 1")
                st.progress(prob1)
            with res_col2:
                st.metric(f"{p2_name}", f"{prob2 * 100:.1f}%", delta="Player 2")
                st.progress(prob2)

            # Interpretation note
            favourite = p1_name if prob1 >= prob2 else p2_name
            fav_prob = max(prob1, prob2)
            if fav_prob >= 0.65:
                confidence = "strong"
            elif fav_prob >= 0.55:
                confidence = "moderate"
            else:
                confidence = "slight"

            st.info(
                f"📊 The model gives **{favourite}** a **{confidence} favourite** "
                f"edge ({fav_prob * 100:.1f}% win probability) based on "
                f"ranking (#{p1_rank} vs #{p2_rank}) and match context."
            )


# ============================================================
# TAB 2 — Live Challenge
# ============================================================
with tab2:
    st.header("Live Challenge — Holdout Test Set")
    st.markdown(
        "Pick a real match from the holdout test set, make your prediction, "
        "then reveal the actual result to see if the model was correct."
    )

    lc = _load_lc_module()

    @st.cache_data(show_spinner="Loading holdout matches…")
    def _get_holdout():
        return lc.get_holdout_matches()

    matches_df = _get_holdout()

    # Build display labels for the selectbox
    labels = [lc.format_match_label(row) for _, row in matches_df.iterrows()]
    selected_label = st.selectbox("Select a match", labels, key="lc_select")
    selected_idx = labels.index(selected_label)
    selected_row = matches_df.iloc[selected_idx]

    st.divider()

    # Show match info
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Player 1", selected_row["Player_1"])
        st.metric("Player 1 Rank", int(selected_row["Rank_1"]))
    with info_col2:
        st.metric("Player 2", selected_row["Player_2"])
        st.metric("Player 2 Rank", int(selected_row["Rank_2"]))
    with info_col3:
        st.metric("Tournament", selected_row.get("Tournament", "—"))
        st.metric("Surface", selected_row["Surface"])

    st.divider()

    # Prediction
    with st.spinner("Predicting…"):
        lc_prob1 = lc.predict_match(selected_row.to_frame().T)
    lc_prob2 = 1.0 - lc_prob1

    pred_col1, pred_col2 = st.columns(2)
    with pred_col1:
        st.metric(f"{selected_row['Player_1']} win prob.", f"{lc_prob1 * 100:.1f}%")
        st.progress(lc_prob1)
    with pred_col2:
        st.metric(f"{selected_row['Player_2']} win prob.", f"{lc_prob2 * 100:.1f}%")
        st.progress(lc_prob2)

    predicted_winner = (
        selected_row["Player_1"] if lc_prob1 >= lc_prob2 else selected_row["Player_2"]
    )
    st.markdown(f"🤖 **Model predicts:** {predicted_winner}")

    st.divider()

    # Reveal button
    if st.button("🔍 Reveal Actual Result", key="reveal_btn"):
        actual_wins = int(selected_row["Player1_wins"])
        actual_winner = (
            selected_row["Player_1"] if actual_wins == 1 else selected_row["Player_2"]
        )
        model_correct = predicted_winner == actual_winner

        if model_correct:
            st.success(f"✅ Actual winner: **{actual_winner}** — Model was **CORRECT**!")
        else:
            st.error(f"❌ Actual winner: **{actual_winner}** — Model was **WRONG**.")
