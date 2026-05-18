"""CricketIQ — Streamlit War Room Dashboard.

Interactive dashboard combining all 4 modules:
  1. Pressure Genome: radar chart, archetype explorer, selection recommendations
  2. Impact Player AI: live substitution recommendation with confidence meter
  3. Broadcast Monetisation: pre-match hot zone report, predicted excitement curves
  4. Fantasy CLV: churn risk rankings, segment distribution, revenue impact
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import CricketDataLoader
from src.pressure_genome import PressureGenome, PressureGenomeConfig
from src.impact_player import ImpactPlayerAI, ImpactPlayerConfig, MatchState
from src.broadcast_monetisation import BroadcastMonetisation, BroadcastConfig
from src.fantasy_clv import FantasyChurnCLV, ChurnCLVConfig

st.set_page_config(
    page_title="CricketIQ — War Room",
    page_icon="🏏",
    layout="wide",
)

st.title("CricketIQ — Enterprise Cricket Analytics War Room")
st.markdown("---")

# ---------------------------------------------------------------------------
# Initialize data + models (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def init_pressure_genome():
    loader = CricketDataLoader()
    df = loader.get_pressure_features(200)
    genome = PressureGenome(PressureGenomeConfig())
    genome.fit(df)
    df["cluster"] = genome._cluster_labels
    archetypes = genome.name_archetypes(df)
    df["archetype"] = df["cluster"].map(archetypes)
    return genome, df

@st.cache_resource
def init_impact_player():
    ai = ImpactPlayerAI()
    ai.run_pipeline()
    return ai

@st.cache_resource
def init_broadcast():
    loader = CricketDataLoader()
    over_stats = loader.get_over_stats()
    bc = BroadcastMonetisation(BroadcastConfig(epochs=20))
    bc.run_pipeline(over_stats)
    return bc

@st.cache_resource
def init_fantasy():
    loader = CricketDataLoader()
    try:
        users = loader.get_fantasy_users(50000)
        fc = FantasyChurnCLV(ChurnCLVConfig())
        fc.run_pipeline(users)
        return fc
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

module = st.sidebar.radio(
    "Select Module",
    ["Pressure Genome", "Impact Player AI", "Broadcast Monetisation", "Fantasy CLV"],
)

# ---------------------------------------------------------------------------
# MODULE 1 — Pressure Genome
# ---------------------------------------------------------------------------

if module == "Pressure Genome":
    st.header("🧬 Pressure Genome — Batsman Psychological DNA")

    genome, df = init_pressure_genome()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Archetype Distribution")
        counts = df["archetype"].value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                      title="Player Archetypes", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Player Comparison")
        players = sorted(df["player"].unique())
        p1 = st.selectbox("Player 1", players, index=players.index("V Kohli") if "V Kohli" in players else 0)
        p2 = st.selectbox("Player 2", players, index=players.index("MS Dhoni") if "MS Dhoni" in players else 0)

        data = genome.comparison_data(df, p1, p2)
        if data:
            d1, d2 = data
            categories = list(d1.keys())
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=list(d1.values()), theta=categories,
                                          fill='toself', name=p1))
            fig.add_trace(go.Scatterpolar(r=list(d2.values()), theta=categories,
                                          fill='toself', name=p2))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                              height=500, margin=dict(l=80, r=80, t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selection Recommendation Engine")
    col3, col4, col5 = st.columns(3)
    with col3:
        rrr = st.number_input("Required Run Rate", value=10.5, step=0.5)
    with col4:
        wkts_left = st.number_input("Wickets Left", value=4, min_value=0, max_value=10)
    with col5:
        overs_left = st.number_input("Overs Remaining", value=4.0, step=0.5, min_value=0.0, max_value=20.0)

    match_state = {
        "required_run_rate": rrr,
        "wickets_left": wkts_left,
        "is_chase": True,
        "overs_remaining": overs_left,
    }
    top_n = genome.rank_for_situation(df, match_state, top_n=5)
    st.dataframe(top_n, use_container_width=True)

    st.subheader("Pressure Mismatch Alert")
    lineup = st.text_input("Available batsmen (comma-separated)", "V Kohli, RG Sharma, SK Yadav, KL Rahul")
    player_list = [p.strip() for p in lineup.split(",") if p.strip()]
    if player_list:
        alert = genome.pressure_mismatch_alert(player_list, df, match_state)
        st.metric("Avg Lineup Compatibility", f"{alert['avg_compatibility']:.0%}",
                  delta="MISMATCH" if alert['mismatch_detected'] else "OK",
                  delta_color="inverse" if alert['mismatch_detected'] else "normal")
        st.info(alert["message"])

# ---------------------------------------------------------------------------
# MODULE 2 — Impact Player AI
# ---------------------------------------------------------------------------

elif module == "Impact Player AI":
    st.header("🤖 Impact Player AI — Strategic Substitution Engine")

    ai = init_impact_player()

    st.subheader("Match State Input")
    col1, col2, col3 = st.columns(3)
    with col1:
        score = st.number_input("Current Score", value=145, step=5)
        wkts = st.number_input("Wickets Fallen", value=4, min_value=0, max_value=9)
        rrr = st.number_input("Required Run Rate", value=12.0, step=0.5)
    with col2:
        overs_left = st.number_input("Overs Remaining", value=6.0, step=0.5, min_value=0.0)
        is_pressure = st.checkbox("Pressure Situation", value=True)
        is_first = st.checkbox("First Innings", value=False)
    with col3:
        available = st.text_area("Available Players (one per line)", "Shivam Dube\nDeepak Chahar\nMoeen Ali\nTim David\nPiyush Chawla")
        avail_list = [p.strip() for p in available.split("\n") if p.strip()]

    state = MatchState(
        current_score=score, wickets_fallen=wkts, required_run_rate=rrr,
        overs_remaining=overs_left, is_first_innings=float(is_first),
        is_pressure_situation=float(is_pressure),
    )

    if st.button("Get Recommendation", type="primary"):
        rec = ai.recommend_substitution(state, avail_list)

        col_a, col_b = st.columns(2)
        with col_a:
            decision = "SUBSTITUTE NOW" if rec["substitute_now"] else "WAIT"
            st.metric("Decision", decision,
                      delta=f"{rec['confidence']*100:.0f}% confidence",
                      delta_color="off" if rec['substitute_now'] else "normal")

        with col_b:
            st.markdown(f"**Recommended action:** {rec['recommended_action'].replace('_', ' ').title()}")
            st.markdown(f"**Baseline benefit prob:** {rec['baseline_benefit_probability']:.1%}")

        st.subheader("Top Candidates")
        for i, c in enumerate(rec["candidates"], 1):
            with st.container():
                st.markdown(f"**{i}. {c['player']}** ({c['role']})")
                st.markdown(f"Expected uplift: **+{c['expected_uplift_runs']:.0f} runs**")
                st.markdown(f"Compatibility: {c['compatibility']:.2f}")
                st.markdown(f"*{c['rationale']}*")

    st.subheader("Counterfactual Analysis")
    cf_results = ai.counterfactual.analyse_all()
    for _, r in cf_results.iterrows():
        verdict_color = "🟢" if r["runs_difference"] > 0 else "🟡" if abs(r["runs_difference"]) <= 3 else "🔴"
        st.markdown(f"{verdict_color} **{r['match']}** — Model: {r['model_recommends']} vs Actual: {r['actual'][:50]}... | Δ: {r['runs_difference']:+.1f} runs")

# ---------------------------------------------------------------------------
# MODULE 3 — Broadcast Monetisation
# ---------------------------------------------------------------------------

elif module == "Broadcast Monetisation":
    st.header("📺 Broadcast Monetisation Predictor")

    bc = init_broadcast()

    st.subheader("Revenue Impact Summary")
    rev = bc._processed_data
    if rev is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Matches Analysed", bc._processed_data["match_id"].nunique())
        with col2:
            st.metric("Peak Window Fraction", f"{bc._processed_data['engagement_window'].mean():.1%}")
        with col3:
            st.metric("Precision@1", f"{bc.detector.precision_at_1(bc._processed_data):.2%}")

    st.subheader("Match Hot Zone Report")
    match_ids = bc._processed_data["match_id"].unique()[:10] if bc._processed_data is not None else []
    if len(match_ids) > 0:
        selected_match = st.selectbox("Select Match", match_ids)
        report = bc.generate_match_report(selected_match)
        if "error" not in report:
            st.metric("Estimated Ad Revenue", f"₹{report['estimated_ad_revenue_cr']} crore")
            st.markdown("**Top 5 Peak Windows:**")
            for w in report["top_5_hot_zones"]:
                st.markdown(f"- Innings {w['innings']}, Over {w['over']}: excitement = {w['excitement_normalised']:.3f}")

    st.subheader("Excitement Curve")
    if bc._processed_data is not None:
        sample_mid = bc._processed_data["match_id"].iloc[0]
        match_df = bc._processed_data[bc._processed_data["match_id"] == sample_mid].sort_values(["innings", "over"])
        fig = px.line(match_df, x="over", y="excitement_normalised", color="innings",
                      title=f"Excitement Density — {sample_mid}",
                      markers=True)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# MODULE 4 — Fantasy CLV
# ---------------------------------------------------------------------------

elif module == "Fantasy CLV":
    st.header("💰 Fantasy Churn & Lifetime Value Engine")

    fc = init_fantasy()

    st.subheader("Segment Distribution")
    segmented = fc.get_segmented_data()
    if segmented is not None:
        seg_counts = segmented["segment"].value_counts()
        fig = px.bar(x=seg_counts.index, y=seg_counts.values,
                     title="User Segments", color=seg_counts.index,
                     labels={"x": "Segment", "y": "Users"})
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Rate", f"{segmented['churned'].mean():.1%}")
        with col2:
            st.metric("Avg CLV", f"₹{segmented['predicted_clv'].mean():.0f}")
        with col3:
            high_clv = segmented[segmented["segment"] == "At-Risk High-CLV"]
            st.metric("At-Risk High-CLV Users", len(high_clv))

    st.subheader("Revenue Impact")
    revenue = fc.intervention.simulate_revenue_impact(segmented) if segmented is not None else {}
    if revenue:
        st.info(revenue.get("message", ""))
        for seg, data in revenue.get("per_segment", {}).items():
            st.markdown(f"- **{seg}**: ₹{data['recovered_annual_cr']} cr/yr (at {data['conversion_rate']:.0%} conversion)")

    st.subheader("Churn Risk Rankings")
    rankings = fc.get_risk_rankings(top_n=20)
    if rankings is not None:
        st.dataframe(rankings, use_container_width=True)

    st.subheader("Intervention Strategy Matrix")
    matrix = fc.intervention.intervention_matrix()
    st.table(matrix[["segment", "priority", "intervention", "expected_conversion"]])
