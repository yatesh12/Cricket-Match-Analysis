"""CricketIQ: Self-contained Streamlit War Room Dashboard.

Loads data from the feature store (parquet) and computes all analysis
on-the-fly. No dependency on pre-run notebooks.
"""

import sys, os, json, warnings
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CricketIQ: War Room", page_icon="🏏", layout="wide")
st.title("CricketIQ: Enterprise Cricket Analytics")
st.markdown("---")

# ---------------------------------------------------------------------------
# Build feature store (fast if already cached)
# ---------------------------------------------------------------------------
from src.data_pipeline import build_feature_store, load_feature_store, aggregate_player_level
store = build_feature_store()
ball_by_ball = store["match_state"]
over_stats = store["over_level"]
player_stats = store["player_level"]

# ---------------------------------------------------------------------------
# Module selector
# ---------------------------------------------------------------------------
module = st.sidebar.radio(
    "Select Module",
    ["Pressure Genome", "Impact Player AI", "Broadcast Monetisation", "Fantasy CLV"],
)

# ===================================================================
# MODULE 1: PRESSURE GENOME
# ===================================================================
if module == "Pressure Genome":
    st.header("Pressure Genome: Batsman Psychological DNA")
    st.markdown(f"Analysing **{len(ball_by_ball)}** deliveries across **{ball_by_ball['match_id'].nunique()}** matches")

    # --- 12 pressure features ---
    df = ball_by_ball.copy()
    df["is_wide"] = (df["wides"].fillna(0) > 0).astype(int)
    df["is_noball"] = (df["noballs"].fillna(0) > 0).astype(int)
    legal = (~df["is_wide"].fillna(0).astype(bool) & ~df["is_noball"].fillna(0).astype(bool))
    df["legal"] = legal
    df["is_boundary"] = (df["runs_off_bat"] >= 4) & ~df["is_wide"].astype(bool)
    df["is_dot"] = (df["runs_off_bat"] == 0) & df["legal"]
    df["death_overs"] = df["over"].fillna(0).astype(int) >= 16
    df["chase_high_rrr"] = (df["innings"] == 2) & (df["required_run_rate"] > 10)
    df["high_rrr"] = df["required_run_rate"] > 12
    df["quick_wickets"] = df.groupby(["match_id", "innings"])["wickets"].transform(
        lambda x: x.rolling(6, min_periods=1).sum()
    ) >= 2

    records = []
    for batter, grp in df.groupby("striker"):
        n_balls = int(grp["legal"].sum())
        if n_balls < 5:
            continue
        tr = grp["runs_off_bat"].sum()
        death = grp[grp["death_overs"] & grp["legal"]]
        dsr = (death["runs_off_bat"].sum() / max(len(death), 1)) * 100
        chase = grp[grp["chase_high_rrr"] & grp["legal"]]
        csr = (chase["runs_off_bat"].sum() / max(len(chase), 1)) * 100
        br = grp[grp["is_boundary"]]["runs_off_bat"].sum() / max(tr, 1)
        grp_s = grp.sort_values(["match_id", "innings", "over", "ball"])
        dot_sh = grp_s["is_dot"].shift(1).fillna(False)
        rec = grp_s[dot_sh & grp_s["legal"]]
        rrec = (rec["runs_off_bat"].sum() / max(len(rec), 1)) * 100
        hrr = grp[grp["high_rrr"] & grp["legal"]]
        hrr_sr = (hrr["runs_off_bat"].sum() / max(len(hrr), 1)) * 100
        clutch = grp[grp["death_overs"] & (grp["innings"] == 2) & grp["legal"]]
        cbp = clutch["is_boundary"].sum() / max(len(clutch), 1)
        coll = grp[grp["quick_wickets"]]
        crs = coll["runs_off_bat"].sum() / max(coll["legal"].sum(), 1) * 100
        pb = grp[grp["death_overs"] | grp["chase_high_rrr"] | grp["high_rrr"]]["runs_off_bat"]
        pc = float(np.std(pb)) if len(pb) > 5 else 0.5
        records.append({"player": batter, "matches": grp["match_id"].nunique(),
            "balls": n_balls, "death_overs_sr": round(dsr, 2), "chase_pressure_sr": round(csr, 2),
            "boundary_dependency": round(br, 3), "dot_ball_recovery_rate": round(rrec, 2),
            "high_rr_performance": round(hrr_sr, 2), "clutch_boundary_rate": round(cbp, 3),
            "collapse_resistance": round(crs, 2), "pressure_consistency": round(pc, 3),
            "overall_sr": (tr / max(n_balls, 1)) * 100})

    pf = pd.DataFrame(records).fillna(0)
    if pf.empty:
        st.error("Not enough data to compute pressure profiles")
        st.stop()

    # --- PCA + KMeans ---
    feat_cols = [c for c in pf.columns if c not in ("player", "matches", "balls", "overall_sr", "cluster", "archetype")]
    X = pf[feat_cols].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca = PCA(n_components=min(3, X_s.shape[1]), random_state=42)
    X_p = pca.fit_transform(X_s)
    km = KMeans(n_clusters=min(3, len(pf)), random_state=42, n_init="auto")
    pf["cluster"] = km.fit_predict(X_p)
    centroids = pf.groupby("cluster")[feat_cols].mean()

    def label_arch(row, cdf):
        """Dynamic archetype labeling using available feature percentiles."""
        # Detect which features are available
        avail_feats = [c for c in feat_cols if c in cdf.columns and c in row.index]
        if len(avail_feats) < 2:
            return "Unclassified"

        # High-value and low-value feature sets (dynamically detected)
        high_variance_feats = [f for f in avail_feats if cdf[f].std() > 0]
        if not high_variance_feats:
            return "Unclassified"

        q75 = {f: cdf[f].quantile(0.75) for f in high_variance_feats}
        q25 = {f: cdf[f].quantile(0.25) for f in high_variance_feats}
        q50 = {f: cdf[f].quantile(0.50) for f in high_variance_feats}

        # Count how many features are above 75th percentile (high performer)
        high_count = sum(1 for f in high_variance_feats if row[f] >= q75[f])
        # Count how many features are below 25th percentile (low performer)
        low_count = sum(1 for f in high_variance_feats if row[f] <= q25[f])
        n = len(high_variance_feats)

        if high_count >= n * 0.6:
            return "Ice Finisher"
        elif low_count >= n * 0.6:
            return "Risk Stabilizer"
        elif high_count >= n * 0.4:
            return "Power Enforcer"
        elif high_count >= n * 0.2 and low_count < n * 0.3:
            return "Collapse Anchor"
        elif high_count >= n * 0.25:
            return "Chaos Accelerator"
        else:
            return "Situational Player"

    arch_map = {c: label_arch(centroids.loc[c], centroids) for c in centroids.index}
    pf["archetype"] = pf["cluster"].map(arch_map)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Archetype Distribution")
        vc = pf["archetype"].value_counts()
        fig = px.pie(values=vc.values, names=vc.index, title="Player Archetypes", hole=0.4)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Player Comparison")
        players = sorted(pf["player"].unique())
        p1 = st.selectbox("Player 1", players, index=0, key="p1")
        p2 = st.selectbox("Player 2", players, index=min(1, len(players) - 1), key="p2")
        r1, r2 = pf[pf["player"] == p1], pf[pf["player"] == p2]
        if not r1.empty and not r2.empty:
            cats = feat_cols[:8]
            v1, v2 = [float(r1.iloc[0][c]) for c in cats], [float(r2.iloc[0][c]) for c in cats]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=v1 + [v1[0]], theta=cats + [cats[0]], fill="toself", name=p1))
            fig.add_trace(go.Scatterpolar(r=v2 + [v2[0]], theta=cats + [cats[0]], fill="toself", name=p2))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), height=500)
            st.plotly_chart(fig, width='stretch')

    st.subheader("Selection Recommendation")
    c3, c4, c5 = st.columns(3)
    with c3:
        rrr = st.number_input("Required Run Rate", value=10.5, step=0.5, key="rrr")
    with c4:
        wkts = st.number_input("Wickets Left", value=4, min_value=0, max_value=10, key="wkts")
    with c5:
        overs = st.number_input("Overs Remaining", value=4.0, step=0.5, key="overs")

    # Dynamic weights based on match situation
    overs_left_factor = overs / 20.0
    death_phase = overs <= 6.0
    chase_pressure = rrr > 8.0
    collapsing = (10 - wkts) >= 5
    weights = {
        "death_overs_sr": 0.30 if death_phase else (0.20 if rrr > 6 else 0.10),
        "chase_pressure_sr": 0.30 if chase_pressure else 0.15,
        "clutch_boundary_rate": 0.20 if chase_pressure else 0.10,
        "boundary_dependency": 0.10 if chase_pressure and not collapsing else 0.05,
        "dot_ball_recovery_rate": 0.15 if collapsing else 0.10,
        "pressure_consistency": 0.15 if not death_phase else 0.05,
    }
    # Normalise weights to sum to 1
    wt = sum(weights.values())
    weights = {k: v / wt for k, v in weights.items()}
    score = sum(pf.get(f, 0) * w for f, w in weights.items() if f in pf.columns)
    score = score / pf["overall_sr"].max() if pf["overall_sr"].max() > 0 else 0
    top_idx = np.argsort(score.values)[::-1][:5]
    top5 = pf.iloc[top_idx][["player", "archetype", "overall_sr"]].copy()
    top5["compatibility"] = score.iloc[top_idx].values
    st.dataframe(top5, width='stretch')
    with st.expander("Why these weights?"):
        st.markdown(
            f"- **Death overs SR** weighted **{weights.get('death_overs_sr',0):.0%}** "
            f"{'(death phase, <6 overs left)' if death_phase else ''}\n"
            f"- **Chase pressure SR** weighted **{weights.get('chase_pressure_sr',0):.0%}** "
            f"{'(high RRR > 8)' if chase_pressure else ''}\n"
            f"- **Boundary dependency** weighted **{weights.get('boundary_dependency',0):.0%}**\n"
            f"- **Pressure consistency** weighted **{weights.get('pressure_consistency',0):.0%}**"
        )

    st.subheader("Lineup Mismatch Alert")
    lineup = st.text_input("Available batsmen (comma-separated)", "Ms Dhoni, V Kohli, RG Sharma, SK Yadav")
    plist = [p.strip() for p in lineup.split(",") if p.strip()]
    avail = pf[pf["player"].isin(plist)]
    if not avail.empty:
        avg_c = avail["overall_sr"].mean() / pf["overall_sr"].max() if pf["overall_sr"].max() > 0 else 0.5
        st.metric("Avg Lineup Compatibility", f"{avg_c:.0%}",
                  delta="MISMATCH" if avg_c < 0.4 else "OK",
                  delta_color="inverse" if avg_c < 0.4 else "normal")

# ===================================================================
# MODULE 2: IMPACT PLAYER AI
# ===================================================================
elif module == "Impact Player AI":
    st.header("Impact Player AI: Strategic Substitution Engine")
    st.markdown(f"Analysing **{len(ball_by_ball)}** deliveries across **{ball_by_ball['match_id'].nunique()}** matches")

    # --- 14-d state vector ---
    data = ball_by_ball.sort_values(["match_id", "innings", "over", "ball"]).copy()
    data["is_boundary_flag"] = (data["runs_off_bat"] >= 4) & (data["wides"].fillna(0) == 0)
    phase_map = {"powerplay": 0, "middle": 1, "death": 2, "chase_death": 2}
    state_df = pd.DataFrame({
        "match_id": data["match_id"], "innings": data["innings"], "over": data["over"],
        "batter": data["striker"], "bowler": data["bowler"],
        "s1_innings": data["innings"],
        "s2_balls_remaining": data["balls_remaining"].fillna(0),
        "s3_wickets_in_hand": data["wickets_in_hand"].fillna(0),
        "s4_required_run_rate": data["required_run_rate"].fillna(0),
        "s5_current_run_rate": data["current_run_rate"].fillna(0),
        "s6_momentum_score": data["momentum_score"].fillna(0),
        "s7_phase_code": data["phase_of_play"].map(phase_map).fillna(1),
        "s8_runs_last_6": data.groupby(["match_id", "innings"])["runs"].transform(lambda x: x.rolling(6, min_periods=1).sum()).fillna(0),
        "s9_wickets_last_6": data.groupby(["match_id", "innings"])["wickets"].transform(lambda x: x.rolling(6, min_periods=1).sum()).fillna(0),
        "s10_boundary_3ov": data.groupby(["match_id", "innings"])["is_boundary_flag"].transform(lambda x: x.rolling(18, min_periods=1).sum()).fillna(0),
        "s11_pressure_idx": data["pressure_index"].fillna(0),
        "s12_win_prob": data["chase_win_probability_proxy"].fillna(0.5),
        "s13_bowling_pressure": data["bowling_pressure_index"].fillna(0),
        "s14_partnership": data["partnership_runs"].fillna(0),
    })
    sc = [c for c in state_df.columns if c.startswith("s")]

    def discretize(row):
        rrr = row["s4_required_run_rate"]
        wkts = row["s3_wickets_in_hand"]
        b = (0 if rrr <= 6 else 1 if rrr <= 10 else 2,
             0 if wkts >= 7 else 1 if wkts >= 4 else 2,
             int(row["s7_phase_code"]),
             0 if row["s6_momentum_score"] <= -2 else 1 if row["s6_momentum_score"] <= 2 else 2,
             0 if row["s11_pressure_idx"] <= 0.3 else 1 if row["s11_pressure_idx"] <= 0.6 else 2,
             0 if row["s12_win_prob"] <= 0.3 else 1 if row["s12_win_prob"] <= 0.7 else 2,
             0 if row["s13_bowling_pressure"] <= 0.3 else 1 if row["s13_bowling_pressure"] <= 0.6 else 2)
        return b

    state_df["state_key"] = state_df[sc].apply(discretize, axis=1)

    st.subheader("Recommendation")
    with st.form("impact_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            rrr = st.number_input("Required Run Rate", value=11.5, step=0.5)
            wkts = st.number_input("Wickets Fallen", value=5, min_value=0, max_value=9)
        with col2:
            overs_left = st.number_input("Overs Remaining", value=4.0, step=0.5)
            pressure = st.checkbox("Pressure Situation", value=True)
        with col3:
            avail = st.text_area("Available Players", "Ms Dhoni\nShivam Dube\nDeepak Chahar\nTim David")
        submitted = st.form_submit_button("Get Recommendation", type="primary")

    if submitted:
        avail_list = [p.strip() for p in avail.split("\n") if p.strip()]
        state_df["dist"] = ((state_df["s4_required_run_rate"] - rrr).abs() + (10 - state_df["s3_wickets_in_hand"] - wkts).abs())
        best = state_df.loc[state_df["dist"].idxmin()]
        q_table_file = Path(__file__).resolve().parent.parent / "models" / "q_table.json"

        if q_table_file.exists():
            with open(q_table_file) as f:
                qt = json.load(f)
            # Q-table trained on 3-part keys: rrr_wicketsInHand_ballsRemaining
            q_table = {}
            for k, v in qt["q_table"].items():
                parts = tuple(map(int, k.split("_")))
                q_table[parts] = np.array(v)
            actions = qt["action_names"]
            # Build matching 3-part key from best state
            sk_rrr = int(round(best.get("s4_required_run_rate", 0)))
            sk_wkts = int(round(best.get("s3_wickets_in_hand", 10)))
            sk_balls = int(round(best.get("s2_balls_remaining", 120) / 10) * 10)
            sk = (sk_rrr, sk_wkts, sk_balls)
            closest = min(q_table.keys(), key=lambda k: sum(abs(a - b) for a, b in zip(k, sk))) if q_table else None
            if closest and closest in q_table:
                qv = q_table[closest]
                ba = int(np.argmax(qv))
                conf = float((np.max(qv) - np.mean(qv)) / (np.ptp(qv) + 1e-8))
                ca, cb = st.columns(2)
                with ca:
                    dec = "SUBSTITUTE NOW" if ba in (1, 2) else "WAIT"
                    st.metric("Decision", dec, delta=f"{conf:.0%} confidence",
                              delta_color="off" if ba in (1, 2) else "normal")
                with cb:
                    st.markdown(f"**Action:** {actions[ba].replace('_', ' ').title()}")
        else:
            st.info("No pre-trained Q-table found. Using heuristic: substitute when RRR > 10 and wickets < 5.")
            should_sub = rrr > 10 and wkts >= 5
            st.metric("Decision", "SUBSTITUTE NOW" if should_sub else "WAIT")

        st.subheader("Top Candidates")
        # Role definitions
        role_map = {
            "ms dhoni": ("Wicketkeeper", 2), "v kohli": ("Batsman", 5),
            "rg sharma": ("Batsman", 5), "sk yadav": ("Batsman", 5),
            "suryakumar yadav": ("Batsman", 5), "tim david": ("Batsman", 5),
            "shivam dube": ("All-rounder", 3), "hardik pandya": ("All-rounder", 3),
            "ravindra jadeja": ("All-rounder", 3), "deepak chahar": ("Bowler", 0),
            "jasprit bumrah": ("Bowler", 0), "yuzvendra chahal": ("Bowler", 0),
            "pat cummins": ("All-rounder", 3), "andre russell": ("All-rounder", 3),
            "sunil narine": ("All-rounder", 3), "kl rahul": ("Wicketkeeper", 2),
            "rishabh pant": ("Wicketkeeper", 2), "sanju samson": ("Wicketkeeper", 2),
            "ishan kishan": ("Wicketkeeper", 2), "jos buttler": ("Wicketkeeper", 2),
        }
        cand = []
        for pl in avail_list:
            pdata = state_df[state_df["batter"] == pl]
            if not pdata.empty:
                uplift = pdata["s8_runs_last_6"].mean() / 6.0 - pdata["s4_required_run_rate"].mean()
                compat = round(1 / (1 + abs(uplift)), 3)
                role_info = role_map.get(pl.lower().strip(), ("Unknown", 0))
                role_name, role_wt = role_info
                cand.append({
                    "player": pl, "role": role_name, "role_wt": role_wt,
                    "expected_uplift_runs": round(uplift, 2),
                    "compatibility": compat,
                    "weighted_score": round((uplift * 0.6) + (compat * 0.4) + role_wt, 2),
                })

        # Filter: exclude pure Bowlers unless pinch-hitter flag set
        pinch_hitters = ["deepak chahar"]
        filtered = [c for c in cand if c["role"] != "Bowler" or c["player"].lower().strip() in pinch_hitters]

        if filtered:
            cdf = pd.DataFrame(filtered).sort_values("weighted_score", ascending=False)
            st.dataframe(cdf[["player", "role", "expected_uplift_runs", "compatibility", "weighted_score"]],
                         width='stretch', column_config={
                "player": "Player", "role": "Role",
                "expected_uplift_runs": st.column_config.NumberColumn("Expected Runs", format="+%.2f"),
                "compatibility": st.column_config.ProgressColumn("Compat.", format="%.1f%%", min_value=0, max_value=1),
                "weighted_score": "Score",
            })
            best = cdf.iloc[0]
            sign = "+" if best["expected_uplift_runs"] >= 0 else ""
            st.success(
                f"**Top Pick: {best['player']}** ({best['role']}) — "
                f"Score: {best['weighted_score']} | "
                f"{sign}{best['expected_uplift_runs']} expected runs | "
                f"{best['compatibility']:.1%} compatible"
            )
            if best["expected_uplift_runs"] < 0:
                st.caption("All available players show negative uplift in this match state. Pick is the least-worst option.")
        else:
            st.warning("No suitable batting substitute available. All candidates are pure bowlers.")

    st.subheader("State Space Summary")
    st.markdown(f"**{len(state_df)}** state vectors, **{state_df['state_key'].nunique()}** unique discretised states")

# ===================================================================
# MODULE 3: BROADCAST MONETISATION
# ===================================================================
elif module == "Broadcast Monetisation":
    st.header("Broadcast Monetisation Predictor")
    st.markdown(f"Analysing **{len(over_stats)}** overs across **{over_stats['match_id'].nunique()}** matches")

    df = over_stats.copy()
    is_chase = df["innings"] == 2
    df["excitement"] = (df["boundaries"] * 2.0 + df["wickets"] * 4.0 + df["dot_balls"] * 0.5 +
                        np.where(is_chase, df["dot_balls"] * 1.5, 0.0))
    df["excitement_norm"] = df.groupby("match_id")["excitement"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))

    p75 = df["excitement"].quantile(0.75)
    p50 = df["excitement"].quantile(0.50)
    df["ad_rate"] = np.select(
        [df["excitement"] >= p75, df["excitement"] >= p50],
        [25_00_000, 8_00_000], default=3_00_000)
    df["ad_over"] = df["ad_rate"] * 4
    df["is_peak"] = (df["excitement"] >= p75).astype(int)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matches Analysed", df["match_id"].nunique())
    with col2:
        st.metric("Peak Windows", f"{df['is_peak'].mean():.1%}")
    with col3:
        avg_rev = df["ad_over"].mean()
        st.metric("Avg Rev/Over", f"₹{avg_rev:,.0f}")

    st.subheader("Match Hot Zone Report")
    mids = sorted(df["match_id"].unique())
    if mids:
        sel = st.selectbox("Select Match", mids, key="m3_mid")
        md = df[df["match_id"] == sel].sort_values(["innings", "over"])
        peak = md[md["is_peak"] == 1]
        total_ad = len(peak) * 25_00_000 * 4 + (len(md) - len(peak)) * 8_00_000 * 4
        st.metric("Estimated Ad Revenue", f"₹{total_ad / 1e7:.2f} crore")
        st.dataframe(md[["innings", "over", "excitement", "runs_scored", "wickets", "is_peak"]].head(20),
                     width='stretch')

        fig = px.line(md, x="over", y="excitement_norm", color=md["innings"].astype(str),
                      title=f"Excitement: Match {sel}", markers=True)
        st.plotly_chart(fig, width='stretch')

        # Revenue simulation
        n_sim, total = 50, 0
        for sim in range(n_sim):
            rng = np.random.default_rng(sim)
            rand_rev = (rng.choice([25_00_000, 8_00_000, 3_00_000], size=len(df)) * 4).sum()
            uniform_rev = len(df) * 8_00_000 * 4
            model_rev = (df["is_peak"] * 25_00_000 + (1 - df["is_peak"]) * 8_00_000).sum() * 4
            total += model_rev - uniform_rev
        avg_uplift_cr = total / n_sim / 1e7
        st.info(f"Model-guided placement beats uniform allocation by **₹{avg_uplift_cr:.2f} crore** "
                f"per season (avg over {n_sim} simulations).")

# ===================================================================
# MODULE 4: FANTASY CLV
# ===================================================================
else:
    st.header("Fantasy Churn & Lifetime Value Engine")

    fantasy_csv = Path(__file__).resolve().parent.parent / "data" / "processed" / "fantasy_users.csv"
    if fantasy_csv.exists():
        users = pd.read_csv(fantasy_csv)
        st.success(f"Loaded {len(users)} users from CSV")
    else:
        st.info("No fantasy_users.csv found. Generating realistic synthetic data for demonstration.")
        rng = np.random.default_rng(42)
        n = 50000
        users = pd.DataFrame({"user_id": range(1, n + 1)})
        users["age_group"] = rng.choice(["18-24", "25-34", "35-44", "45+"], n, p=[0.35, 0.35, 0.20, 0.10])
        users["contests_entered_per_week"] = np.clip(rng.lognormal(1.5, 1.0, n).astype(int), 1, 100)
        users["win_rate"] = np.clip(rng.normal(0.48, 0.12, n), 0.05, 0.95)
        users["days_since_last_login"] = np.clip(rng.lognormal(2.5, 1.5, n).astype(int), 0, 365)
        users["total_deposits"] = np.clip(rng.lognormal(7.5, 2.0, n).astype(int), 100, 10_000_000)
        users["loss_streak_length"] = rng.geometric(0.3, n) - 1
        users["team_diversity_score"] = np.clip(rng.beta(3, 4, n), 0.1, 0.9)
        logit = -2.5 + 0.15 * users["loss_streak_length"] + 0.02 * (users["days_since_last_login"] / 7) - 0.3 * users["win_rate"] - 0.5 * np.log1p(users["contests_entered_per_week"])
        users["churned"] = rng.binomial(1, p=1 / (1 + np.exp(-logit)))

    # Ensure required derived columns exist
    if "contests_entered_per_week" not in users.columns:
        users["contests_entered_per_week"] = (users["contests_joined"] / users["lifetime_days"].clip(lower=1) * 7).round(1)
    if "loss_streak_length" not in users.columns:
        users["loss_streak_length"] = ((1 - users["win_rate"]) * 5).round().astype(int)
    if "days_since_last_login" not in users.columns:
        users["days_since_last_login"] = np.random.default_rng(42).poisson(lam=14, size=len(users))

    # --- Feature engineering ---
    data = users.copy()
    data["log_deposits"] = np.log1p(data["total_deposits"])
    data["log_recency"] = np.log1p(data["days_since_last_login"])
    data["log_frequency"] = np.log1p(data["contests_entered_per_week"])
    data["loss_risk"] = data["loss_streak_length"] / data["loss_streak_length"].max()
    data["inactivity_risk"] = (data["days_since_last_login"] > 30).astype(int)
    data["churn_class"] = data["churned"]
    data["duration_days"] = data["days_since_last_login"].clip(1)
    data["event_observed"] = data["churned"]
    data["monetary"] = data["log_deposits"]

    st.subheader("Segment Distribution")
    high_clv = data["total_deposits"] > data["total_deposits"].quantile(0.75)
    high_risk = data["churned"] == 1
    high_win = data["win_rate"] > 0.55
    mega = data.get("mega_contest_ratio", data["contests_entered_per_week"] > 20)
    data["segment"] = np.select(
        [high_clv & high_risk, high_clv & ~high_risk & high_win, high_clv & ~high_risk & ~high_win, ~high_clv & mega],
        ["Churn Risk", "High Roller", "Loyal Grinder", "Promo Hunter"], default="Casual")
    data["churn_risk"] = np.where(data["segment"] == "Churn Risk", 0.8, np.where(data["segment"] == "Casual", 0.5, 0.2))

    vc = data["segment"].value_counts()
    fig = px.bar(x=vc.index, y=vc.values, title="User Segments", color=vc.index,
                 labels={"x": "Segment", "y": "Users"})
    st.plotly_chart(fig, width='stretch')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Churn Rate", f"{data['churned'].mean():.1%}")
    with col2:
        avg_dep = data["total_deposits"].mean()
        st.metric("Avg Deposit", f"₹{avg_dep:,.0f}")
    with col3:
        at_risk = len(data[data["segment"] == "Churn Risk"])
        st.metric("At-Risk Users", at_risk)

    st.subheader("Revenue Impact")
    seg_map = {"Churn Risk": 0.35, "High Roller": 0.20, "Loyal Grinder": 0.15,
               "Promo Hunter": 0.10, "Casual": 0.08}
    cost_map = {"Churn Risk": 150, "High Roller": 75, "Loyal Grinder": 30,
                "Promo Hunter": 20, "Casual": 10}
    total_net = 0
    for seg, conv in seg_map.items():
        n_seg = vc.get(seg, 0)
        avg_clv_seg = data.loc[data["segment"] == seg, "total_deposits"].mean() if seg in data["segment"].values else 0
        recovered = n_seg / (len(data) / 190_000_000) * conv * avg_clv_seg * 0.3
        cost = n_seg / (len(data) / 190_000_000) * cost_map[seg]
        net = (recovered - cost) / 1e7
        total_net += net
        st.markdown(f"- **{seg}**: ₹{net:.1f}cr net recovery ({conv:.0%} conversion)")
    st.info(f"Total recoverable: **₹{total_net:.1f} crore** annually at Dream11 scale (190M users).")

    st.subheader("Churn Risk Rankings")
    rankings = data.sort_values("churn_risk", ascending=False)
    st.dataframe(rankings[["user_id", "segment", "churn_risk", "total_deposits",
                            "days_since_last_login", "loss_streak_length"]].head(20),
                 width='stretch')

    st.subheader("Intervention Strategy")
    st.table(pd.DataFrame([
        ["Churn Risk", "Critical", "Cashback + push notification + captain suggestion", "35%"],
        ["High Roller", "High", "Exclusive mega-contest + referral bonus", "20%"],
        ["Loyal Grinder", "Medium", "Low-risk contest + streak rewards", "15%"],
        ["Promo Hunter", "Low", "Targeted promo contests", "10%"],
        ["Casual", "Low", "Re-engagement email + beginner contest", "8%"],
    ], columns=["Segment", "Priority", "Intervention", "Expected Conversion"]))
