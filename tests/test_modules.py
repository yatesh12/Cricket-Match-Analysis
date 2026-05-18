"""CricketIQ — Integration tests for all 4 modules.

Run: python -m pytest tests/test_modules.py -v
or:  python tests/test_modules.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.data_loader import CricketDataLoader, PRESSURE_RULES
from src.pressure_genome import PressureGenome, PressureGenomeConfig
from src.impact_player import (
    ImpactPlayerAI, ImpactPlayerConfig, MatchState,
    SupervisedBaseline, SubstitutionQLearning, CandidateRanker,
)
from src.broadcast_monetisation import BroadcastMonetisation, BroadcastConfig, ExcitementEngine
from src.fantasy_clv import FantasyChurnCLV, ChurnCLVConfig, CoxSurvivalModel, XGBoostChurnModel


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_loader():
    return CricketDataLoader()


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------

def test_real_ball_by_ball():
    loader = _get_loader()
    df = loader.load_ball_by_ball("ashwin")
    assert len(df) > 0
    assert "match_id" in df.columns
    assert "batter" in df.columns
    assert "bowler" in df.columns
    assert "runs" in df.columns
    print(f"  [PASS] Real data: {len(df)} balls across {df['match_id'].nunique()} matches")


def test_pressure_rules():
    loader = _get_loader()
    df = loader.load_ball_by_ball("ashwin")
    over_stats = loader.get_over_stats()
    from src.data_loader import compute_match_context
    ctx = compute_match_context(over_stats)

    for _, row in ctx.iterrows():
        for rule_name, rule_fn in PRESSURE_RULES.items():
            _ = rule_fn(row)  # should not raise
    print(f"  [PASS] All {len(PRESSURE_RULES)} pressure rules execute without error")


def test_over_aggregation():
    loader = _get_loader()
    over_stats = loader.get_over_stats(force_rebuild=True)
    assert "runs_scored" in over_stats.columns
    assert "wickets" in over_stats.columns
    assert "boundaries" in over_stats.columns
    if len(over_stats) > 0:
        assert over_stats["over"].max() <= 20
    print(f"  [PASS] Over aggregation: {len(over_stats)} overs, {over_stats['match_id'].nunique()} matches")


def test_fantasy_data_generation():
    loader = _get_loader()
    try:
        users = loader.get_fantasy_users(n_users=1000)
        assert len(users) == 1000
        assert "user_id" in users.columns
        assert "churned" in users.columns
        assert "total_deposits" in users.columns
        churn_rate = users["churned"].mean()
        assert 0.05 < churn_rate < 0.60, f"Churn rate {churn_rate} unrealistic"
        print(f"  [PASS] Fantasy data: {len(users)} users, churn rate {churn_rate:.1%}")
    except FileNotFoundError:
        pytest.skip("Fantasy user CSV not available — this module requires separate user data")


# ---------------------------------------------------------------------------
# Module 1 — Pressure Genome tests
# ---------------------------------------------------------------------------

def test_pressure_genome_fit():
    loader = _get_loader()
    df = loader.get_pressure_features(n_batsmen=100)
    genome = PressureGenome(PressureGenomeConfig(n_components=3))
    genome.fit(df)
    assert genome._fitted
    assert genome.kmeans is not None
    assert len(np.unique(genome._cluster_labels)) >= 2
    print(f"  [PASS] Pressure Genome: {len(df)} batsmen -> {len(np.unique(genome._cluster_labels))} clusters")


def test_pressure_genome_archetypes():
    loader = _get_loader()
    df = loader.get_pressure_features(n_batsmen=100)
    genome = PressureGenome(PressureGenomeConfig())
    genome.fit(df)
    archetypes = genome.name_archetypes(df)
    assert len(archetypes) >= 2
    names = set(archetypes.values())
    assert len(names) >= 2
    print(f"  [PASS] Archetypes: {names}")


def test_pressure_ranking():
    loader = _get_loader()
    df = loader.get_pressure_features(n_batsmen=100)
    genome = PressureGenome(PressureGenomeConfig())
    genome.fit(df)
    match_state = {"required_run_rate": 10, "wickets_left": 4, "is_chase": True, "overs_remaining": 4}
    rankings = genome.rank_for_situation(df, match_state, top_n=3)
    assert len(rankings) == 3
    assert rankings["compatibility"].iloc[0] >= rankings["compatibility"].iloc[-1]
    print(f"  [PASS] Ranking: top player = {rankings.iloc[0]['player']} ({rankings.iloc[0]['compatibility']:.2%})")


def test_pressure_mismatch_alert():
    loader = _get_loader()
    df = loader.get_pressure_features(n_batsmen=100)
    genome = PressureGenome(PressureGenomeConfig())
    genome.fit(df)
    match_state = {"required_run_rate": 12, "wickets_left": 3, "is_chase": True, "overs_remaining": 3}
    lineup = df["player"].sample(6).tolist()
    alert = genome.pressure_mismatch_alert(lineup, df, match_state)
    assert "mismatch_detected" in alert
    assert "avg_compatibility" in alert
    print(f"  [PASS] Mismatch alert: detected={alert['mismatch_detected']}, avg={alert['avg_compatibility']:.2%}")


# ---------------------------------------------------------------------------
# Module 2 — Impact Player AI tests
# ---------------------------------------------------------------------------

def test_supervised_baseline():
    baseline = SupervisedBaseline()
    baseline.fit()
    assert baseline._fitted
    eval_result = baseline.evaluate(n_test=500)
    assert eval_result["roc_auc"] > 0.5
    print(f"  [PASS] Baseline: ROC-AUC = {eval_result['roc_auc']:.4f}")


def test_q_learning():
    q_agent = SubstitutionQLearning(ImpactPlayerConfig(q_episodes=500))
    rewards = q_agent.train(episodes=500)
    assert len(q_agent.q_table) > 0
    state = MatchState(wickets_fallen=4, required_run_rate=11, overs_remaining=5,
                       is_first_innings=0, is_pressure_situation=1)
    rec = q_agent.recommend_action(state)
    assert "recommended_action" in rec
    assert "confidence" in rec
    print(f"  [PASS] Q-Learning: {len(q_agent.q_table)} states, action={rec['recommended_action']}")


def test_candidate_ranker():
    ranker = CandidateRanker()
    state = MatchState(wickets_fallen=5, required_run_rate=10, overs_remaining=4)
    players = ["Player A", "Player B", "Player C", "Player D"]
    rankings = ranker.rank_candidates(state, players, top_n=3)
    assert len(rankings) == 3
    assert "expected_uplift_runs" in rankings.columns
    assert rankings["expected_uplift_runs"].iloc[0] >= rankings["expected_uplift_runs"].iloc[-1]
    print(f"  [PASS] Ranker: top candidate = {rankings.iloc[0]['player']} (+{rankings.iloc[0]['expected_uplift_runs']} runs)")


def test_counterfactual():
    from src.impact_player import CounterfactualAnalyser
    analyser = CounterfactualAnalyser()
    results = analyser.analyse_all()
    assert len(results) == 5
    assert "match" in results.columns
    assert "runs_difference" in results.columns
    print(f"  [PASS] Counterfactual: {len(results)} matches analysed")


def test_full_impact_player_pipeline():
    ai = ImpactPlayerAI(ImpactPlayerConfig(q_episodes=500))
    result = ai.run_pipeline()
    assert "baseline_roc_auc" in result
    assert "q_learning_converged" in result
    state = MatchState(wickets_fallen=4, required_run_rate=11, overs_remaining=5)
    rec = ai.recommend_substitution(state, ["Player A", "Player B", "Player C"])
    assert "substitute_now" in rec
    assert "candidates" in rec
    print(f"  [PASS] Full pipeline: baseline={result['baseline_roc_auc']:.4f}, {result['matches_where_model_wins']}/{result['matches_analysed']} counterfactual wins")


# ---------------------------------------------------------------------------
# Module 3 — Broadcast Monetisation tests
# ---------------------------------------------------------------------------

def test_excitement_engineering():
    loader = _get_loader()
    over_stats = loader.get_over_stats()
    engine = ExcitementEngine()
    excited = engine.compute_excitement_density(over_stats)
    assert "excitement_density" in excited.columns
    assert "excitement_normalised" in excited.columns
    assert excited["excitement_density"].min() >= 0
    print(f"  [PASS] Excitement: mean={excited['excitement_density'].mean():.2f}, max={excited['excitement_density'].max():.2f}")


def test_revenue_mapping():
    loader = _get_loader()
    over_stats = loader.get_over_stats()
    engine = ExcitementEngine()
    excited = engine.compute_excitement_density(over_stats)
    mapped = engine.map_ad_revenue(excited)
    assert "ad_rate_per_30s" in mapped.columns
    assert "ad_rate_per_over" in mapped.columns
    assert "is_peak_window" in mapped.columns
    assert mapped["ad_rate_per_over"].min() >= 3_00_000 * 4
    print(f"  [PASS] Revenue mapping: peak windows = {mapped['is_peak_window'].mean():.1%}")


def test_broadcast_pipeline():
    loader = _get_loader()
    over_stats = loader.get_over_stats()
    bc = BroadcastMonetisation(BroadcastConfig(epochs=5))
    result = bc.run_pipeline(over_stats)
    assert result["n_matches"] > 0
    assert "revenue_impact" in result
    assert "trends_validation" in result
    print(f"  [PASS] Broadcast pipeline: {result['n_matches']} matches, precision@1={result['precision_at_1']:.4f}")


def test_hot_zone_report():
    loader = _get_loader()
    over_stats = loader.get_over_stats()
    bc = BroadcastMonetisation(BroadcastConfig(epochs=5))
    bc.run_pipeline(over_stats)
    import pandas as pd
    match_ids = bc._processed_data["match_id"].unique() if bc._processed_data is not None else []
    if len(match_ids) > 0:
        report = bc.generate_match_report(match_ids[0])
        assert "error" not in report
        assert "estimated_ad_revenue_cr" in report
        assert "top_5_hot_zones" in report
        print(f"  [PASS] Hot zone report: rev=Rs.{report['estimated_ad_revenue_cr']}cr, zones={len(report['top_5_hot_zones'])}")


# ---------------------------------------------------------------------------
# Module 4 — Fantasy CLV tests
# ---------------------------------------------------------------------------

def _require_fantasy_users(loader, n_users=1000):
    """Load fantasy users or raise SKIP Exception."""
    try:
        return loader.get_fantasy_users(n_users=n_users)
    except FileNotFoundError as e:
        pytest.skip(str(e))


def test_feature_engineering():
    loader = _get_loader()
    users = _require_fantasy_users(loader, n_users=1000)
    from src.fantasy_clv import FantasyFeatureEngineer
    engineer = FantasyFeatureEngineer()
    processed = engineer.engineer_features(users)
    assert "recency_days" in processed.columns
    assert "churn_class" in processed.columns
    assert "duration_days" in processed.columns
    assert "event_observed" in processed.columns
    print(f"  [PASS] Feature engineering: {len(processed)} users, {len(processed.columns)} columns")


def test_cox_model():
    loader = _get_loader()
    users = _require_fantasy_users(loader, n_users=2000)
    from src.fantasy_clv import FantasyFeatureEngineer, CoxSurvivalModel
    processed = FantasyFeatureEngineer().engineer_features(users)
    cox = CoxSurvivalModel()
    cox.fit(processed)
    if cox.is_fitted:
        hazard_ratios = cox.hazard_ratios()
        assert hazard_ratios is not None
        assert len(hazard_ratios) > 0
        c_index = cox.concordance_index(processed)
        print(f"  [PASS] Cox PH: {len(hazard_ratios)} features, C-index={c_index:.4f}")
    else:
        print(f"  [SKIP] Cox PH: lifelines not available")


def test_xgb_model():
    loader = _get_loader()
    users = _require_fantasy_users(loader, n_users=2000)
    from src.fantasy_clv import FantasyFeatureEngineer, XGBoostChurnModel
    processed = FantasyFeatureEngineer().engineer_features(users)
    xgb = XGBoostChurnModel()
    xgb.fit(processed)
    if xgb._fitted:
        preds = xgb.predict(processed.head(100))
        assert len(preds) == 100
        print(f"  [PASS] XGBoost: feature importance available = {xgb.feature_importance() is not None}")
    else:
        print(f"  [SKIP] XGBoost: not installed")


def test_clv_model():
    loader = _get_loader()
    users = _require_fantasy_users(loader, n_users=2000)
    from src.fantasy_clv import FantasyFeatureEngineer, CLVModel
    processed = FantasyFeatureEngineer().engineer_features(users)
    clv = CLVModel()
    clv.fit(processed)
    clv_predictions = clv.predict_clv(processed, time_periods=12)
    assert "predicted_clv" in clv_predictions.columns
    assert "predicted_purchases" in clv_predictions.columns
    print(f"  [PASS] CLV: mean predicted CLV = Rs.{clv_predictions['predicted_clv'].mean():.0f}")


def test_intervention_engine():
    loader = _get_loader()
    users = _require_fantasy_users(loader, n_users=2000)
    from src.fantasy_clv import FantasyFeatureEngineer, CLVModel, InterventionEngine
    processed = FantasyFeatureEngineer().engineer_features(users)
    clv = CLVModel()
    clv.fit(processed)
    clv_preds = clv.predict_clv(processed)
    engine = InterventionEngine()
    segmented = engine.segment_users(clv_preds)
    assert "segment" in segmented.columns
    assert "churn_risk" in segmented.columns
    matrix = engine.intervention_matrix()
    assert len(matrix) == 4
    impact = engine.simulate_revenue_impact(segmented, platform_user_count=190_000_000)
    assert "per_segment" in impact
    assert "total_recovered_cr_sample" in impact
    print(f"  [PASS] Intervention: {segmented['segment'].nunique()} segments, recovery=Rs.{impact['total_recovered_cr_sample']}cr sample")


def test_full_fantasy_pipeline():
    loader = _get_loader()
    users = _require_fantasy_users(loader, n_users=5000)
    fc = FantasyChurnCLV(ChurnCLVConfig())
    result = fc.run_pipeline(users)
    assert result["n_users"] == 5000
    assert "churn_rate" in result
    assert "revenue_impact" in result
    assert "intervention_matrix" in result
    rankings = fc.get_risk_rankings(top_n=10)
    assert rankings is not None
    assert len(rankings) <= 10
    print(f"  [PASS] Full fantasy pipeline: churn={result['churn_rate']:.1%}, segments={len(result['segment_distribution'])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Real Data", test_real_ball_by_ball),
        ("Pressure Rules", test_pressure_rules),
        ("Over Aggregation", test_over_aggregation),
        ("Fantasy Data", test_fantasy_data_generation),
        ("", None),
        ("MODULE 1 — Pressure Genome", None),
        ("Fit & Cluster", test_pressure_genome_fit),
        ("Archetype Naming", test_pressure_genome_archetypes),
        ("Player Ranking", test_pressure_ranking),
        ("Mismatch Alert", test_pressure_mismatch_alert),
        ("", None),
        ("MODULE 2 — Impact Player AI", None),
        ("Supervised Baseline", test_supervised_baseline),
        ("Q-Learning", test_q_learning),
        ("Candidate Ranker", test_candidate_ranker),
        ("Counterfactual Analysis", test_counterfactual),
        ("Full Pipeline", test_full_impact_player_pipeline),
        ("", None),
        ("MODULE 3 — Broadcast Monetisation", None),
        ("Excitement Engineering", test_excitement_engineering),
        ("Revenue Mapping", test_revenue_mapping),
        ("Broadcast Pipeline", test_broadcast_pipeline),
        ("Hot Zone Report", test_hot_zone_report),
        ("", None),
        ("MODULE 4 — Fantasy CLV", None),
        ("Feature Engineering", test_feature_engineering),
        ("Cox Model", test_cox_model),
        ("XGBoost Model", test_xgb_model),
        ("CLV Model", test_clv_model),
        ("Intervention Engine", test_intervention_engine),
        ("Full Pipeline", test_full_fantasy_pipeline),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, fn in tests:
        if fn is None:
            if name:
                print(f"\n{'='*50}")
                print(f"  {name}")
                print(f"{'='*50}")
            continue
        try:
            fn()
            passed += 1
        except Exception as e:
            if "SKIP" in str(e):
                print(f"  [SKIP] {name}: {e}")
                skipped += 1
            else:
                print(f"  [FAIL] {name}: {e}")
                failed += 1

    total = passed + failed + skipped
    print(f"\n{'='*50}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*50}")
    sys.exit(1 if failed > 0 else 0)

# Use pytest.mark annotations for pytest compatibility
import pytest
test_real_ball_by_ball = pytest.mark.data(test_real_ball_by_ball)
test_pressure_genome_fit = pytest.mark.pressure(test_pressure_genome_fit)
test_broadcast_pipeline = pytest.mark.broadcast(test_broadcast_pipeline)
test_full_fantasy_pipeline = pytest.mark.fantasy(test_full_fantasy_pipeline)
