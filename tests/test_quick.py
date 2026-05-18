"""Quick validation tests — skips heavy models requiring extra dependencies."""
import sys
sys.path.insert(0, ".")

def run(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        return False

tests = []

# ---- Data loader ----
def test_load_real_data():
    from src.data_loader import CricketDataLoader
    loader = CricketDataLoader()
    df = loader.load_ball_by_ball("ashwin")
    assert len(df) > 0
    os = loader.get_over_stats()
    assert os["match_id"].nunique() >= 1

def test_pressure_rules():
    from src.data_loader import CricketDataLoader, PRESSURE_RULES, compute_match_context, aggregate_over_stats
    loader = CricketDataLoader()
    bbb = loader.load_ball_by_ball("ashwin")
    os = aggregate_over_stats(bbb)
    ctx = compute_match_context(os)
    for _, row in ctx.iterrows():
        for rule_fn in PRESSURE_RULES.values():
            rule_fn(row)

def test_fantasy_data():
    from src.data_loader import CricketDataLoader
    loader = CricketDataLoader()
    try:
        users = loader.get_fantasy_users(n_users=500)
        assert len(users) == 500
    except FileNotFoundError as e:
        print(f"  SKIP  Fantasy Data: {e}")
        raise Exception("SKIP: Real fantasy user CSV not available")

# ---- Module 1 ----
def test_pressure_genome():
    from src.data_loader import CricketDataLoader
    from src.pressure_genome import PressureGenome, PressureGenomeConfig
    loader = CricketDataLoader()
    df = loader.get_pressure_features(n_batsmen=30)
    genome = PressureGenome(PressureGenomeConfig())
    genome.fit(df)
    assert genome._fitted

def test_pressure_ranking():
    from src.data_loader import CricketDataLoader
    from src.pressure_genome import PressureGenome, PressureGenomeConfig
    loader = CricketDataLoader()
    df = loader.get_pressure_features(n_batsmen=30)
    genome = PressureGenome(PressureGenomeConfig())
    genome.fit(df)
    rankings = genome.rank_for_situation(df, {"required_run_rate": 10, "wickets_left": 4, "is_chase": True}, top_n=3)
    assert len(rankings) == 3

# ---- Module 2 ----
def test_q_learning():
    from src.impact_player import SubstitutionQLearning, ImpactPlayerConfig, MatchState
    q = SubstitutionQLearning(ImpactPlayerConfig(q_episodes=200))
    q.train(episodes=200)
    state = MatchState(wickets_fallen=4, required_run_rate=11, overs_remaining=5)
    rec = q.recommend_action(state)
    assert "recommended_action" in rec

def test_ranker():
    from src.impact_player import CandidateRanker, MatchState
    ranker = CandidateRanker()
    state = MatchState(wickets_fallen=5, required_run_rate=10)
    rankings = ranker.rank_candidates(state, ["A", "B", "C"], top_n=2)
    assert len(rankings) == 2

# ---- Module 3 ----
def test_excitement():
    from src.data_loader import CricketDataLoader
    from src.broadcast_monetisation import ExcitementEngine
    loader = CricketDataLoader()
    os = loader.get_over_stats()
    engine = ExcitementEngine()
    excited = engine.compute_excitement_density(os)
    assert "excitement_density" in excited.columns

def test_revenue_mapping():
    from src.data_loader import CricketDataLoader
    from src.broadcast_monetisation import ExcitementEngine
    loader = CricketDataLoader()
    excited = ExcitementEngine().compute_excitement_density(loader.get_over_stats())
    mapped = ExcitementEngine().map_ad_revenue(excited)
    assert "ad_rate_per_over" in mapped.columns

# ---- Module 4 ----
def test_feature_engineering():
    from src.data_loader import CricketDataLoader
    from src.fantasy_clv import FantasyFeatureEngineer
    loader = CricketDataLoader()
    users = loader.get_fantasy_users(n_users=500)
    processed = FantasyFeatureEngineer().engineer_features(users)
    assert "recency_days" in processed.columns
    assert "churn_class" in processed.columns

def test_intervention():
    from src.data_loader import CricketDataLoader
    from src.fantasy_clv import InterventionEngine, FantasyFeatureEngineer, CLVModel
    loader = CricketDataLoader()
    users = FantasyFeatureEngineer().engineer_features(loader.get_fantasy_users(n_users=500))
    clv = CLVModel()
    clv.fit(users)
    clv_preds = clv.predict_clv(users)
    engine = InterventionEngine()
    segmented = engine.segment_users(clv_preds)
    assert "segment" in segmented.columns
    matrix = engine.intervention_matrix()
    assert len(matrix) == 4

# ---- Run ----
if __name__ == "__main__":
    print("=" * 50)
    print("  CricketIQ — Quick Validation Tests")
    print("=" * 50)
    all_tests = [
        ("Data: Real Ball-by-Ball", test_load_real_data),
        ("Data: Pressure Rules", test_pressure_rules),
        ("Data: Fantasy Users", test_fantasy_data),
        ("M1: Pressure Genome", test_pressure_genome),
        ("M1: Player Ranking", test_pressure_ranking),
        ("M2: Q-Learning", test_q_learning),
        ("M2: Candidate Ranker", test_ranker),
        ("M3: Excitement Density", test_excitement),
        ("M3: Revenue Mapping", test_revenue_mapping),
        ("M4: Feature Engineering", test_feature_engineering),
        ("M4: Intervention Engine", test_intervention),
    ]
    passed = sum(1 for n, fn in all_tests if run(n, fn))
    print(f"\n  Results: {passed}/{len(all_tests)} passed")
    print("=" * 50)
