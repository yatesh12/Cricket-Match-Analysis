# %% [markdown]
# # Module 2 — Impact Player AI: Strategic Substitution Intelligence
# 
# **What this does:** An RL-inspired decision engine that recommends the optimal
# substitution moment and candidate during a live IPL match under the Impact Player rule.
# 
# **Business value:** No public implementation of this exists for the IPL 2023+ rule.
# Teams using this gain a strategic edge in live match management.

# %% [markdown]
# ## 1. Setup

# %%
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
from src.impact_player import (
    ImpactPlayerAI, ImpactPlayerConfig, MatchState,
    MatchStateBuilder, SupervisedBaseline, SubstitutionQLearning,
    CandidateRanker, CounterfactualAnalyser,
)

# %% [markdown]
# ## 2. Match State Representation
# 
# Every over in a match is encoded as a 14-dimensional state vector:

# %%
state = MatchState(
    current_score=145, wickets_fallen=4, required_run_rate=12.0,
    overs_remaining=6, is_first_innings=0, is_pressure_situation=1,
)
print(f"State vector ({len(state.to_vector())} dimensions):")
for k, v in state.__dataclass_fields__.items():
    print(f"  {k}: {getattr(state, k)}")

# Discretised buckets for Q-learning
buckets = MatchStateBuilder.discretise_state(state)
print(f"\nDiscretised state: RRR={buckets[0]}, Over={buckets[1]}, Wickets={buckets[2]}")

# %% [markdown]
# ## 3. Supervised Baseline (XGBoost)

# %%
baseline = SupervisedBaseline()
baseline.fit()
eval_result = baseline.evaluate(n_test=2000)
print(f"XGBoost Baseline ROC-AUC: {eval_result['roc_auc']:.4f}")
print("(This is the benchmark we beat with the RL approach)")

# %% [markdown]
# ## 4. Q-Learning for Substitution Timing

# %%
q_agent = SubstitutionQLearning(ImpactPlayerConfig(q_episodes=2000))
reward_history = q_agent.train(episodes=2000)
print(f"Q-table size: {len(q_agent.q_table)} state-action pairs")

# Recommendation for a match state
state = MatchState(wickets_fallen=5, required_run_rate=11.0, overs_remaining=4,
                   is_first_innings=0, is_pressure_situation=1)
timing = q_agent.recommend_action(state)
print(f"\nRecommended action: {timing['recommended_action'].replace('_', ' ').title()}")
print(f"Confidence: {timing['confidence']*100:.1f}%")
print(f"Q-values: {timing['q_values']}")

# %% [markdown]
# ## 5. Candidate Ranking (LightGBM-style)

# %%
ranker = CandidateRanker()
available = ["Shivam Dube", "Deepak Chahar", "Moeen Ali", "Maheesh Theekshana",
             "Tim David", "Piyush Chawla"]
rankings = ranker.rank_candidates(state, available, top_n=3)
print("Top 3 candidates to bring in:")
for _, r in rankings.iterrows():
    print(f"  {r['rank']}. {r['player']} ({r['primary_role']}) — "
          f"+{r['expected_uplift_runs']} runs expected (compat: {r['compatibility']:.2f})")

# %% [markdown]
# ## 6. Full Recommendation Pipeline

# %%
ai = ImpactPlayerAI()
ai.run_pipeline()
rec = ai.recommend_substitution(state, available)
print(f"Substitute now? {'YES' if rec['substitute_now'] else 'NO'}")
print(f"Confidence: {rec['confidence']*100:.1f}%")
print(f"Our pick: {rec['candidates'][0]['player']} → +{rec['candidates'][0]['expected_uplift_runs']} runs")

# %% [markdown]
# ## 7. Counterfactual Analysis — Famous IPL Matches

# %%
analyser = CounterfactualAnalyser()
results = analyser.analyse_all()
for _, r in results.iterrows():
    print(f"\n{r['match']}")
    print(f"  Actual: {r['actual']}")
    print(f"  Model: {r['model_recommends']} (+{r['model_expected_uplift']} runs expected)")
    print(f"  Actual runs in next 5 overs: {r['actual_runs']}")
    print(f"  Runs difference: {r['runs_difference']:+.1f}")
    print(f"  Verdict: {r['verdict']}")

# %% [markdown]
# ## Summary
# 
# **Key outputs for Cognizant panel:**
# 1. Match state encoded as 14-dim vector + discretised into 27 RL states
# 2. XGBoost baseline establishes benchmark ROC-AUC
# 3. Q-learning converges to optimal substitution timing policy
# 4. LightGBM-style ranker outputs top-3 candidates with expected runs uplift
# 5. Counterfactual analysis of 5 real IPL matches shows model vs reality
# 6. Ready-to-deploy FastAPI endpoint: POST /recommend-substitution
