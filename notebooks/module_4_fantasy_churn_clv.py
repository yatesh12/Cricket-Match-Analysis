# %% [markdown]
# # Module 4 — Fantasy Churn & Lifetime Value Engine
# 
# **What this does:** Survival-analysis-powered churn prediction + BG/NBD customer
# lifetime value modelling for fantasy cricket platforms. Outputs actionable
# intervention strategies per user segment with revenue impact in ₹.
# 
# **Business value:** Dream11-scale platform (190M users) can recover ₹X crore in
# annual deposits by proactive intervention on at-risk high-CLV users.
# 
# **Note:** This module requires a separate `fantasy_users.csv` file in `data/processed/`
# with platform user activity data. This is not included in the public cricket datasets.
# Export user data from your platform and save it as `data/processed/fantasy_users.csv`.

# %% [markdown]
# ## 1. Setup & Data Loading

# %%
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
from src.data_loader import CricketDataLoader
from src.fantasy_clv import (
    FantasyChurnCLV, ChurnCLVConfig,
    CoxSurvivalModel, XGBoostChurnModel, CLVModel,
    FantasyFeatureEngineer, InterventionEngine,
)

loader = CricketDataLoader()

try:
    users = loader.get_fantasy_users(n_users=50000)
    print(f"Loaded {len(users)} fantasy platform users from CSV")
    print(f"Churn rate: {users['churned'].mean():.1%}")
    print(f"Partial churn rate: {users['partial_churn'].mean():.1%}")
except FileNotFoundError:
    print("=" * 60)
    print("  Fantasy user data not available.")
    print("  This module requires a 'fantasy_users.csv' file in data/processed/")
    print("  with columns: user_id, city, age_group, contests_entered_per_week,")
    print("  avg_team_score_percentile, days_since_last_login, total_deposits,")
    print("  churned, favourite_player, etc.")
    print("=" * 60)
    raise  # re-raise to stop execution — this module needs real data

# %% [markdown]
# ## 2. Feature Engineering — RFM+ Features

# %%
engineer = FantasyFeatureEngineer()
processed = engineer.engineer_features(users)
features = [
    "recency_days", "frequency_contests_per_week", "monetary_deposits",
    "win_rate", "team_diversity_score", "loss_streak_length",
]
processed[["user_id"] + features + ["churn_class"]].head(10)

# %% [markdown]
# ## 3. Cox Proportional Hazards Model (Survival Analysis)
# 
# Most DS teams use binary classifiers for churn. Survival analysis handles
# **censored data** — users who haven't churned yet — correctly.

# %%
cox = CoxSurvivalModel()
cox.fit(processed)
if cox.is_fitted:
    hazard_ratios = cox.hazard_ratios()
    c_index = cox.concordance_index(processed)
    print("Cox PH — Hazard Ratios:")
    for feat, hr in hazard_ratios.items():
        print(f"  {feat}: {hr:.3f}x {'↑' if hr > 1 else '↓'} churn risk")
    print(f"\nConcordance Index (C-index): {c_index:.4f}")
else:
    print("Cox model not fitted (lifelines may not be installed)")

# %% [markdown]
# ## 4. Kaplan-Meier Curves (Survival Analysis)

# %%
try:
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt

    kmf = KaplanMeierFitter()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By age group
    for i, group in enumerate(["18-24", "25-34", "35-44"]):
        mask = processed["age_group"] == group
        kmf.fit(processed.loc[mask, "duration_days"],
                processed.loc[mask, "event_observed"],
                label=group)
        kmf.plot_survival_function(ax=axes[0])

    axes[0].set_title("Survival by Age Group")
    axes[0].set_xlabel("Days Since Last Login")
    axes[0].set_ylabel("Retention Probability")

    # By loss streak
    for streak in [0, 1, 3, 5]:
        mask = processed["loss_streak_length"] >= streak
        kmf.fit(processed.loc[mask, "duration_days"],
                processed.loc[mask, "event_observed"],
                label=f"Loss streak >= {streak}")
        kmf.plot_survival_function(ax=axes[1])

    axes[1].set_title("Survival by Loss Streak Length")
    axes[1].set_xlabel("Days Since Last Login")

    plt.tight_layout()
    plt.show()
except ImportError:
    print("lifelines or matplotlib not installed — skipping KM curves")

# %% [markdown]
# ## 5. XGBoost Classifier (Comparison)

# %%
xgb_model = XGBoostChurnModel()
xgb_model.fit(processed)
importance = xgb_model.feature_importance()
if importance is not None:
    print("XGBoost Feature Importance:")
    print(importance)

# %% [markdown]
# ## 6. SHAP Explainability

# %%
shap_data = xgb_model.shap_values(processed.head(1000))
if shap_data:
    vals = np.array(shap_data["values"])
    mean_abs = np.mean(np.abs(vals), axis=0)
    shap_df = pd.DataFrame({
        "feature": shap_data["features"],
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    print("Top SHAP features pushing users toward churn:")
    print(shap_df.head(6))

# %% [markdown]
# ## 7. Customer Lifetime Value (BG/NBD Model)

# %%
clv = CLVModel()
clv.fit(processed)
clv_predictions = clv.predict_clv(processed, time_periods=12)
print("CLV Distribution:")
print(clv_predictions["predicted_clv"].describe())

# %% [markdown]
# ## 8. User Segmentation & Intervention Strategy

# %%
engine = InterventionEngine()
segmented = engine.segment_users(clv_predictions)
print("Segment Distribution:")
print(segmented["segment"].value_counts())

interventions = engine.intervention_matrix()
print("\nIntervention Strategy Matrix:")
interventions[["segment", "priority", "intervention", "expected_conversion"]]

# %% [markdown]
# ## 9. Revenue Impact Simulation

# %%
impact = engine.simulate_revenue_impact(segmented, platform_user_count=190_000_000)
print("Revenue Impact per Segment:")
for segment, data in impact["per_segment"].items():
    print(f"  {segment}: ₹{data['recovered_annual_cr']} cr/yr recovered")
print(f"\n{impact['message']}")

# %% [markdown]
# ## 10. Churn Risk Rankings (CRM-Ready Output)

# %%
rankings = FantasyChurnCLV().get_risk_rankings(top_n=20)
if rankings is not None:
    print("Top 20 at-risk users for CRM intervention:")
    rankings[["user_id", "segment", "churn_risk", "predicted_clv",
              "days_since_last_login", "loss_streak_length", "favourite_player"]]

# %% [markdown]
# ## Summary
# 
# **Key outputs for Cognizant panel:**
# 1. Cox PH model handles censored data correctly (most competitors use binary classifiers)
# 2. Concordance Index (C-index) is the right metric — not just AUC
# 3. Kaplan-Meier curves show segment-level survival patterns
# 4. BG/NBD model computes expected future value per user
# 5. CLV × Churn segment matrix drives automated intervention campaigns
# 6. Headline: ₹X crore recoverable annually on Dream11-scale platform
# 7. Ready for CRM integration (Clevertap/MoEngage) via ranked daily pipeline
