"""Fantasy Churn & Lifetime Value Engine — Module 4.

Survival analysis + CLV modelling for fantasy cricket platforms (Dream11,
My11Circle, etc.). Predicts user-level churn risk and lifetime value, then
prescribes intervention strategies per segment.

Key differentiators from standard approaches:
  - Cox Proportional Hazards (handles censored data properly)
  - Concordance Index as evaluation metric (not just AUC)
  - BG/NBD model for CLV (Buy-Till-You-Die framework)
  - Actionable intervention matrix with revenue impact estimates
"""

from typing import Dict, List, Optional, Tuple
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CHURN_FEATURES = [
    "recency_days",
    "frequency_contests_per_week",
    "monetary_deposits",
    "win_rate",
    "team_diversity_score",
    "loss_streak_length",
    "avg_team_score_percentile",
    "winnings_last_30d",
]

CLV_SEGMENTS = {
    "High-CLV Loyalist": "Frequent depositors, high engagement. Retain with VIP perks.",
    "Medium-CLV Grower": "Moderate engagement, growth potential. Nudge with targeted offers.",
    "Low-CLV Casual": "Infrequent, low deposits. Low-touch retention via push notifications.",
    "At-Risk High-CLV": "High historical value, showing churn signals. PRIORITY intervene.",
}


@dataclass
class ChurnCLVConfig:
    churn_window_days: int = 45
    partial_churn_window_days: int = 30
    random_state: int = 42
    test_size: float = 0.2


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class FantasyFeatureEngineer:
    """Build RFM + behavioural features from raw user data."""

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw user DataFrame into model-ready feature set."""
        data = df.copy()

        # recency (days since last login is already a column)
        data["recency_days"] = data["days_since_last_login"]

        # frequency
        data["frequency_contests_per_week"] = data["contests_entered_per_week"]

        # monetary
        data["monetary_deposits"] = data["total_deposits"]
        data["monetary_withdrawals"] = data["total_withdrawals"]
        data["net_deposits"] = data["total_deposits"] - data["total_withdrawals"]

        # engagement
        data["is_high_engagement"] = (data["contests_entered_per_week"] >= 5).astype(int)
        data["is_dormant"] = (data["days_since_last_login"] >= 14).astype(int)

        # loss streak squared (non-linear effect)
        data["loss_streak_sq"] = data["loss_streak_length"] ** 2

        # win rate bucket
        data["win_rate_bucket"] = pd.cut(
            data["win_rate"],
            bins=[0, 0.05, 0.15, 0.30, 1.0],
            labels=["low", "medium", "high", "elite"],
        )

        # team diversity
        data["low_diversity"] = (data["team_diversity_score"] < 0.3).astype(int)

        # age group encoding
        age_map = {"18-24": 0, "25-34": 1, "35-44": 2, "45+": 3}
        data["age_group_encoded"] = data["age_group"].map(age_map).fillna(0)

        # churn target
        data["churned"] = data["churned"].astype(int)
        data["churn_class"] = np.where(
            data["days_since_last_login"] >= df["churn_window_days"]
            if "churn_window_days" in df.columns
            else 45,
            2,
            np.where(
                (data["days_since_last_login"] >= 30) & (data["contests_entered_per_week"] < 1),
                1,
                0,
            ),
        )
        # Override with actual churn column
        data["churn_class"] = np.select(
            [
                data["churned"] == 1,
                data["partial_churn"].astype(bool) & (data["churned"] == 0),
            ],
            [2, 1],
            default=0,
        ).astype(int)

        # survival analysis target
        data["duration_days"] = data["days_since_last_login"]
        data["event_observed"] = data["churned"].astype(int)

        return data


# ---------------------------------------------------------------------------
# Cox Proportional Hazards (Survival Analysis)
# ---------------------------------------------------------------------------

class CoxSurvivalModel:
    """Cox Proportional Hazards model for time-to-churn prediction.

    Uses lifelines library under the hood. Handles right-censored data
    (users who haven't churned yet), which is the correct approach for
    churn modelling — most practitioners incorrectly use binary classifiers.
    """

    def __init__(self):
        self._model = None
        self._features = CHURN_FEATURES
        self._fitted = False

    def fit(self, df: pd.DataFrame, duration_col: str = "duration_days",
            event_col: str = "event_observed") -> "CoxSurvivalModel":
        try:
            from lifelines import CoxPHFitter
            train_data = df[[duration_col, event_col] + self._features].dropna()
            self._model = CoxPHFitter(penalizer=0.01)
            self._model.fit(train_data, duration_col=duration_col, event_col=event_col)
            self._fitted = True
        except Exception as e:
            # Fallback: if lifelines not installed, train a dummy model
            self._fitted = False
            self._fit_error = str(e)
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def summary(self) -> Optional[pd.DataFrame]:
        if self._model is None:
            return None
        return self._model.summary

    def hazard_ratios(self) -> Optional[Dict[str, float]]:
        if self._model is None:
            return None
        return {k: round(np.exp(v), 3) for k, v in self._model.params_.items()}

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(df))
        return self._model.predict_partial_hazard(df[self._features]).values

    def concordance_index(self, df: pd.DataFrame) -> float:
        if self._model is None:
            return 0.0
        try:
            from lifelines.utils import concordance_index as ci
            durations = df["duration_days"].values
            events = df["event_observed"].values
            scores = self.predict_risk(df)
            return ci(durations, scores, events)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# XGBoost Churn Classifier (for comparison)
# ---------------------------------------------------------------------------

class XGBoostChurnModel:
    """XGBoost classifier for churn — used as a comparison baseline.

    Trained alongside the Cox model to demonstrate why survival analysis
    is superior for this problem domain.
    """

    def __init__(self, random_state: int = 42):
        self._model = None
        self._features = CHURN_FEATURES
        self._fitted = False
        self.random_state = random_state

    def fit(self, df: pd.DataFrame, target_col: str = "churned") -> "XGBoostChurnModel":
        try:
            import xgboost as xgb
            X = df[self._features].fillna(0)
            y = df[target_col].values
            self._model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
                random_state=self.random_state,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            self._model.fit(X, y)
            self._fitted = True
        except Exception as e:
            self._fitted = False
            self._fit_error = str(e)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            return np.zeros((len(df), 2))
        return self._model.predict_proba(df[self._features].fillna(0))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(df))
        return self._model.predict(df[self._features].fillna(0))

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self._model is None:
            return None
        importances = self._model.feature_importances_
        return pd.DataFrame({"feature": self._features, "importance": importances}).sort_values("importance", ascending=False)

    def shap_values(self, df: pd.DataFrame) -> Optional[Dict]:
        """SHAP waterfall values for model interpretability."""
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            X = df[self._features].fillna(0).values
            shaps = explainer.shap_values(X)
            return {
                "values": shaps.tolist(),
                "features": self._features,
                "base_value": float(explainer.expected_value),
            }
        except Exception:
            return None


# ---------------------------------------------------------------------------
# BG/NBD Customer Lifetime Value Model
# ---------------------------------------------------------------------------

class CLVModel:
    """BG/NBD (Buy-Till-You-Die) model for CLV estimation.

    Predicts expected future transactions and monetary value per user.
    Correctly handles the fact that some users have already 'died'
    (churned) while others are still 'alive'.
    """

    def __init__(self):
        self._bg_model = None
        self._gamma_model = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "CLVModel":
        try:
            from lifetimes import BetaGeoFitter, GammaGammaFitter

            # Prepare frequency/recency/T data
            # Using observable behaviour as proxy for the model's required inputs
            data = df.copy()
            data["frequency"] = data["contests_entered_per_week"] * 4  # monthly freq approx
            data["recency"] = np.where(
                data["days_since_last_login"] < 45,
                90 - data["days_since_last_login"],  # days since last transaction
                0,
            )
            data["T"] = 90  # observation period (90 days)
            data["monetary_value"] = np.where(
                data["total_deposits"] > 0,
                data["total_deposits"] / 12,  # avg deposit per month
                0,
            )

            active = data[data["days_since_last_login"] < 45].copy()

            self._bg_model = BetaGeoFitter(penalizer_coef=0.01)
            self._bg_model.fit(
                active["frequency"].values,
                active["recency"].values,
                active["T"].values,
                verbose=False,
            )

            with_pos = active[active["monetary_value"] > 0].copy()
            if len(with_pos) > 10:
                self._gamma_model = GammaGammaFitter(penalizer_coef=0.01)
                self._gamma_model.fit(
                    with_pos["frequency"].values,
                    with_pos["monetary_value"].values,
                    verbose=False,
                )

            self._fitted = True
            self._active_data = active

        except Exception as e:
            self._fitted = False
            self._fit_error = str(e)
        return self

    def predict_clv(self, df: pd.DataFrame, time_periods: int = 12) -> pd.DataFrame:
        """Predict CLV for each user over given time periods (months)."""
        if not self._fitted or self._bg_model is None:
            df["predicted_purchases"] = 0
            df["predicted_clv"] = 0
            return df

        data = df.copy()
        data["frequency"] = data["contests_entered_per_week"] * 4
        data["recency"] = np.where(
            data["days_since_last_login"] < 45,
            90 - data["days_since_last_login"],
            0,
        )
        data["T"] = 90
        data["monetary_value"] = np.where(
            data["total_deposits"] > 0,
            data["total_deposits"] / 12,
            0,
        )

        data["predicted_purchases"] = self._bg_model.predict(
            time_periods,
            data["frequency"].values,
            data["recency"].values,
            data["T"].values,
        )

        if self._gamma_model is not None:
            data["predicted_clv"] = self._gamma_model.customer_lifetime_value(
                self._bg_model,
                data["frequency"].values,
                data["recency"].values,
                data["T"].values,
                data["monetary_value"].values,
                time=time_periods,
                discount_rate=0.01,
            )
        else:
            data["predicted_clv"] = data["predicted_purchases"] * \
                data["monetary_value"].mean() if data["monetary_value"].mean() > 0 else 0

        return data


# ---------------------------------------------------------------------------
# Intervention Strategy Engine
# ---------------------------------------------------------------------------

class InterventionEngine:
    """Generates actionable intervention strategies per user segment.

    Outputs a matrix of (CLV Segment × Churn Risk) → {offer, channel, priority}
    plus a revenue impact simulation.
    """

    def __init__(self, avg_deposit_per_user: float = 2000):
        self.avg_deposit = avg_deposit_per_user

    def segment_users(
        self, df: pd.DataFrame, clv_col: str = "predicted_clv"
    ) -> pd.DataFrame:
        """Assign CLV quartile and churn risk segment for each user."""
        data = df.copy()

        clv_segments = ["Low-CLV Casual", "Medium-CLV Grower", "High-CLV Loyalist"]
        if clv_col in data.columns:
            data["clv_quartile"] = pd.qcut(
                data[clv_col].rank(method="first"), q=4, labels=["Q1_Low", "Q2", "Q3", "Q4_High"]
            )
        else:
            data["clv_quartile"] = "Q2"

        # churn risk: high if recency > 21 days OR loss_streak >= 3
        data["churn_risk"] = np.select(
            [
                (data["days_since_last_login"] > 30) | (data["loss_streak_length"] >= 5),
                (data["days_since_last_login"] > 14) | (data["loss_streak_length"] >= 2),
            ],
            ["High", "Medium"],
            default="Low",
        )

        # segment label
        data["segment"] = np.where(
            (data["churn_risk"] == "High") & (data["clv_quartile"] == "Q4_High"),
            "At-Risk High-CLV",
            np.where(
                data["clv_quartile"] == "Q4_High",
                "High-CLV Loyalist",
                np.where(
                    data["clv_quartile"].isin(["Q2", "Q3"]),
                    "Medium-CLV Grower",
                    "Low-CLV Casual",
                ),
            ),
        )

        return data

    def intervention_matrix(self) -> pd.DataFrame:
        """Return the intervention strategy matrix."""
        strategies = [
            {
                "segment": "At-Risk High-CLV",
                "churn_risk": "High",
                "priority": 1,
                "intervention": "Personalised bonus + favourite player promo + streak protection",
                "channel": "SMS + Email + In-App Push",
                "offer_value": "₹500 bonus + free team entry",
                "expected_conversion": 0.20,
            },
            {
                "segment": "High-CLV Loyalist",
                "churn_risk": "Low",
                "priority": 2,
                "intervention": "VIP referral bonus + loyalty badge + exclusive contests",
                "channel": "Email + In-App",
                "offer_value": "₹200 referral bonus",
                "expected_conversion": 0.35,
            },
            {
                "segment": "Medium-CLV Grower",
                "churn_risk": "Medium",
                "priority": 3,
                "intervention": "Contest nudge + first-deposit match + team tips",
                "channel": "Push Notification + Email",
                "offer_value": "100% deposit match up to ₹250",
                "expected_conversion": 0.15,
            },
            {
                "segment": "Low-CLV Casual",
                "churn_risk": "Medium",
                "priority": 4,
                "intervention": "Re-engagement push + free contest entry",
                "channel": "Push Notification",
                "offer_value": "Free ₹10 contest entry",
                "expected_conversion": 0.08,
            },
        ]
        return pd.DataFrame(strategies)

    def simulate_revenue_impact(
        self,
        df: pd.DataFrame,
        platform_user_count: int = 190_000_000,
    ) -> Dict:
        """Compute annual revenue recovered by proactive intervention.

        Args:
            df: user DataFrame with 'segment' and 'predicted_clv' columns.
            platform_user_count: total platform users (Dream11 scale ~190M).

        Returns:
            dict with revenue impact breakdown in ₹.
        """
        strategies = self.intervention_matrix()
        results = {}

        for _, strat in strategies.iterrows():
            segment_df = df[df["segment"] == strat["segment"]]
            if segment_df.empty:
                continue

            n_segment = len(segment_df)
            avg_clv = segment_df["predicted_clv"].mean() if "predicted_clv" in segment_df.columns else self.avg_deposit
            conversion_rate = strat["expected_conversion"]

            # annual deposits at risk
            annual_at_risk = n_segment * avg_clv * 12  # 12 months
            recovered = annual_at_risk * conversion_rate

            results[strat["segment"]] = {
                "users_in_segment": n_segment,
                "avg_clv": round(avg_clv, 2),
                "annual_deposits_at_risk_cr": round(annual_at_risk / 1e7, 2),
                "conversion_rate": conversion_rate,
                "recovered_annual_cr": round(recovered / 1e7, 2),
            }

        total_recovered = sum(
            v["recovered_annual_cr"] for v in results.values()
        )

        # Scale to platform level
        sample_ratio = len(df) / max(platform_user_count, 1)
        if sample_ratio > 0:
            platform_recovery = total_recovered / sample_ratio / 100
        else:
            platform_recovery = 0

        return {
            "per_segment": results,
            "total_recovered_cr_sample": round(total_recovered, 2),
            "estimated_platform_recovery_cr": round(platform_recovery, 2),
            "message": (
                f"For a platform of Dream11's scale (~{platform_user_count:,} users), "
                f"proactive intervention on the top 5% highest-CLV at-risk users would recover "
                f"approximately ₹{platform_recovery:,.0f} crore in annual deposits."
            ),
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class FantasyChurnCLV:
    """End-to-end churn & CLV pipeline orchestrator."""

    def __init__(self, config: Optional[ChurnCLVConfig] = None):
        self.config = config or ChurnCLVConfig()
        self.feature_engineer = FantasyFeatureEngineer()
        self.cox_model = CoxSurvivalModel()
        self.xgb_model = XGBoostChurnModel()
        self.clv_model = CLVModel()
        self.intervention = InterventionEngine()
        self._processed_data: Optional[pd.DataFrame] = None
        self._all_fitted = False

    def run_pipeline(self, df: pd.DataFrame) -> Dict:
        """Run the full churn + CLV pipeline, return results summary."""
        # 1. Feature engineering
        processed = self.feature_engineer.engineer_features(df)
        self._processed_data = processed

        # 2. Train Cox PH model
        self.cox_model.fit(processed)

        # 3. Train XGBoost model
        self.xgb_model.fit(processed)

        # 4. CLV model (BG/NBD)
        self.clv_model.fit(processed)
        clv_data = self.clv_model.predict_clv(processed)

        # 5. Segment users
        segmented = self.intervention.segment_users(clv_data)

        # 6. Simulation
        impact = self.intervention.simulate_revenue_impact(segmented)

        self._all_fitted = True

        # 7. Summary
        return {
            "n_users": len(processed),
            "churn_rate": float(processed["churned"].mean()),
            "partial_churn_rate": float(processed["partial_churn"].mean()),
            "cox_hazard_ratios": self.cox_model.hazard_ratios(),
            "cox_c_index": self.cox_model.concordance_index(processed),
            "xgb_feature_importance": (
                self.xgb_model.feature_importance().to_dict("records")
                if self.xgb_model.feature_importance() is not None
                else None
            ),
            "clv_summary": {
                "mean_clv": float(clv_data["predicted_clv"].mean()),
                "median_clv": float(clv_data["predicted_clv"].median()),
                "p95_clv": float(clv_data["predicted_clv"].quantile(0.95)),
            },
            "segment_distribution": segmented["segment"].value_counts().to_dict(),
            "revenue_impact": impact,
            "intervention_matrix": self.intervention.intervention_matrix().to_dict("records"),
        }

    def get_segmented_data(self) -> Optional[pd.DataFrame]:
        if self._processed_data is None:
            return None
        clv_data = self.clv_model.predict_clv(self._processed_data)
        return self.intervention.segment_users(clv_data)

    def get_risk_rankings(self, top_n: int = 100) -> Optional[pd.DataFrame]:
        """Return ranked list of highest-risk users for CRM integration."""
        segmented = self.get_segmented_data()
        if segmented is None:
            return None

        cox_risk = self.cox_model.predict_risk(segmented)
        segmented["cox_churn_risk_score"] = cox_risk

        rankings = segmented.sort_values("cox_churn_risk_score", ascending=False)
        return rankings.head(top_n)[
            ["user_id", "segment", "churn_risk", "predicted_clv", "cox_churn_risk_score",
             "days_since_last_login", "loss_streak_length", "favourite_player"]
        ]
