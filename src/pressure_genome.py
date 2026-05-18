"""Pressure Genome — Module 1: Batsman Psychological DNA under match pressure.

Quantifies batsman performance under pressure using 12 contextual features,
then clusters players into archetypes via unsupervised learning. Delivers
actionable selection intelligence for coaches and selectors.

Glossary:
  - Pressure ball: delivery where match context is high-stakes (death overs chase,
    early collapse, tight finish). Defined in data_loader.PRESSURE_RULES.
  - Archetype: cluster label assigned by K-Means on PCA-reduced feature space.
  - Compatibility score: cosine similarity between match-state vector and
    archetype centroid weighted by situational relevance.
"""

from typing import Dict, List, Optional, Tuple
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

PRESSURE_FEATURES = [
    "pressure_sr",
    "pressure_dot_pct",
    "pressure_boundary_pct",
    "pressure_dismissal_rate",
    "consistency_index",
    "clutch_runs_above_expected",
    "performance_decay_slope",
    "venue_pressure_delta",
    "vs_pace_pressure_sr",
    "vs_spin_pressure_sr",
    "late_innings_fatigue_index",
    "high_stakes_match_multiplier",
]

FEATURE_DISPLAY_NAMES = {
    "pressure_sr": "Pressure Strike Rate",
    "pressure_dot_pct": "Dot Ball % Under Pressure",
    "pressure_boundary_pct": "Boundary % Under Pressure",
    "pressure_dismissal_rate": "Dismissal Rate Under Pressure",
    "consistency_index": "Consistency Index (lower = more consistent)",
    "clutch_runs_above_expected": "Clutch Runs Above Expectation",
    "performance_decay_slope": "Performance Decay Slope",
    "venue_pressure_delta": "Venue Pressure Delta",
    "vs_pace_pressure_sr": "vs Pace Pressure SR",
    "vs_spin_pressure_sr": "vs Spin Pressure SR",
    "late_innings_fatigue_index": "Late Innings Fatigue Index",
    "high_stakes_match_multiplier": "High-Stakes Match Multiplier",
}

ARCHETYPE_DESCRIPTIONS = {
    "Ice-blooded finisher": (
        "Thrives under pressure — elevated SR, low dot%, low dismissal rate. "
        "The player you want at the crease in a final-over chase."
    ),
    "Steady accumulator": (
        "Moderate SR with excellent consistency and low dismissal rate. "
        "Reliable in a rebuilding phase but may not accelerate when needed."
    ),
    "Panic scorer": (
        "SR rises under pressure but at the cost of high dismissal rate and "
        "poor consistency. Entertaining but high-risk in crunch moments."
    ),
    "Pressure crumbler": (
        "All metrics degrade under pressure — lower SR, higher dot%, higher "
        "dismissal rate. Needs mental conditioning or a lower-pressure role."
    ),
}


@dataclass
class PressureGenomeConfig:
    n_components: int = 3
    min_clusters: int = 2
    max_clusters: int = 8
    random_state: int = 42
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class PressureGenome:
    """End-to-end Pressure Genome pipeline."""

    def __init__(self, config: Optional[PressureGenomeConfig] = None):
        self.config = config or PressureGenomeConfig()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.config.n_components, random_state=self.config.random_state)
        self.kmeans: Optional[KMeans] = None
        self.feature_names = PRESSURE_FEATURES
        self._cluster_labels: Optional[np.ndarray] = None
        self._pca_loaded = False
        self._fitted = False

    # -- Preprocessing -------------------------------------------------------

    def normalize(self, df: pd.DataFrame) -> np.ndarray:
        """Scale pressure features to zero mean, unit variance."""
        X = df[self.feature_names].values.copy()
        return self.scaler.fit_transform(X)

    def reduce(self, X_scaled: np.ndarray) -> np.ndarray:
        """PCA dimensionality reduction."""
        return self.pca.fit_transform(X_scaled)

    # -- Clustering ----------------------------------------------------------

    def find_optimal_k(self, X_scaled: np.ndarray, plot: bool = False) -> Dict[int, float]:
        """Sweep k and return silhouette scores for each cluster count."""
        scores = {}
        for k in range(self.config.min_clusters, self.config.max_clusters + 1):
            km = KMeans(n_clusters=k, random_state=self.config.random_state, n_init="auto")
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            scores[k] = sil
        return scores

    def fit(self, df: pd.DataFrame, n_clusters: Optional[int] = None) -> "PressureGenome":
        """Run the full pipeline: normalize → PCA → K-Means."""
        X_scaled = self.normalize(df)
        X_pca = self.reduce(X_scaled)

        if n_clusters is None:
            sil_scores = self.find_optimal_k(X_scaled)
            n_clusters = max(sil_scores, key=sil_scores.get)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_state, n_init="auto")
        self._cluster_labels = self.kmeans.fit_predict(X_pca)
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self.kmeans is None:
            raise RuntimeError("PressureGenome must be fitted before predict.")
        X_scaled = self.scaler.transform(df[self.feature_names].values)
        X_pca = self.pca.transform(X_scaled)
        return self.kmeans.predict(X_pca)

    # -- Archetype naming ----------------------------------------------------

    def name_archetypes(self, df: pd.DataFrame) -> Dict[int, str]:
        """Assign human-readable names to clusters based on centroid profiles."""
        if self._cluster_labels is None or self.kmeans is None:
            raise RuntimeError("Must fit before naming archetypes.")

        X_scaled = self.scaler.transform(df[self.feature_names].values)
        df_temp = df.copy()
        df_temp["cluster"] = self._cluster_labels

        centroids = df_temp.groupby("cluster")[self.feature_names].mean()
        archetype_map = {}

        for cluster_id in centroids.index:
            row = centroids.loc[cluster_id]
            sr = row["pressure_sr"]
            dot = row["pressure_dot_pct"]
            dr = row["pressure_dismissal_rate"]
            decay = row["performance_decay_slope"]
            clutch = row["clutch_runs_above_expected"]

            if sr > 135 and dot < 0.40 and dr < 0.12 and decay > -0.3 and clutch > 0:
                archetype_map[cluster_id] = "Ice-blooded finisher"
            elif sr < 115 or dot > 0.50:
                archetype_map[cluster_id] = "Pressure crumbler"
            elif dr < 0.10 and abs(decay) < 0.5:
                archetype_map[cluster_id] = "Steady accumulator"
            else:
                archetype_map[cluster_id] = "Panic scorer"

        return archetype_map

    # -- Compatibility scoring ------------------------------------------------

    def compatibility_score(
        self,
        player_features: pd.Series,
        match_state: Dict[str, float],
    ) -> float:
        """Score a player's fit for a given match situation.

        Args:
            player_features: row from the pressure features DataFrame.
            match_state: dict with keys like 'required_run_rate', 'overs_remaining',
                        'wickets_left', 'is_death_overs', 'is_chase'.

        Returns:
            Compatibility score ∈ [0, 1]. Higher = better fit.
        """
        if not self._fitted or self.kmeans is None:
            raise RuntimeError("Must fit before computing compatibility.")

        player_vec = self.scaler.transform(
            player_features[self.feature_names].values.reshape(1, -1)
        )
        player_pca = self.pca.transform(player_vec).flatten()

        cluster = self.kmeans.predict(player_pca.reshape(1, -1))[0]
        centroid = self.kmeans.cluster_centers_[cluster]

        # distance to centroid (closer = more archetypical)
        dist = np.linalg.norm(player_pca - centroid)
        max_dist = np.linalg.norm(
            np.ones(self.config.n_components) * 3 -
            np.ones(self.config.n_components) * -3
        )
        archetype_purity = 1.0 - min(dist / max_dist, 1.0)

        # situational bonus
        rrr = match_state.get("required_run_rate", 0)
        wickets_left = match_state.get("wickets_left", 10)
        is_chase = match_state.get("is_chase", True)

        sr_bonus = 0.0
        if rrr > 10 and wickets_left <= 3:
            # crunch time — weight pressure_sr heavily
            sr_norm = min(max((player_features["pressure_sr"] - 80) / 80, 0), 1)
            sr_bonus = sr_norm * 0.3
        elif is_chase:
            sr_norm = min(max((player_features["pressure_sr"] - 80) / 80, 0), 1)
            sr_bonus = sr_norm * 0.15

        return min(archetype_purity * 0.7 + sr_bonus, 1.0)

    def rank_for_situation(
        self,
        df: pd.DataFrame,
        match_state: Dict[str, float],
        top_n: int = 3,
        exclude_players: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Rank available batsmen by compatibility with match situation."""
        results = []
        exclude = set(exclude_players or [])
        for _, row in df.iterrows():
            if row["player"] in exclude:
                continue
            score = self.compatibility_score(row, match_state)
            results.append({"player": row["player"], "compatibility": score})

        rankings = pd.DataFrame(results).sort_values("compatibility", ascending=False)
        return rankings.head(top_n)

    # -- Pressure mismatch alert ---------------------------------------------

    def pressure_mismatch_alert(
        self,
        lineup: List[str],
        squad_df: pd.DataFrame,
        match_state: Dict[str, float],
        threshold: float = 0.35,
    ) -> Dict:
        """Flag if a team's remaining batting lineup is ill-suited for the situation.

        Args:
            lineup: list of player names available to bat.
            squad_df: DataFrame with pressure features for all squad players.
            match_state: current match situation.
            threshold: minimum average compatibility to avoid alert.

        Returns:
            dict with 'mismatch_detected', 'avg_compatibility', 'best_xi', 'worst_xi'.
        """
        available = squad_df[squad_df["player"].isin(lineup)].copy()
        if available.empty:
            return {"mismatch_detected": True, "avg_compatibility": 0.0, "message": "No players found in pressure database."}

        available["compatibility"] = available.apply(
            lambda r: self.compatibility_score(r, match_state), axis=1
        )
        avg_comp = available["compatibility"].mean()
        best = available.nlargest(3, "compatibility")["player"].tolist()
        worst = available.nsmallest(3, "compatibility")["player"].tolist()

        return {
            "mismatch_detected": avg_comp < threshold,
            "avg_compatibility": round(avg_comp, 3),
            "best_xi": best,
            "worst_xi": worst,
            "message": (
                f"MISMATCH: Average lineup compatibility {avg_comp:.2f} is below {threshold}. "
                f"Consider promoting {best[0]} up the order."
                if avg_comp < threshold
                else f"Lineup OK. Avg compatibility {avg_comp:.2f}."
            ),
        }

    # -- Radar chart data -----------------------------------------------------

    def radar_data(self, player_row: pd.Series) -> Dict[str, float]:
        """Extract normalized radar chart values for a single player."""
        vals = {}
        for feat in self.feature_names:
            vals[FEATURE_DISPLAY_NAMES.get(feat, feat)] = round(player_row[feat], 2)
        return vals

    def comparison_data(
        self, df: pd.DataFrame, player_a: str, player_b: str
    ) -> Optional[Tuple[Dict, Dict]]:
        """Get radar chart data for two players."""
        p1 = df[df["player"] == player_a]
        p2 = df[df["player"] == player_b]
        if p1.empty or p2.empty:
            return None
        return self.radar_data(p1.iloc[0]), self.radar_data(p2.iloc[0])

    # -- PCA biplot data -----------------------------------------------------

    def biplot_data(self) -> pd.DataFrame:
        """Return PCA component loadings for feature interpretation."""
        loadings = self.pca.components_.T
        return pd.DataFrame(
            loadings,
            index=self.feature_names,
            columns=[f"PC{i+1}" for i in range(self.config.n_components)],
        )

    # -- Summary output ------------------------------------------------------

    def summary(self, df: pd.DataFrame) -> Dict:
        """Return a summary dict with all key results for dashboard consumption."""
        if not self._fitted:
            return {"status": "not fitted"}
        if self._cluster_labels is None:
            return {"status": "no cluster labels"}

        archetypes = self.name_archetypes(df)
        sil_scores = self.find_optimal_k(
            self.scaler.transform(df[self.feature_names].values)
        )

        cluster_counts = pd.Series(self._cluster_labels).value_counts().sort_index()
        cluster_summary = {}
        for cid in cluster_counts.index:
            name = archetypes.get(cid, f"Cluster {cid}")
            members = df.loc[self._cluster_labels == cid, "player"].tolist()
            cluster_summary[int(cid)] = {
                "name": name,
                "description": ARCHETYPE_DESCRIPTIONS.get(name, ""),
                "size": int(cluster_counts[cid]),
                "members": members[:10],
            }

        return {
            "status": "fitted",
            "n_clusters": len(cluster_counts),
            "n_players": len(df),
            "silhouette_scores": {str(k): round(v, 4) for k, v in sil_scores.items()},
            "explained_variance_ratio": {
                f"PC{i+1}": round(self.pca.explained_variance_ratio_[i], 4)
                for i in range(self.config.n_components)
            },
            "clusters": cluster_summary,
            "features": self.feature_names,
        }
