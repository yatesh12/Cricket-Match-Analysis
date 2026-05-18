"""CricketIQ Data Loader — shared data foundation for all 4 modules.

Handles:
  - Loading real cricket data from CSV files in data/processed/
  - Standardising column names across different CSV formats
  - Computing derived features (pressure stats, over-level aggregates)
  - Shared feature engineering utilities
  - Pressure state taxonomy definitions
"""

import glob
import warnings
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Pressure state taxonomy — every rule is a testable function
# ---------------------------------------------------------------------------

def is_death_overs_chase(row: pd.Series) -> bool:
    """Last 4 overs of a chase with RRR > 10."""
    if row.get("innings") != 2:
        return False
    overs_left = row.get("overs_remaining", 0)
    rrr = row.get("required_run_rate", 0)
    return overs_left <= 4 and rrr > 10


def is_early_collapse(row: pd.Series) -> bool:
    """Wickets fallen >= 6 in first 15 overs."""
    overs_elapsed = row.get("over", 0)
    wkts = row.get("wickets_fallen", 0)
    return overs_elapsed <= 15 and wkts >= 6


def is_tight_finish(row: pd.Series) -> bool:
    """Score within 10 of target with 2 wickets left."""
    if row.get("innings") != 2:
        return False
    target = row.get("target", 0)
    score = row.get("current_score", 0)
    wkts = row.get("wickets_fallen", 0)
    runs_needed = target - score
    return 0 < runs_needed <= 10 and wkts >= 8


def is_pressure_ball(row: pd.Series) -> bool:
    """Composite pressure flag — True if ANY pressure condition is met."""
    return any([
        is_death_overs_chase(row),
        is_early_collapse(row),
        is_tight_finish(row),
    ])


PRESSURE_RULES = {
    "death_overs_chase": is_death_overs_chase,
    "early_collapse": is_early_collapse,
    "tight_finish": is_tight_finish,
}


# ---------------------------------------------------------------------------
# 2. Ball-by-ball to over-level aggregation
# ---------------------------------------------------------------------------

def aggregate_over_stats(ball_by_ball: pd.DataFrame) -> pd.DataFrame:
    """Convert ball-by-ball to over-by-over DataFrame."""
    df = ball_by_ball.copy()
    if "is_wide" not in df.columns:
        df["is_wide"] = False
    if "is_noball" not in df.columns:
        df["is_noball"] = False
    if "is_wicket" not in df.columns:
        df["is_wicket"] = False

    df["legal_delivery"] = (~df["is_wide"].fillna(0).astype(bool) & ~df["is_noball"].fillna(0).astype(bool))

    over_stats = df.groupby(
        ["match_id", "innings", "batting_team", "over"]
    ).agg(
        runs_scored=("runs", "sum"),
        wickets=("is_wicket", "sum"),
        boundaries=("runs", lambda x: ((x >= 4) & (~df.loc[x.index, "is_wide"])).sum()),
        sixes=("runs", lambda x: ((x >= 6) & (~df.loc[x.index, "is_wide"])).sum()),
        dot_balls=("legal_delivery", lambda x: (
            (x) & (df.loc[x.index, "runs"] == 0)
        ).sum()),
        total_balls=("legal_delivery", "sum"),
    ).reset_index()

    over_stats["cumulative_score"] = over_stats.groupby(
        ["match_id", "innings"]
    )["runs_scored"].cumsum()

    over_stats["cumulative_wickets"] = over_stats.groupby(
        ["match_id", "innings"]
    )["wickets"].cumsum()

    return over_stats


def compute_match_context(over_stats: pd.DataFrame) -> pd.DataFrame:
    """Add contextual columns: RRR, overs remaining, target, etc."""
    df = over_stats.copy()
    if df.empty:
        return df

    if "inning1_total" not in df.columns:
        match_targets = df[df["innings"] == 1].groupby("match_id")["runs_scored"].sum().reset_index()
        if match_targets.empty:
            match_targets = pd.DataFrame({"match_id": df["match_id"].unique(), "inning1_total": 0})
        else:
            match_targets.columns = ["match_id", "inning1_total"]
        df = df.merge(match_targets, on="match_id", how="left")
    elif "target" not in df.columns:
        pass  # already computed

    df["target"] = df["inning1_total"].fillna(0).astype(int) + 1
    df["overs_remaining"] = 20 - df["over"]

    is_chase = df["innings"] == 2
    df["runs_needed"] = np.where(is_chase, df["target"] - df["cumulative_score"], np.nan)
    df["required_run_rate"] = np.where(
        is_chase & (df["overs_remaining"] > 0),
        df["runs_needed"] / df["overs_remaining"],
        np.nan,
    )
    df["current_run_rate"] = np.where(
        df["over"] > 0,
        df["cumulative_score"] / df["over"],
        np.nan,
    )
    df["wickets_fallen"] = df["cumulative_wickets"]

    pressure_mask = df.apply(is_pressure_ball, axis=1)
    df["pressure_ball"] = pressure_mask

    return df


# ---------------------------------------------------------------------------
# 3. Pressure feature derivation from real ball-by-ball data
# ---------------------------------------------------------------------------

def derive_pressure_features(ball_by_ball: pd.DataFrame) -> pd.DataFrame:
    """Compute per-player pressure performance metrics from real data.

    For each batsman, calculates:
      - overall strike rate
      - pressure strike rate (death overs + chase)
      - dot ball percentage under pressure
      - boundary percentage under pressure
      - dismissal rate
      - consistency index
    """
    df = ball_by_ball.copy()

    if "is_wide" not in df.columns:
        df["is_wide"] = False
    if "is_noball" not in df.columns:
        df["is_noball"] = False
    if "is_wicket" not in df.columns:
        df["is_wicket"] = False

    df["legal_delivery"] = (~df["is_wide"].fillna(0).astype(bool) & ~df["is_noball"].fillna(0).astype(bool))
    df["is_boundary"] = (df["runs"] >= 4) & ~df["is_wide"]
    df["is_dot"] = (df["runs"] == 0) & df["legal_delivery"]

    # death overs (over >= 16)
    df["is_death_over"] = df["over"] >= 16

    # pressure ball defined by match context (chase with high RRR etc.)
    # compute over-level context first
    over_stats = aggregate_over_stats(df)
    ctx = compute_match_context(over_stats).set_index(["match_id", "innings", "over"])

    df["is_pressure_over"] = df.set_index(["match_id", "innings", "over"]).index.isin(
        ctx[ctx["pressure_ball"]].index
    )

    # aggregate per batter
    records = []
    for batter, group in df.groupby("batter"):
        total_balls = group["legal_delivery"].sum()
        if total_balls < 5:
            continue

        total_runs = group["runs"].sum()
        total_wickets = group["is_wicket"].sum()
        total_dots = group["is_dot"].sum()
        total_boundaries = group["is_boundary"].sum()
        total_sixes = group["is_boundary"].sum()

        pressure_group = group[group["is_pressure_over"] | group["is_death_over"]]
        pressure_balls = pressure_group["legal_delivery"].sum()
        pressure_runs = pressure_group["runs"].sum()
        pressure_wickets = pressure_group["is_wicket"].sum()
        pressure_dots = pressure_group["is_dot"].sum()
        pressure_boundaries = pressure_group["is_boundary"].sum()

        non_pressure = group[~group["is_pressure_over"] & ~group["is_death_over"]]
        non_pressure_sr = (non_pressure["runs"].sum() / max(non_pressure["legal_delivery"].sum(), 1)) * 100

        # compute consistency as std dev of runs per ball
        runs_per_ball = group[group["legal_delivery"]]["runs"]

        records.append({
            "player": batter,
            "appearances": group["match_id"].nunique(),
            "total_runs": total_runs,
            "total_balls_faced": int(total_balls),
            "strike_rate": (total_runs / max(total_balls, 1)) * 100,
            "pressure_sr": (pressure_runs / max(pressure_balls, 1)) * 100 if pressure_balls > 0 else 0,
            "pressure_dot_pct": pressure_dots / max(pressure_balls, 1) if pressure_balls > 0 else 0.5,
            "pressure_boundary_pct": pressure_boundaries / max(pressure_balls, 1) if pressure_balls > 0 else 0,
            "pressure_dismissal_rate": pressure_wickets / max(pressure_balls, 1) if pressure_balls > 0 else 0,
            "consistency_index": float(np.std(runs_per_ball)) if len(runs_per_ball) > 1 else 0.5,
            "clutch_runs_above_expected": pressure_runs - non_pressure_sr * pressure_balls / 100 if pressure_balls > 0 else 0,
            "performance_decay_slope": non_pressure_sr - ((pressure_runs / max(pressure_balls, 1)) * 100) if pressure_balls > 0 else 0,
            "venue_pressure_delta": 0.0,
            "vs_pace_pressure_sr": 0.0,
            "vs_spin_pressure_sr": 0.0,
            "late_innings_fatigue_index": 0.0,
            "high_stakes_match_multiplier": 1.0,
            "known_archetype": "Unclassified",
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    # assign archetypes based on derived stats
    def assign_archetype(row):
        sr = row["pressure_sr"]
        dot = row["pressure_dot_pct"]
        dr = row["pressure_dismissal_rate"]
        decay = row["performance_decay_slope"]
        if sr > 140 and dot < 0.35 and dr < 0.12 and decay > -20:
            return "Ice-blooded finisher"
        elif sr < 100 or dot > 0.55:
            return "Pressure crumbler"
        elif 100 <= sr <= 135 and dr < 0.10:
            return "Steady accumulator"
        else:
            return "Panic scorer"

    result["known_archetype"] = result.apply(assign_archetype, axis=1)
    return result


# ---------------------------------------------------------------------------
# 4. Shared data loader (facade)
# ---------------------------------------------------------------------------

class CricketDataLoader:
    """Central data loading interface for all CricketIQ modules.

    Loads real cricket match data from CSV files in data/processed/.
    Falls back to available files with clear error messages if data is missing.
    """

    # CSV files available in data/processed/ with their column schemas
    AVAILABLE_CSV = {
        "ashwin": "cricsheet_ashwin_ball_by_ball.csv",
        "enriched": "enriched_ball_by_ball_57cols.csv",
        "ipl_2025": "ipl_2025_ball_by_ball_40cols.csv",
        "kaggle": "kaggle_ipl_2008_2025_41cols.csv",
    }

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
        self._ball_by_ball: Optional[pd.DataFrame] = None
        self._over_stats: Optional[pd.DataFrame] = None

    def _find_csv(self, *preferred_names: str) -> Optional[Path]:
        """Find the first available CSV from a list of preferred names."""
        if not self.data_dir.exists():
            return None
        for name in preferred_names:
            path = self.data_dir / name
            if path.exists():
                return path
        return None

    def load_ball_by_ball(self, source: str = "enriched") -> pd.DataFrame:
        """Load real ball-by-ball data from CSV files in data/processed/.

        Parameters
        ----------
        source : str
            Which CSV format to load:
            - "enriched"  (57 cols, most features, recommended)
            - "ashwin"    (22 cols, clean Cricsheet format)
            - "ipl_2025"  (40 cols, IPL-specific)
            - "kaggle"    (41 cols, Kaggle format)

        Returns
        -------
        pd.DataFrame with standardised columns:
            match_id, innings, batting_team, bowling_team, over, ball,
            batter, bowler, runs, is_wide, is_noball, is_wicket, ...
        """
        csv_map = {
            "enriched": ("enriched_ball_by_ball_57cols.csv",),
            "ashwin": ("cricsheet_ashwin_ball_by_ball.csv",),
            "ipl_2025": ("ipl_2025_ball_by_ball_40cols.csv",),
            "kaggle": ("kaggle_ipl_2008_2025_41cols.csv",),
        }

        if source not in csv_map:
            valid = list(csv_map.keys())
            raise ValueError(f"Unknown source '{source}'. Valid options: {valid}")

        filepath = self._find_csv(*csv_map[source])
        if filepath is None:
            alt_path = self.data_dir / csv_map[source][0]
            raise FileNotFoundError(
                f"Real data file not found at {alt_path}. "
                f"Ensure a CSV file exists in {self.data_dir}/"
            )

        df = pd.read_csv(filepath)

        if source == "ashwin":
            df = self._standardise_ashwin(df)
        elif source == "enriched":
            df = self._standardise_enriched(df)
        elif source == "ipl_2025":
            df = self._standardise_ipl_2025(df)
        elif source == "kaggle":
            df = self._standardise_kaggle(df)

        self._ball_by_ball = df
        return df

    def load_all_matches(self) -> pd.DataFrame:
        """Load and combine all available CSV files into one unified DataFrame."""
        frames = []
        loaded_sources = []
        for source in ["enriched", "ashwin", "ipl_2025", "kaggle"]:
            try:
                df = self.load_ball_by_ball(source)
                frames.append(df)
                loaded_sources.append(source)
            except FileNotFoundError:
                continue

        if not frames:
            raise FileNotFoundError(
                f"No CSV data files found in {self.data_dir}. "
                f"Please add at least one ball-by-ball CSV file to data/processed/"
            )

        combined = pd.concat(frames, ignore_index=True)
        self._ball_by_ball = combined
        return combined

    @staticmethod
    def _standardise_ashwin(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise Cricsheet Ashwin CSV columns to internal format."""
        df = df.rename(columns={
            "striker": "batter",
            "runs_off_bat": "runs",
        })
        ball_str = df["ball"].astype(str)
        df["over"] = ball_str.str.split(".").str[0].astype(int)
        df["ball_in_over"] = ball_str.str.split(".").str[1].astype(int)
        df["is_wide"] = (df["wides"].fillna(0).astype(int) > 0).astype(int)
        df["is_noball"] = (df["noballs"].fillna(0).astype(int) > 0).astype(int)
        df["is_wicket"] = (df["wicket_type"].notna() & (df["wicket_type"] != "")).astype(int)
        df["runs"] = df["runs"].fillna(0).astype(int)
        df["match_type"] = "T20"
        df["city"] = ""
        df["teams"] = df.apply(
            lambda r: f"[{r['batting_team']}, {r['bowling_team']}]", axis=1
        )
        for col in ["toss_winner", "toss_decision", "target"]:
            if col not in df.columns:
                df[col] = ""
        return df

    @staticmethod
    def _standardise_enriched(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise enriched 57-col CSV to internal format."""
        df = df.rename(columns={
            "striker": "batter",
            "date": "start_date",
        })
        if "over" not in df.columns:
            ball_series = df["ball"].astype(str)
            df["over"] = ball_series.str.split(".").str[0].fillna("0").astype(int)
        else:
            df["over"] = df["over"].fillna(0).astype(int)
        def _parse_ball(over, ball):
            if ball and str(ball) != "nan":
                b = str(ball)
                if "." in b:
                    return int(b.split(".")[-1])
                return int(b)
            return 1
        df["ball_in_over"] = df.apply(
            lambda r: int(_parse_ball(r.get("over", 0), r.get("ball", 1))),
            axis=1,
        )
        df["is_wide"] = (df["wides"].fillna(0).astype(int) > 0).astype(int) if "wides" in df.columns else 0
        df["is_noball"] = (df["noballs"].fillna(0).astype(int) > 0).astype(int) if "noballs" in df.columns else 0
        if "is_wicket" in df.columns:
            df["is_wicket"] = df["is_wicket"].fillna(0).astype(int)
        else:
            df["is_wicket"] = (df["wicket_type"].notna() & (df["wicket_type"] != "")).astype(int)
        df["runs"] = df["runs_off_bat"].fillna(0).astype(int)
        if "match_type" not in df.columns:
            df["match_type"] = "T20"
        if "city" not in df.columns:
            df["city"] = ""
        df["teams"] = df.apply(
            lambda r: f"[{r.get('team1', r.get('batting_team', ''))}, "
                      f"{r.get('team2', r.get('bowling_team', ''))}]", axis=1
        )
        for col in ["toss_winner", "toss_decision", "target"]:
            if col not in df.columns:
                df[col] = ""
        return df

    @staticmethod
    def _standardise_ipl_2025(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise IPL 2025 40-col CSV to internal format."""
        df = df.rename(columns={
            "striker": "batter",
            "runs_of_bat": "runs",
            "date_x": "start_date",
            "venue_x": "venue",
        })
        df["innings"] = df["innings"].astype(int)
        if "over" in df.columns and df["over"].dtype.kind in ("f", "i"):
            df["over"] = df["over"].fillna(0).astype(int)
        df["ball_in_over"] = 1
        df["is_wide"] = (df["wide"].fillna(0).astype(int) > 0).astype(int) if "wide" in df.columns else 0
        df["is_noball"] = df["noballs"].fillna(0).astype(int) if "noballs" in df.columns else 0
        df["is_wicket"] = (df["wicket_type"].notna() & (df["wicket_type"] != "")).astype(int)
        df["runs"] = df["runs"].fillna(0).astype(int)
        df["city"] = df.get("venue_x", "")
        df["match_type"] = "IPL"
        df["teams"] = df.apply(lambda r: f"[{r['team1']}, {r['team2']}]", axis=1)
        df["toss_winner"] = df.get("toss_winner", "")
        df["toss_decision"] = df.get("toss_decision", "")
        target_col = "first_ings_score" if "first_ings_score" in df.columns else "target"
        df["target"] = df.get(target_col, 0)
        return df

    @staticmethod
    def _standardise_kaggle(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise Kaggle IPL 2008-2025 CSV to internal format."""
        df = df.rename(columns={
            "striker": "batter",
            "runs_off_bat": "runs",
        })
        df["innings"] = df["innings"].astype(int)
        if "over" in df.columns and df["over"].dtype.kind in ("f", "i"):
            df["over"] = df["over"].fillna(0).astype(int)
        df["ball_in_over"] = df["ball"].fillna(1).astype(int) if "ball" in df.columns else 1
        df["is_wide"] = (df["wides"].fillna(0).astype(int) > 0).astype(int) if "wides" in df.columns else 0
        df["is_noball"] = (df["noballs"].fillna(0).astype(int) > 0).astype(int) if "noballs" in df.columns else 0
        if "is_wicket" in df.columns:
            df["is_wicket"] = df["is_wicket"].fillna(0).astype(int)
        else:
            df["is_wicket"] = (df["wicket_type"].notna() & (df["wicket_type"] != "")).astype(int)
        df["runs"] = df["runs"].fillna(0).astype(int)
        df["city"] = df.get("city", "")
        df["match_type"] = df.get("match_type", "T20")
        df["teams"] = df.apply(lambda r: f"[{r['team1']}, {r['team2']}]", axis=1)
        df["toss_winner"] = df.get("toss_winner", "")
        df["toss_decision"] = df.get("toss_decision", "")
        df["target"] = df.get("target", 0)
        return df

    def get_over_stats(self, force_rebuild: bool = False) -> pd.DataFrame:
        """Get or compute over-level stats from real ball-by-ball data."""
        if self._over_stats is not None and not force_rebuild:
            return self._over_stats

        if self._ball_by_ball is None:
            try:
                self.load_ball_by_ball("enriched")
            except FileNotFoundError:
                try:
                    self.load_ball_by_ball("ashwin")
                except FileNotFoundError:
                    raise FileNotFoundError(
                        "No ball-by-ball CSV found in data/processed/. "
                        "Place at least one CSV file (cricsheet_ashwin_ball_by_ball.csv "
                        "or enriched_ball_by_ball_57cols.csv) in data/processed/"
                    )

        aggregated = aggregate_over_stats(self._ball_by_ball)
        match_targets = aggregated[aggregated["innings"] == 1].groupby("match_id")["runs_scored"].sum()
        aggregated["target"] = aggregated["match_id"].map(match_targets).fillna(0)

        self._over_stats = compute_match_context(aggregated)
        return self._over_stats

    def get_pressure_features(self, n_batsmen: int = 200) -> pd.DataFrame:
        """Derive per-player pressure features from real ball-by-ball data.

        Parameters
        ----------
        n_batsmen : int
            Minimum number of batsmen to return. If real data has fewer,
            all available are returned with a warning.

        Returns
        -------
        pd.DataFrame with player pressure metrics for Module 1.
        """
        if self._ball_by_ball is None:
            try:
                self.load_ball_by_ball("enriched")
            except FileNotFoundError:
                try:
                    self.load_ball_by_ball("ashwin")
                except FileNotFoundError:
                    raise FileNotFoundError(
                        "No ball-by-ball CSV found. Cannot derive pressure features "
                        "without real match data."
                    )

        df = derive_pressure_features(self._ball_by_ball)
        if df.empty:
            raise ValueError(
                "Could not derive pressure features from the available data. "
                "Ensure the CSV contains valid ball-by-ball data with batter names and runs."
            )

        if len(df) < n_batsmen:
            print(f"  [INFO] Only {len(df)} batsmen available in real data "
                  f"(requested {n_batsmen}). Using all available.")

        # Select columns expected by PressureGenome
        required_cols = [
            "player", "pressure_sr", "pressure_dot_pct", "pressure_boundary_pct",
            "pressure_dismissal_rate", "consistency_index", "clutch_runs_above_expected",
            "performance_decay_slope", "venue_pressure_delta", "vs_pace_pressure_sr",
            "vs_spin_pressure_sr", "late_innings_fatigue_index",
            "high_stakes_match_multiplier", "known_archetype",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0 if col not in ("player", "known_archetype") else ""

        df = df.drop_duplicates(subset="player")
        return df[required_cols]

    def get_fantasy_users(self, n_users: int = 50000) -> pd.DataFrame:
        """Load fantasy platform user data.

        Real user data is not included in the cricket ball-by-ball CSV datasets.
        To use this module, place a 'fantasy_users.csv' file in data/processed/
        with columns: user_id, city, age_group, registration_date, contests_entered_per_week,
        avg_team_score_percentile, winnings_last_30d, days_since_last_login, total_deposits,
        total_withdrawals, win_rate, team_diversity_score, loss_streak_length, churned,
        partial_churn, favourite_player.

        Returns
        -------
        pd.DataFrame with user data.
        """
        fantasy_csv = self.data_dir / "fantasy_users.csv"
        if fantasy_csv.exists():
            df = pd.read_csv(fantasy_csv)
            if len(df) >= n_users:
                return df.head(n_users)
            return df

        raise FileNotFoundError(
            f"Fantasy user data not found at {fantasy_csv}. "
            f"Module 4 (Fantasy Churn CLV) requires a separate CSV file with "
            f"user activity data, which is not included in the public cricket "
            f"datasets. To use this module, export user data from your platform "
            f"and save it as data/processed/fantasy_users.csv"
        )

    def get_match_schedule(self, n_matches: int = 74) -> pd.DataFrame:
        """Load match schedule from real data or derive from available matches.

        Parameters
        ----------
        n_matches : int
            Desired number of matches (used only as upper bound).

        Returns
        -------
        pd.DataFrame with match schedule information.
        """
        if self._ball_by_ball is None:
            try:
                self.load_ball_by_ball("enriched")
            except FileNotFoundError:
                try:
                    self.load_ball_by_ball("ashwin")
                except FileNotFoundError:
                    return pd.DataFrame()

        schedule = self._ball_by_ball.groupby("match_id").agg(
            team1=("batting_team", "first"),
            city=("city", "first"),
            season=("season", "first"),
        ).reset_index()

        schedule["team2"] = ""
        schedule["is_knockout"] = False

        return schedule.head(n_matches)
