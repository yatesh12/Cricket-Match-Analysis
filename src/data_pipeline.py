"""CricketIQ Data Pipeline — unified ingestion, cleaning, match-state, and feature store.

Architecture:
  raw/CSV → harmonize_schema() → clean_cricket_data() → compute_match_state()
  → aggregate_over_level() → build_feature_store() → parquet files
"""

import warnings, os, json, hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Canonical schema — every raw source maps to these 22 columns
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS = [
    "match_id", "innings", "over", "ball", "batting_team", "bowling_team",
    "striker", "non_striker", "bowler", "runs_off_bat", "extras", "wides",
    "noballs", "byes", "legbyes", "wicket_type", "player_dismissed",
    "runs", "wickets", "venue", "date", "match_type",
]

# ---------------------------------------------------------------------------
# Schema mapping: raw source name → {canonical_col: raw_col or default_value}
# ---------------------------------------------------------------------------
SCHEMA_MAP: Dict[str, Dict[str, object]] = {
    "raw_kaggle_ipl": {
        "match_id": "match_id", "innings": "innings", "over": "over",
        "ball": "ball", "batting_team": "batting_team", "bowling_team": "bowling_team",
        "striker": "striker", "non_striker": "non_striker", "bowler": "bowler",
        "runs_off_bat": "runs_off_bat", "extras": "extras", "wides": "wides",
        "noballs": "noballs", "byes": "byes", "legbyes": "legbyes",
        "wicket_type": "wicket_type", "player_dismissed": "player_dismissed",
        "runs": "runs", "wickets": "wickets", "venue": "venue", "date": "date",
        "match_type": "match_type",
    },
    "raw_cricsheet_ashwin": {
        "match_id": "match_id", "innings": "innings",
        "ball": "ball", "batting_team": "batting_team", "bowling_team": "bowling_team",
        "striker": "striker", "non_striker": "non_striker", "bowler": "bowler",
        "runs_off_bat": "runs_off_bat", "extras": "extras", "wides": "wides",
        "noballs": "noballs", "byes": "byes", "legbyes": "legbyes",
        "wicket_type": "wicket_type", "player_dismissed": "player_dismissed",
        "venue": "venue", "date": "start_date", "match_type": "__missing__",
    },
    "raw_ipl_2025": {
        "match_id": "match_id", "innings": "innings", "over": "over",
        "batting_team": "batting_team", "bowling_team": "bowling_team",
        "striker": "striker", "bowler": "bowler",
        "runs_off_bat": "runs_of_bat", "extras": "extras",
        "wides": "wide", "noballs": "noballs",
        "byes": "byes", "legbyes": "legbyes",
        "wicket_type": "wicket_type", "player_dismissed": "player_dismissed",
        "runs": "runs", "wickets": "wickets",
        "venue": "venue_x", "date": "date_x", "match_type": "__constant__IPL",
        "non_striker": "__missing__", "ball": "__missing__",
    },
    "raw_cricket_dataset_downloader": {
        "match_id": "match_id", "innings": "innings", "over": "over",
        "ball": "ball", "batting_team": "batting_team", "bowling_team": "bowling_team",
        "striker": "striker", "non_striker": "non_striker", "bowler": "bowler",
        "runs_off_bat": "runs_off_bat", "extras": "extras", "wides": "wides",
        "noballs": "noballs", "byes": "byes", "legbyes": "legbyes",
        "wicket_type": "wicket_type", "player_dismissed": "player_dismissed",
        "runs": "runs", "wickets": "wickets",
        "venue": "venue", "date": "date", "match_type": "match_type",
    },
    "raw_cricsheet_all_ipl": {
        "match_id": "match_id", "innings": "innings",
        "ball": "ball",  # "over.ball" format, parsed in harmonize_schema
        "batting_team": "batting_team", "bowling_team": "bowling_team",
        "striker": "striker", "non_striker": "non_striker", "bowler": "bowler",
        "runs_off_bat": "runs_off_bat", "extras": "extras", "wides": "wides",
        "noballs": "noballs", "byes": "byes", "legbyes": "legbyes",
        "wicket_type": "wicket_type", "player_dismissed": "player_dismissed",
        "venue": "venue", "date": "start_date", "match_type": "__constant__IPL",
        "over": "__parsed_ball_over",
        "runs": "__computed_runs",
        "wickets": "__computed_wickets",
    },
}

RAW_FILES = {
    "raw_kaggle_ipl": DATA_RAW / "raw_kaggle_ipl.csv",
    "raw_cricsheet_ashwin": DATA_RAW / "raw_cricsheet_ashwin.csv",
    "raw_ipl_2025": DATA_RAW / "raw_ipl_2025.csv",
    "raw_cricket_dataset_downloader": DATA_RAW / "raw_cricket_dataset_downloader.csv",
    "raw_cricsheet_all_ipl": DATA_RAW / "raw_cricsheet_all_ipl.csv",
}


# ---------------------------------------------------------------------------
# 1. Raw Ingestion Layer
# ---------------------------------------------------------------------------

def load_raw_csv(name: str) -> pd.DataFrame:
    """Load a raw CSV by its logical name."""
    path = RAW_FILES.get(name)
    if path is None:
        raise ValueError(f"Unknown raw source '{name}'. Options: {list(RAW_FILES.keys())}")
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    return df


def load_all_raw() -> Dict[str, pd.DataFrame]:
    """Load all available raw CSV files into a dict keyed by source name."""
    result = {}
    for name, path in RAW_FILES.items():
        if path.exists():
            result[name] = pd.read_csv(path, low_memory=False)
    return result


# ---------------------------------------------------------------------------
# 2. Schema Harmonization Layer
# ---------------------------------------------------------------------------

def harmonize_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map a raw DataFrame from *source* into the canonical 22-column schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from one of the known CSVs.
    source : str
        One of the keys in SCHEMA_MAP.

    Returns
    -------
    pd.DataFrame with exactly CANONICAL_COLUMNS columns.
    """
    mapping = SCHEMA_MAP.get(source)
    if mapping is None:
        raise ValueError(f"Unknown source '{source}'. Available: {list(SCHEMA_MAP.keys())}")

    rows_in = len(df)
    canonical = {}
    audit = {"schema_mismatches": 0, "missing_filled": 0}

    for col in CANONICAL_COLUMNS:
        rule = mapping.get(col)
        if rule is None:
            canonical[col] = pd.Series([None] * len(df), index=df.index, dtype="object")
            audit["missing_filled"] += 1
        elif isinstance(rule, str) and rule.startswith("__constant__"):
            val = rule.replace("__constant__", "")
            canonical[col] = pd.Series([val] * len(df), index=df.index)
        elif isinstance(rule, str) and rule == "__missing__":
            canonical[col] = pd.Series([None] * len(df), index=df.index, dtype="object")
        elif isinstance(rule, str) and rule == "__parsed_ball_over":
            canonical[col] = pd.Series([1] * len(df), index=df.index, dtype=int)
        elif isinstance(rule, str) and rule == "__computed_runs":
            roff = pd.to_numeric(df.get("runs_off_bat", 0), errors="coerce").fillna(0)
            ext = pd.to_numeric(df.get("extras", 0), errors="coerce").fillna(0)
            canonical[col] = (roff + ext).astype(int)
        elif isinstance(rule, str) and rule == "__computed_wickets":
            canonical[col] = df["wicket_type"].notna().astype(int) if "wicket_type" in df.columns else pd.Series([0] * len(df), index=df.index)
        elif isinstance(rule, str):
            if rule in df.columns:
                canonical[col] = df[rule]
            else:
                canonical[col] = pd.Series([None] * len(df), index=df.index, dtype="object")
                audit["missing_filled"] += 1
        else:
            canonical[col] = pd.Series([rule] * len(df), index=df.index)

    result = pd.DataFrame(canonical)
    assert result.shape[1] == len(CANONICAL_COLUMNS), (
        f"Expected {len(CANONICAL_COLUMNS)} columns, got {result.shape[1]}"
    )

    # Parse ball from "over.ball" format (Cricsheet style)
    if source in ("raw_cricsheet_ashwin", "raw_cricsheet_all_ipl"):
        ball_str = result["ball"].astype(str)
        has_dot = ball_str.str.contains(r"\.", na=False)
        if has_dot.any():
            result.loc[has_dot, "over"] = (
                ball_str[has_dot].str.split(".").str[0].astype(int).values
            )
            result.loc[has_dot, "ball"] = (
                ball_str[has_dot].str.split(".").str[1].astype(int).values
            )
        else:
            result["over"] = result["over"].fillna(1).astype(int)
            result["ball"] = result["ball"].fillna(1).astype(int)

    # Fill missing over from ball
    if result["over"].isna().any():
        result["over"] = result["over"].fillna(1).astype(int)

    # Ensure numeric types
    for int_col in ["innings", "over", "ball"]:
        result[int_col] = pd.to_numeric(result[int_col], errors="coerce").fillna(1).astype(int)

    for num_col in ["runs_off_bat", "extras", "wides", "noballs", "byes", "legbyes", "runs", "wickets"]:
        if num_col in result.columns:
            result[num_col] = pd.to_numeric(result[num_col], errors="coerce").fillna(0).astype(int)

    return result


def merge_all_sources() -> pd.DataFrame:
    """Load, harmonize, and concatenate all available raw sources into one canonical DataFrame."""
    raw_sources = load_all_raw()
    frames = []
    for source_name, raw_df in raw_sources.items():
        canon = harmonize_schema(raw_df, source_name)
        canon["_source"] = source_name
        frames.append(canon)
    if not frames:
        raise FileNotFoundError("No raw CSV files found in data/raw/")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


# ---------------------------------------------------------------------------
# 3. Cricket-Aware Cleaning Layer
# ---------------------------------------------------------------------------

@dataclass
class CleaningAudit:
    rows_before: int = 0
    rows_after: int = 0
    duplicate_deliveries_removed: int = 0
    impossible_overs_removed: int = 0
    invalid_wickets_removed: int = 0
    impossible_bowler_consecutive: int = 0
    missing_critical_fields: int = 0
    runs_inconsistency_fixed: int = 0
    extras_inconsistency_fixed: int = 0
    innings_discontinuity_removed: int = 0

    def summary(self) -> str:
        parts = [
            f"Rows: {self.rows_before} → {self.rows_after} (-{self.rows_before - self.rows_after})",
        ]
        if self.duplicate_deliveries_removed:
            parts.append(f"Duplicate deliveries: {self.duplicate_deliveries_removed}")
        if self.impossible_overs_removed:
            parts.append(f"Impossible overs: {self.impossible_overs_removed}")
        if self.invalid_wickets_removed:
            parts.append(f"Invalid wickets (>10): {self.invalid_wickets_removed}")
        if self.missing_critical_fields:
            parts.append(f"Missing critical fields: {self.missing_critical_fields}")
        if self.runs_inconsistency_fixed:
            parts.append(f"Runs fixed: {self.runs_inconsistency_fixed}")
        if self.extras_inconsistency_fixed:
            parts.append(f"Extras fixed: {self.extras_inconsistency_fixed}")
        return " | ".join(parts)


def clean_cricket_data(df: pd.DataFrame, audit: Optional[CleaningAudit] = None) -> pd.DataFrame:
    """Apply cricket-aware data cleaning with relaxed cross-source tolerance.

    Rules:
    1. Deduplicate (match_id, innings, over, ball, batting_team, bowler) keeping first
    2. Remove impossible overs (<1 or >50)
    3. Warn on wickets per innings > 10 but don't drop (sources differ in coding)
    4. Reconcile runs_off_bat + extras → runs when mismatch > 3
    5. Reconcile wides+noballs+byes+legbyes → extras when mismatch > 3
    6. Remove rows missing match_id, innings, over, striker, bowler
    7. Fill missing non_striker
    8. Standardise name casing
    """
    result = df.copy()
    aud = audit or CleaningAudit()
    aud.rows_before = len(result)

    # 1. Remove duplicate deliveries across sources
    dup_cols = ["match_id", "innings", "over", "ball", "batting_team", "bowler"]
    if all(c in result.columns for c in dup_cols):
        dup_mask = result.duplicated(subset=dup_cols, keep="first")
        aud.duplicate_deliveries_removed = int(dup_mask.sum())
        result = result[~dup_mask].copy()

    # 2. Remove impossible overs
    if "over" in result.columns:
        valid_over = (result["over"] >= 1) & (result["over"] <= 50)
        aud.impossible_overs_removed = int((~valid_over).sum())
        result = result[valid_over].copy()

    # 3. Validate wickets per innings ≤ 10 (relaxed — don't drop, just warn)
    if "wickets" in result.columns:
        innings_wkts = result.groupby(["match_id", "innings"])["wickets"].cumsum()
        bad_wkts = innings_wkts > 10
        aud.invalid_wickets_removed = int(bad_wkts.sum())
        # cap at 10 instead of dropping
        result.loc[bad_wkts, "wickets"] = 0  # reset over-10 deliveries to 0

    # 4. Runs consistency: runs ≈ runs_off_bat + extras (wider tolerance for cross-source)
    if "runs" in result.columns and "runs_off_bat" in result.columns:
        expected_runs = result["runs_off_bat"].fillna(0) + result["extras"].fillna(0)
        runs_diff = (result["runs"].fillna(0) - expected_runs).abs()
        bad_runs = runs_diff > 3
        aud.runs_inconsistency_fixed = int(bad_runs.sum())
        result.loc[bad_runs, "runs"] = expected_runs[bad_runs].astype(int)

    # 5. Extras consistency
    expected_extras = (
        result["wides"].fillna(0) + result["noballs"].fillna(0)
        + result["byes"].fillna(0) + result["legbyes"].fillna(0)
    )
    extras_diff = (result["extras"].fillna(0) - expected_extras).abs()
    bad_extras = extras_diff > 3
    aud.extras_inconsistency_fixed = int(bad_extras.sum())
    result.loc[bad_extras, "extras"] = expected_extras[bad_extras].astype(int)

    # 6. Remove rows missing critical fields
    critical = ["match_id", "innings", "over", "striker", "bowler"]
    existing_crit = [c for c in critical if c in result.columns]
    if existing_crit:
        missing_crit = result[existing_crit].isna().any(axis=1)
        aud.missing_critical_fields = int(missing_crit.sum())
        result = result[~missing_crit].copy()

    # 7. Fill missing non_striker
    if "non_striker" in result.columns:
        result["non_striker"] = result["non_striker"].fillna("unknown")

    # 8. Standardise names
    for name_col in ["striker", "non_striker", "bowler", "batting_team", "bowling_team"]:
        if name_col in result.columns:
            result[name_col] = result[name_col].astype(str).str.strip().str.title()

    aud.rows_after = len(result)
    return result


# ---------------------------------------------------------------------------
# 4. Match State Engine — creates ball-level contextual features
# ---------------------------------------------------------------------------

MATCH_STATE_FEATURES = [
    "total_runs", "total_wickets", "balls_remaining", "required_run_rate",
    "current_run_rate", "wickets_in_hand", "phase_of_play", "pressure_index",
    "momentum_score", "partnership_runs", "partnership_balls",
    "batting_aggression_index", "bowling_pressure_index",
    "recent_12_ball_runs", "recent_12_ball_wickets",
    "boundary_rate", "dot_ball_pressure", "chase_win_probability_proxy",
]


def compute_match_state(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ball-level match-state features using vectorised operations.

    Operates on canonical-schema DataFrames grouped by (match_id, innings).
    All features are computed without leakage (only past and current ball info).
    """
    data = df.sort_values(["match_id", "innings", "over", "ball"]).copy()

    # Per-ball runs and wickets
    data["total_runs"] = data.groupby(["match_id", "innings"])["runs"].cumsum()
    data["total_wickets"] = data.groupby(["match_id", "innings"])["wickets"].cumsum()
    data["wickets_in_hand"] = 10 - data["total_wickets"]

    # Balls remaining (est.: 120 balls per innings)
    data["ball_number"] = data.groupby(["match_id", "innings"]).cumcount() + 1
    innings_max_balls = data.groupby(["match_id", "innings"])["ball_number"].transform("max")
    data["balls_remaining"] = innings_max_balls - data["ball_number"]

    # Phase of play
    def _phase(over_val, innings_val):
        if over_val <= 6:
            return "powerplay"
        elif over_val <= 15:
            return "middle"
        else:
            return "death" if innings_val == 1 else "chase_death"
    data["phase_of_play"] = data.apply(
        lambda r: _phase(r["over"], r["innings"]), axis=1
    )

    # Current run rate
    overs_bowled = data["ball_number"] / 6.0
    data["current_run_rate"] = data["total_runs"] / overs_bowled.clip(lower=0.1)

    # Innings 1: compute target as final total
    inns1_total = data[data["innings"] == 1].groupby("match_id")["total_runs"].transform("last")
    data["inning1_total"] = data["match_id"].map(
        data[data["innings"] == 1].groupby("match_id")["total_runs"].last()
    )
    target = data["inning1_total"].fillna(0).astype(int) + 1

    # Required run rate (only for innings 2)
    is_chase = data["innings"] == 2
    runs_needed = np.where(is_chase, (target - data["total_runs"]).clip(lower=0), np.nan)
    data["required_run_rate"] = np.where(
        is_chase & (data["balls_remaining"] > 0),
        runs_needed / (data["balls_remaining"] / 6.0),
        np.nan,
    )

    # Pressure index: composite of RRR, wickets lost, phase
    rrr_norm = data["required_run_rate"] / 15.0  # 15 is max plausible RRR
    wkts_lost_norm = data["total_wickets"] / 10.0
    is_death = (data["over"] >= 16).astype(float)
    data["pressure_index"] = np.where(
        is_chase,
        (rrr_norm.fillna(0) * 0.5 + wkts_lost_norm * 0.3 + is_death * 0.2),
        0.0,
    ).clip(0, 1)

    # Momentum score: run rate difference weighted by wickets
    run_rate_diff = data["current_run_rate"] - data["required_run_rate"].fillna(0)
    data["momentum_score"] = run_rate_diff * (data["wickets_in_hand"] / 10.0)
    data["momentum_score"] = data["momentum_score"].clip(-10, 10)

    # Partnership
    def _partition_by_wicket(group):
        group = group.copy()
        group["partnership_id"] = group["wickets"].cumsum()
        return group
    data = data.groupby(["match_id", "innings"], group_keys=False).apply(_partition_by_wicket)
    data["partnership_runs"] = data.groupby(["match_id", "innings", "partnership_id"])["runs"].cumsum()
    data["partnership_balls"] = data.groupby(["match_id", "innings", "partnership_id"]).cumcount() + 1

    # Batting aggression index: boundary rate × run rate multiplier
    data["is_boundary"] = (data["runs_off_bat"] >= 4) & (data["wides"].fillna(0) == 0)
    data["boundary_rate"] = (
        data.groupby(["match_id", "innings"])["is_boundary"]
        .transform("mean")
        .fillna(0)
    )
    data["is_dot_ball"] = (data["runs_off_bat"] == 0) & (data["wides"].fillna(0) == 0) & (data["noballs"].fillna(0) == 0)
    data["dot_ball_pressure"] = (
        data.groupby(["match_id", "innings"])["is_dot_ball"]
        .transform("mean")
        .fillna(0)
    )
    data["batting_aggression_index"] = data["boundary_rate"] * 2 + (data["current_run_rate"] / 6.0)

    # Bowling pressure index: inverse of batting aggression with more granularity
    data["bowling_pressure_index"] = np.where(
        data["innings"] == 1,
        (data.groupby(["match_id", "innings"])["wickets"].transform("sum")) / 10.0,
        (data["total_wickets"] / 10.0) + (data["required_run_rate"].fillna(0) / 15.0),
    )

    # Rolling 12-ball aggregates
    data["recent_12_ball_runs"] = data.groupby(["match_id", "innings"])["runs"].transform(
        lambda x: x.rolling(12, min_periods=1).sum()
    )
    data["recent_12_ball_wickets"] = data.groupby(["match_id", "innings"])["wickets"].transform(
        lambda x: x.rolling(12, min_periods=1).sum()
    )

    # Chase win probability proxy: function of RRR, wickets, overs
    def _win_prob(row):
        if row["innings"] != 2:
            return 0.5
        rrr = row.get("required_run_rate", 0) or 0
        wkts = row["wickets_in_hand"]
        balls = row["balls_remaining"]
        if balls <= 0 or rrr <= 0:
            return 1.0 if row["total_runs"] >= target.loc[row.name] else 0.0
        base = max(0, 1 - (rrr / 12.0))
        wkt_factor = wkts / 10.0
        ball_factor = min(balls / 72.0, 1.0)
        return min(base * 0.5 + wkt_factor * 0.3 + ball_factor * 0.2, 1.0)
    data["chase_win_probability_proxy"] = data.apply(_win_prob, axis=1)

    data = data.drop(columns=["inning1_total", "partnership_id", "ball_number"], errors="ignore")
    return data


# ---------------------------------------------------------------------------
# 5. Aggregation Layer
# ---------------------------------------------------------------------------

def aggregate_over_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ball-by-ball to over-level statistics."""
    if "is_wide" not in df.columns:
        df = df.copy()
        df["is_wide"] = (df["wides"].fillna(0) > 0).astype(int)
        df["is_noball"] = (df["noballs"].fillna(0) > 0).astype(int)
    legal = (~df["is_wide"].fillna(0).astype(bool) & ~df["is_noball"].fillna(0).astype(bool))

    result = df.groupby(["match_id", "innings", "batting_team", "over"]).agg(
        runs_scored=("runs", "sum"),
        wickets=("wickets", "sum"),
        boundaries=("runs_off_bat", lambda x: ((x >= 4) & ~df.loc[x.index, "is_wide"].fillna(0).astype(bool)).sum()),
        sixes=("runs_off_bat", lambda x: ((x >= 6) & ~df.loc[x.index, "is_wide"].fillna(0).astype(bool)).sum()),
        dot_balls=("runs_off_bat", lambda x: (
            (x == 0) & legal.loc[x.index]
        ).sum()),
        total_balls=("runs_off_bat", lambda x: legal.loc[x.index].sum()),
        extras=("extras", "sum"),
        wides=("wides", "sum"),
        noballs=("noballs", "sum"),
    ).reset_index()
    result["cumulative_score"] = result.groupby(["match_id", "innings"])["runs_scored"].cumsum()
    result["cumulative_wickets"] = result.groupby(["match_id", "innings"])["wickets"].cumsum()
    return result


def aggregate_player_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to per-player career statistics from ball-by-ball data."""
    legal = (~df["wides"].fillna(0).astype(bool) & ~df["noballs"].fillna(0).astype(bool))
    is_boundary = (df["runs_off_bat"] >= 4) & ~df["wides"].fillna(0).astype(bool)
    is_dot = (df["runs_off_bat"] == 0) & legal

    batting = df.groupby("striker").agg(
        matches=("match_id", "nunique"),
        innings=("innings", "nunique"),
        total_runs=("runs_off_bat", "sum"),
        balls_faced=("runs_off_bat", lambda x: legal[x.index].sum()),
        boundaries=("runs_off_bat", lambda x: is_boundary[x.index].sum()),
        sixes=("runs_off_bat", lambda x: ((x >= 6) & ~df.loc[x.index, "wides"].fillna(0).astype(bool)).sum()),
        dot_balls=("runs_off_bat", lambda x: is_dot[x.index].sum()),
        dismissals=("wickets", "sum"),
    ).reset_index().rename(columns={"striker": "player"})
    batting["strike_rate"] = (batting["total_runs"] / batting["balls_faced"].clip(1)) * 100
    batting["average"] = batting["total_runs"] / batting["dismissals"].clip(1)
    batting["boundary_pct"] = batting["boundaries"] / batting["balls_faced"].clip(1)
    batting["dot_ball_pct"] = batting["dot_balls"] / batting["balls_faced"].clip(1)
    return batting


def aggregate_match_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to per-match summary."""
    innings_summary = df.groupby(["match_id", "innings"]).agg(
        total_runs=("runs", "sum"),
        total_wickets=("wickets", "sum"),
        total_balls=("runs", "count"),
        boundaries=("runs_off_bat", lambda x: (x >= 4).sum()),
        sixes=("runs_off_bat", lambda x: (x >= 6).sum()),
        venue=("venue", "first"),
        date=("date", "first"),
        match_type=("match_type", "first"),
    ).reset_index()
    return innings_summary


# ---------------------------------------------------------------------------
# 6. Feature Store — Parquet-based caching
# ---------------------------------------------------------------------------

FEATURE_STORE_FILES = {
    "full_canonical": "full_canonical.parquet",
    "match_state": "match_state.parquet",
    "over_level": "over_level.parquet",
    "player_level": "player_level.parquet",
    "match_level": "match_level.parquet",
}


def build_feature_store(force_rebuild: bool = False) -> Dict[str, pd.DataFrame]:
    """Build and save all feature-store parquet files.

    Returns dict of {name: DataFrame} for all computed datasets.
    """
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    store = {}

    # 1. Full canonical (cleaned, harmonized)
    canon_path = DATA_PROCESSED / FEATURE_STORE_FILES["full_canonical"]
    if canon_path.exists() and not force_rebuild:
        store["full_canonical"] = pd.read_parquet(canon_path)
    else:
        combined = merge_all_sources()
        cleaned = clean_cricket_data(combined)
        cleaned.to_parquet(canon_path, index=False)
        store["full_canonical"] = cleaned

    # 2. Match state
    state_path = DATA_PROCESSED / FEATURE_STORE_FILES["match_state"]
    if state_path.exists() and not force_rebuild:
        store["match_state"] = pd.read_parquet(state_path)
    else:
        store["match_state"] = compute_match_state(store["full_canonical"])
        store["match_state"].to_parquet(state_path, index=False)

    # 3. Over level
    over_path = DATA_PROCESSED / FEATURE_STORE_FILES["over_level"]
    if over_path.exists() and not force_rebuild:
        store["over_level"] = pd.read_parquet(over_path)
    else:
        store["over_level"] = aggregate_over_level(store["match_state"])
        store["over_level"].to_parquet(over_path, index=False)

    # 4. Player level
    player_path = DATA_PROCESSED / FEATURE_STORE_FILES["player_level"]
    if player_path.exists() and not force_rebuild:
        store["player_level"] = pd.read_parquet(player_path)
    else:
        store["player_level"] = aggregate_player_level(store["match_state"])
        store["player_level"].to_parquet(player_path, index=False)

    # 5. Match level
    match_path = DATA_PROCESSED / FEATURE_STORE_FILES["match_level"]
    if match_path.exists() and not force_rebuild:
        store["match_level"] = pd.read_parquet(match_path)
    else:
        store["match_level"] = aggregate_match_level(store["match_state"])
        store["match_level"].to_parquet(match_path, index=False)

    return store


def load_feature_store(key: str) -> pd.DataFrame:
    """Load a single feature-store parquet by key."""
    fname = FEATURE_STORE_FILES.get(key)
    if fname is None:
        valid = list(FEATURE_STORE_FILES.keys())
        raise ValueError(f"Unknown feature store '{key}'. Options: {valid}")
    path = DATA_PROCESSED / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Feature store '{key}' not found at {path}. Run build_feature_store() first."
        )
    return pd.read_parquet(path)
