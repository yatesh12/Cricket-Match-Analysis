"""CricketIQ — FastAPI Service.

Exposes model predictions as production-ready REST endpoints:
  - POST /recommend-substitution (Module 2)
  - POST /churn-score (Module 4)
  - GET /pressure-compatibility (Module 1)
  - GET /match-hot-zones (Module 3)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
import numpy as np

from src.data_loader import CricketDataLoader
from src.pressure_genome import PressureGenome, PressureGenomeConfig
from src.impact_player import ImpactPlayerAI, ImpactPlayerConfig, MatchState
from src.broadcast_monetisation import BroadcastMonetisation, BroadcastConfig
from src.fantasy_clv import FantasyChurnCLV, ChurnCLVConfig

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CricketIQ API",
    description="Enterprise Cricket Analytics — Pressure Genome, Impact Player AI, "
                "Broadcast Monetisation, Fantasy CLV",
    version="1.0.0",
)

# Lazy-loaded models
_models: Dict[str, Any] = {}


def get_model(name: str):
    """Get or initialize a model."""
    if name not in _models:
        loader = CricketDataLoader()
        if name == "pressure_genome":
            df = loader.get_pressure_features(200)
            g = PressureGenome(PressureGenomeConfig())
            g.fit(df)
            _models[name] = {"model": g, "data": df}
        elif name == "impact_player":
            ai = ImpactPlayerAI()
            ai.run_pipeline()
            _models[name] = {"model": ai}
        elif name == "broadcast":
            os = loader.get_over_stats()
            bc = BroadcastMonetisation(BroadcastConfig(epochs=20))
            bc.run_pipeline(os)
            _models[name] = {"model": bc}
        elif name == "fantasy":
            users = loader.get_fantasy_users(50000)
            fc = FantasyChurnCLV(ChurnCLVConfig())
            fc.run_pipeline(users)
            _models[name] = {"model": fc}
    return _models.get(name)


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class MatchStateRequest(BaseModel):
    current_score: float = Field(0, description="Current batting score")
    wickets_fallen: float = Field(0, description="Wickets lost so far")
    current_run_rate: float = Field(0, description="Current run rate")
    required_run_rate: float = Field(0, description="Required run rate (0 if 1st innings)")
    overs_remaining: float = Field(20, description="Overs remaining in innings")
    wickets_remaining: float = Field(10, description="Wickets in hand")
    is_first_innings: float = Field(1.0, description="1.0 for 1st inns, 0.0 for 2nd")
    is_pressure_situation: float = Field(0.0, description="1.0 if high-pressure context")

class SubstitutionRequest(BaseModel):
    match_state: MatchStateRequest
    available_players: List[str] = Field(..., description="Players eligible for substitution")

class ChurnScoreRequest(BaseModel):
    user_ids: List[str] = Field(..., description="List of user IDs to score", max_items=1000)

class PressureQueryRequest(BaseModel):
    required_run_rate: float = Field(10.0)
    wickets_left: float = Field(4)
    overs_remaining: float = Field(4.0)
    is_chase: bool = Field(True)
    top_n: int = Field(5, ge=1, le=20)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "CricketIQ API",
        "version": "1.0.0",
        "endpoints": {
            "POST /recommend-substitution": "Impact Player AI — optimal substitution timing + candidate",
            "POST /churn-score": "Fantasy CLV — user-level churn risk + predicted CLV",
            "GET /pressure-compatibility": "Pressure Genome — rank batsmen for match situation",
            "GET /match-hot-zones": "Broadcast Monetisation — predicted peak engagement windows",
            "GET /health": "Health check",
        },
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/recommend-substitution")
def recommend_substitution(req: SubstitutionRequest):
    """Recommend substitution timing and top candidates."""
    try:
        ai_data = get_model("impact_player")
        if ai_data is None:
            raise HTTPException(500, "Impact Player AI not loaded")
        ai = ai_data["model"]

        state = MatchState(
            current_score=req.match_state.current_score,
            wickets_fallen=req.match_state.wickets_fallen,
            current_run_rate=req.match_state.current_run_rate,
            required_run_rate=req.match_state.required_run_rate,
            overs_remaining=req.match_state.overs_remaining,
            wickets_remaining=req.match_state.wickets_remaining,
            is_first_innings=req.match_state.is_first_innings,
            is_pressure_situation=req.match_state.is_pressure_situation,
        )

        rec = ai.recommend_substitution(state, req.available_players)
        return {"success": True, "data": rec}

    except Exception as e:
        raise HTTPException(500, f"Substitution recommendation failed: {str(e)}")


@app.post("/churn-score")
def churn_score(req: ChurnScoreRequest):
    """Return churn risk score and CLV for requested users."""
    try:
        fc_data = get_model("fantasy")
        if fc_data is None:
            raise HTTPException(500, "Fantasy CLV model not loaded")
        fc = fc_data["model"]

        rankings = fc.get_risk_rankings(top_n=len(req.user_ids))
        if rankings is None:
            return {"success": True, "data": []}

        result = rankings[rankings["user_id"].isin(req.user_ids)]
        return {
            "success": True,
            "data": result.to_dict("records"),
        }

    except Exception as e:
        raise HTTPException(500, f"Churn scoring failed: {str(e)}")


@app.get("/pressure-compatibility")
def pressure_compatibility(
    required_run_rate: float = Query(10.0),
    wickets_left: float = Query(4),
    overs_remaining: float = Query(4.0),
    is_chase: bool = Query(True),
    top_n: int = Query(5),
):
    """Rank available batsmen by pressure compatibility for a match situation."""
    try:
        pg_data = get_model("pressure_genome")
        if pg_data is None:
            raise HTTPException(500, "Pressure Genome not loaded")
        genome, df = pg_data["model"], pg_data["data"]

        match_state = {
            "required_run_rate": required_run_rate,
            "wickets_left": wickets_left,
            "overs_remaining": overs_remaining,
            "is_chase": is_chase,
        }

        rankings = genome.rank_for_situation(df, match_state, top_n=top_n)
        return {
            "success": True,
            "match_state": match_state,
            "recommendations": rankings.to_dict("records"),
        }

    except Exception as e:
        raise HTTPException(500, f"Pressure compatibility failed: {str(e)}")


@app.get("/match-hot-zones")
def match_hot_zones(match_id: str = Query(..., description="Match ID")):
    """Get predicted peak engagement windows for a match."""
    try:
        bc_data = get_model("broadcast")
        if bc_data is None:
            raise HTTPException(500, "Broadcast model not loaded")
        bc = bc_data["model"]

        report = bc.generate_match_report(match_id)
        if "error" in report:
            raise HTTPException(404, report["error"])

        return {"success": True, "data": report}

    except Exception as e:
        raise HTTPException(500, f"Hot zone report failed: {str(e)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
