from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nfl_ml.service import load_feature_importance, load_metrics, predict_matchup


ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT_DIR / "web"


class MatchupRequest(BaseModel):
    model_mode: str = "market"
    home_team: str = Field(..., min_length=2, max_length=3)
    away_team: str = Field(..., min_length=2, max_length=3)
    spread_line: float = 0
    total_line: float = Field(44, ge=20, le=80)
    home_rest: float = Field(..., ge=0, le=30)
    away_rest: float = Field(..., ge=0, le=30)
    home_moneyline: float = -110
    away_moneyline: float = -110
    div_game: int = Field(..., ge=0, le=1)
    roof: str = "outdoors"
    surface: str = "grass"
    temp: float = Field(70, ge=-20, le=120)
    wind: float = Field(0, ge=0, le=60)


app = FastAPI(
    title="Game Script",
    description="Interactive API for the NFL machine learning project.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics(model_mode: str = "market") -> dict:
    return load_metrics(model_mode)


@app.get("/api/feature-importance")
def feature_importance(model_mode: str = "market") -> list[dict]:
    return load_feature_importance(model_mode)


@app.post("/api/predict")
def predict(payload: MatchupRequest) -> dict:
    data = payload.dict()
    return predict_matchup(data, model_mode=data.pop("model_mode", "market"))
