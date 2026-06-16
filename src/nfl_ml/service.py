from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from nfl_ml.features import add_matchup_features, add_team_stat_matchup_features


MODEL_PATHS = {
    "market": Path("models/nfl_win_model.joblib"),
    "team": Path("models/nfl_team_model.joblib"),
}
METRICS_PATHS = {
    "market": Path("reports/metrics.json"),
    "team": Path("reports/team_metrics.json"),
}
IMPORTANCE_PATHS = {
    "market": Path("reports/feature_importance.csv"),
    "team": Path("reports/team_feature_importance.csv"),
}


def confidence_label(probability: float) -> str:
    edge = abs(probability - 0.5)
    if edge >= 0.2:
        return "High"
    if edge >= 0.1:
        return "Medium"
    return "Low"


@lru_cache(maxsize=4)
def load_model_artifact(model_path: str) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Train first with python -m nfl_ml.train."
        )
    return joblib.load(path)


def normalize_model_mode(model_mode: str | None) -> str:
    mode = (model_mode or "market").lower()
    if mode not in MODEL_PATHS:
        raise ValueError(f"Unsupported model mode: {model_mode}")
    return mode


def build_prediction_features(
    input_data: dict[str, Any],
    artifact: dict[str, Any],
    model_mode: str,
) -> pd.DataFrame:
    row = pd.DataFrame([input_data])
    feature_columns = artifact["feature_columns"]
    if model_mode == "team":
        return add_team_stat_matchup_features(row, artifact["team_profiles"])[feature_columns]
    return add_matchup_features(row)[feature_columns]


def predict_matchup(
    input_data: dict[str, Any],
    model_mode: str | None = "market",
    model_path: str | None = None,
) -> dict[str, Any]:
    mode = normalize_model_mode(model_mode)
    artifact_path = str(model_path or MODEL_PATHS[mode])
    artifact = load_model_artifact(artifact_path)
    feature_columns = artifact["feature_columns"]
    features = build_prediction_features(input_data, artifact, mode)
    probability = float(artifact["pipeline"].predict_proba(features)[:, 1][0])
    away_probability = 1 - probability
    home_team = str(input_data["home_team"]).upper()
    away_team = str(input_data["away_team"]).upper()
    pick = home_team if probability >= 0.5 else away_team

    return {
        "matchup": f"{away_team} at {home_team}",
        "model": artifact["model_name"],
        "model_mode": mode,
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": round(probability, 4),
        "away_win_probability": round(away_probability, 4),
        "pick": pick,
        "confidence": confidence_label(probability),
        "features": {
            column: round(float(features.iloc[0][column]), 4) for column in feature_columns
        },
    }


def load_metrics(
    model_mode: str | None = "market",
    metrics_path: str | None = None,
) -> dict[str, Any]:
    mode = normalize_model_mode(model_mode)
    path = Path(metrics_path) if metrics_path else METRICS_PATHS[mode]
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found at {path}. Train the model first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_importance(
    model_mode: str | None = "market",
    importance_path: str | None = None,
) -> list[dict[str, Any]]:
    mode = normalize_model_mode(model_mode)
    path = Path(importance_path) if importance_path else IMPORTANCE_PATHS[mode]
    if not path.exists():
        raise FileNotFoundError(
            f"Feature importance file not found at {path}. Train the model first."
        )
    rows = pd.read_csv(path).to_dict(orient="records")
    return [
        {
            "feature": str(row["feature"]),
            "importance": round(float(row["importance"]), 6),
            "model": str(row["model"]),
        }
        for row in rows
    ]
