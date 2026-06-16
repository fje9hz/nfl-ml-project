from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from nfl_ml.features import add_matchup_features


DEFAULT_MODEL_PATH = Path("models/nfl_win_model.joblib")
DEFAULT_METRICS_PATH = Path("reports/metrics.json")
DEFAULT_IMPORTANCE_PATH = Path("reports/feature_importance.csv")


def confidence_label(probability: float) -> str:
    edge = abs(probability - 0.5)
    if edge >= 0.2:
        return "High"
    if edge >= 0.1:
        return "Medium"
    return "Low"


@lru_cache(maxsize=4)
def load_model_artifact(model_path: str = str(DEFAULT_MODEL_PATH)) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Train first with python -m nfl_ml.train."
        )
    return joblib.load(path)


def predict_matchup(input_data: dict[str, Any], model_path: str = str(DEFAULT_MODEL_PATH)) -> dict[str, Any]:
    artifact = load_model_artifact(model_path)
    row = pd.DataFrame([input_data])
    feature_columns = artifact["feature_columns"]
    features = add_matchup_features(row)[feature_columns]
    probability = float(artifact["pipeline"].predict_proba(features)[:, 1][0])
    away_probability = 1 - probability
    home_team = str(input_data["home_team"]).upper()
    away_team = str(input_data["away_team"]).upper()
    pick = home_team if probability >= 0.5 else away_team

    return {
        "matchup": f"{away_team} at {home_team}",
        "model": artifact["model_name"],
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


def load_metrics(metrics_path: str = str(DEFAULT_METRICS_PATH)) -> dict[str, Any]:
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found at {path}. Train the model first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_importance(
    importance_path: str = str(DEFAULT_IMPORTANCE_PATH),
) -> list[dict[str, Any]]:
    path = Path(importance_path)
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
