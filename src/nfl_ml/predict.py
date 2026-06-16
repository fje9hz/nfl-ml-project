from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from nfl_ml.features import add_matchup_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict an NFL home-team win probability.")
    parser.add_argument("--model", default="models/nfl_win_model.joblib")
    parser.add_argument("--home-team", required=True)
    parser.add_argument("--away-team", required=True)
    parser.add_argument("--spread-line", type=float, required=True)
    parser.add_argument("--total-line", type=float, required=True)
    parser.add_argument("--home-rest", type=float, required=True)
    parser.add_argument("--away-rest", type=float, required=True)
    parser.add_argument("--home-moneyline", type=float, required=True)
    parser.add_argument("--away-moneyline", type=float, required=True)
    parser.add_argument("--div-game", type=int, choices=[0, 1], required=True)
    parser.add_argument("--roof", default="outdoors")
    parser.add_argument("--surface", default="grass")
    parser.add_argument("--temp", type=float, default=70)
    parser.add_argument("--wind", type=float, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train first with python -m nfl_ml.train."
        )

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]

    row = pd.DataFrame(
        [
            {
                "spread_line": args.spread_line,
                "total_line": args.total_line,
                "home_rest": args.home_rest,
                "away_rest": args.away_rest,
                "home_moneyline": args.home_moneyline,
                "away_moneyline": args.away_moneyline,
                "div_game": args.div_game,
                "roof": args.roof,
                "surface": args.surface,
                "temp": args.temp,
                "wind": args.wind,
            }
        ]
    )
    feature_columns = artifact["feature_columns"]
    x = add_matchup_features(row)[feature_columns]
    probability = float(pipeline.predict_proba(x)[:, 1][0])

    result = {
        "matchup": f"{args.away_team} at {args.home_team}",
        "model": artifact["model_name"],
        "home_win_probability": round(probability, 4),
        "away_win_probability": round(1 - probability, 4),
        "pick": args.home_team if probability >= 0.5 else args.away_team,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
