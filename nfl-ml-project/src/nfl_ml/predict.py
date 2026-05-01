from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from nfl_ml.features import MODEL_FEATURE_COLUMNS, add_matchup_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict an NFL home-team win probability.")
    parser.add_argument("--model", default="models/nfl_win_model.joblib")
    parser.add_argument("--home-team", required=True)
    parser.add_argument("--away-team", required=True)
    parser.add_argument("--home-elo", type=float, required=True)
    parser.add_argument("--away-elo", type=float, required=True)
    parser.add_argument("--home-rest-days", type=float, required=True)
    parser.add_argument("--away-rest-days", type=float, required=True)
    parser.add_argument("--home-offense-epa", type=float, required=True)
    parser.add_argument("--away-offense-epa", type=float, required=True)
    parser.add_argument("--home-defense-epa", type=float, required=True)
    parser.add_argument("--away-defense-epa", type=float, required=True)
    parser.add_argument("--home-turnover-margin", type=float, required=True)
    parser.add_argument("--away-turnover-margin", type=float, required=True)
    parser.add_argument("--home-injury-score", type=float, required=True)
    parser.add_argument("--away-injury-score", type=float, required=True)
    parser.add_argument("--is-division-game", type=int, choices=[0, 1], required=True)
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
                "home_elo": args.home_elo,
                "away_elo": args.away_elo,
                "home_rest_days": args.home_rest_days,
                "away_rest_days": args.away_rest_days,
                "home_offense_epa": args.home_offense_epa,
                "away_offense_epa": args.away_offense_epa,
                "home_defense_epa": args.home_defense_epa,
                "away_defense_epa": args.away_defense_epa,
                "home_turnover_margin": args.home_turnover_margin,
                "away_turnover_margin": args.away_turnover_margin,
                "home_injury_score": args.home_injury_score,
                "away_injury_score": args.away_injury_score,
                "is_division_game": args.is_division_game,
            }
        ]
    )
    x = add_matchup_features(row)[MODEL_FEATURE_COLUMNS]
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
