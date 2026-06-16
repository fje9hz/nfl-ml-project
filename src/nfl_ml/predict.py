from __future__ import annotations

import argparse
import json

from nfl_ml.service import predict_matchup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict an NFL home-team win probability.")
    parser.add_argument("--model", default="models/nfl_win_model.joblib")
    parser.add_argument("--model-mode", choices=["market", "team"], default="market")
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
    result = predict_matchup(
        {
            "home_team": args.home_team,
            "away_team": args.away_team,
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
        },
        model_mode=args.model_mode,
        model_path=args.model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
