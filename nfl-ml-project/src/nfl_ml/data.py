from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NFLVERSE_GAMES_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)

OUTPUT_COLUMNS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "home_win",
    "away_rest",
    "home_rest",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "total_line",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare real NFL game data from nflverse."
    )
    parser.add_argument("--out", default="data/nfl_games_real.csv")
    parser.add_argument("--start-season", type=int, default=2010)
    parser.add_argument("--end-season", type=int)
    parser.add_argument("--include-playoffs", action="store_true")
    parser.add_argument("--source-url", default=NFLVERSE_GAMES_URL)
    return parser.parse_args()


def load_nflverse_games(source_url: str = NFLVERSE_GAMES_URL) -> pd.DataFrame:
    return pd.read_csv(source_url, low_memory=False)


def prepare_games(
    games: pd.DataFrame,
    start_season: int = 2010,
    end_season: int | None = None,
    include_playoffs: bool = False,
) -> pd.DataFrame:
    required = sorted(set(OUTPUT_COLUMNS) - {"home_win"})
    missing = sorted(set(required) - set(games.columns))
    if missing:
        raise ValueError(f"Missing required nflverse columns: {', '.join(missing)}")

    prepared = games.copy()
    prepared = prepared[prepared["season"] >= start_season]
    if end_season is not None:
        prepared = prepared[prepared["season"] <= end_season]
    if not include_playoffs:
        prepared = prepared[prepared["game_type"] == "REG"]

    prepared = prepared.dropna(subset=["home_score", "away_score"])
    prepared["home_win"] = (prepared["home_score"] > prepared["away_score"]).astype(int)
    prepared = prepared[OUTPUT_COLUMNS].sort_values(["season", "week", "game_id"])
    return prepared.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_path = Path(args.out)

    games = load_nflverse_games(args.source_url)
    prepared = prepare_games(
        games,
        start_season=args.start_season,
        end_season=args.end_season,
        include_playoffs=args.include_playoffs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    print(
        f"Wrote {len(prepared):,} games from {prepared['season'].min()}-"
        f"{prepared['season'].max()} to {output_path}"
    )


if __name__ == "__main__":
    main()
