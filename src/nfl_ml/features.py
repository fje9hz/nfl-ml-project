from __future__ import annotations

import pandas as pd


RAW_FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "home_rest_days",
    "away_rest_days",
    "home_offense_epa",
    "away_offense_epa",
    "home_defense_epa",
    "away_defense_epa",
    "home_turnover_margin",
    "away_turnover_margin",
    "home_injury_score",
    "away_injury_score",
    "is_division_game",
]

MODEL_FEATURE_COLUMNS = [
    "elo_diff",
    "rest_diff",
    "offense_epa_diff",
    "defense_epa_diff",
    "turnover_margin_diff",
    "injury_score_diff",
    "is_division_game",
]

TARGET_COLUMN = "home_win"


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create home-minus-away matchup features from raw team inputs."""
    validate_columns(df, RAW_FEATURE_COLUMNS)

    featured = df.copy()
    featured["elo_diff"] = featured["home_elo"] - featured["away_elo"]
    featured["rest_diff"] = featured["home_rest_days"] - featured["away_rest_days"]
    featured["offense_epa_diff"] = (
        featured["home_offense_epa"] - featured["away_offense_epa"]
    )
    featured["defense_epa_diff"] = (
        featured["away_defense_epa"] - featured["home_defense_epa"]
    )
    featured["turnover_margin_diff"] = (
        featured["home_turnover_margin"] - featured["away_turnover_margin"]
    )
    featured["injury_score_diff"] = (
        featured["away_injury_score"] - featured["home_injury_score"]
    )
    return featured


def build_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_columns(df, RAW_FEATURE_COLUMNS + [TARGET_COLUMN])
    featured = add_matchup_features(df)
    x = featured[MODEL_FEATURE_COLUMNS]
    y = featured[TARGET_COLUMN].astype(int)
    return x, y
