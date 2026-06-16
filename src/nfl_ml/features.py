from __future__ import annotations

import pandas as pd


SAMPLE_RAW_FEATURE_COLUMNS = [
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

SAMPLE_MODEL_FEATURE_COLUMNS = [
    "elo_diff",
    "rest_diff",
    "offense_epa_diff",
    "defense_epa_diff",
    "turnover_margin_diff",
    "injury_score_diff",
    "is_division_game",
]

REAL_RAW_FEATURE_COLUMNS = [
    "spread_line",
    "total_line",
    "home_rest",
    "away_rest",
    "home_moneyline",
    "away_moneyline",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
]

REAL_MODEL_FEATURE_COLUMNS = [
    "spread_line",
    "total_line",
    "rest_diff",
    "home_implied_prob",
    "implied_prob_diff",
    "div_game",
    "is_dome",
    "is_turf",
    "temp",
    "wind",
]

MODEL_FEATURE_COLUMNS = SAMPLE_MODEL_FEATURE_COLUMNS
TARGET_COLUMN = "home_win"


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def moneyline_to_implied_probability(value: float | int | None) -> float | None:
    if pd.isna(value) or value == 0:
        return None
    moneyline = float(value)
    if moneyline > 0:
        return 100 / (moneyline + 100)
    return abs(moneyline) / (abs(moneyline) + 100)


def add_sample_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create home-minus-away matchup features from raw team inputs."""
    validate_columns(df, SAMPLE_RAW_FEATURE_COLUMNS)

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


def add_real_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from real nflverse game rows."""
    validate_columns(df, REAL_RAW_FEATURE_COLUMNS)

    featured = df.copy()
    featured["rest_diff"] = featured["home_rest"] - featured["away_rest"]
    featured["home_implied_prob"] = featured["home_moneyline"].map(
        moneyline_to_implied_probability
    )
    featured["away_implied_prob"] = featured["away_moneyline"].map(
        moneyline_to_implied_probability
    )
    featured["implied_prob_diff"] = (
        featured["home_implied_prob"] - featured["away_implied_prob"]
    )
    featured["is_dome"] = featured["roof"].isin(["dome", "closed", "retractable"]).astype(int)
    featured["is_turf"] = (
        featured["surface"].fillna("").str.contains("turf", case=False, regex=False)
    ).astype(int)
    featured["div_game"] = featured["div_game"].astype(int)
    return featured


def detect_schema(df: pd.DataFrame) -> str:
    if set(REAL_RAW_FEATURE_COLUMNS).issubset(df.columns):
        return "real_nflverse_games"
    if set(SAMPLE_RAW_FEATURE_COLUMNS).issubset(df.columns):
        return "sample"
    raise ValueError(
        "Missing required columns for a supported schema. Expected either nflverse "
        "game columns or the original sample feature columns."
    )


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    schema = detect_schema(df)
    if schema == "real_nflverse_games":
        return add_real_matchup_features(df)
    return add_sample_matchup_features(df)


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    schema = detect_schema(df)
    if schema == "real_nflverse_games":
        return REAL_MODEL_FEATURE_COLUMNS
    return SAMPLE_MODEL_FEATURE_COLUMNS


def build_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_columns(df, [TARGET_COLUMN])
    featured = add_matchup_features(df)
    x = featured[get_model_feature_columns(df)]
    y = featured[TARGET_COLUMN].astype(int)
    return x, y
