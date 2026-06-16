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

TEAM_STAT_RAW_FEATURE_COLUMNS = [
    "home_team",
    "away_team",
    "home_rest",
    "away_rest",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
]

TEAM_STAT_MODEL_FEATURE_COLUMNS = [
    "home_win_pct",
    "away_win_pct",
    "win_pct_diff",
    "home_points_for_avg",
    "away_points_for_avg",
    "points_for_diff",
    "home_points_allowed_avg",
    "away_points_allowed_avg",
    "points_allowed_diff",
    "home_point_diff_avg",
    "away_point_diff_avg",
    "point_diff_diff",
    "games_played_diff",
    "rest_diff",
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


def add_environment_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df, ["home_rest", "away_rest", "div_game", "roof", "surface"])

    featured = df.copy()
    featured["rest_diff"] = featured["home_rest"] - featured["away_rest"]
    featured["is_dome"] = featured["roof"].isin(["dome", "closed", "retractable"]).astype(int)
    featured["is_turf"] = (
        featured["surface"].fillna("").str.contains("turf", case=False, regex=False)
    ).astype(int)
    featured["div_game"] = featured["div_game"].astype(int)
    return featured


def empty_team_state() -> dict[str, float]:
    return {
        "games_played": 0,
        "wins": 0,
        "points_for": 0.0,
        "points_allowed": 0.0,
    }


def team_profile_from_state(state: dict[str, float]) -> dict[str, float]:
    games = state["games_played"]
    if games == 0:
        return {
            "win_pct": 0.5,
            "points_for_avg": 22.0,
            "points_allowed_avg": 22.0,
            "point_diff_avg": 0.0,
            "games_played": 0.0,
        }

    points_for_avg = state["points_for"] / games
    points_allowed_avg = state["points_allowed"] / games
    return {
        "win_pct": state["wins"] / games,
        "points_for_avg": points_for_avg,
        "points_allowed_avg": points_allowed_avg,
        "point_diff_avg": points_for_avg - points_allowed_avg,
        "games_played": float(games),
    }


def add_profile_columns(
    row: dict,
    home_profile: dict[str, float],
    away_profile: dict[str, float],
) -> dict:
    row.update(
        {
            "home_win_pct": home_profile["win_pct"],
            "away_win_pct": away_profile["win_pct"],
            "win_pct_diff": home_profile["win_pct"] - away_profile["win_pct"],
            "home_points_for_avg": home_profile["points_for_avg"],
            "away_points_for_avg": away_profile["points_for_avg"],
            "points_for_diff": home_profile["points_for_avg"]
            - away_profile["points_for_avg"],
            "home_points_allowed_avg": home_profile["points_allowed_avg"],
            "away_points_allowed_avg": away_profile["points_allowed_avg"],
            "points_allowed_diff": away_profile["points_allowed_avg"]
            - home_profile["points_allowed_avg"],
            "home_point_diff_avg": home_profile["point_diff_avg"],
            "away_point_diff_avg": away_profile["point_diff_avg"],
            "point_diff_diff": home_profile["point_diff_avg"]
            - away_profile["point_diff_avg"],
            "games_played_diff": home_profile["games_played"]
            - away_profile["games_played"],
        }
    )
    return row


def build_team_stat_rows(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(
        df,
        TEAM_STAT_RAW_FEATURE_COLUMNS
        + ["season", "week", "home_score", "away_score", TARGET_COLUMN],
    )

    team_states: dict[str, dict[str, float]] = {}
    rows = []
    ordered = df.sort_values(["season", "week", "gameday", "game_id"])

    for _, game in ordered.iterrows():
        home_team = game["home_team"]
        away_team = game["away_team"]
        home_state = team_states.setdefault(home_team, empty_team_state())
        away_state = team_states.setdefault(away_team, empty_team_state())

        row = game.to_dict()
        add_profile_columns(
            row,
            team_profile_from_state(home_state),
            team_profile_from_state(away_state),
        )
        rows.append(row)

        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        home_state["games_played"] += 1
        home_state["wins"] += int(home_score > away_score)
        home_state["points_for"] += home_score
        home_state["points_allowed"] += away_score

        away_state["games_played"] += 1
        away_state["wins"] += int(away_score > home_score)
        away_state["points_for"] += away_score
        away_state["points_allowed"] += home_score

    return pd.DataFrame(rows).sort_index()


def build_latest_team_profiles(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    validate_columns(df, ["home_team", "away_team", "home_score", "away_score"])

    team_states: dict[str, dict[str, float]] = {}
    ordered = df.sort_values(["season", "week", "gameday", "game_id"])
    for _, game in ordered.iterrows():
        home_team = game["home_team"]
        away_team = game["away_team"]
        home_state = team_states.setdefault(home_team, empty_team_state())
        away_state = team_states.setdefault(away_team, empty_team_state())
        home_score = float(game["home_score"])
        away_score = float(game["away_score"])

        home_state["games_played"] += 1
        home_state["wins"] += int(home_score > away_score)
        home_state["points_for"] += home_score
        home_state["points_allowed"] += away_score

        away_state["games_played"] += 1
        away_state["wins"] += int(away_score > home_score)
        away_state["points_for"] += away_score
        away_state["points_allowed"] += home_score

    return {
        team: team_profile_from_state(state)
        for team, state in sorted(team_states.items())
    }


def add_team_stat_matchup_features(
    df: pd.DataFrame,
    team_profiles: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    validate_columns(df, TEAM_STAT_RAW_FEATURE_COLUMNS)

    featured = add_environment_features(df)
    if team_profiles is None:
        validate_columns(featured, TEAM_STAT_MODEL_FEATURE_COLUMNS)
        return featured

    rows = []
    neutral = team_profile_from_state(empty_team_state())
    for _, row in featured.iterrows():
        row_dict = row.to_dict()
        home_profile = team_profiles.get(str(row["home_team"]).upper(), neutral)
        away_profile = team_profiles.get(str(row["away_team"]).upper(), neutral)
        rows.append(add_profile_columns(row_dict, home_profile, away_profile))
    return pd.DataFrame(rows)


def detect_schema(df: pd.DataFrame) -> str:
    if set(TEAM_STAT_MODEL_FEATURE_COLUMNS).issubset(df.columns):
        return "team_stat"
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
    if schema == "team_stat":
        return add_team_stat_matchup_features(df)
    if schema == "real_nflverse_games":
        return add_real_matchup_features(df)
    return add_sample_matchup_features(df)


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    schema = detect_schema(df)
    if schema == "team_stat":
        return TEAM_STAT_MODEL_FEATURE_COLUMNS
    if schema == "real_nflverse_games":
        return REAL_MODEL_FEATURE_COLUMNS
    return SAMPLE_MODEL_FEATURE_COLUMNS


def build_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_columns(df, [TARGET_COLUMN])
    featured = add_matchup_features(df)
    x = featured[get_model_feature_columns(df)]
    y = featured[TARGET_COLUMN].astype(int)
    return x, y


def build_team_stat_model_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, dict[str, float]]]:
    featured = add_environment_features(build_team_stat_rows(df))
    x = featured[TEAM_STAT_MODEL_FEATURE_COLUMNS]
    y = featured[TARGET_COLUMN].astype(int)
    team_profiles = build_latest_team_profiles(df)
    return x, y, featured, team_profiles
