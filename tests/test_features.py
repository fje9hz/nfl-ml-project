import pandas as pd
import pytest

from nfl_ml.data import prepare_games
from nfl_ml.features import (
    MODEL_FEATURE_COLUMNS,
    REAL_MODEL_FEATURE_COLUMNS,
    add_matchup_features,
    build_model_matrix,
)
from nfl_ml.service import confidence_label


def sample_row(home_win=1):
    return {
        "home_elo": 1600,
        "away_elo": 1550,
        "home_rest_days": 7,
        "away_rest_days": 6,
        "home_offense_epa": 0.12,
        "away_offense_epa": 0.04,
        "home_defense_epa": -0.03,
        "away_defense_epa": 0.01,
        "home_turnover_margin": 3,
        "away_turnover_margin": -1,
        "home_injury_score": 2,
        "away_injury_score": 5,
        "is_division_game": 1,
        "home_win": home_win,
    }


def test_add_matchup_features_creates_expected_diffs():
    df = pd.DataFrame([sample_row()])

    featured = add_matchup_features(df)

    assert featured.loc[0, "elo_diff"] == 50
    assert featured.loc[0, "rest_diff"] == 1
    assert featured.loc[0, "offense_epa_diff"] == pytest.approx(0.08)
    assert featured.loc[0, "defense_epa_diff"] == pytest.approx(0.04)
    assert featured.loc[0, "turnover_margin_diff"] == 4
    assert featured.loc[0, "injury_score_diff"] == 3


def test_build_model_matrix_returns_features_and_target():
    df = pd.DataFrame([sample_row(1), sample_row(0)])

    x, y = build_model_matrix(df)

    assert list(x.columns) == MODEL_FEATURE_COLUMNS
    assert y.tolist() == [1, 0]


def test_missing_columns_raise_clear_error():
    with pytest.raises(ValueError, match="Missing required columns"):
        add_matchup_features(pd.DataFrame([{"home_elo": 1500}]))


def test_real_nflverse_features_are_supported():
    df = pd.DataFrame(
        [
            {
                "spread_line": -3.5,
                "total_line": 47.5,
                "home_rest": 7,
                "away_rest": 6,
                "home_moneyline": -170,
                "away_moneyline": 150,
                "div_game": 1,
                "roof": "dome",
                "surface": "fieldturf",
                "temp": 72,
                "wind": 0,
                "home_win": 1,
            }
        ]
    )

    x, y = build_model_matrix(df)

    assert list(x.columns) == REAL_MODEL_FEATURE_COLUMNS
    assert x.loc[0, "rest_diff"] == 1
    assert x.loc[0, "is_dome"] == 1
    assert x.loc[0, "is_turf"] == 1
    assert y.tolist() == [1]


def test_prepare_games_filters_completed_regular_season_games():
    games = pd.DataFrame(
        [
            {
                "game_id": "2024_01_AAA_BBB",
                "season": 2024,
                "game_type": "REG",
                "week": 1,
                "gameday": "2024-09-08",
                "away_team": "AAA",
                "home_team": "BBB",
                "away_score": 17,
                "home_score": 24,
                "away_rest": 7,
                "home_rest": 7,
                "away_moneyline": 140,
                "home_moneyline": -160,
                "spread_line": -3.0,
                "total_line": 44.5,
                "div_game": 0,
                "roof": "outdoors",
                "surface": "grass",
                "temp": 65,
                "wind": 5,
            },
            {
                "game_id": "2024_02_AAA_BBB",
                "season": 2024,
                "game_type": "POST",
                "week": 2,
                "gameday": "2025-01-12",
                "away_team": "AAA",
                "home_team": "BBB",
                "away_score": 21,
                "home_score": 20,
                "away_rest": 7,
                "home_rest": 7,
                "away_moneyline": -110,
                "home_moneyline": -110,
                "spread_line": 0.0,
                "total_line": 42.0,
                "div_game": 0,
                "roof": "outdoors",
                "surface": "grass",
                "temp": 40,
                "wind": 8,
            },
        ]
    )

    prepared = prepare_games(games, start_season=2024)

    assert len(prepared) == 1
    assert prepared.loc[0, "home_win"] == 1


def test_confidence_label_tracks_probability_edge():
    assert confidence_label(0.51) == "Low"
    assert confidence_label(0.61) == "Medium"
    assert confidence_label(0.75) == "High"
