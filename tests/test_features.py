import pandas as pd
import pytest

from nfl_ml.features import MODEL_FEATURE_COLUMNS, add_matchup_features, build_model_matrix


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
