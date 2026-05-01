# NFL Game Outcome Machine Learning Project

This project trains a machine learning model to predict whether the home team wins an NFL game from pregame and season-to-date team metrics.

It is intentionally self-contained: the repository includes a small NFL-style sample dataset so the pipeline runs without API keys or internet access. You can replace `data/nfl_games_sample.csv` with real historical game data later.

## What It Does

- Loads game-level NFL data
- Engineers matchup features such as Elo gap, rest gap, efficiency gap, and turnover gap
- Trains a logistic regression model and a random forest model
- Selects the best model by ROC AUC
- Saves the trained model pipeline to `models/nfl_win_model.joblib`
- Writes evaluation metrics and feature importance outputs to `reports/`
- Runs single-game predictions from the command line

## Project Structure

```text
nfl-ml-project/
  data/
    nfl_games_sample.csv
  models/
  reports/
  src/
    nfl_ml/
      features.py
      predict.py
      train.py
  tests/
    test_features.py
```

## Setup

```bash
cd nfl-ml-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

That installs the local `nfl_ml` package in editable mode, so the `python -m nfl_ml...` commands below work from the project folder.

## Train the Model

```bash
python -m nfl_ml.train --data data/nfl_games_sample.csv
```

After training, check:

- `models/nfl_win_model.joblib`
- `reports/metrics.json`
- `reports/feature_importance.csv`

## Make a Prediction

```bash
python -m nfl_ml.predict \
  --home-team SF \
  --away-team DAL \
  --home-elo 1685 \
  --away-elo 1605 \
  --home-rest-days 7 \
  --away-rest-days 6 \
  --home-offense-epa 0.13 \
  --away-offense-epa 0.09 \
  --home-defense-epa -0.05 \
  --away-defense-epa -0.02 \
  --home-turnover-margin 4 \
  --away-turnover-margin 1 \
  --home-injury-score 2 \
  --away-injury-score 4 \
  --is-division-game 1
```

## Run Tests

```bash
pytest
```

## Data Dictionary

| Column | Meaning |
| --- | --- |
| `season` | NFL season year |
| `week` | Week number |
| `home_team`, `away_team` | Team abbreviations |
| `home_elo`, `away_elo` | Team strength ratings before the game |
| `home_rest_days`, `away_rest_days` | Days of rest before kickoff |
| `home_offense_epa`, `away_offense_epa` | Season-to-date offensive EPA/play estimate |
| `home_defense_epa`, `away_defense_epa` | Season-to-date defensive EPA/play estimate, lower is better |
| `home_turnover_margin`, `away_turnover_margin` | Season-to-date turnover margin |
| `home_injury_score`, `away_injury_score` | Approximate injury burden, lower is healthier |
| `is_division_game` | 1 if teams are in the same division, else 0 |
| `home_win` | Target variable: 1 if home team won, else 0 |

## Next Upgrades

- Swap the sample CSV for `nflverse` or `nflfastR` historical data
- Add betting spread and closing total features
- Use time-series validation by season/week
- Build a Streamlit dashboard for matchup predictions
