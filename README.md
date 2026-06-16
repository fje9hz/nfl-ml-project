# NFL Game Outcome Machine Learning Project

This project trains a machine learning model to predict whether the home team wins an NFL game from real historical NFL game data.

The real dataset comes from the public `nflverse`/Lee Sharpe NFL game file. It includes completed NFL games, scores, rest days, betting lines, moneylines, weather, roof/surface, and division-game flags.

## What It Does

- Downloads real NFL game data
- Engineers matchup features such as spread, total, rest differential, implied moneyline probabilities, weather, roof/surface, and division-game status
- Trains a logistic regression model and a random forest model
- Selects the best model by ROC AUC
- Saves the trained model pipeline to `models/nfl_win_model.joblib`
- Writes evaluation metrics and feature importance outputs to `reports/`
- Runs single-game predictions from the command line
- Serves an interactive web app with FastAPI

## Project Structure

```text
nfl-ml-project/
  data/
    nfl_games_real.csv
    nfl_games_sample.csv
  models/
  reports/
  src/
    nfl_ml/
      features.py
      data.py
      predict.py
      train.py
      service.py
      web.py
  tests/
    test_features.py
  web/
    index.html
    styles.css
    app.js
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
python -m nfl_ml.data --start-season 2010
python -m nfl_ml.train --data data/nfl_games_real.csv
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
  --spread-line 3.5 \
  --total-line 47.5 \
  --home-rest 7 \
  --away-rest 6 \
  --home-moneyline -170 \
  --away-moneyline 150 \
  --div-game 1 \
  --roof outdoors \
  --surface grass \
  --temp 65 \
  --wind 8
```

## Launch the Web App

```bash
python3 -m uvicorn nfl_ml.web:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The app includes matchup inputs, win probabilities, model metrics, and a feature importance chart.

## Deploy Online

This repo includes a `render.yaml` file for Render deployment. See `DEPLOYMENT.md` for the exact steps and production start command.

## Run With Docker

```bash
docker build -t nfl-win-probability .
docker run --rm -p 8000:8000 nfl-win-probability
```

Then open:

```text
http://127.0.0.1:8000
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
| `home_score`, `away_score` | Final score |
| `home_rest`, `away_rest` | Days of rest before kickoff |
| `home_moneyline`, `away_moneyline` | Closing moneyline odds |
| `spread_line` | Closing spread line from the away-team perspective; positive means the home team is favored |
| `total_line` | Closing over/under total |
| `div_game` | 1 if teams are in the same division, else 0 |
| `roof`, `surface`, `temp`, `wind` | Game environment fields |
| `home_win` | Target variable: 1 if home team won, else 0 |

## Next Upgrades

- Add rolling team performance features from `nflfastR` play-by-play EPA
- Add quarterback injuries and starter changes
- Use time-series validation by season/week
- Build a Streamlit dashboard for matchup predictions
