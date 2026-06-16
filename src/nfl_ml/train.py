from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_ml.features import build_model_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an NFL home-win model.")
    parser.add_argument("--data", default="data/nfl_games_real.csv", help="CSV dataset path")
    parser.add_argument("--model-out", default="models/nfl_win_model.joblib")
    parser.add_argument("--metrics-out", default="reports/metrics.json")
    parser.add_argument("--importance-out", default="reports/feature_importance.csv")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Use a random split instead of the default chronological split.",
    )
    return parser.parse_args()


def make_candidates(random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "brier_score": round(brier_score_loss(y_test, probabilities), 4),
        "log_loss": round(log_loss(y_test, probabilities), 4),
    }


def split_data(
    df: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    random_split: bool,
):
    if random_split or not {"season", "week"}.issubset(df.columns):
        return train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    ordered = df.sort_values(["season", "week"]).index
    test_start = int(len(ordered) * (1 - test_size))
    train_index = ordered[:test_start]
    test_index = ordered[test_start:]
    return x.loc[train_index], x.loc[test_index], y.loc[train_index], y.loc[test_index]


def feature_importance(
    model_name: str, model: Pipeline, feature_columns: list[str]
) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
    else:
        values = [0.0] * len(feature_columns)

    return (
        pd.DataFrame({"feature": feature_columns, "importance": values})
        .assign(model=model_name)
        .sort_values("importance", key=lambda s: s.abs(), ascending=False)
    )


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    importance_path = Path(args.importance_out)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run python -m nfl_ml.data first."
        )

    df = pd.read_csv(data_path)
    x, y = build_model_matrix(df)
    feature_columns = list(x.columns)

    x_train, x_test, y_train, y_test = split_data(
        df, x, y, args.test_size, args.random_state, args.random_split
    )

    results = {}
    trained_models = {}
    for model_name, model in make_candidates(args.random_state).items():
        model.fit(x_train, y_train)
        results[model_name] = evaluate_model(model, x_test, y_test)
        trained_models[model_name] = model

    best_model_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = trained_models[best_model_name]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    importance_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model_name": best_model_name,
            "pipeline": best_model,
            "feature_columns": feature_columns,
        },
        model_path,
    )

    payload = {
        "best_model": best_model_name,
        "rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "target_rate_home_win": round(float(y.mean()), 4),
        "feature_columns": feature_columns,
        "split": "random" if args.random_split else "chronological",
        "models": results,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    feature_importance(best_model_name, best_model, feature_columns).to_csv(
        importance_path, index=False
    )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
