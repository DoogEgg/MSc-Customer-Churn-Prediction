import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_cleaning import clean_data
from src.data_loader import load_data
from src.preprocessing import prepare_data


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def main():
    project_root = Path(__file__).resolve().parents[1]

    file_path = (
        project_root
        / "data"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    output_path = (
        project_root
        / "results"
        / "experiments"
        / "cross_validation_results.csv"
    )

    data = load_data(file_path)
    cleaned_data = clean_data(data)

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        cleaned_data
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    experiments = {
        "Logistic Regression": {
            "model": LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                random_state=42
            ),
            "parameters": {
                "model__C": [0.1, 1, 10],
                "model__penalty": ["l1", "l2"]
            }
        },

        "Random Forest": {
            "model": RandomForestClassifier(
                random_state=42
            ),
            "parameters": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [10, 15, None],
                "model__min_samples_split": [2, 10, 20]
            }
        },

        "XGBoost": {
            "model": XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=1
            ),
            "parameters": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 6]
            }
        }
    }

    results = []

    for model_name, experiment in experiments.items():
        print(f"\nRunning cross-validation for {model_name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", experiment["model"])
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=experiment["parameters"],
            scoring="f1",
            cv=cv,
            n_jobs=1,
            refit=True
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        validation_time = time.time() - start_time

        y_pred = grid_search.predict(X_test)
        y_proba = grid_search.predict_proba(X_test)[:, 1]

        best_parameters = {
            name.replace("model__", ""): value
            for name, value in grid_search.best_params_.items()
        }

        results.append({
            "model": model_name,
            "best_parameters": str(best_parameters),
            "mean_cv_f1": grid_search.best_score_,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_proba),
            "validation_time": validation_time
        })

        print(f"Best parameters: {best_parameters}")
        print(f"Mean CV F1: {grid_search.best_score_:.6f}")
        print(f"Test F1: {f1_score(y_test, y_pred):.6f}")

    results = pd.DataFrame(results)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nCross-validation results:")
    print(results)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()