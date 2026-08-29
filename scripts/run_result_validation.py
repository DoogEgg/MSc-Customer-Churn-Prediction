import time
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_cleaning import clean_data
from src.data_loader import load_data
from src.feature_engineering.feature_construction import (
    add_constructed_features
)
from src.feature_engineering.feature_selection import (
    create_feature_selector
)
from src.preprocessing import prepare_data


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def create_model(model_name):
    if model_name == "Logistic Regression":
        return LogisticRegression(
            C=10,
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            random_state=42
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )

    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1
        )

    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(
    random_state,
    model_name,
    feature_set,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    selected_feature_count=None
):
    pipeline_steps = [
        ("preprocessor", clone(preprocessor))
    ]

    if selected_feature_count is not None:
        pipeline_steps.append(
            (
                "selector",
                create_feature_selector(
                    selected_feature_count
                )
            )
        )

    pipeline_steps.append(
        ("model", create_model(model_name))
    )

    pipeline = Pipeline(pipeline_steps)

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "random_state": random_state,
        "model": model_name,
        "feature_set": feature_set,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "training_time": training_time
    }


def main():
    project_root = Path(__file__).resolve().parents[1]

    file_path = (
        project_root
        / "data"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    runs_output_path = (
        project_root
        / "results"
        / "experiments"
        / "result_validation_runs.csv"
    )

    summary_output_path = (
        project_root
        / "results"
        / "experiments"
        / "result_validation_summary.csv"
    )

    data = load_data(file_path)
    cleaned_data = clean_data(data)

    constructed_data = add_constructed_features(
        cleaned_data
    )

    extra_numeric_features = [
        "NumberOfServices",
        "ChargePerService",
        "IsAutomaticPayment",
        "HasSupportService"
    ]

    random_states = [42, 52, 62, 72, 82]
    results = []

    for random_state in random_states:
        print(
            f"\nRunning validation with "
            f"random_state={random_state}..."
        )

        baseline_split = prepare_data(
            cleaned_data,
            random_state=random_state
        )

        constructed_split = prepare_data(
            constructed_data,
            extra_numeric_features=extra_numeric_features,
            random_state=random_state
        )

        (
            baseline_X_train,
            baseline_X_test,
            baseline_y_train,
            baseline_y_test,
            baseline_preprocessor
        ) = baseline_split

        (
            constructed_X_train,
            constructed_X_test,
            constructed_y_train,
            constructed_y_test,
            constructed_preprocessor
        ) = constructed_split

        experiments = [
            {
                "model_name": "Logistic Regression",
                "feature_set": "Baseline",
                "split": baseline_split,
                "selected_feature_count": None
            },
            {
                "model_name": "Logistic Regression",
                "feature_set": "Selected Top 30",
                "split": baseline_split,
                "selected_feature_count": 30
            },
            {
                "model_name": "Random Forest",
                "feature_set": "Baseline",
                "split": baseline_split,
                "selected_feature_count": None
            },
            {
                "model_name": "Random Forest",
                "feature_set": "Selected Top 30",
                "split": baseline_split,
                "selected_feature_count": 30
            },
            {
                "model_name": "XGBoost",
                "feature_set": "Baseline",
                "split": baseline_split,
                "selected_feature_count": None
            },
            {
                "model_name": "XGBoost",
                "feature_set": "Constructed",
                "split": constructed_split,
                "selected_feature_count": None
            }
        ]

        for experiment in experiments:
            (
                X_train,
                X_test,
                y_train,
                y_test,
                preprocessor
            ) = experiment["split"]

            result = evaluate_model(
                random_state=random_state,
                model_name=experiment["model_name"],
                feature_set=experiment["feature_set"],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                preprocessor=preprocessor,
                selected_feature_count=experiment[
                    "selected_feature_count"
                ]
            )

            results.append(result)

    results = pd.DataFrame(results)

    summary = (
        results
        .groupby(
            ["model", "feature_set"],
            as_index=False
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_training_time=(
                "training_time",
                "mean"
            )
        )
    )

    baseline_f1 = (
        summary[
            summary["feature_set"] == "Baseline"
        ]
        .set_index("model")["mean_f1"]
    )

    baseline_roc_auc = (
        summary[
            summary["feature_set"] == "Baseline"
        ]
        .set_index("model")["mean_roc_auc"]
    )

    summary["f1_change_vs_baseline"] = summary.apply(
        lambda row: (
            row["mean_f1"]
            - baseline_f1[row["model"]]
        ),
        axis=1
    )

    summary["roc_auc_change_vs_baseline"] = (
        summary.apply(
            lambda row: (
                row["mean_roc_auc"]
                - baseline_roc_auc[row["model"]]
            ),
            axis=1
        )
    )

    runs_output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        runs_output_path,
        index=False
    )

    summary.to_csv(
        summary_output_path,
        index=False
    )

    print("\nValidation summary:")
    print(summary)

    print(
        f"\nDetailed results saved to: "
        f"{runs_output_path}"
    )

    print(
        f"Summary saved to: "
        f"{summary_output_path}"
    )


if __name__ == "__main__":
    main()