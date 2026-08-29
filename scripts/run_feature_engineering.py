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


def run_models(
    feature_set,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    selected_feature_count=None
):
    models = {
        "Logistic Regression": LogisticRegression(
            C=10,
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1
        )
    }

    results = []

    for model_name, model in models.items():
        pipeline_steps = [
            ("preprocessor", preprocessor)
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
            ("model", model)
        )

        pipeline = Pipeline(pipeline_steps)

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        if selected_feature_count is None:
            number_of_features = len(
                pipeline.named_steps[
                    "preprocessor"
                ].get_feature_names_out()
            )
        else:
            number_of_features = selected_feature_count

        results.append({
            "feature_set": feature_set,
            "model": model_name,
            "number_of_features": number_of_features,
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
        })

    return pd.DataFrame(results)


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
        / "feature_engineering_final_results.csv"
    )

    data = load_data(file_path)
    cleaned_data = clean_data(data)

    # Baseline data
    (
        baseline_X_train,
        baseline_X_test,
        baseline_y_train,
        baseline_y_test,
        baseline_preprocessor
    ) = prepare_data(cleaned_data)

    baseline_results = run_models(
        "Baseline",
        baseline_X_train,
        baseline_X_test,
        baseline_y_train,
        baseline_y_test,
        baseline_preprocessor
    )

    # Constructed data
    constructed_data = add_constructed_features(
        cleaned_data
    )

    extra_numeric_features = [
        "NumberOfServices",
        "ChargePerService",
        "IsAutomaticPayment",
        "HasSupportService"
    ]

    (
        constructed_X_train,
        constructed_X_test,
        constructed_y_train,
        constructed_y_test,
        constructed_preprocessor
    ) = prepare_data(
        constructed_data,
        extra_numeric_features=extra_numeric_features
    )

    constructed_results = run_models(
        "Constructed",
        constructed_X_train,
        constructed_X_test,
        constructed_y_train,
        constructed_y_test,
        constructed_preprocessor
    )

    # Feature selection using baseline features
    selected_10_results = run_models(
        "Selected Top 10",
        baseline_X_train,
        baseline_X_test,
        baseline_y_train,
        baseline_y_test,
        baseline_preprocessor,
        selected_feature_count=10
    )

    selected_20_results = run_models(
        "Selected Top 20",
        baseline_X_train,
        baseline_X_test,
        baseline_y_train,
        baseline_y_test,
        baseline_preprocessor,
        selected_feature_count=20
    )

    selected_30_results = run_models(
        "Selected Top 30",
        baseline_X_train,
        baseline_X_test,
        baseline_y_train,
        baseline_y_test,
        baseline_preprocessor,
        selected_feature_count=30
    )

    # Constructed features followed by feature selection
    combined_10_results = run_models(
        "Combined Top 10",
        constructed_X_train,
        constructed_X_test,
        constructed_y_train,
        constructed_y_test,
        constructed_preprocessor,
        selected_feature_count=10
    )

    combined_20_results = run_models(
        "Combined Top 20",
        constructed_X_train,
        constructed_X_test,
        constructed_y_train,
        constructed_y_test,
        constructed_preprocessor,
        selected_feature_count=20
    )

    combined_30_results = run_models(
        "Combined Top 30",
        constructed_X_train,
        constructed_X_test,
        constructed_y_train,
        constructed_y_test,
        constructed_preprocessor,
        selected_feature_count=30
    )

    blank_row = pd.DataFrame(
        [[None] * len(baseline_results.columns)],
        columns=baseline_results.columns
    )

    results = pd.concat(
        [
            baseline_results,
            blank_row,
            constructed_results,
            blank_row,
            selected_10_results,
            blank_row,
            selected_20_results,
            blank_row,
            selected_30_results,
            blank_row,
            combined_10_results,
            blank_row,
            combined_20_results,
            blank_row,
            combined_30_results
        ],
        ignore_index=True
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nBaseline feature results:")
    print(baseline_results)

    print("\nConstructed feature results:")
    print(constructed_results)

    print("\nSelected Top 10 feature results:")
    print(selected_10_results)

    print("\nSelected Top 20 feature results:")
    print(selected_20_results)

    print("\nSelected Top 30 feature results:")
    print(selected_30_results)

    print("\nCombined Top 10 feature results:")
    print(combined_10_results)

    print("\nCombined Top 20 feature results:")
    print(combined_20_results)

    print("\nCombined Top 30 feature results:")
    print(combined_30_results)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()