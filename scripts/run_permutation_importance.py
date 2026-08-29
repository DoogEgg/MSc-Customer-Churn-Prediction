from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import issparse
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data_cleaning import clean_data
from src.data_loader import load_data
from src.preprocessing import prepare_data


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


def clean_feature_names(feature_names):
    cleaned_names = []

    for feature_name in feature_names:
        feature_name = feature_name.replace(
            "numeric__",
            ""
        )
        feature_name = feature_name.replace(
            "categorical__",
            ""
        )

        cleaned_names.append(feature_name)

    return cleaned_names


def calculate_permutation_importance(
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    # Fit preprocessing only on training data
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    # Convert sparse matrices if necessary
    if issparse(X_train_processed):
        X_train_processed = (
            X_train_processed.toarray()
        )

    if issparse(X_test_processed):
        X_test_processed = (
            X_test_processed.toarray()
        )

    # Get names of the 46 processed features
    feature_names = (
        preprocessor.get_feature_names_out()
    )

    feature_names = clean_feature_names(
        feature_names
    )

    # Train model using fixed final parameters
    model = create_model(model_name)

    model.fit(
        X_train_processed,
        y_train
    )

    # Calculate permutation importance
    importance = permutation_importance(
        model,
        X_test_processed,
        y_test,
        scoring="f1",
        n_repeats=10,
        random_state=42,
        n_jobs=1
    )

    # Create result table
    result = pd.DataFrame({
        "Model": model_name,
        "Feature": feature_names,
        "Importance_Mean": (
            importance.importances_mean
        ),
        "Importance_SD": (
            importance.importances_std
        )
    })

    result = result.sort_values(
        by="Importance_Mean",
        ascending=False
    ).reset_index(drop=True)

    result.insert(
        1,
        "Rank",
        range(1, len(result) + 1)
    )

    return result


def save_top_10_plot(
    result,
    model_name,
    output_path
):
    top_10 = result.head(10).copy()

    top_10 = top_10.sort_values(
        by="Importance_Mean",
        ascending=True
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        top_10["Feature"],
        top_10["Importance_Mean"],
        xerr=top_10["Importance_SD"]
    )

    plt.xlabel(
        "Decrease in F1-score"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Top 10 Permutation Feature Importances - "
        f"{model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():
    project_root = (
        Path(__file__).resolve().parents[1]
    )

    file_path = (
        project_root
        / "data"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    output_dir = (
        project_root
        / "results"
        / "interpretability"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_output_path = (
        output_dir
        / "permutation_importance.csv"
    )

    # Load and clean data
    data = load_data(file_path)
    cleaned_data = clean_data(data)

    # Use baseline features and the same data split
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data(
        cleaned_data,
        random_state=42
    )

    model_names = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ]

    all_results = []

    for model_name in model_names:
        print(
            f"\nCalculating permutation importance "
            f"for {model_name}..."
        )

        result = (
            calculate_permutation_importance(
                model_name=model_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                preprocessor=preprocessor
            )
        )

        all_results.append(result)

        safe_model_name = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        figure_path = (
            output_dir
            / f"{safe_model_name}_"
              f"permutation_importance.png"
        )

        save_top_10_plot(
            result=result,
            model_name=model_name,
            output_path=figure_path
        )

        print()
        print(result.head(10))

    # Combine results from all three models
    all_results = pd.concat(
        all_results,
        ignore_index=True
    )

    all_results.to_csv(
        csv_output_path,
        index=False
    )

    print(
        f"\nResults saved to: "
        f"{csv_output_path}"
    )

    print(
        f"Figures saved to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()