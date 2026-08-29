from pathlib import Path

import pandas as pd

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


def get_selected_features(
    feature_set,
    X_train,
    y_train,
    preprocessor
):
    selected_feature_counts = [10, 20, 30]
    results = []

    # Fit preprocessing using training data only
    X_train_processed = preprocessor.fit_transform(
        X_train,
        y_train
    )

    feature_names = preprocessor.get_feature_names_out()

    # Remove prefixes added by ColumnTransformer
    feature_names = [
        name.replace("numeric__", "")
        .replace("categorical__", "")
        for name in feature_names
    ]

    for feature_count in selected_feature_counts:
        selector = create_feature_selector(
            feature_count
        )

        selector.fit(
            X_train_processed,
            y_train
        )

        selected_mask = selector.get_support()
        selected_scores = selector.scores_[
            selected_mask
        ]

        selected_names = [
            name
            for name, selected
            in zip(feature_names, selected_mask)
            if selected
        ]

        selected_results = pd.DataFrame({
            "feature_name": selected_names,
            "score": selected_scores
        })

        selected_results = selected_results.sort_values(
            by="score",
            ascending=False
        ).reset_index(drop=True)

        selected_results["rank"] = (
            selected_results.index + 1
        )

        selected_results["feature_set"] = feature_set
        selected_results["top_k"] = feature_count

        selected_results = selected_results[
            [
                "feature_set",
                "top_k",
                "rank",
                "feature_name",
                "score"
            ]
        ]

        results.append(selected_results)

    return pd.concat(
        results,
        ignore_index=True
    )


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
        / "selected_features.csv"
    )

    data = load_data(file_path)
    cleaned_data = clean_data(data)

    # Baseline features
    (
        baseline_X_train,
        baseline_X_test,
        baseline_y_train,
        baseline_y_test,
        baseline_preprocessor
    ) = prepare_data(cleaned_data)

    baseline_features = get_selected_features(
        "Selected",
        baseline_X_train,
        baseline_y_train,
        baseline_preprocessor
    )

    # Constructed features
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

    combined_features = get_selected_features(
        "Combined",
        constructed_X_train,
        constructed_y_train,
        constructed_preprocessor
    )

    results = pd.concat(
        [
            baseline_features,
            combined_features
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

    print("\nSelected features:")
    print(results)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()