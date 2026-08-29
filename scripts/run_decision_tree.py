 from pathlib import Path

import pandas as pd

from src.data_cleaning import clean_data
from src.data_loader import load_data
from src.experiments.decision_tree import (
    test_combined_values,
    test_max_depth_values,
    test_min_samples_leaf_values,
    test_min_samples_split_values
)
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
        / "decision_tree_results.csv"
    )

    data = load_data(file_path)
    cleaned_data = clean_data(data)

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        cleaned_data
    )

    max_depth_results = test_max_depth_values(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )
    max_depth_results["experiment"] = "max_depth"

    min_samples_split_results = test_min_samples_split_values(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )
    min_samples_split_results["experiment"] = "min_samples_split"

    min_samples_leaf_results = test_min_samples_leaf_values(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )
    min_samples_leaf_results["experiment"] = "min_samples_leaf"

    combined_results = test_combined_values(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )
    combined_results["experiment"] = "combined"

    blank_row = pd.DataFrame(
        [[None] * len(max_depth_results.columns)],
        columns=max_depth_results.columns
    )

    results = pd.concat(
        [
            max_depth_results,
            blank_row,
            min_samples_split_results,
            blank_row,
            min_samples_leaf_results,
            blank_row,
            combined_results
        ],
        ignore_index=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\nDecision Tree max_depth experiment:")
    print(max_depth_results)

    print("\nDecision Tree min_samples_split experiment:")
    print(min_samples_split_results)

    print("\nDecision Tree min_samples_leaf experiment:")
    print(min_samples_leaf_results)

    print("\nDecision Tree combined experiment:")
    print(combined_results)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()