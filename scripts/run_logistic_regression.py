from pathlib import Path

import pandas as pd

from src.data_cleaning import clean_data
from src.data_loader import load_data
from src.experiments.logistic_regression import (
    test_c_values,
    test_combined_values,
    test_penalty_values
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
        / "logistic_regression_results.csv"
    )

    data = load_data(file_path)
    cleaned_data = clean_data(data)

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        cleaned_data
    )

    c_results = test_c_values(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )
    c_results["experiment"] = "C"

    penalty_results = test_penalty_values(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )
    penalty_results["experiment"] = "penalty"

    combined_results = test_combined_values(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )
    combined_results["experiment"] = "combined"

    blank_row = pd.DataFrame(
        [[None] * len(c_results.columns)],
        columns=c_results.columns
    )

    results = pd.concat(
        [
            c_results,
            blank_row,
            penalty_results,
            blank_row,
            combined_results
        ],
        ignore_index=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\nLogistic Regression experiments:")
    print(results)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()