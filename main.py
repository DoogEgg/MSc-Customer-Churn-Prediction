from src.data_loader import inspect_data, load_data
from src.model_training import train_and_evaluate_models
from src.data_cleaning import clean_data
from src.eda import run_eda
from src.preprocessing import prepare_data
import pandas as pd
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

def main():
    file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    data = load_data(file_path)
    inspect_data(data)

    cleaned_data = clean_data(data)

    print(f"\nOriginal dataset shape: {data.shape}")
    print(f"Cleaned dataset shape: {cleaned_data.shape}")
    print(f"\nTotalCharges data type: {cleaned_data['TotalCharges'].dtype}")
    print(
        "Remaining missing values: "
        f"{cleaned_data['TotalCharges'].isna().sum()}"
    )

    run_eda(cleaned_data)

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        cleaned_data
    )

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTest target distribution:")
    print(y_test.value_counts(normalize=True))

    baseline_results = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    print("\nBaseline model results:")
    print(baseline_results)

    baseline_output = Path(
        "results/experiments/baseline_results.csv"
    )
    baseline_output.parent.mkdir(parents=True, exist_ok=True)

    baseline_results.to_csv(
        baseline_output,
        index=False
    )

    print(f"\nBaseline results saved to: {baseline_output}")


if __name__ == "__main__":
    main()