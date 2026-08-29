from pathlib import Path

import pandas as pd


def load_data(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    return pd.read_csv(path)


def inspect_data(df):
    print(f"Dataset shape: {df.shape}")

    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nChurn distribution:")
    print(df["Churn"].value_counts())

    print("\nChurn proportion:")
    print(df["Churn"].value_counts(normalize=True))

    invalid_total_charges = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    ).isna().sum()

    print(f"\nInvalid TotalCharges values: {invalid_total_charges}")