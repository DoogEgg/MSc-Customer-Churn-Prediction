from pathlib import Path

from src.data_cleaning import clean_data
from src.data_loader import load_data
from src.feature_engineering.feature_construction import (
    add_constructed_features
)


def main():
    project_root = Path(__file__).resolve().parents[1]

    file_path = (
        project_root
        / "data"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    data = load_data(file_path)
    cleaned_data = clean_data(data)

    constructed_data = add_constructed_features(cleaned_data)

    new_features = [
        "NumberOfServices",
        "ChargePerService",
        "IsAutomaticPayment",
        "HasSupportService",
        "TenureGroup"
    ]

    print("\nConstructed features:")
    print(constructed_data[new_features].head(10))

    print("\nMissing values:")
    print(constructed_data[new_features].isnull().sum())

    print("\nDataset shape before feature construction:")
    print(cleaned_data.shape)

    print("\nDataset shape after feature construction:")
    print(constructed_data.shape)


if __name__ == "__main__":
    main()