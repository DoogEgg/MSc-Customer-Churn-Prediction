import pandas as pd


def clean_data(df):
    cleaned_data = df.copy()

    cleaned_data["TotalCharges"] = pd.to_numeric(
        cleaned_data["TotalCharges"],
        errors="coerce"
    )

    cleaned_data = cleaned_data.dropna(subset=["TotalCharges"])

    return cleaned_data