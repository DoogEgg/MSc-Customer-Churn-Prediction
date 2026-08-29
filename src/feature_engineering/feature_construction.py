import pandas as pd


def add_constructed_features(data):
    data = data.copy()

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    data["NumberOfServices"] = 0

    for column in service_columns:
        data["NumberOfServices"] += (
            data[column] == "Yes"
        ).astype(int)

    data["ChargePerService"] = (
        data["MonthlyCharges"]
        / (data["NumberOfServices"] + 1)
    )

    data["IsAutomaticPayment"] = data[
        "PaymentMethod"
    ].isin([
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]).astype(int)

    data["HasSupportService"] = (
        (data["OnlineSecurity"] == "Yes")
        | (data["TechSupport"] == "Yes")
    ).astype(int)

    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=[
            "0-12",
            "13-24",
            "25-48",
            "49-60",
            "61-72"
        ]
    )

    return data