from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data(
    data,
    extra_numeric_features=None,
    random_state=42
):
    data = data.copy()

    X = data.drop(columns=["customerID", "Churn"])

    y = data["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    if extra_numeric_features is not None:
        numeric_features += extra_numeric_features

    categorical_features = [
        column
        for column in X.columns
        if column not in numeric_features
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    preprocessor = ColumnTransformer([
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ])

    return X_train, X_test, y_train, y_test, preprocessor