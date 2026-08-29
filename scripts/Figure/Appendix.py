from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# --------------------------------------------------
# Paths
# --------------------------------------------------

# Current script:
# E:\MSc_project\scripts\Figure\Appendix.py
project_root = Path(__file__).resolve().parents[2]

data_path = (
    project_root
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

output_dir = (
    project_root
    / "results"
    / "figures"
    / "appendix"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

print("Project root:", project_root)
print("Data path:", data_path)
print("Output directory:", output_dir)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(data_path)


# --------------------------------------------------
# Data cleaning
# --------------------------------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(
    subset=["TotalCharges"]
).copy()

df = df.drop(
    columns=["customerID"]
)

df["Churn"] = df["Churn"].map(
    {
        "No": 0,
        "Yes": 1
    }
)

print("Cleaned dataset shape:", df.shape)


# --------------------------------------------------
# Features and target
# --------------------------------------------------

X = df.drop(
    columns=["Churn"]
)

y = df["Churn"]


# --------------------------------------------------
# Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


# --------------------------------------------------
# Numerical and categorical features
# --------------------------------------------------

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    column
    for column in X.columns
    if column not in numeric_features
]


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


feature_names = preprocessor.get_feature_names_out()

print(
    "Number of processed features:",
    len(feature_names)
)


# --------------------------------------------------
# Representative models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        C=10,
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
}


# --------------------------------------------------
# Output file names
# --------------------------------------------------

file_names = {
    "Logistic Regression":
        "figure_c_1_logistic_regression_confusion_matrix",

    "Random Forest":
        "figure_c_2_random_forest_confusion_matrix",

    "XGBoost":
        "figure_c_3_xgboost_confusion_matrix"
}


# --------------------------------------------------
# Train models and generate confusion matrices
# --------------------------------------------------

for model_name, model in models.items():

    print("\n----------------------------------------")
    print(model_name)
    print("----------------------------------------")

    # Train model
    model.fit(
        X_train_processed,
        y_train
    )

    # Predictions
    y_pred = model.predict(
        X_test_processed
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print("Confusion matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    print(f"True Negative:  {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    print(f"True Positive:  {tp}")
    print(f"Total:          {cm.sum()}")


    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(5.5, 4.8)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Non-Churn",
            "Churn"
        ]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d"
    )

    ax.set_title(
        f"Confusion Matrix for {model_name}"
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    fig.tight_layout()


    # --------------------------------------------------
    # Save PNG and SVG
    # --------------------------------------------------

    base_name = file_names[
        model_name
    ]

    png_path = (
        output_dir
        / f"{base_name}.png"
    )

    svg_path = (
        output_dir
        / f"{base_name}.svg"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:")
    print(png_path)
    print(svg_path)


# --------------------------------------------------
# Finished
# --------------------------------------------------

print("\n========================================")
print("All confusion matrix figures generated.")
print("========================================")

print("Output directory:")
print(output_dir)