from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# --------------------------------------------------
# Paths
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[3]

data_path = (
    project_root
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

output_dir = (
    project_root
    / "results"
    / "figures"
    / "chapter4"
)

output_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load and clean data
# --------------------------------------------------

df = pd.read_csv(data_path)

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


# --------------------------------------------------
# Preprocessing
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


X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


feature_names = preprocessor.get_feature_names_out()

print(
    f"Number of processed features: "
    f"{len(feature_names)}"
)


# --------------------------------------------------
# Representative models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        C=10,
        penalty="l2",
        solver="liblinear",
        max_iter=1000
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
# ROC curves
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.2, 5.6)
)


line_styles = {
    "Logistic Regression": "-",
    "Random Forest": "--",
    "XGBoost": "-."
}


for model_name, model in models.items():

    model.fit(
        X_train_processed,
        y_train
    )

    y_probability = model.predict_proba(
        X_test_processed
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    print(
        f"{model_name}: "
        f"ROC-AUC = {auc:.4f}"
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        linestyle=line_styles[model_name],
        label=(
            f"{model_name} "
            f"(AUC = {auc:.4f})"
        )
    )


# Random classifier reference
ax.plot(
    [0, 1],
    [0, 1],
    linestyle=":",
    linewidth=1.3,
    label="Random Classifier"
)


ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    "ROC Curves of Representative Models"
)

ax.legend(
    loc="lower right",
    frameon=True
)

ax.grid(
    alpha=0.20
)

fig.tight_layout()


# --------------------------------------------------
# Save figure
# --------------------------------------------------

png_path = (
    output_dir
    / "figure_4_15_roc_comparison_representative_models.png"
)

svg_path = (
    output_dir
    / "figure_4_15_roc_comparison_representative_models.svg"
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


print("\nSaved:")
print(png_path)
print(svg_path)