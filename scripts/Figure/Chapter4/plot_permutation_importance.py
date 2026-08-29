from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# ---------------------------------------------------------
# Global figure settings
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5
})


# ---------------------------------------------------------
# Set safe temporary folder for Windows / joblib
# ---------------------------------------------------------
def configure_temp_folder(project_root):
    temp_folder = (
        project_root
        / "temp_joblib"
    )

    temp_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    os.environ["JOBLIB_TEMP_FOLDER"] = str(
        temp_folder
    )

    os.environ["TEMP"] = str(
        temp_folder
    )

    os.environ["TMP"] = str(
        temp_folder
    )

    print(
        f"Temporary folder: "
        f"{temp_folder}"
    )


# ---------------------------------------------------------
# Load and clean dataset
# ---------------------------------------------------------
def load_and_clean_data(project_root):
    data_path = (
        project_root
        / "data"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    print(
        f"Loading dataset: "
        f"{data_path}"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{data_path}"
        )

    df = pd.read_csv(
        data_path
    )

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove the 11 records with missing TotalCharges
    df = df.dropna(
        subset=["TotalCharges"]
    ).copy()

    # Encode target
    df["Churn"] = (
        df["Churn"]
        .map({
            "No": 0,
            "Yes": 1
        })
    )

    # Remove customer identifier
    df = df.drop(
        columns=["customerID"]
    )

    print(
        f"Cleaned dataset shape: "
        f"{df.shape}"
    )

    return df


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------
def split_data(df):
    X = df.drop(
        columns=["Churn"]
    )

    y = df["Churn"]

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"Training set shape: "
        f"{X_train.shape}"
    )

    print(
        f"Test set shape: "
        f"{X_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
def create_preprocessor(X_train):
    numerical_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    categorical_features = [
        column
        for column in X_train.columns
        if column not in numerical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    return preprocessor


# ---------------------------------------------------------
# Get transformed feature names
# ---------------------------------------------------------
def get_feature_names(preprocessor):
    raw_names = (
        preprocessor
        .get_feature_names_out()
    )

    feature_names = []

    for name in raw_names:
        if name.startswith("num__"):
            clean_name = name.replace(
                "num__",
                ""
            )

        elif name.startswith("cat__"):
            clean_name = name.replace(
                "cat__",
                ""
            )

        else:
            clean_name = name

        feature_names.append(
            clean_name
        )

    return np.array(
        feature_names
    )


# ---------------------------------------------------------
# Convert transformed data to dense arrays
# ---------------------------------------------------------
def convert_to_dense(X):
    if hasattr(
        X,
        "toarray"
    ):
        return X.toarray()

    return np.asarray(X)


# ---------------------------------------------------------
# Create final models
# ---------------------------------------------------------
def create_models():
    models = {
        "Logistic Regression":
            LogisticRegression(
                C=10,
                penalty="l2",
                solver="liblinear",
                max_iter=1000,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=1
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                eval_metric="logloss",
                n_jobs=1
            )
    }

    return models


# ---------------------------------------------------------
# Calculate permutation importance
# ---------------------------------------------------------
def calculate_permutation_importance(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names
):
    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Important:
    # n_jobs=1 avoids Windows joblib multiprocessing
    # problems caused by non-ASCII temporary paths.
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=10,
        random_state=42,
        n_jobs=1
    )

    importance_df = pd.DataFrame({
        "feature":
            feature_names,

        "importance_mean":
            result.importances_mean,

        "importance_std":
            result.importances_std
    })

    importance_df = (
        importance_df
        .sort_values(
            by="importance_mean",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    importance_df["rank"] = (
        np.arange(
            1,
            len(importance_df) + 1
        )
    )

    return importance_df


# ---------------------------------------------------------
# Save complete PI results
# ---------------------------------------------------------
def save_importance_results(
    importance_df,
    output_dir,
    filename
):
    csv_path = (
        output_dir
        / f"{filename}.csv"
    )

    importance_df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"Saved results: "
        f"{csv_path}"
    )


# ---------------------------------------------------------
# Save figure
# ---------------------------------------------------------
def save_figure(
    fig,
    output_dir,
    filename
):
    png_path = (
        output_dir
        / f"{filename}.png"
    )

    svg_path = (
        output_dir
        / f"{filename}.svg"
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

    plt.close(
        fig
    )

    print(
        f"Saved: "
        f"{png_path}"
    )

    print(
        f"Saved: "
        f"{svg_path}"
    )


# ---------------------------------------------------------
# Plot Top 10 Permutation Importance
# ---------------------------------------------------------
def plot_top10_importance(
    importance_df,
    output_dir,
    filename
):
    top10 = (
        importance_df
        .head(10)
        .copy()
    )

    # Reverse order so the highest-ranked
    # feature appears at the top
    plot_df = (
        top10
        .sort_values(
            by="importance_mean",
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 5.4)
    )

    ax.barh(
        plot_df["feature"],
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
        capsize=3
    )

    ax.set_xlabel(
        "Decrease in F1-score",
        fontsize=11
    )

    ax.set_ylabel(
        "Feature",
        fontsize=11
    )

    ax.tick_params(
        axis="x",
        labelsize=10.5
    )

    ax.tick_params(
        axis="y",
        labelsize=10.5
    )

    # Zero reference line
    ax.axvline(
        x=0,
        linewidth=1
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        filename
    )


# ---------------------------------------------------------
# Print Top 10 PI results
# ---------------------------------------------------------
def print_top10(
    model_name,
    importance_df
):
    print(
        f"\nTop 10 Permutation Importances "
        f"for {model_name}:"
    )

    print(
        importance_df[
            [
                "rank",
                "feature",
                "importance_mean",
                "importance_std"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    # Script location:
    # E:\MSc_project\scripts\Figure\Chapter4\
    #
    # parents[3]:
    # E:\MSc_project
    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    figure_output_dir = (
        project_root
        / "results"
        / "figures"
        / "chapter4"
    )

    result_output_dir = (
        project_root
        / "results"
        / "experiments"
        / "permutation_importance"
    )

    figure_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    result_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Project root: "
        f"{project_root}"
    )

    # Configure an ASCII-only temporary folder
    configure_temp_folder(
        project_root
    )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    df = load_and_clean_data(
        project_root
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        df
    )

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------
    preprocessor = create_preprocessor(
        X_train
    )

    # Fit preprocessing only on training data
    X_train_processed = (
        preprocessor
        .fit_transform(
            X_train
        )
    )

    # Use exactly the same fitted preprocessing
    # on the test data
    X_test_processed = (
        preprocessor
        .transform(
            X_test
        )
    )

    feature_names = get_feature_names(
        preprocessor
    )

    print(
        f"Number of processed features: "
        f"{len(feature_names)}"
    )

    if len(feature_names) != 46:
        print(
            "Warning: Expected 46 baseline "
            f"features, but found "
            f"{len(feature_names)}."
        )

    # Convert sparse matrices to dense arrays
    X_train_processed = convert_to_dense(
        X_train_processed
    )

    X_test_processed = convert_to_dense(
        X_test_processed
    )

    print(
        f"Processed training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed test shape: "
        f"{X_test_processed.shape}"
    )

    # -----------------------------------------------------
    # Final model configurations
    # -----------------------------------------------------
    models = create_models()

    figure_files = {
        "Logistic Regression":
            (
                "figure_4_12_"
                "logistic_regression_"
                "permutation_importance"
            ),

        "Random Forest":
            (
                "figure_4_13_"
                "random_forest_"
                "permutation_importance"
            ),

        "XGBoost":
            (
                "figure_4_14_"
                "xgboost_"
                "permutation_importance"
            )
    }

    result_files = {
        "Logistic Regression":
            (
                "logistic_regression_"
                "permutation_importance"
            ),

        "Random Forest":
            (
                "random_forest_"
                "permutation_importance"
            ),

        "XGBoost":
            (
                "xgboost_"
                "permutation_importance"
            )
    }

    # -----------------------------------------------------
    # Run Permutation Importance
    # -----------------------------------------------------
    for (
        model_name,
        model
    ) in models.items():

        print(
            "\n--------------------------------"
        )

        print(
            f"Calculating Permutation "
            f"Importance for "
            f"{model_name}..."
        )

        importance_df = (
            calculate_permutation_importance(
                model=model,
                X_train=X_train_processed,
                X_test=X_test_processed,
                y_train=y_train,
                y_test=y_test,
                feature_names=feature_names
            )
        )

        print_top10(
            model_name,
            importance_df
        )

        save_importance_results(
            importance_df,
            result_output_dir,
            result_files[
                model_name
            ]
        )

        plot_top10_importance(
            importance_df,
            figure_output_dir,
            figure_files[
                model_name
            ]
        )

    print(
        "\nAll Permutation Importance "
        "results and figures were "
        "generated successfully."
    )


if __name__ == "__main__":
    main()