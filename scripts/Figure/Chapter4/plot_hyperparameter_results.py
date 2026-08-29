from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# Global figure settings
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5
})


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
def load_result_file(project_root, filename):
    file_path = (
        project_root
        / "results"
        / "experiments"
        / filename
    )

    print(f"Loading: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Result file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    # Remove completely empty separator rows
    df = df.dropna(how="all").copy()

    # Standardise experiment names
    if "experiment" in df.columns:
        df["experiment"] = (
            df["experiment"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return df


# ---------------------------------------------------------
# Save figure
# ---------------------------------------------------------
def save_figure(fig, output_dir, filename):
    png_path = output_dir / f"{filename}.png"
    svg_path = output_dir / f"{filename}.svg"

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

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


# ---------------------------------------------------------
# General plotting function
# ---------------------------------------------------------
def plot_parameter_performance(
    df,
    experiment_name,
    parameter_column,
    parameter_labels,
    x_label,
    title,
    output_dir,
    output_filename
):
    experiment_df = (
        df[
            df["experiment"] == experiment_name.lower()
        ]
        .copy()
        .reset_index(drop=True)
    )

    if experiment_df.empty:
        raise ValueError(
            f"No experiment data found for: {experiment_name}"
        )

    if len(experiment_df) != len(parameter_labels):
        raise ValueError(
            f"{experiment_name}: "
            f"{len(experiment_df)} rows found, "
            f"but {len(parameter_labels)} labels provided."
        )

    # Use equally spaced positions rather than raw numeric values.
    # This prevents values such as 0.001, 0.01 and 0.1
    # from being compressed on the left side of the plot.
    x_positions = range(len(experiment_df))

    fig, ax = plt.subplots(
        figsize=(6.5, 4.2)
    )

    ax.plot(
        x_positions,
        experiment_df["f1"],
        marker="o",
        markersize=7,
        linewidth=2.2,
        label="F1-score"
    )

    ax.plot(
        x_positions,
        experiment_df["roc_auc"],
        marker="s",
        markersize=7,
        linewidth=2.2,
        label="ROC-AUC"
    )

    ax.set_xticks(
        list(x_positions)
    )

    ax.set_xticklabels(
        parameter_labels
    )

    ax.set_xlabel(
        x_label,
        fontsize=11
    )

    ax.set_ylabel(
        "Score",
        fontsize=11
    )

    ax.set_title(
        title,
        fontsize=12,
        pad=10
    )

    ax.set_ylim(
        0,
        0.9
    )

    ax.tick_params(
        axis="both",
        labelsize=10.5
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="best",
        frameon=True
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        output_filename
    )


# ---------------------------------------------------------
# Figure 4.6
# Logistic Regression
# ---------------------------------------------------------
def plot_logistic_regression(
    df,
    output_dir
):
    plot_parameter_performance(
        df=df,
        experiment_name="c",
        parameter_column="C",
        parameter_labels=[
            "0.001",
            "0.01",
            "0.1",
            "1",
            "10",
            "100"
        ],
        x_label="C",
        title="Effect of C on Logistic Regression Performance",
        output_dir=output_dir,
        output_filename=(
            "figure_4_6_logistic_regression_c"
        )
    )


# ---------------------------------------------------------
# Figure 4.7
# Random Forest
# ---------------------------------------------------------
def plot_random_forest(
    df,
    output_dir
):
    plot_parameter_performance(
        df=df,
        experiment_name="max_depth",
        parameter_column="max_depth",
        parameter_labels=[
            "None",
            "5",
            "10",
            "15",
            "20"
        ],
        x_label="Maximum Depth",
        title="Effect of Maximum Depth on Random Forest Performance",
        output_dir=output_dir,
        output_filename=(
            "figure_4_7_random_forest_max_depth"
        )
    )


# ---------------------------------------------------------
# Figure 4.8
# Support Vector Machine
# ---------------------------------------------------------
def plot_svm(
    df,
    output_dir
):
    plot_parameter_performance(
        df=df,
        experiment_name="c",
        parameter_column="C",
        parameter_labels=[
            "0.01",
            "0.1",
            "1",
            "10",
            "100"
        ],
        x_label="C",
        title="Effect of C on SVM Performance",
        output_dir=output_dir,
        output_filename=(
            "figure_4_8_svm_c"
        )
    )


# ---------------------------------------------------------
# Figure 4.9
# XGBoost
# ---------------------------------------------------------
def plot_xgboost(
    df,
    output_dir
):
    plot_parameter_performance(
        df=df,
        experiment_name="max_depth",
        parameter_column="max_depth",
        parameter_labels=[
            "2",
            "3",
            "4",
            "6",
            "8",
            "10"
        ],
        x_label="Maximum Depth",
        title="Effect of Maximum Depth on XGBoost Performance",
        output_dir=output_dir,
        output_filename=(
            "figure_4_9_xgboost_max_depth"
        )
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    # Script location:
    # E:\MSc_project\scripts\Figure\Chapter4\
    #
    # parents[3] = E:\MSc_project
    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    output_dir = (
        project_root
        / "results"
        / "figures"
        / "chapter4"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Project root: {project_root}")
    print(f"Output directory: {output_dir}")

    # -----------------------------------------------------
    # Load experiment results
    # -----------------------------------------------------
    logistic_df = load_result_file(
        project_root,
        "logistic_regression_results.csv"
    )

    random_forest_df = load_result_file(
        project_root,
        "random_forest_results.csv"
    )

    svm_df = load_result_file(
        project_root,
        "svm_results.csv"
    )

    xgboost_df = load_result_file(
        project_root,
        "xgboost_results.csv"
    )

    # -----------------------------------------------------
    # Generate figures
    # -----------------------------------------------------
    print("\nGenerating Figure 4.6...")
    plot_logistic_regression(
        logistic_df,
        output_dir
    )

    print("\nGenerating Figure 4.7...")
    plot_random_forest(
        random_forest_df,
        output_dir
    )

    print("\nGenerating Figure 4.8...")
    plot_svm(
        svm_df,
        output_dir
    )

    print("\nGenerating Figure 4.9...")
    plot_xgboost(
        xgboost_df,
        output_dir
    )

    print(
        "\nAll selected hyperparameter "
        "experiment figures were generated successfully."
    )


if __name__ == "__main__":
    main()