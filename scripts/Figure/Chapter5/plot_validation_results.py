from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# Global figure settings
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5
})


# ---------------------------------------------------------
# Load validation results
# ---------------------------------------------------------
def load_validation_results(project_root):
    file_path = (
        project_root
        / "results"
        / "experiments"
        / "result_validation_runs.csv"
    )

    print(f"Loading: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Validation result file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "random_state",
        "model",
        "feature_set",
        "f1"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print(
        f"Validation rows loaded: {len(df)}"
    )

    return df


# ---------------------------------------------------------
# Prepare paired validation data
# ---------------------------------------------------------
def prepare_model_data(
    df,
    model_name,
    baseline_scheme,
    compared_scheme
):
    model_df = df[
        (df["model"] == model_name)
        & (
            df["feature_set"].isin(
                [
                    baseline_scheme,
                    compared_scheme
                ]
            )
        )
    ].copy()

    pivot_df = model_df.pivot(
        index="random_state",
        columns="feature_set",
        values="f1"
    )

    pivot_df = (
        pivot_df
        .reset_index()
        .sort_values(
            by="random_state"
        )
    )

    expected_columns = [
        baseline_scheme,
        compared_scheme
    ]

    for column in expected_columns:
        if column not in pivot_df.columns:
            raise ValueError(
                f"{column} not found for {model_name}"
            )

    return pivot_df


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

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


# ---------------------------------------------------------
# Draw paired dot plot
# ---------------------------------------------------------
def plot_paired_validation(
    plot_df,
    baseline_scheme,
    compared_scheme,
    output_dir,
    filename
):
    seeds = plot_df["random_state"].tolist()

    baseline_values = (
        plot_df[baseline_scheme]
        .tolist()
    )

    compared_values = (
        plot_df[compared_scheme]
        .tolist()
    )

    y_positions = list(
        range(len(seeds))
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.6)
    )

    # Draw connecting lines
    for i in range(len(seeds)):
        ax.hlines(
            y=i,
            xmin=min(
                baseline_values[i],
                compared_values[i]
            ),
            xmax=max(
                baseline_values[i],
                compared_values[i]
            ),
            linewidth=2,
            alpha=0.6
        )

    # Baseline points
    ax.scatter(
        baseline_values,
        y_positions,
        s=75,
        marker="o",
        label=baseline_scheme,
        zorder=3
    )

    # Feature engineering points
    ax.scatter(
        compared_values,
        y_positions,
        s=75,
        marker="s",
        label=compared_scheme,
        zorder=3
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        [
            f"Seed {seed}"
            for seed in seeds
        ]
    )

    ax.set_xlabel(
        "F1-score",
        fontsize=11
    )

    ax.set_ylabel(
        "Random Seed",
        fontsize=11
    )

    ax.tick_params(
        axis="both",
        labelsize=10.5
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        fontsize=10,
        frameon=True
    )

    # Keep some space around the points
    all_values = (
        baseline_values
        + compared_values
    )

    minimum_value = min(all_values)
    maximum_value = max(all_values)

    margin = max(
        (maximum_value - minimum_value) * 0.15,
        0.005
    )

    ax.set_xlim(
        minimum_value - margin,
        maximum_value + margin
    )

    # Seed 42 appears at the top
    ax.invert_yaxis()

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        filename
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    # Script location:
    #
    # E:\MSc_project\
    # scripts\
    # Figure\
    # Chapter5\
    # plot_validation_results.py
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
        / "chapter5"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Project root: {project_root}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    # -----------------------------------------------------
    # Load validation results
    # -----------------------------------------------------
    df = load_validation_results(
        project_root
    )

    # -----------------------------------------------------
    # Figure 5.1
    # Logistic Regression
    # -----------------------------------------------------
    lr_df = prepare_model_data(
        df=df,
        model_name="Logistic Regression",
        baseline_scheme="Baseline",
        compared_scheme="Selected Top 30"
    )

    print(
        "\nLogistic Regression validation:"
    )

    print(
        lr_df[
            [
                "random_state",
                "Baseline",
                "Selected Top 30"
            ]
        ].to_string(index=False)
    )

    plot_paired_validation(
        plot_df=lr_df,
        baseline_scheme="Baseline",
        compared_scheme="Selected Top 30",
        output_dir=output_dir,
        filename=(
            "figure_5_1_"
            "logistic_regression_validation"
        )
    )

    # -----------------------------------------------------
    # Figure 5.2
    # Random Forest
    # -----------------------------------------------------
    rf_df = prepare_model_data(
        df=df,
        model_name="Random Forest",
        baseline_scheme="Baseline",
        compared_scheme="Selected Top 30"
    )

    print(
        "\nRandom Forest validation:"
    )

    print(
        rf_df[
            [
                "random_state",
                "Baseline",
                "Selected Top 30"
            ]
        ].to_string(index=False)
    )

    plot_paired_validation(
        plot_df=rf_df,
        baseline_scheme="Baseline",
        compared_scheme="Selected Top 30",
        output_dir=output_dir,
        filename=(
            "figure_5_2_"
            "random_forest_validation"
        )
    )

    # -----------------------------------------------------
    # Figure 5.3
    # XGBoost
    # -----------------------------------------------------
    xgb_df = prepare_model_data(
        df=df,
        model_name="XGBoost",
        baseline_scheme="Baseline",
        compared_scheme="Constructed"
    )

    print(
        "\nXGBoost validation:"
    )

    print(
        xgb_df[
            [
                "random_state",
                "Baseline",
                "Constructed"
            ]
        ].to_string(index=False)
    )

    plot_paired_validation(
        plot_df=xgb_df,
        baseline_scheme="Baseline",
        compared_scheme="Constructed",
        output_dir=output_dir,
        filename=(
            "figure_5_3_"
            "xgboost_validation"
        )
    )

    print(
        "\nAll Chapter 5 validation "
        "figures generated successfully."
    )


if __name__ == "__main__":
    main()