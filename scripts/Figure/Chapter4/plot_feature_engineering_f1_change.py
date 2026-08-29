from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
# Load feature engineering results
# ---------------------------------------------------------
def load_feature_engineering_results(project_root):
    file_path = (
        project_root
        / "results"
        / "experiments"
        / "feature_engineering_final_results.csv"
    )

    print(f"Loading: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Result file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    # Remove separator rows in the CSV
    df = df.dropna(
        subset=["feature_set", "model"]
    ).copy()

    # Remove unnecessary spaces
    df["feature_set"] = (
        df["feature_set"]
        .astype(str)
        .str.strip()
    )

    df["model"] = (
        df["model"]
        .astype(str)
        .str.strip()
    )

    print(
        f"Valid experiment rows: {len(df)}"
    )

    return df


# ---------------------------------------------------------
# Prepare F1-score changes
# ---------------------------------------------------------
def prepare_f1_change_data(df):
    # Obtain the baseline F1-score for each model
    baseline_df = df[
        df["feature_set"] == "Baseline"
    ].copy()

    baseline_f1 = (
        baseline_df
        .set_index("model")["f1"]
        .to_dict()
    )

    print("\nBaseline F1-scores:")

    for model, score in baseline_f1.items():
        print(
            f"{model}: {score:.6f}"
        )

    # Remove baseline because its change is always zero
    plot_df = df[
        df["feature_set"] != "Baseline"
    ].copy()

    # Calculate change relative to each model's own baseline
    plot_df["f1_change"] = plot_df.apply(
        lambda row:
        row["f1"] - baseline_f1[row["model"]],
        axis=1
    )

    feature_order = [
        "Constructed",
        "Selected Top 10",
        "Selected Top 20",
        "Selected Top 30",
        "Combined Top 10",
        "Combined Top 20",
        "Combined Top 30"
    ]

    model_order = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ]

    return (
        plot_df,
        feature_order,
        model_order
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

    plt.close(fig)

    print(f"\nSaved: {png_path}")
    print(f"Saved: {svg_path}")


# ---------------------------------------------------------
# Plot Figure 4.11
# ---------------------------------------------------------
def plot_f1_change(
    plot_df,
    feature_order,
    model_order,
    output_dir
):
    pivot_df = plot_df.pivot(
        index="feature_set",
        columns="model",
        values="f1_change"
    )

    pivot_df = pivot_df.loc[
        feature_order,
        model_order
    ]

    print(
        "\nF1-score changes relative "
        "to baseline:"
    )

    print(
        pivot_df.round(4)
    )

    x = np.arange(
        len(feature_order)
    )

    bar_width = 0.24

    fig, ax = plt.subplots(
        figsize=(7.2, 4.6)
    )

    for index, model in enumerate(
        model_order
    ):
        offset = (
            index - 1
        ) * bar_width

        ax.bar(
            x + offset,
            pivot_df[model],
            width=bar_width,
            label=model
        )

    # Baseline reference line
    ax.axhline(
        y=0,
        linewidth=1.2
    )

    ax.set_ylabel(
        "Change in F1-score",
        fontsize=11
    )

    ax.set_xlabel(
        "Feature Engineering Scheme",
        fontsize=11
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            "Constructed",
            "Selected\nTop 10",
            "Selected\nTop 20",
            "Selected\nTop 30",
            "Combined\nTop 10",
            "Combined\nTop 20",
            "Combined\nTop 30"
        ],
        fontsize=10
    )

    ax.tick_params(
        axis="y",
        labelsize=10.5
    )

    ax.legend(
        loc="lower left",
        fontsize=10,
        frameon=True
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    # Give the bars some space above and below
    minimum_value = (
        pivot_df.min().min()
    )

    maximum_value = (
        pivot_df.max().max()
    )

    lower_limit = min(
        minimum_value * 1.20,
        -0.01
    )

    upper_limit = max(
        maximum_value * 1.30,
        0.01
    )

    ax.set_ylim(
        lower_limit,
        upper_limit
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "figure_4_10_f1_change_relative_to_baseline"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    # Script:
    # E:\MSc_project\scripts\Figure\Chapter4\
    #
    # parents[3]:
    # E:\MSc_project
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

    print(
        f"Project root: "
        f"{project_root}"
    )

    df = (
        load_feature_engineering_results(
            project_root
        )
    )

    (
        plot_df,
        feature_order,
        model_order
    ) = prepare_f1_change_data(
        df
    )

    plot_f1_change(
        plot_df,
        feature_order,
        model_order,
        output_dir
    )

    print(
        "\nFigure 4.10 generated "
        "successfully."
    )


if __name__ == "__main__":
    main()