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
    "ytick.labelsize": 10.5
})


# ---------------------------------------------------------
# Load feature ranking results
# ---------------------------------------------------------
def load_selected_features(project_root):
    file_path = (
        project_root
        / "results"
        / "experiments"
        / "selected_features.csv"
    )

    print(f"Loading: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Result file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    df = df.dropna(
        subset=[
            "feature_set",
            "top_k",
            "rank",
            "feature_name",
            "score"
        ]
    ).copy()

    return df


# ---------------------------------------------------------
# Prepare Top 10 ANOVA ranking
# ---------------------------------------------------------
def prepare_top10_features(df):
    top10_df = df[
        (df["feature_set"] == "Selected")
        & (df["top_k"] == 10)
    ].copy()

    top10_df = top10_df.sort_values(
        by="rank",
        ascending=True
    )

    if len(top10_df) != 10:
        raise ValueError(
            f"Expected 10 Top 10 features, "
            f"but found {len(top10_df)}."
        )

    return top10_df


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
# Plot Figure 4.12
# ---------------------------------------------------------
def plot_anova_ranking(top10_df, output_dir):
    # Reverse order so rank 1 appears at the top
    plot_df = top10_df.sort_values(
        by="score",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 5.2)
    )

    bars = ax.barh(
        plot_df["feature_name"],
        plot_df["score"]
    )

    ax.set_xlabel(
        "ANOVA F-score",
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

    ax.grid(
        axis="x",
        alpha=0.2
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    maximum_score = plot_df["score"].max()

    ax.set_xlim(
        0,
        maximum_score * 1.15
    )

    for bar, value in zip(
        bars,
        plot_df["score"]
    ):
        ax.text(
            value + maximum_score * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=10
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "figure_4_11_anova_top10_feature_ranking"
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

    df = load_selected_features(
        project_root
    )

    top10_df = prepare_top10_features(
        df
    )

    print("\nTop 10 ANOVA features:")
    print(
        top10_df[
            [
                "rank",
                "feature_name",
                "score"
            ]
        ].to_string(index=False)
    )

    plot_anova_ranking(
        top10_df,
        output_dir
    )

    print(
        "\nFigure 4.11 generated successfully."
    )


if __name__ == "__main__":
    main()