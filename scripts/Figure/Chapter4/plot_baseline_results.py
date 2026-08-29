from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})


def load_baseline_results(project_root):
    file_path = (
        project_root
        / "results"
        / "experiments"
        / "baseline_results.csv"
    )

    df = pd.read_csv(file_path)

    # Standardise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("-", "_")
        .str.replace(" ", "_")
    )

    return df


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


def shorten_model_names(df):
    model_name_map = {
        "Logistic Regression": "Logistic Regression",
        "Decision Tree": "Decision Tree",
        "Random Forest": "Random Forest",
        "Support Vector Machine": "SVM",
        "SVM": "SVM",
        "XGBoost": "XGBoost"
    }

    df = df.copy()

    df["display_model"] = (
        df["model"]
        .map(model_name_map)
        .fillna(df["model"])
    )

    return df


def plot_baseline_model_performance(df, output_dir):
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    metric_labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ]

    y = np.arange(len(df))
    bar_height = 0.14

    fig, ax = plt.subplots(
        figsize=(10, 6.5)
    )

    for i, (metric, label) in enumerate(
        zip(metric_columns, metric_labels)
    ):
        ax.barh(
            y + (i - 2) * bar_height,
            df[metric],
            height=bar_height,
            label=label
        )

    ax.set_yticks(y)

    ax.set_yticklabels(
        df["display_model"],
        fontsize=11
    )

    ax.set_xlabel(
        "Score",
        fontsize=11
    )

    ax.set_ylabel(
        "Model",
        fontsize=11
    )

    ax.set_title(
        "Baseline Model Performance Comparison",
        fontsize=11,
        pad=10
    )

    ax.set_xlim(
        0,
        1.0
    )

    ax.tick_params(
        axis="x",
        labelsize=10
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        fontsize=10,
        loc="lower right",
        frameon=True
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "figure_4_4_baseline_model_performance"
    )


def plot_baseline_training_time(df, output_dir):
    sorted_df = (
        df.sort_values(
            by="training_time",
            ascending=True
        )
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = ax.barh(
        sorted_df["display_model"],
        sorted_df["training_time"]
    )

    ax.set_xlabel(
        "Training Time (seconds)",
        fontsize=11
    )

    ax.set_ylabel(
        "Model",
        fontsize=11
    )

    ax.set_title(
        "Baseline Model Training Time",
        fontsize=11,
        pad=10
    )

    ax.tick_params(
        axis="x",
        labelsize=10
    )

    ax.tick_params(
        axis="y",
        labelsize=11
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_time = sorted_df["training_time"].max()

    ax.set_xlim(
        0,
        max_time * 1.16
    )

    for bar, value in zip(
        bars,
        sorted_df["training_time"]
    ):
        ax.text(
            value + max_time * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=10
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "figure_4_5_baseline_training_time"
    )


def main():
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

    baseline_results = load_baseline_results(
        project_root
    )

    baseline_results = shorten_model_names(
        baseline_results
    )

    print("\nBaseline results:")
    print(
        baseline_results[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "training_time"
            ]
        ]
    )

    plot_baseline_model_performance(
        baseline_results,
        output_dir
    )

    plot_baseline_training_time(
        baseline_results,
        output_dir
    )

    print(
        "\nFigure 4.4 and Figure 4.5 "
        "were generated successfully."
    )


if __name__ == "__main__":
    main()