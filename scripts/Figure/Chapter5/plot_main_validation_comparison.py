from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10


def load_csv(project_root, filename):
    file_path = (
        project_root
        / "results"
        / "experiments"
        / filename
    )
    return pd.read_csv(file_path)


def get_main_experiment_changes(feature_df):
    target_schemes = {
        "Logistic Regression": "Selected Top 30",
        "Random Forest": "Selected Top 30",
        "XGBoost": "Constructed"
    }

    results = []

    for model_name, feature_set in target_schemes.items():
        baseline_row = feature_df[
            (feature_df["model"] == model_name)
            & (feature_df["feature_set"] == "Baseline")
        ]

        target_row = feature_df[
            (feature_df["model"] == model_name)
            & (feature_df["feature_set"] == feature_set)
        ]

        if baseline_row.empty or target_row.empty:
            raise ValueError(
                f"Missing data for {model_name} - {feature_set}"
            )

        baseline_f1 = baseline_row.iloc[0]["f1"]
        target_f1 = target_row.iloc[0]["f1"]
        f1_change = target_f1 - baseline_f1

        results.append(
            {
                "model": model_name,
                "feature_set": feature_set,
                "main_f1_change": f1_change
            }
        )

    return pd.DataFrame(results)


def get_validation_changes(validation_df):
    target_schemes = {
        "Logistic Regression": "Selected Top 30",
        "Random Forest": "Selected Top 30",
        "XGBoost": "Constructed"
    }

    results = []

    for model_name, feature_set in target_schemes.items():
        target_row = validation_df[
            (validation_df["model"] == model_name)
            & (validation_df["feature_set"] == feature_set)
        ]

        if target_row.empty:
            raise ValueError(
                f"Missing validation data for {model_name} - {feature_set}"
            )

        if "f1_change_vs_baseline" in target_row.columns:
            f1_change = target_row.iloc[0]["f1_change_vs_baseline"]
        else:
            baseline_row = validation_df[
                (validation_df["model"] == model_name)
                & (validation_df["feature_set"] == "Baseline")
            ]

            if baseline_row.empty:
                raise ValueError(
                    f"Missing baseline validation data for {model_name}"
                )

            baseline_f1 = baseline_row.iloc[0]["mean_f1"]
            target_f1 = target_row.iloc[0]["mean_f1"]
            f1_change = target_f1 - baseline_f1

        results.append(
            {
                "model": model_name,
                "feature_set": feature_set,
                "validation_f1_change": f1_change
            }
        )

    return pd.DataFrame(results)


def prepare_plot_data(feature_df, validation_df):
    main_df = get_main_experiment_changes(feature_df)
    validation_change_df = get_validation_changes(validation_df)

    comparison_df = main_df.merge(
        validation_change_df,
        on=["model", "feature_set"]
    )

    model_order = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ]

    comparison_df["model"] = pd.Categorical(
        comparison_df["model"],
        categories=model_order,
        ordered=True
    )

    comparison_df = comparison_df.sort_values("model")

    return comparison_df


def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()

        if height >= 0:
            y = height + 0.001
            va = "bottom"
        else:
            y = height - 0.001
            va = "top"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{height:.4f}",
            ha="center",
            va=va,
            fontsize=9
        )


def plot_comparison(comparison_df, output_path):
    models = comparison_df["model"].tolist()
    main_values = comparison_df["main_f1_change"].tolist()
    validation_values = comparison_df["validation_f1_change"].tolist()

    x = np.arange(len(models))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars1 = ax.bar(
        x - width / 2,
        main_values,
        width,
        label="Main experiment"
    )

    bars2 = ax.bar(
        x + width / 2,
        validation_values,
        width,
        label="Validation experiment"
    )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "Comparison of F1-score Changes between Main and Validation Experiments",
        pad=12
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Change in F1-score relative to Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

    add_value_labels(ax, bars1)
    add_value_labels(ax, bars2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    project_root = Path(__file__).resolve().parents[3]

    feature_df = load_csv(
        project_root,
        "feature_engineering_final_results.csv"
    )

    validation_df = load_csv(
        project_root,
        "result_validation_summary.csv"
    )

    comparison_df = prepare_plot_data(
        feature_df,
        validation_df
    )

    output_dir = (
        project_root
        / "results"
        / "figures"
        / "chapter5"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / "figure_5_4_main_validation_f1_comparison.png"
    )

    plot_comparison(comparison_df, output_path)

    print("\nComparison data:")
    print(comparison_df)

    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()