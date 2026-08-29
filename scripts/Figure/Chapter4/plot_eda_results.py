from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_and_clean_data(project_root):
    data_path = (
        project_root
        / "data"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df = pd.read_csv(data_path)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.dropna(subset=["TotalCharges"]).copy()

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


def plot_churn_distribution(df, output_dir):
    counts = (
        df["Churn"]
        .value_counts()
        .reindex(["No", "Yes"])
    )

    labels = [
        f"Non-Churn\n{counts['No']:,}",
        f"Churn\n{counts['Yes']:,}"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        wedgeprops={
            "width": 0.45,
            "edgecolor": "white"
        },
        textprops={
            "fontsize": 15
        }
    )

    ax.set_aspect("equal")

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "figure_4_1_churn_distribution"
    )


def plot_numerical_distributions(df, output_dir):
    numerical_features = [
        (
            "tenure",
            "Tenure Distribution",
            "Tenure (Months)"
        ),
        (
            "MonthlyCharges",
            "Monthly Charges Distribution",
            "Monthly Charges"
        ),
        (
            "TotalCharges",
            "Total Charges Distribution",
            "Total Charges"
        )
    ]

    fig = plt.figure(
        figsize=(11, 9)
    )

    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1, 1]
    )

    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :])
    ]

    for ax, (
        feature,
        title,
        x_label
    ) in zip(
        axes,
        numerical_features
    ):
        ax.hist(
            df[feature],
            bins=30,
            edgecolor="black",
            linewidth=0.6
        )

        ax.set_title(
            title,
            fontsize=16,
            pad=10
        )

        ax.set_xlabel(
            x_label,
            fontsize=14
        )

        ax.set_ylabel(
            "Number of Customers",
            fontsize=14
        )

        ax.tick_params(
            axis="both",
            labelsize=12
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            axis="y",
            alpha=0.2
        )

    fig.tight_layout(
        h_pad=2.5,
        w_pad=2.5
    )

    save_figure(
        fig,
        output_dir,
        "figure_4_2_numerical_distributions"
    )


def calculate_churn_rate(df, feature):
    churn_rate = (
        df.groupby(feature)["Churn"]
        .apply(
            lambda values:
            (values == "Yes").mean() * 100
        )
        .sort_values(ascending=True)
    )

    return churn_rate


def plot_horizontal_churn_rate(
    ax,
    df,
    feature,
    title
):
    churn_rate = calculate_churn_rate(
        df,
        feature
    )

    bars = ax.barh(
        churn_rate.index,
        churn_rate.values
    )

    ax.set_title(
        title,
        fontsize=16,
        pad=10
    )

    ax.set_xlabel(
        "Churn Rate (%)",
        fontsize=14
    )

    ax.set_ylabel("")

    ax.tick_params(
        axis="both",
        labelsize=12
    )

    maximum_rate = churn_rate.max()

    ax.set_xlim(
        0,
        maximum_rate * 1.30
    )

    for bar, rate in zip(
        bars,
        churn_rate.values
    ):
        ax.text(
            rate + maximum_rate * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}%",
            va="center",
            fontsize=12
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="x",
        alpha=0.2
    )


def plot_categorical_churn_rates(
    df,
    output_dir
):
    categorical_features = [
        (
            "Contract",
            "Customer Churn Rate by Contract"
        ),
        (
            "InternetService",
            "Customer Churn Rate by Internet Service"
        ),
        (
            "PaymentMethod",
            "Customer Churn Rate by Payment Method"
        )
    ]

    fig = plt.figure(
        figsize=(11, 9)
    )

    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1, 1]
    )

    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :])
    ]

    for ax, (
        feature,
        title
    ) in zip(
        axes,
        categorical_features
    ):
        plot_horizontal_churn_rate(
            ax,
            df,
            feature,
            title
        )

    fig.tight_layout(
        h_pad=2.5,
        w_pad=3.0
    )

    save_figure(
        fig,
        output_dir,
        "figure_4_3_categorical_churn_rates"
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

    df = load_and_clean_data(
        project_root
    )

    print(
        f"Cleaned dataset shape: "
        f"{df.shape}"
    )

    plot_churn_distribution(
        df,
        output_dir
    )

    plot_numerical_distributions(
        df,
        output_dir
    )

    plot_categorical_churn_rates(
        df,
        output_dir
    )

    print(
        "All Chapter 4.1 EDA figures "
        "were generated successfully."
    )


if __name__ == "__main__":
    main()