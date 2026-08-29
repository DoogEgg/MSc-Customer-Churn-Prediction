from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def run_eda(df, output_dir="results/eda"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_churn_distribution(df, output_path)
    plot_numeric_distributions(df, output_path)
    plot_churn_by_category(df, "Contract", output_path)
    plot_churn_by_category(df, "InternetService", output_path)
    plot_churn_by_category(df, "PaymentMethod", output_path)


def plot_churn_distribution(df, output_path):
    churn_counts = df["Churn"].value_counts()

    churn_counts.plot(kind="bar")
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(output_path / "churn_distribution.png")
    plt.close()


def plot_numeric_distributions(df, output_path):
    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

    for column in numeric_columns:
        df[column].plot(kind="hist", bins=30)

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_path / f"{column.lower()}_distribution.png")
        plt.close()


def plot_churn_by_category(df, column, output_path):
    churn_rates = pd.crosstab(
        df[column],
        df["Churn"],
        normalize="index"
    )

    churn_rates.plot(kind="bar")
    plt.title(f"Churn Rate by {column}")
    plt.xlabel(column)
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(output_path / f"churn_by_{column.lower()}.png")
    plt.close()