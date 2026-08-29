from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from src.data_loader import load_data
from src.data_cleaning import clean_data
from src.preprocessing import prepare_data


def main():
    project_root = Path(__file__).resolve().parents[1]

    data_path = (
        project_root
        / "data"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    output_dir = project_root / "results" / "feature_engineering"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "feature_ranking.csv"

    # Load and clean data
    data = load_data(data_path)
    data = clean_data(data)

    # Prepare data using the same process as the main experiments
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        data,
        random_state=42
    )

    # Fit preprocessing only on the training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # Calculate ANOVA F-scores for all features
    selector = SelectKBest(
        score_func=f_classif,
        k="all"
    )

    selector.fit(X_train_processed, y_train)

    # Create ranking table
    feature_ranking = pd.DataFrame({
        "Feature": feature_names,
        "F_score": selector.scores_,
        "P_value": selector.pvalues_
    })

    feature_ranking = feature_ranking.sort_values(
        by="F_score",
        ascending=False
    ).reset_index(drop=True)

    feature_ranking.insert(
        0,
        "Rank",
        range(1, len(feature_ranking) + 1)
    )

    feature_ranking.to_csv(
        output_path,
        index=False
    )

    print(feature_ranking)
    print()
    print(f"Feature ranking saved to: {output_path}")
    print(f"Total features: {len(feature_ranking)}")


if __name__ == "__main__":
    main()