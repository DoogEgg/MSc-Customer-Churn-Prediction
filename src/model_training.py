import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline

from src.model_definitions import get_baseline_models


def train_and_evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    results = []
    models = get_baseline_models()

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "training_time": training_time
            }
        )

    return pd.DataFrame(results)