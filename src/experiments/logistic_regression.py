import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline


def test_c_values(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    c_values = [0.001, 0.01, 0.1, 1, 10, 100]
    results = []

    for c_value in c_values:
        model = LogisticRegression(
            C=c_value,
            max_iter=1000,
            random_state=42
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        results.append({
            "C": c_value,
            "penalty": model.penalty,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "training_time": training_time
        })

    return pd.DataFrame(results)


def test_penalty_values(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    penalty_values = ["l1", "l2"]
    results = []

    for penalty_value in penalty_values:
        model = LogisticRegression(
            C=1.0,
            penalty=penalty_value,
            solver="liblinear",
            max_iter=1000,
            random_state=42
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        results.append({
            "C": 1.0,
            "penalty": penalty_value,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "training_time": training_time
        })

    return pd.DataFrame(results)


def test_combined_values(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    c_values = [0.01, 0.1, 1, 10]
    penalty_values = ["l1", "l2"]
    results = []

    for c_value in c_values:
        for penalty_value in penalty_values:
            model = LogisticRegression(
                C=c_value,
                penalty=penalty_value,
                solver="liblinear",
                max_iter=1000,
                random_state=42
            )

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            start_time = time.time()
            pipeline.fit(X_train, y_train)
            training_time = time.time() - start_time

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            results.append({
                "C": c_value,
                "penalty": penalty_value,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "training_time": training_time
            })

    return pd.DataFrame(results)