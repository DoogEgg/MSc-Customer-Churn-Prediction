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
from sklearn.svm import SVC


def test_c_values(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    c_values = [0.01, 0.1, 1, 10, 100]
    results = []

    for c_value in c_values:
        model = SVC(
            C=c_value,
            gamma="scale",
            probability=True,
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
            "gamma": "scale",
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "training_time": training_time
        })

    return pd.DataFrame(results)


def test_gamma_values(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    gamma_values = ["scale", "auto", 0.001, 0.01, 0.1, 1]
    results = []

    for gamma_value in gamma_values:
        model = SVC(
            C=1,
            gamma=gamma_value,
            probability=True,
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
            "C": 1,
            "gamma": gamma_value,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),
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
    c_values = [0.1, 1, 10]
    gamma_values = ["scale", 0.01, 0.1, 1]
    results = []

    for c_value in c_values:
        for gamma_value in gamma_values:
            model = SVC(
                C=c_value,
                gamma=gamma_value,
                probability=True,
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
                "gamma": gamma_value,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "training_time": training_time
            })

    return pd.DataFrame(results)