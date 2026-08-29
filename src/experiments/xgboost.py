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
from xgboost import XGBClassifier


def test_n_estimators_values(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    n_estimators_values = [50, 100, 200, 300]
    results = []

    for n_estimators_value in n_estimators_values:
        model = XGBClassifier(
            n_estimators=n_estimators_value,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric="logloss"
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
            "n_estimators": n_estimators_value,
            "learning_rate": 0.1,
            "max_depth": 6,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "training_time": training_time
        })

    return pd.DataFrame(results)


def test_learning_rate_values(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    learning_rate_values = [0.01, 0.05, 0.1, 0.2, 0.3]
    results = []

    for learning_rate_value in learning_rate_values:
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=learning_rate_value,
            max_depth=6,
            random_state=42,
            eval_metric="logloss"
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
            "n_estimators": 100,
            "learning_rate": learning_rate_value,
            "max_depth": 6,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "training_time": training_time
        })

    return pd.DataFrame(results)


def test_max_depth_values(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    max_depth_values = [2, 3, 4, 6, 8, 10]
    results = []

    for max_depth_value in max_depth_values:
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=max_depth_value,
            random_state=42,
            eval_metric="logloss"
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
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": max_depth_value,
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
    n_estimators_values = [100, 200]
    learning_rate_values = [0.05, 0.1, 0.2]
    max_depth_values = [3, 6, 9]
    results = []

    for n_estimators_value in n_estimators_values:
        for learning_rate_value in learning_rate_values:
            for max_depth_value in max_depth_values:
                model = XGBClassifier(
                    n_estimators=n_estimators_value,
                    learning_rate=learning_rate_value,
                    max_depth=max_depth_value,
                    random_state=42,
                    eval_metric="logloss"
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
                    "n_estimators": n_estimators_value,
                    "learning_rate": learning_rate_value,
                    "max_depth": max_depth_value,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_proba),
                    "training_time": training_time
                })

    return pd.DataFrame(results)