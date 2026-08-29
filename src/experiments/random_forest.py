import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline


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
        model = RandomForestClassifier(
            n_estimators=n_estimators_value,
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
            "n_estimators": n_estimators_value,
            "max_depth": None,
            "min_samples_split": 2,
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
    max_depth_values = [None, 5, 10, 15, 20]
    results = []

    for max_depth_value in max_depth_values:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=max_depth_value,
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
            "n_estimators": 100,
            "max_depth": max_depth_value,
            "min_samples_split": 2,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "training_time": training_time
        })

    return pd.DataFrame(results)


def test_min_samples_split_values(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    min_samples_split_values = [2, 5, 10, 20]
    results = []

    for min_samples_split_value in min_samples_split_values:
        model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=min_samples_split_value,
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
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": min_samples_split_value,
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
    max_depth_values = [5, 10, 15, None]
    min_samples_split_values = [2, 10, 20]
    results = []

    for n_estimators_value in n_estimators_values:
        for max_depth_value in max_depth_values:
            for min_samples_split_value in min_samples_split_values:
                model = RandomForestClassifier(
                    n_estimators=n_estimators_value,
                    max_depth=max_depth_value,
                    min_samples_split=min_samples_split_value,
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
                    "n_estimators": n_estimators_value,
                    "max_depth": max_depth_value,
                    "min_samples_split": min_samples_split_value,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_proba),
                    "training_time": training_time
                })

    return pd.DataFrame(results)