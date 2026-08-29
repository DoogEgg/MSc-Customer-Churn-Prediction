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
from sklearn.tree import DecisionTreeClassifier


def test_max_depth_values(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    max_depth_values = [None, 3, 5, 7, 10, 15, 20]
    results = []

    for max_depth_value in max_depth_values:
        model = DecisionTreeClassifier(
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
            "max_depth": max_depth_value,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
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
    min_samples_split_values = [2, 5, 10, 20, 50]
    results = []

    for min_samples_split_value in min_samples_split_values:
        model = DecisionTreeClassifier(
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
            "max_depth": None,
            "min_samples_split": min_samples_split_value,
            "min_samples_leaf": 1,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "training_time": training_time
        })

    return pd.DataFrame(results)


def test_min_samples_leaf_values(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    min_samples_leaf_values = [1, 2, 5, 10, 20, 50]
    results = []

    for min_samples_leaf_value in min_samples_leaf_values:
        model = DecisionTreeClassifier(
            min_samples_leaf=min_samples_leaf_value,
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
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": min_samples_leaf_value,
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
    max_depth_values = [3, 5, 7, 10]
    min_samples_split_values = [2, 10, 20]
    min_samples_leaf_values = [1, 5, 10]
    results = []

    for max_depth_value in max_depth_values:
        for min_samples_split_value in min_samples_split_values:
            for min_samples_leaf_value in min_samples_leaf_values:
                model = DecisionTreeClassifier(
                    max_depth=max_depth_value,
                    min_samples_split=min_samples_split_value,
                    min_samples_leaf=min_samples_leaf_value,
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
                    "max_depth": max_depth_value,
                    "min_samples_split": min_samples_split_value,
                    "min_samples_leaf": min_samples_leaf_value,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_proba),
                    "training_time": training_time
                })

    return pd.DataFrame(results)