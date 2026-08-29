from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def get_baseline_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42
        ),
        "SVM": SVC(
            probability=True,
            random_state=42
        )
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )

    return models