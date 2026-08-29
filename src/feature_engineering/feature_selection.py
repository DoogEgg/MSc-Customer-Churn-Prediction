from sklearn.feature_selection import SelectKBest, f_classif


SELECTED_FEATURE_COUNTS = [10, 20, 30]


def create_feature_selector(number_of_features):
    return SelectKBest(
        score_func=f_classif,
        k=number_of_features
    )