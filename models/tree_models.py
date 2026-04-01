"""Tree and boosting regression models."""

from __future__ import annotations

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)


def get_tree_models():
    return {
        "random_forest": {
            "constructor": RandomForestRegressor,
            "search_space": {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [1, 3, 5],
            },
        },
        "extra_trees": {
            "constructor": ExtraTreesRegressor,
            "search_space": {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [1, 3, 5],
            },
        },
        "gradient_boosting": {
            "constructor": GradientBoostingRegressor,
            "search_space": {
                "model__n_estimators": [100, 300],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            },
        },
        "hist_gradient_boosting": {
            "constructor": HistGradientBoostingRegressor,
            "search_space": {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [None, 6, 12],
                "model__max_leaf_nodes": [15, 31, 63],
            },
        },
    }
