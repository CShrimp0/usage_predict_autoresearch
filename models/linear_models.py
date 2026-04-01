"""Linear and sparse regression models."""

from __future__ import annotations

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge


def get_linear_models():
    return {
        "linear_regression": {
            "constructor": LinearRegression,
            "search_space": {},
        },
        "ridge": {
            "constructor": Ridge,
            "search_space": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        },
        "lasso": {
            "constructor": Lasso,
            "search_space": {
                "model__alpha": [0.0005, 0.001, 0.01, 0.1, 1.0],
                "model__max_iter": [5000],
            },
        },
        "elasticnet": {
            "constructor": ElasticNet,
            "search_space": {
                "model__alpha": [0.0005, 0.001, 0.01, 0.1, 1.0],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "model__max_iter": [5000],
            },
        },
    }
