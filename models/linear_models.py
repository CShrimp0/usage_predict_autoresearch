"""Linear and sparse regression models."""

from __future__ import annotations

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge


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
        "huber": {
            "constructor": HuberRegressor,
            "search_space": {
                "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "model__epsilon": [1.1, 1.35, 1.5, 1.75],
                "model__max_iter": [1000],
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
        "pls_regression": {
            "constructor": PLSRegression,
            "search_space": {
                "model__n_components": [2, 4, 8, 16, 32],
                "model__scale": [False],
                "model__max_iter": [1000],
            },
        },
    }
