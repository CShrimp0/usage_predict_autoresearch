"""Optional third-party ensemble regressors."""

from __future__ import annotations


def get_optional_ensemble_models():
    models = {}

    try:
        from xgboost import XGBRegressor

        models["xgboost"] = {
            "constructor": XGBRegressor,
            "search_space": {
                "model__n_estimators": [200, 500],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.03, 0.1],
            },
        }
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor

        models["lightgbm"] = {
            "constructor": LGBMRegressor,
            "search_space": {
                "model__n_estimators": [200, 500],
                "model__num_leaves": [31, 63],
                "model__learning_rate": [0.03, 0.1],
            },
        }
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor

        models["catboost"] = {
            "constructor": CatBoostRegressor,
            "search_space": {
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.1],
                "model__iterations": [300, 500],
            },
        }
    except ImportError:
        pass

    return models
