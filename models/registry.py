"""Central model registry."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from models.ensemble_models import get_optional_ensemble_models
from models.kernel_models import get_kernel_models
from models.linear_models import get_linear_models
from models.tree_models import get_tree_models

LOGGER = logging.getLogger("usage_predict_feature_engineering")


def get_model_registry() -> dict[str, dict[str, Any]]:
    registry = {}
    registry.update(get_linear_models())
    registry.update(get_kernel_models())
    registry.update(get_tree_models())
    registry.update(get_optional_ensemble_models())
    return registry


def build_model(name: str, params: dict[str, Any] | None = None, random_state: int | None = None):
    """Instantiate a registered model."""
    registry = get_model_registry()
    if name not in registry:
        raise KeyError(f"Unknown model: {name}. Available: {sorted(registry)}")
    constructor = registry[name]["constructor"]
    kwargs = dict(params or {})

    signature = inspect.signature(constructor)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    valid_names = set(signature.parameters)

    filtered_kwargs = {}
    dropped_kwargs = {}
    for key, value in kwargs.items():
        if accepts_var_kwargs or key in valid_names:
            filtered_kwargs[key] = value
        else:
            dropped_kwargs[key] = value

    if dropped_kwargs:
        LOGGER.warning(
            "Dropping unsupported constructor params for model '%s': %s",
            name,
            dropped_kwargs,
        )

    if random_state is not None and "random_state" in valid_names and "random_state" not in filtered_kwargs:
        filtered_kwargs["random_state"] = random_state

    return constructor(**filtered_kwargs)


def get_search_space(name: str) -> dict[str, Any]:
    """Return default hyperparameter search space for a model."""
    registry = get_model_registry()
    if name not in registry:
        raise KeyError(f"Unknown model: {name}")
    return dict(registry[name].get("search_space", {}))
