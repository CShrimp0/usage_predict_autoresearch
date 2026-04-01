"""Central model registry."""

from __future__ import annotations

from typing import Any

from models.ensemble_models import get_optional_ensemble_models
from models.kernel_models import get_kernel_models
from models.linear_models import get_linear_models
from models.tree_models import get_tree_models


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
    if random_state is not None:
        try:
            return constructor(random_state=random_state, **kwargs)
        except TypeError:
            return constructor(**kwargs)
    return constructor(**kwargs)


def get_search_space(name: str) -> dict[str, Any]:
    """Return default hyperparameter search space for a model."""
    registry = get_model_registry()
    if name not in registry:
        raise KeyError(f"Unknown model: {name}")
    return dict(registry[name].get("search_space", {}))
