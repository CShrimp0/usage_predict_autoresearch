"""Parallel backend helpers to avoid noisy process-shutdown issues."""

from __future__ import annotations

from contextlib import nullcontext

try:
    from joblib import parallel_backend
except ImportError:  # pragma: no cover
    parallel_backend = None


def threading_backend_context(n_jobs=None):
    """Prefer joblib's threading backend over process-based loky."""
    if parallel_backend is None:
        return nullcontext()
    return parallel_backend("threading", n_jobs=n_jobs)
