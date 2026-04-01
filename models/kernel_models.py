"""Kernel-based regression models."""

from __future__ import annotations

from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR


def get_kernel_models():
    return {
        "svr": {
            "constructor": SVR,
            "search_space": {
                "model__C": [0.1, 1.0, 10.0],
                "model__epsilon": [0.1, 0.5, 1.0],
                "model__kernel": ["rbf", "linear"],
            },
        },
        "kernel_ridge": {
            "constructor": KernelRidge,
            "search_space": {
                "model__alpha": [0.01, 0.1, 1.0],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": [None, 0.01, 0.1],
            },
        },
    }
