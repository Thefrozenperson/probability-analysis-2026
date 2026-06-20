"""Poisson goal models.

This module provides basic stubs for fitting Poisson GLMs to football goal counts
and predicting scoreline distributions.

These stubs are for research/teaching use with public datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None  # type: ignore

try:
    from scipy.stats import poisson
except Exception:  # pragma: no cover
    poisson = None  # type: ignore


def fit_poisson_goals(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    features: np.ndarray,
) -> Dict[str, Any]:
    """Fit independent Poisson GLMs for home/away goals."""
    if sm is None:
        raise RuntimeError("statsmodels not installed; install from requirements.txt")

    home_model = sm.GLM(home_goals, features, family=sm.families.Poisson()).fit()
    away_model = sm.GLM(away_goals, features, family=sm.families.Poisson()).fit()
    return {"home_model": home_model, "away_model": away_model}


def predict_scoreline(
    params: Dict[str, Any],
    features: np.ndarray,
    max_goals: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict scoreline probabilities up to max_goals."""
    if sm is None:
        raise RuntimeError("statsmodels not installed; install from requirements.txt")
    if poisson is None:
        raise RuntimeError("scipy not installed; install from requirements.txt")

    home_lambda = np.asarray(params["home_model"].predict(features))
    away_lambda = np.asarray(params["away_model"].predict(features))

    k = np.arange(0, max_goals + 1)

    home_pmf = poisson.pmf(k[None, :], home_lambda[:, None])
    away_pmf = poisson.pmf(k[None, :], away_lambda[:, None])

    prob_matrices = home_pmf[:, :, None] * away_pmf[:, None, :]
    expected_goals = np.stack([home_lambda, away_lambda], axis=1)
    return prob_matrices, expected_goals
