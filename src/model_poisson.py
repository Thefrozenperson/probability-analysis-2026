"""Poisson goal models.

This module provides basic stubs for fitting Poisson GLMs to football goal counts
and predicting scoreline distributions.

These stubs are for research/teaching use with public datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _extract_numeric_feature_cols(df, drop_cols: Sequence[str]) -> List[str]:
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    return [c for c in num_cols if c not in drop_cols]


def _to_design_matrix(
    df_or_array: Any,
    feature_cols: Sequence[str],
    add_constant: bool,
) -> np.ndarray:
    if isinstance(df_or_array, np.ndarray):
        X = df_or_array
    else:
        X = df_or_array[list(feature_cols)].to_numpy()

    if add_constant:
        if sm is None:
            raise RuntimeError("statsmodels not installed; install from requirements.txt")
        X = sm.add_constant(X, prepend=True, has_constant="add")
    return X


@dataclass
class PoissonModel:
    """Container for a Poisson goal model.

    This is a lightweight service wrapper aimed at skill/distillation scenarios:
    - keeps track of which feature columns were used
    - optionally adds an intercept column at fit/predict time
    - exposes max_goals for downstream market probability derivation
    """

    home_model: Any
    away_model: Any
    feature_cols: List[str]
    add_constant: bool = True
    max_goals: int = 6


def fit_poisson_model(
    df,
    *,
    home_goals_col: str = "home_goals",
    away_goals_col: str = "away_goals",
    feature_cols: Optional[Sequence[str]] = None,
    add_constant: bool = True,
    max_goals: int = 6,
) -> PoissonModel:
    """Fit a Poisson model from a DataFrame.

    If `feature_cols` is None, numeric columns (excluding goal columns) are used.
    If no features are available, an intercept-only model is fit.

    This is intended for baseline research usage; production-quality feature
    selection/standardization should live in feature_engineering.
    """

    if sm is None:
        raise RuntimeError("statsmodels not installed; install from requirements.txt")

    df = df.copy()

    drop_cols = {home_goals_col, away_goals_col, "home_goals", "away_goals"}

    if feature_cols is None:
        feature_cols = _extract_numeric_feature_cols(df, drop_cols)

    feature_cols = list(feature_cols)

    if len(feature_cols) == 0:
        # Intercept-only model.
        feature_cols = []
        add_constant = False
        X = np.ones((len(df), 1))
    else:
        X = df[feature_cols].to_numpy()

    if add_constant:
        X = sm.add_constant(X, prepend=True, has_constant="add")

    y_home = df[home_goals_col].to_numpy()
    y_away = df[away_goals_col].to_numpy()

    params = fit_poisson_goals(y_home, y_away, X)
    return PoissonModel(
        home_model=params["home_model"],
        away_model=params["away_model"],
        feature_cols=feature_cols,
        add_constant=add_constant,
        max_goals=max_goals,
    )


def predict_scoreline_probs(
    model: PoissonModel,
    df_or_array: Any,
    *,
    max_goals: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict scoreline probability matrices.

    Returns
    -------
    prob_matrices : np.ndarray
        Shape (n_matches, max_goals+1, max_goals+1)
    expected_goals : np.ndarray
        Shape (n_matches, 2): lambdas / expected home/away goals
    """

    mg = model.max_goals if max_goals is None else int(max_goals)

    if isinstance(df_or_array, np.ndarray):
        X = df_or_array
    else:
        if model.feature_cols:
            X = df_or_array[model.feature_cols].to_numpy()
        else:
            X = np.ones((len(df_or_array), 1))

    if model.add_constant:
        if sm is None:
            raise RuntimeError("statsmodels not installed; install from requirements.txt")
        X = sm.add_constant(X, prepend=True, has_constant="add")

    params = {"home_model": model.home_model, "away_model": model.away_model}
    return predict_scoreline(params, X, max_goals=mg)
