"""Evaluation metrics for probabilistic models.

This module provides helpers for evaluating probabilistic forecasts using
log loss, Brier score, accuracy, and simple calibration diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
except Exception:  # pragma: no cover
    accuracy_score = None  # type: ignore
    brier_score_loss = None  # type: ignore
    log_loss = None  # type: ignore


@dataclass
class CalibrationBin:
    prob_mean: float
    observed_rate: float
    count: int


def _check_sklearn() -> None:
    if any(x is None for x in (accuracy_score, brier_score_loss, log_loss)):
        raise RuntimeError("scikit-learn not installed; install from requirements.txt")


def score_binary(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Return standard metrics for binary event probabilities."""
    _check_sklearn()
    probs = np.asarray(probs).ravel()
    labels = np.asarray(labels).ravel()
    acc = float(accuracy_score(labels, probs >= 0.5))
    bri = float(brier_score_loss(labels, probs))
    ll = float(log_loss(labels, probs, eps=1e-15))
    return {"accuracy": acc, "brier": bri, "log_loss": ll}


def calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return calibration curve (mean predicted prob, observed rate) for bins."""
    probs = np.asarray(probs).ravel()
    labels = np.asarray(labels).ravel()

    order = np.argsort(probs)
    probs_sorted = probs[order]
    labels_sorted = labels[order]

    bins = np.array_split(np.arange(len(probs_sorted)), n_bins)

    prob_means = np.array([float(probs_sorted[b].mean()) for b in bins])
    obs_rates = np.array([float(labels_sorted[b].mean()) for b in bins])

    return prob_means, obs_rates
