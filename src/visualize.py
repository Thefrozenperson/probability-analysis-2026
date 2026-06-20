"""Visualization helpers for probability-analysis-2026.

The goal of this module is exploratory analysis: plotting distributions,
residuals, and calibration curves. It intentionally avoids any code that
suggests wagering strategies.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_scoreline_probs(probs: dict, *, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot a heatmap-like visualization for scoreline probabilities.

    Parameters
    ----------
    probs:
        Mapping (home_goals, away_goals) -> probability.
    ax:
        Optional Matplotlib axes.

    Returns
    -------
    plt.Axes
        The axes containing the plot.
    """

    ax = ax if ax is not None else plt.gca()

    # Determine extents
    scorelines = list(probs.keys())
    if not scorelines:
        raise ValueError("probs must not be empty")

    max_home = max(h for h, _ in scorelines)
    max_away = max(a for _, a in scorelines)

    grid = np.zeros((max_home + 1, max_away + 1), dtype=float)
    for (h, a), p in probs.items():
        if h < 0 or a < 0:
            continue
        grid[h, a] = float(p)

    im = ax.imshow(grid, origin="lower")
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    ax.set_title("Scoreline probabilities")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax

def plot_calibration_curve(
    y_true: pd.Series,
    y_prob: pd.Series,
    *,
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a calibration curve from observed outcomes and predicted probabilities."""

    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    ax = ax if ax is not None else plt.gca()
    ax.plot(prob_pred, prob_true, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curve")
    return ax
