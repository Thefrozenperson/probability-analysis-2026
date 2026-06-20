"""Bradley-Terry / Elo-style rating models.

This module includes minimal stubs for estimating team strength parameters
from pairwise match outcomes and converting strength differences into win
probabilities.

All implementations are simplified for research/teaching use with public data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

try:
    from scipy.special import expit
except Exception:  # pragma: no cover
    def expit(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))


@dataclass
class EloConfig:
    k_factor: float = 20.0
    home_advantage: float = 60.0


def elo_expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for team A vs team B using Elo formula."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def elo_update(
    rating_a: float,
    rating_b: float,
    score_a: float,
    score_b: float,
    cfg: EloConfig = EloConfig(),
) -> Tuple[float, float]:
    """Update Elo ratings based on observed scores (0/0.5/1)."""
    expected_a = elo_expected_score(rating_a + cfg.home_advantage, rating_b)
    expected_b = 1.0 - expected_a
    new_a = rating_a + cfg.k_factor * (score_a - expected_a)
    new_b = rating_b + cfg.k_factor * (score_b - expected_b)
    return new_a, new_b


def bt_probabilities(strength_diff: np.ndarray) -> np.ndarray:
    """Bradley-Terry win probability (team A)."""
    return expit(strength_diff)


def bt_strength_from_elo(rating_a: float, rating_b: float) -> float:
    """Convenience: convert Elo difference to a logistic strength diff."""
    # Map Elo diff to log-odds under canonical formula.
    return np.log(10) * (rating_a - rating_b) / 400
