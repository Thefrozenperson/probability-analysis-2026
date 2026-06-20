"""Monte Carlo simulation utilities for football probabilistic models.

This module provides small, reusable helpers for sampling scorelines from a
probability mass function and aggregating summary statistics. The design goal
is to quantify uncertainty via simulation/bootstrapping rather than implement
any betting strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    """Simulation output.

    Attributes
    ----------
    samples:
        Simulated scorelines (home_goals, away_goals) with derived totals.
    aggregates:
        One-row DataFrame of summary statistics.
    """

    samples: pd.DataFrame
    aggregates: pd.DataFrame


def simulate_scorelines(
    probs: Dict[Tuple[int, int], float],
    *,
    n_sim: int = 10_000,
    random_state: Optional[int] = None,
) -> SimulationResult:
    """Simulate scorelines from a discrete probability distribution.

    Parameters
    ----------
    probs:
        Mapping from (home_goals, away_goals) -> probability.
    n_sim:
        Number of draws.
    random_state:
        Optional seed.

    Returns
    -------
    SimulationResult
        Simulated samples and summary aggregates.
    """

    rng = np.random.default_rng(random_state)

    scorelines = list(probs.keys())
    p = np.array([probs[s] for s in scorelines], dtype=float)

    mass = p.sum()
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Probability mass must be positive and finite.")

    p = p / mass
    idx = rng.choice(len(scorelines), size=n_sim, p=p)
    arr = np.array(scorelines)[idx]

    df = pd.DataFrame(arr, columns=["home_goals", "away_goals"])
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["goal_diff"] = df["home_goals"] - df["away_goals"]

    aggregates = pd.DataFrame(
        {
            "mean_home_goals": [df["home_goals"].mean()],
            "mean_away_goals": [df["away_goals"].mean()],
            "mean_total_goals": [df["total_goals"].mean()],
            "mean_goal_diff": [df["goal_diff"].mean()],
            "var_total_goals": [df["total_goals"].var(ddof=1)],
        }
    )

    return SimulationResult(samples=df, aggregates=aggregates)
