"""
Feature engineering utilities for football match datasets.

This module adds derived features such as goal difference and total goals.
"""

from __future__ import annotations

import pandas as pd


def add_basic_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with additional derived features."""

    out = matches.copy()
    out["goal_diff"] = out["home_goals"] - out["away_goals"]
    out["total_goals"] = out["home_goals"] + out["away_goals"]
    return out
