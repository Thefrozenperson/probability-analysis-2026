"""
Data cleaning utilities for football match datasets.

This module defines transformations to standardize team names, parse dates,
and handle missing values and basic outlier filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import pandas as pd


@dataclass(slots=True)
class CleanConfig:
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    team_map: Optional[Mapping[str, str]] = None


def clean_matches(df: pd.DataFrame, config: CleanConfig) -> pd.DataFrame:
    """Return a cleaned match-level dataframe."""

    cleaned = df.copy()

    # Standardize team names
    if config.team_map:
        cleaned["home_team"] = cleaned["home_team"].replace(config.team_map)
        cleaned["away_team"] = cleaned["away_team"].replace(config.team_map)

    # Parse dates
    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    # Drop rows missing core fields
    cleaned = cleaned.dropna(
        subset=["home_team", "away_team", "home_goals", "away_goals"]
    )

    # Optional year filtering
    if "date" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["date"])
        if config.min_year is not None:
            cleaned = cleaned[cleaned["date"].dt.year >= config.min_year]
        if config.max_year is not None:
            cleaned = cleaned[cleaned["date"].dt.year <= config.max_year]

    cleaned = cleaned.reset_index(drop=True)
    return cleaned
