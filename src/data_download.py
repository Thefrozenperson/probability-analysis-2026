"""
data_download.py

Download or load public football datasets (matches/teams/goals).

Only public data sources are supported; scraping logic should comply with the
source site's terms.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class DataSource:
    """Metadata for a public data source."""

    name: str
    url: str
    description: str = ""


def load_csv(path: str, *, encoding: str = "utf-8") -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""

    return pd.read_csv(path, encoding=encoding)


def expected_fields() -> list[str]:
    """Return the canonical field list for match-level datasets."""

    return [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "tournament",
        "city",
        "country",
        "neutral",
    ]
