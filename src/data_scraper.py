"""
data_scraper.py

Optional module for scraping football match data from public sites.

NOTE: scraping must comply with terms of use; prefer published datasets
when possible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScrapeResult:
    """Container for scraped content or parsed records."""

    raw: str


def scrape(url: str) -> ScrapeResult:
    """Download raw text from a public URL."""

    raise NotImplementedError("Implement scraping with an appropriate HTTP client")
