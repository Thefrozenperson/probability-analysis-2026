"""Pipeline runner for probability-analysis-2026.

This module stitches together the key steps:

- load datasets
- clean and standardize raw data
- construct baseline features
- fit probabilistic models
- evaluate forecasts

It is intentionally lightweight: experiment-specific code should live in notebooks/report.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_download import load_csv
from src.data_clean import clean_matches
from src.feature_engineering import basic_features
from src.model_poisson import fit_poisson_model, predict_scoreline_probs
from src.model_bt import elo_ratings, bt_probabilities
from src.evaluation import brier_score, log_loss_score


def run_pipeline(matches_path: str | Path) -> dict:
    """Run the end-to-end baseline pipeline on a match-level dataset.

    Parameters
    ----------
    matches_path:
        Path to a CSV file with at least: date, home_team, away_team, home_goals, away_goals.

    Returns
    -------
    dict
        Aggregated results (e.g., model objects, evaluation metrics).
    """

    df = load_csv(matches_path)
    df = clean_matches(df)
    df = basic_features(df)

    poisson_model = fit_poisson_model(df)
    elo = elo_ratings(df)

    # Example: compute a few evaluation metrics using simplified probabilities.
    # In practice you may want full scoreline distributions.
    dummy_probs = [0.5] * len(df)
    brier = brier_score(df["home_goals"], dummy_probs)

    return {
        "poisson_model": poisson_model,
        "elo": elo,
        "brier": brier,
    }


def main() -> None:
    # Placeholder: wire your dataset path here.
    raise NotImplementedError("Set matches_path and customize run_pipeline().")


if __name__ == "__main__":
    main()
