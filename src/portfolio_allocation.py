"""src/portfolio_allocation.py

Minimal wallet/portfolio allocation utilities for probability-analysis-2026.

This module is intentionally lightweight: it works with the repo's core
requirements. If you want a full Efficient Frontier or CVaR optimizer,
you can install optional libraries (e.g., PyPortfolioOpt, Riskfolio-Lib,
or Keeks-style run tracking) and extend the functions here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.value_betting import expected_value, kelly_fraction
from src.risk_metrics import conditional_value_at_risk, max_drawdown, value_at_risk


@dataclass(frozen=True)
class WalletPlan:
    positions: pd.DataFrame
    exposure_sum: float


def _risk_proxy(probability: pd.Series, odds: pd.Series) -> pd.Series:
    """Return standard deviation of the binary return distribution per unit stake."""

    win_r = odds.astype(float) - 1.0
    lose_r = -1.0

    p = probability.astype(float)
    mean = p * win_r + (1.0 - p) * lose_r
    var = p * win_r**2 + (1.0 - p) * lose_r**2 - mean**2
    var = var.clip(lower=0.0)
    return np.sqrt(var)


def kelly_plan(
    df: pd.DataFrame,
    probability_col: str,
    odds_col: str,
    id_col: Optional[str] = None,
    fractional_kelly: float = 0.25,
    max_exposure: float = 0.25,
    min_ev: float = 0.0,
) -> WalletPlan:
    """Kelly-based capital plan.

    - keeps only EV >= min_ev
    - stakes = fractional_kelly * KellyFraction
    - scales down by a simple risk proxy
    - caps total exposure to max_exposure
    """

    df = df.copy()

    df["ev"] = expected_value(df[probability_col], df[odds_col])
    df["kelly_full"] = kelly_fraction(df[probability_col], df[odds_col])

    df = df[(df["ev"] >= min_ev) & (df["kelly_full"] > 0)]
    if df.empty:
        return WalletPlan(positions=df, exposure_sum=0.0)

    df["stake_fraction"] = df["kelly_full"] * float(fractional_kelly)

    risk = _risk_proxy(df[probability_col], df[odds_col])
    risk = risk.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    df["stake_fraction"] = df["stake_fraction"] / risk

    exposure_sum = float(df["stake_fraction"].sum())
    if exposure_sum > max_exposure and exposure_sum > 0:
        df["stake_fraction"] *= max_exposure / exposure_sum
        exposure_sum = max_exposure

    cols = [probability_col, odds_col, "ev", "kelly_full", "stake_fraction"]
    if id_col is not None:
        cols = [id_col] + cols

    df = df[cols].reset_index(drop=True)
    return WalletPlan(positions=df, exposure_sum=exposure_sum)


def equity_curve(returns: pd.Series) -> pd.Series:
    """Turn a realized return series into an equity curve (starting at 1)."""

    returns = returns.astype(float)
    return (1.0 + returns).cumprod()


def wallet_risk_summary(returns: pd.Series, alpha: float = 0.95) -> dict:
    """Risk summary (drawdown, VaR/CVaR) computed with src.risk_metrics."""

    curve = equity_curve(returns)
    return {
        "max_drawdown": float(max_drawdown(curve)),
        "var": float(value_at_risk(returns, alpha=alpha)),
        "cvar": float(conditional_value_at_risk(returns, alpha=alpha)),
    }


__all__ = [
    "WalletPlan",
    "kelly_plan",
    "equity_curve",
    "wallet_risk_summary",
]
