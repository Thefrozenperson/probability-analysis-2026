"""Risk metrics for an abstract return series.

The functions below operate on a user-provided return series (e.g. daily log returns)
and compute risk measures like volatility, max drawdown, and Value at Risk.
No trading strategy is implemented; these are purely statistical summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RiskMetrics:
    volatility: float
    max_drawdown: float
    var: float
    cvar: float


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Compute maximum drawdown from a cumulative equity curve."""
    eq = np.asarray(equity_curve, dtype=float)
    if eq.ndim != 1:
        raise ValueError("equity_curve must be 1-D")

    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    return float(np.min(drawdowns)) if drawdowns.size else 0.0


def value_at_risk(returns: Sequence[float], alpha: float = 0.95) -> float:
    """Historical VaR (alpha quantile) on a return series."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1:
        raise ValueError("returns must be 1-D")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    return float(np.quantile(r, 1 - alpha))


def conditional_value_at_risk(returns: Sequence[float], alpha: float = 0.95) -> float:
    """Historical CVaR (expected shortfall) on a return series."""
    r = np.asarray(returns, dtype=float)
    var = value_at_risk(r, alpha=alpha)

    tail = r[r <= var]
    return float(tail.mean()) if tail.size else float('nan')


def summarize(returns: Sequence[float], alpha: float = 0.95) -> RiskMetrics:
    """Summarize risk metrics for a return series."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("returns must be 1-D and have at least 2 elements")

    vol = float(r.std(ddof=1))

    eq = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    mdd = float(max_drawdown(eq))

    var = float(value_at_risk(r, alpha=alpha))
    cvar = float(conditional_value_at_risk(r, alpha=alpha))

    return RiskMetrics(volatility=vol, max_drawdown=mdd, var=var, cvar=cvar)
