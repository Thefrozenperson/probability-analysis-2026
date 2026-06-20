"""
value_betting.py

Educational utilities for betting analysis (implied probability, expected value, Kelly criterion).
No guarantee of profit. Use responsibly and follow local laws/regulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


def implied_prob(decimal_odds: float) -> float:
  if decimal_odds <= 0:
    raise ValueError("decimal_odds must be positive.")
  return 1.0 / decimal_odds


def expected_value(prob: float, decimal_odds: float) -> float:
  """Expected value (EV) per unit stake for decimal odds."""
  return prob * (decimal_odds - 1.0) - (1.0 - prob)


def kelly_fraction(prob: float, decimal_odds: float) -> float:
  """
  Kelly fraction for decimal odds:

  b = decimal_odds - 1
  f* = (p*b - q) / b

  Only returns positive values; if the edge is negative, returns 0.
  """
  b = decimal_odds - 1.0
  if b <= 0:
    return 0.0
  q = 1.0 - prob
  f_star = (prob * b - q) / b
  return max(0.0, f_star)


@dataclass
class Bet:
  prob: float
  decimal_odds: float
  label: str = ""  # e.g. "home", "draw", "away"

  def ev(self) -> float:
    return expected_value(self.prob, self.decimal_odds)

  def kelly(self) -> float:
    return kelly_fraction(self.prob, self.decimal_odds)


def kelly_stake(bankroll: float, kelly_frac: float, fraction: float = 1.0) -> float:
  """Stake size given bankroll and Kelly fraction.

  fraction can be <1.0 for risk control (half Kelly, quarter Kelly, etc.).
  """
  if bankroll <= 0:
    return 0.0
  if fraction <= 0:
    return 0.0
  return bankroll * kelly_frac * fraction


def summarize_bets(bets: Sequence[Bet], bankroll: float, fraction: float = 0.5) -> List[dict]:
  """Produce a concise summary table for a list of bets."""
  summary = []
  for bet in bets:
    k_frac = bet.kelly()
    stake = kelly_stake(bankroll, k_frac, fraction=fraction)
    summary.append(
      {
        "label": bet.label,
        "prob": bet.prob,
        "odds": bet.decimal_odds,
        "implied_prob": implied_prob(bet.decimal_odds),
        "ev": bet.ev(),
        "kelly_fraction": k_frac,
        "stake": stake,
      }
    )
  return summary
