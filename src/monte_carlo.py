"""Monte Carlo estimators for European option prices."""

from __future__ import annotations

from math import exp

import numpy as np


def _price_discounted_payoff(payoff: np.ndarray, r: float, time_to_maturity: float) -> float:
    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive")
    if payoff.size == 0:
        raise ValueError("terminal_prices must not be empty")
    return float(exp(-r * time_to_maturity) * np.mean(payoff))


def monte_carlo_call_price(terminal_prices: np.ndarray, strike: float, time_to_maturity: float, r: float) -> float:
    """Price a European call from simulated terminal asset prices."""
    if strike <= 0:
        raise ValueError("strike must be positive")
    prices = np.asarray(terminal_prices, dtype=float)
    return _price_discounted_payoff(np.maximum(prices - strike, 0.0), r, time_to_maturity)


def monte_carlo_put_price(terminal_prices: np.ndarray, strike: float, time_to_maturity: float, r: float) -> float:
    """Price a European put from simulated terminal asset prices."""
    if strike <= 0:
        raise ValueError("strike must be positive")
    prices = np.asarray(terminal_prices, dtype=float)
    return _price_discounted_payoff(np.maximum(strike - prices, 0.0), r, time_to_maturity)
