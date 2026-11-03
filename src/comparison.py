"""Helpers for comparing simulated and closed-form option prices."""

from __future__ import annotations

import numpy as np

from .black_scholes import black_scholes_call, black_scholes_put
from .monte_carlo import monte_carlo_call_price, monte_carlo_put_price


def _comparison(monte_carlo_price: float, black_scholes_price: float) -> dict[str, float]:
    """Return consistently named prices and their absolute difference."""
    return {
        "monte_carlo_price": monte_carlo_price,
        "black_scholes_price": black_scholes_price,
        "absolute_difference": abs(monte_carlo_price - black_scholes_price),
    }


def compare_call_prices(
    terminal_prices: np.ndarray, s0: float, strike: float, time_to_maturity: float, r: float, sigma: float
) -> dict[str, float]:
    """Compare a simulated European call estimate with Black--Scholes."""
    mc_price = monte_carlo_call_price(terminal_prices, strike, time_to_maturity, r)
    bs_price = black_scholes_call(s0, strike, time_to_maturity, r, sigma)
    return _comparison(mc_price, bs_price)


def compare_put_prices(
    terminal_prices: np.ndarray, s0: float, strike: float, time_to_maturity: float, r: float, sigma: float
) -> dict[str, float]:
    """Compare a simulated European put estimate with Black--Scholes."""
    mc_price = monte_carlo_put_price(terminal_prices, strike, time_to_maturity, r)
    bs_price = black_scholes_put(s0, strike, time_to_maturity, r, sigma)
    return _comparison(mc_price, bs_price)
