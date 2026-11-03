"""Closed-form Black--Scholes prices for European options."""

from __future__ import annotations

from math import exp, log, sqrt

from scipy.stats import norm


def _d1_d2(s0: float, strike: float, time_to_maturity: float, r: float, sigma: float) -> tuple[float, float]:
    if s0 <= 0 or strike <= 0:
        raise ValueError("s0 and strike must be positive")
    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    volatility_term = sigma * sqrt(time_to_maturity)
    d1 = (log(s0 / strike) + (r + 0.5 * sigma**2) * time_to_maturity) / volatility_term
    return d1, d1 - volatility_term


def black_scholes_call(s0: float, strike: float, time_to_maturity: float, r: float, sigma: float) -> float:
    """Return the Black--Scholes value of a European call with no dividends."""
    d1, d2 = _d1_d2(s0, strike, time_to_maturity, r, sigma)
    return float(s0 * norm.cdf(d1) - strike * exp(-r * time_to_maturity) * norm.cdf(d2))


def black_scholes_put(s0: float, strike: float, time_to_maturity: float, r: float, sigma: float) -> float:
    """Return the Black--Scholes value of a European put with no dividends."""
    d1, d2 = _d1_d2(s0, strike, time_to_maturity, r, sigma)
    return float(strike * exp(-r * time_to_maturity) * norm.cdf(-d2) - s0 * norm.cdf(-d1))
