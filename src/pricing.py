"""Price European options two ways: the Black-Scholes formula and Monte Carlo.

Both compute the same thing, the discounted expected payoff

    price = e^(-rT) * E[payoff(S_T)]

Black-Scholes evaluates the expectation exactly; Monte Carlo estimates it
with a sample average (Law of Large Numbers) and reports a standard error
(Central Limit Theorem).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def payoff(terminal_prices, K, kind="call"):
    """Option payoff at expiry: max(S_T - K, 0) for a call, max(K - S_T, 0) for a put."""
    s = np.asarray(terminal_prices, dtype=float)
    if kind == "call":
        return np.maximum(s - K, 0.0)
    if kind == "put":
        return np.maximum(K - s, 0.0)
    raise ValueError("kind must be 'call' or 'put'")


def black_scholes(s0, K, T, r, sigma, kind="call"):
    """Exact Black-Scholes price of a European call or put (no dividends)."""
    d1 = (np.log(s0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return float(s0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    if kind == "put":
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1))
    raise ValueError("kind must be 'call' or 'put'")


def monte_carlo(terminal_prices, K, T, r, kind="call"):
    """Monte Carlo price and its standard error.

    price = e^(-rT) * mean(payoffs)
    SE    = e^(-rT) * sd(payoffs) / sqrt(N)
    A 95% confidence interval is price +/- 1.96 * SE.
    """
    discounted = np.exp(-r * T) * payoff(terminal_prices, K, kind)
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(discounted.size)
    return float(price), float(std_error)


def price_table(terminal_prices, s0, strikes, T, r, sigma, kinds=("call", "put")):
    """Monte Carlo price, 95% CI and Black-Scholes price for every (kind, strike).

    All options are priced from the same simulated `terminal_prices`.
    """
    rows = []
    for kind in kinds:
        for K in strikes:
            mc, se = monte_carlo(terminal_prices, K, T, r, kind)
            rows.append({
                "kind": kind,
                "strike": float(K),
                "mc_price": mc,
                "std_error": se,
                "ci_low": mc - 1.96 * se,
                "ci_high": mc + 1.96 * se,
                "bs_price": black_scholes(s0, K, T, r, sigma, kind),
            })
    return pd.DataFrame(rows)
