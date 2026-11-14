"""Simulate stock prices with geometric Brownian motion (GBM).

Under GBM the price at time T is

    S_T = S0 * exp((r - sigma^2 / 2) * T + sigma * sqrt(T) * Z),   Z ~ N(0, 1)

so ln(S_T) is normally distributed and S_T is lognormal.
"""

from __future__ import annotations

import numpy as np


def simulate_terminal_prices(s0, r, sigma, T, n_sims, rng=None):
    """Return `n_sims` independent draws of the price at time T."""
    rng = rng or np.random.default_rng()
    z = rng.standard_normal(n_sims)
    return s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def simulate_paths(s0, r, sigma, T, n_steps, n_paths, rng=None):
    """Return price paths of shape (n_paths, n_steps + 1); column 0 is S0.

    Each step multiplies the price by an independent lognormal factor.
    Only needed for pictures: option prices only depend on the final price.
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_steps = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.hstack([np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)])
    return s0 * np.exp(log_paths)
