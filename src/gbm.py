"""Geometric Brownian motion simulation utilities."""

from __future__ import annotations

import numpy as np


def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    time_to_maturity: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate risk-neutral geometric Brownian motion price paths.

    Returns an array of shape ``(n_paths, n_steps + 1)``.  Column zero is
    the initial price, and every later column is one equally spaced time step.
    """
    if s0 <= 0:
        raise ValueError("s0 must be positive")
    if sigma < 0:
        raise ValueError("sigma cannot be negative")
    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must both be at least 1")

    generator = rng if rng is not None else np.random.default_rng()
    dt = time_to_maturity / n_steps
    shocks = generator.standard_normal((n_paths, n_steps))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_returns, axis=1))
    return paths


def simulate_terminal_prices(
    s0: float,
    r: float,
    sigma: float,
    time_to_maturity: float,
    n_paths: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate terminal prices directly, without constructing full paths."""
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")
    if s0 <= 0:
        raise ValueError("s0 must be positive")
    if sigma < 0:
        raise ValueError("sigma cannot be negative")
    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive")

    generator = rng if rng is not None else np.random.default_rng()
    shocks = generator.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * time_to_maturity
    diffusion = sigma * np.sqrt(time_to_maturity) * shocks
    return s0 * np.exp(drift + diffusion)
