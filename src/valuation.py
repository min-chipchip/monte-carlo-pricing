"""Option valuation diagnostics: confidence intervals, strike chains, convergence and parity."""

from __future__ import annotations

from math import exp

import numpy as np
import pandas as pd

from .black_scholes import black_scholes_call, black_scholes_put
from .comparison import compare_call_prices, compare_put_prices
from .gbm import simulate_terminal_prices
from .monte_carlo import monte_carlo_call_price, monte_carlo_put_price

_COMPARE = {"call": compare_call_prices, "put": compare_put_prices}
_MC_PRICE = {"call": monte_carlo_call_price, "put": monte_carlo_put_price}
_BS_PRICE = {"call": black_scholes_call, "put": black_scholes_put}


def _check_kind(kind: str) -> None:
    if kind not in _COMPARE:
        raise ValueError("kind must be 'call' or 'put'")


def payoff(terminal_prices: np.ndarray, strike: float, kind: str = "call") -> np.ndarray:
    """Undiscounted European payoff for each simulated terminal price."""
    _check_kind(kind)
    prices = np.asarray(terminal_prices, dtype=float)
    return np.maximum(prices - strike, 0.0) if kind == "call" else np.maximum(strike - prices, 0.0)


def monte_carlo_standard_error(
    terminal_prices: np.ndarray, strike: float, time_to_maturity: float, r: float, kind: str = "call"
) -> float:
    """SE of the MC price: e^{-rT} * sd(payoff) / sqrt(N)."""
    p = payoff(terminal_prices, strike, kind)
    if p.size < 2:
        raise ValueError("need at least 2 simulated prices for a standard error")
    return float(exp(-r * time_to_maturity) * p.std(ddof=1) / np.sqrt(p.size))


def price_with_confidence(
    terminal_prices: np.ndarray,
    s0: float,
    strike: float,
    time_to_maturity: float,
    r: float,
    sigma: float,
    kind: str = "call",
    z: float = 1.96,
) -> dict:
    """MC price with standard error and CI, benchmarked against Black--Scholes."""
    _check_kind(kind)
    result = _COMPARE[kind](terminal_prices, s0, strike, time_to_maturity, r, sigma)
    se = monte_carlo_standard_error(terminal_prices, strike, time_to_maturity, r, kind)
    mc, bs = result["monte_carlo_price"], result["black_scholes_price"]
    low, high = mc - z * se, mc + z * se
    return {
        "kind": kind,
        "strike": float(strike),
        "mc_price": mc,
        "bs_price": bs,
        "std_error": se,
        "ci_low": low,
        "ci_high": high,
        "abs_diff": result["absolute_difference"],
        "z_score": (mc - bs) / se if se > 0 else np.nan,
        "bs_in_ci": bool(low <= bs <= high),
    }


def price_chain(
    terminal_prices: np.ndarray,
    s0: float,
    strikes,
    time_to_maturity: float,
    r: float,
    sigma: float,
    kinds=("call", "put"),
    z: float = 1.96,
) -> pd.DataFrame:
    """price_with_confidence for every (kind, strike), using the same simulated sample."""
    return pd.DataFrame(
        [
            price_with_confidence(terminal_prices, s0, float(k), time_to_maturity, r, sigma, kind, z)
            for kind in kinds
            for k in strikes
        ]
    )


def convergence_study(
    s0: float,
    strike: float,
    time_to_maturity: float,
    r: float,
    sigma: float,
    n_grid,
    n_seeds: int = 20,
    kind: str = "call",
    reference_paths: int = 200_000,
    seed: int = 0,
) -> dict:
    """Pricing error vs number of simulations, over independent seeds.

    Returns ``n_grid``, ``errors`` (shape ``(len(n_grid), n_seeds)``), observed
    ``rmse``, the ``theoretical_rmse`` sd/sqrt(N), the discounted ``payoff_sd``
    and the Black--Scholes price ``bs_price``.
    """
    _check_kind(kind)
    n_grid = np.asarray(n_grid, dtype=int)
    bs_price = _BS_PRICE[kind](s0, strike, time_to_maturity, r, sigma)
    seeds = np.random.SeedSequence(seed).spawn(len(n_grid) * n_seeds + 1)

    errors = np.empty((len(n_grid), n_seeds))
    for i, n in enumerate(n_grid):
        for j in range(n_seeds):
            rng = np.random.default_rng(seeds[i * n_seeds + j])
            st = simulate_terminal_prices(s0, r, sigma, time_to_maturity, int(n), rng=rng)
            errors[i, j] = _MC_PRICE[kind](st, strike, time_to_maturity, r) - bs_price

    ref = simulate_terminal_prices(s0, r, sigma, time_to_maturity, reference_paths, rng=np.random.default_rng(seeds[-1]))
    payoff_sd = float(exp(-r * time_to_maturity) * payoff(ref, strike, kind).std())
    return {
        "n_grid": n_grid,
        "errors": errors,
        "rmse": np.sqrt((errors**2).mean(axis=1)),
        "theoretical_rmse": payoff_sd / np.sqrt(n_grid),
        "payoff_sd": payoff_sd,
        "bs_price": bs_price,
    }


def put_call_parity_table(
    terminal_prices: np.ndarray, s0: float, strikes, time_to_maturity: float, r: float, sigma: float
) -> pd.DataFrame:
    """C - P from MC and Black--Scholes against the model-free S0 - K e^{-rT}."""
    strikes = np.asarray(strikes, dtype=float)
    return pd.DataFrame(
        {
            "strike": strikes,
            "MC  C - P": [
                monte_carlo_call_price(terminal_prices, k, time_to_maturity, r)
                - monte_carlo_put_price(terminal_prices, k, time_to_maturity, r)
                for k in strikes
            ],
            "BS  C - P": [
                black_scholes_call(s0, k, time_to_maturity, r, sigma) - black_scholes_put(s0, k, time_to_maturity, r, sigma)
                for k in strikes
            ],
            "S0 - K e^{-rT}": s0 - strikes * exp(-r * time_to_maturity),
        }
    ).set_index("strike")


def parity_sampling_noise(terminal_prices: np.ndarray, s0: float, time_to_maturity: float, r: float) -> float:
    """e^{-rT} * mean(S_T) - S0: the exact size of the MC parity gap."""
    return float(exp(-r * time_to_maturity) * np.mean(terminal_prices) - s0)
