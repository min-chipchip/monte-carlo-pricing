"""Reusable models and pricing utilities for the Monte Carlo project."""

from .analytics import (
    dispersion_over_time,
    lognormal_centres,
    percentile_bands,
    terminal_lognormal,
    terminal_summary,
    theoretical_log_sd,
    theoretical_mean,
    theoretical_price_sd,
)
from .black_scholes import black_scholes_call, black_scholes_put
from .comparison import compare_call_prices, compare_put_prices
from .gbm import simulate_gbm_paths, simulate_terminal_prices
from .monte_carlo import monte_carlo_call_price, monte_carlo_put_price
from .valuation import (
    convergence_study,
    monte_carlo_standard_error,
    parity_sampling_noise,
    payoff,
    price_chain,
    price_with_confidence,
    put_call_parity_table,
)

__all__ = [
    # models
    "black_scholes_call",
    "black_scholes_put",
    "simulate_gbm_paths",
    "simulate_terminal_prices",
    # Monte Carlo pricing
    "monte_carlo_call_price",
    "monte_carlo_put_price",
    "compare_call_prices",
    "compare_put_prices",
    # analytics
    "dispersion_over_time",
    "lognormal_centres",
    "percentile_bands",
    "terminal_lognormal",
    "terminal_summary",
    "theoretical_log_sd",
    "theoretical_mean",
    "theoretical_price_sd",
    # valuation diagnostics
    "convergence_study",
    "monte_carlo_standard_error",
    "parity_sampling_noise",
    "payoff",
    "price_chain",
    "price_with_confidence",
    "put_call_parity_table",
]
