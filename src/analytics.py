"""Statistics of simulated GBM paths and terminal prices, with closed-form benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- paths
def percentile_bands(paths: np.ndarray, percentiles=(5, 25, 50, 75, 95)) -> dict[int, np.ndarray]:
    """Cross-sectional percentiles at every time step, keyed by percentile."""
    values = np.percentile(paths, percentiles, axis=0)
    return {p: v for p, v in zip(percentiles, values)}


def theoretical_mean(s0: float, r: float, t: np.ndarray | float) -> np.ndarray:
    """E[S_t] = S0 * exp(r t)."""
    return s0 * np.exp(r * np.asarray(t, dtype=float))


def theoretical_log_sd(sigma: float, t: np.ndarray | float) -> np.ndarray:
    """sd(ln S_t) = sigma * sqrt(t)."""
    return sigma * np.sqrt(np.asarray(t, dtype=float))


def theoretical_price_sd(s0: float, r: float, sigma: float, t: np.ndarray | float) -> np.ndarray:
    """sd(S_t) = S0 e^{rt} sqrt(e^{sigma^2 t} - 1)."""
    t = np.asarray(t, dtype=float)
    return s0 * np.exp(r * t) * np.sqrt(np.exp(sigma**2 * t) - 1)


def dispersion_over_time(paths: np.ndarray, time_grid: np.ndarray, s0: float, r: float, sigma: float) -> pd.DataFrame:
    """Simulated vs theoretical spread of log-price and price at every time step."""
    return pd.DataFrame(
        {
            "sim_log_sd": np.log(paths).std(axis=0, ddof=1),
            "theo_log_sd": theoretical_log_sd(sigma, time_grid),
            "sim_price_sd": paths.std(axis=0, ddof=1),
            "theo_price_sd": theoretical_price_sd(s0, r, sigma, time_grid),
        },
        index=pd.Index(time_grid, name="t"),
    )


# ------------------------------------------------------------------ terminal prices
def terminal_lognormal(s0: float, r: float, sigma: float, time_to_maturity: float):
    """Frozen scipy lognormal distribution of S_T under GBM."""
    return stats.lognorm(
        s=sigma * np.sqrt(time_to_maturity),
        scale=s0 * np.exp((r - 0.5 * sigma**2) * time_to_maturity),
    )


def lognormal_centres(s0: float, r: float, sigma: float, time_to_maturity: float) -> dict[str, float]:
    """Mode, median and mean of S_T (mode < median < mean for sigma > 0)."""
    s = sigma * np.sqrt(time_to_maturity)
    median = s0 * np.exp((r - 0.5 * sigma**2) * time_to_maturity)
    return {
        "mode": float(median * np.exp(-(s**2))),
        "median": float(median),
        "mean": float(s0 * np.exp(r * time_to_maturity)),
    }


def terminal_summary(
    terminal_prices: np.ndarray,
    s0: float,
    r: float,
    sigma: float,
    time_to_maturity: float,
    drop_levels=(0.7, 0.5),
    gain_levels=(1.5,),
) -> pd.DataFrame:
    """Simulated vs theoretical moments, tail quantiles and move probabilities of S_T."""
    st = np.asarray(terminal_prices, dtype=float)
    dist = terminal_lognormal(s0, r, sigma, time_to_maturity)

    rows = {
        "Mean": (st.mean(), dist.mean()),
        "Median": (np.median(st), dist.median()),
        "Std. dev.": (st.std(ddof=1), dist.std()),
        "Skewness": (stats.skew(st), float(dist.stats(moments="s"))),
        "1% quantile": (np.quantile(st, 0.01), dist.ppf(0.01)),
        "99% quantile": (np.quantile(st, 0.99), dist.ppf(0.99)),
    }
    for level in drop_levels:
        rows[f"P(S_T < {level:g}·S0)"] = (np.mean(st < level * s0), dist.cdf(level * s0))
    for level in gain_levels:
        rows[f"P(S_T > {level:g}·S0)"] = (np.mean(st > level * s0), dist.sf(level * s0))

    table = pd.DataFrame(rows, index=["Simulated", "Theoretical"]).T.astype(float)
    table["Difference"] = table["Simulated"] - table["Theoretical"]
    return table
