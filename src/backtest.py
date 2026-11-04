"""Walk-forward check of whether GBM simulations predict where a stock actually ends up.

The idea ("Test 2"): stand at a past date, use only data available up to that day
to estimate volatility, simulate the price ``horizon`` trading days ahead, and
record where the *real* price landed inside the simulated distribution.
Repeat for many past dates and count how often reality fell inside each range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .gbm import simulate_terminal_prices
from .market_data import TRADING_DAYS_PER_YEAR, realized_volatility


def rolling_forecast_check(
    prices: pd.DataFrame,
    rates: pd.DataFrame,
    horizon: int = 21,
    step: int = 5,
    vol_window: int = 60,
    n_sims: int = 10_000,
    percentiles=(5, 25, 50, 75, 95),
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate forward from many past dates and compare with the real outcome.

    Parameters
    ----------
    prices : pandas.DataFrame
        Output of ``load_price_history`` (needs ``date`` and ``adj_close``).
    rates : pandas.DataFrame
        Output of ``load_risk_free_rate`` (needs ``date`` and ``r_cont``).
    horizon : int
        How many trading days ahead to forecast (21 ~ one month).
    step : int
        Days between start dates (5 = one forecast per week).
    vol_window : int
        Number of past daily returns used to estimate sigma. Pass ``0`` to use
        *all* history available up to the start date (an expanding window).
    n_sims : int
        Number of simulated end prices per start date.
    percentiles : sequence of int
        Percentiles of the simulated distribution to store.
    seed : int
        Random seed, for reproducible results.

    Returns
    -------
    pandas.DataFrame
        One row per start date with columns:

        * ``start_date, end_date`` - forecast made on start_date, checked on end_date
        * ``s0, actual`` - price on start_date and the real price on end_date
        * ``sigma, r`` - the inputs used for that forecast
        * ``p5, p25, ...`` - simulated percentiles of the end price
        * ``rank`` - fraction of simulations *below* the real price (0-1).
          If the model is right, ranks are spread evenly between 0 and 1.
        * ``z`` - the real log-return measured in "model standard deviations".
          If the model is right, z looks like a standard normal (mean 0, sd 1).

    Notes
    -----
    Only information up to and including ``start_date`` is used to build each
    forecast (no look-ahead). Prices are dividend-adjusted closes, so dividend
    drops don't count as losses.
    """
    px = prices[["date", "adj_close"]].sort_values("date").reset_index(drop=True)
    rt = rates[["date", "r_cont"]].sort_values("date")
    px = pd.merge_asof(px, rt, on="date", direction="backward")
    px["r_cont"] = px["r_cont"].ffill().bfill()

    if vol_window > 0:
        px["sigma"] = realized_volatility(px["adj_close"], window=vol_window)
    else:
        log_ret = np.log(px["adj_close"]).diff()
        px["sigma"] = log_ret.expanding(min_periods=20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    first = int(px["sigma"].first_valid_index())
    starts = range(first, len(px) - horizon, step)
    rng = np.random.default_rng(seed)
    T = horizon / TRADING_DAYS_PER_YEAR

    rows = []
    for i in starts:
        s0, sigma, r = px.at[i, "adj_close"], px.at[i, "sigma"], px.at[i, "r_cont"]
        actual = px.at[i + horizon, "adj_close"]
        sims = simulate_terminal_prices(s0, r, sigma, T, n_sims, rng=rng)
        row = {
            "start_date": px.at[i, "date"],
            "end_date": px.at[i + horizon, "date"],
            "s0": s0,
            "actual": actual,
            "sigma": sigma,
            "r": r,
            "rank": float(np.mean(sims < actual)),
            "z": (np.log(actual / s0) - (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)),
        }
        row.update({f"p{p}": float(v) for p, v in zip(percentiles, np.percentile(sims, percentiles))})
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_summary(check: pd.DataFrame) -> pd.DataFrame:
    """Compare how often reality landed in each range with how often the model said it would.

    Uses the ``rank`` column of :func:`rolling_forecast_check`. For each range the
    table shows the expected %, the actual %, the count, and a binomial-test
    p-value: a small p-value (below ~0.05) means the gap is bigger than luck
    alone would usually produce.

    Caution: with overlapping forecast windows (``step < horizon``) the checks
    are not independent, so the p-values are optimistic (too small).
    """
    ranges = {
        "below 5th pct (big miss down)": (0.00, 0.05),
        "5th-25th pct": (0.05, 0.25),
        "25th-75th pct (middle half)": (0.25, 0.75),
        "75th-95th pct": (0.75, 0.95),
        "above 95th pct (big miss up)": (0.95, 1.00),
        "inside 5th-95th (the 90% range)": (0.05, 0.95),
    }
    n = len(check)
    out = []
    for name, (lo, hi) in ranges.items():
        inside = ((check["rank"] >= lo) & (check["rank"] < hi)) if hi < 1 else (check["rank"] >= lo)
        k = int(inside.sum())
        expected = hi - lo
        out.append(
            {
                "range": name,
                "expected": expected,
                "actual": k / n,
                "count": f"{k} / {n}",
                "p_value": stats.binomtest(k, n, expected).pvalue,
            }
        )
    return pd.DataFrame(out).set_index("range")
