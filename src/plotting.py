"""Matplotlib figures for the simulation analysis.

Every function returns a ``matplotlib.figure.Figure`` so it works the same in a
notebook (displayed automatically) and in Streamlit (``st.pyplot(fig)``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .analytics import (
    dispersion_over_time,
    lognormal_centres,
    percentile_bands,
    terminal_lognormal,
    theoretical_mean,
)


def plot_spaghetti(paths, time_grid, s0, r, sigma, max_paths_drawn: int | None = None):
    """GBM paths with 5-95% / 25-75% bands, simulated mean/median and E[S_t]."""
    bands = percentile_bands(paths, (5, 25, 50, 75, 95))
    drawn = paths if max_paths_drawn is None else paths[:max_paths_drawn]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_grid, drawn.T, color="steelblue", alpha=0.06, linewidth=0.6)
    ax.fill_between(time_grid, bands[5], bands[95], color="orange", alpha=0.15, label="5–95% band")
    ax.fill_between(time_grid, bands[25], bands[75], color="orange", alpha=0.30, label="25–75% band")
    ax.plot(time_grid, paths.mean(axis=0), color="black", linewidth=2, label="Simulated mean")
    ax.plot(time_grid, theoretical_mean(s0, r, time_grid), "r--", linewidth=2, label=r"Theoretical mean $S_0e^{rt}$")
    ax.plot(time_grid, bands[50], color="darkorange", linewidth=1.5, label="Simulated median")
    ax.set(xlabel="Time (years)", ylabel="Price", title=f"{len(paths):,} GBM paths (σ={sigma:.0%}, r={r:.0%})")
    ax.legend(loc="upper left")
    return fig


def plot_dispersion(paths, time_grid, s0, r, sigma):
    """Simulated vs theoretical std. dev. of log-price and price over time."""
    d = dispersion_over_time(paths, time_grid, s0, r, sigma)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(d.index, d.sim_log_sd, label="Simulated")
    ax1.plot(d.index, d.theo_log_sd, "r--", label=r"$\sigma\sqrt{t}$")
    ax1.set(title="Std. dev. of log-price", xlabel="Time (years)")
    ax1.legend()
    ax2.plot(d.index, d.sim_price_sd, label="Simulated")
    ax2.plot(d.index, d.theo_price_sd, "r--", label=r"$S_0e^{rt}\sqrt{e^{\sigma^2t}-1}$")
    ax2.set(title="Std. dev. of price", xlabel="Time (years)")
    ax2.legend()
    fig.tight_layout()
    return fig


def plot_terminal_histogram(terminal_prices, s0, r, sigma, time_to_maturity, bins: int = 120, tail: float = 0.01):
    """Histogram of S_T with the lognormal pdf, mode/median/mean and shaded tails."""
    st = np.asarray(terminal_prices, dtype=float)
    dist = terminal_lognormal(s0, r, sigma, time_to_maturity)
    centres = lognormal_centres(s0, r, sigma, time_to_maturity)
    q_low, q_high = np.quantile(st, [tail, 1 - tail])

    x = np.linspace(st.min(), np.quantile(st, 0.999), 600)
    pdf = dist.pdf(x)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(st, bins=bins, density=True, color="steelblue", alpha=0.55, label=f"Simulated $S_T$ ({st.size:,} runs)")
    ax.plot(x, pdf, "k", linewidth=2, label="Theoretical lognormal pdf")
    ax.fill_between(x, pdf, where=x <= q_low, color="red", alpha=0.5, label=f"Extreme {tail:.0%} tails")
    ax.fill_between(x, pdf, where=x >= q_high, color="red", alpha=0.5)
    for (name, value), style in zip(centres.items(), [":", "-.", "--"]):
        ax.axvline(value, color="darkorange", linestyle=style, linewidth=2, label=f"{name.title()} = {value:.2f}")
    ax.axvline(s0, color="grey", linewidth=1, label=f"$S_0$ = {s0:g}")
    ax.set(xlabel="Terminal price $S_T$", ylabel="Density",
           title=f"Distribution of terminal prices at T = {time_to_maturity:g} yr")
    ax.legend()
    return fig


def plot_log_qq(terminal_prices):
    """Normal Q-Q plot of ln S_T (should be a straight line under GBM)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(np.log(terminal_prices), dist="norm", plot=ax)
    ax.set_title(r"Normal Q–Q plot of $\ln S_T$")
    return fig


def plot_strike_comparison(chain, s0, z: float = 1.96):
    """MC (with CI) vs Black--Scholes across strikes, plus the error in standard errors.

    ``chain`` is the DataFrame returned by ``valuation.price_chain``.
    """
    colors = {"call": "tab:blue", "put": "tab:red"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for kind, sub in chain.groupby("kind"):
        color = colors.get(kind)
        ax1.plot(sub.strike, sub.bs_price, color=color, linewidth=2, label=f"Black–Scholes {kind}")
        ax1.errorbar(sub.strike, sub.mc_price, yerr=z * sub.std_error, fmt="o", color=color,
                     markersize=4, capsize=3, alpha=0.8, label=f"Monte Carlo {kind} (95% CI)")
        ax2.plot(sub.strike, sub.z_score, "o-", color=color, label=kind)
    ax1.axvline(s0, color="grey", linestyle=":", label="ATM")
    ax1.set(xlabel="Strike", ylabel="Premium", title="Option premium vs strike")
    ax1.legend()
    ax2.axhspan(-z, z, color="green", alpha=0.1, label=f"±{z:g} SE")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set(xlabel="Strike", ylabel="(MC − BS) / SE", title="Pricing error in standard errors")
    ax2.legend()
    fig.tight_layout()
    return fig


def plot_convergence(study, z: float = 1.96):
    """Price estimates and RMSE vs N. ``study`` is the dict from ``valuation.convergence_study``."""
    n, bs, errors = study["n_grid"], study["bs_price"], study["errors"]
    envelope = z * study["theoretical_rmse"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(n, bs + errors, color="steelblue", alpha=0.3)
    ax1.axhline(bs, color="red", linestyle="--", label=f"Black–Scholes = {bs:.4f}")
    ax1.fill_between(n, bs - envelope, bs + envelope, color="red", alpha=0.1, label="95% envelope")
    ax1.set(xscale="log", xlabel="Number of simulations N", ylabel="MC price",
            title=f"Estimates across {errors.shape[1]} seeds")
    ax1.legend()
    ax2.loglog(n, study["rmse"], "o-", label="Observed RMSE")
    ax2.loglog(n, study["theoretical_rmse"], "r--", label=r"$\mathrm{sd}/\sqrt{N}$")
    ax2.set(xlabel="Number of simulations N", ylabel="RMSE", title="Error decays like $1/\\sqrt{N}$")
    ax2.legend()
    fig.tight_layout()
    return fig
