"""Streamlit dashboard: Monte Carlo (GBM) vs Black--Scholes European option pricing.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from math import exp

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from src import (
    black_scholes_call,
    black_scholes_put,
    convergence_study,
    payoff,
    price_chain,
    price_with_confidence,
    put_call_parity_table,
    simulate_gbm_paths,
    simulate_terminal_prices,
    terminal_lognormal,
)

st.set_page_config(page_title="MC vs Black–Scholes", page_icon="📈", layout="wide")

MC_COLOR, BS_COLOR, BAND_COLOR = "#1f77b4", "#d62728", "rgba(31,119,180,0.18)"


# --------------------------------------------------------------------------- inputs
with st.sidebar:
    st.header("Option")
    kind = st.radio("Type", ["call", "put"], horizontal=True)
    s0 = st.number_input("Spot price S₀", min_value=0.01, value=100.0, step=1.0)
    strike = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0)
    maturity = st.number_input("Maturity T (years)", min_value=0.01, value=1.0, step=0.25)
    r = st.number_input("Risk-free rate r", value=0.05, step=0.005, format="%.4f")
    sigma = st.number_input("Volatility σ", min_value=0.001, value=0.20, step=0.01, format="%.4f")

    st.header("Simulation")
    n_paths = st.select_slider(
        "Number of paths N", options=[1_000, 5_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000], value=100_000
    )
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    confidence = st.select_slider("Confidence level", options=[0.90, 0.95, 0.99], value=0.95)
    z = float(norm.ppf(0.5 + confidence / 2))


# --------------------------------------------------------------------------- cached computations
@st.cache_data(show_spinner=False)
def terminal_prices(s0, r, sigma, t, n, seed):
    return simulate_terminal_prices(s0, r, sigma, t, n, rng=np.random.default_rng(seed))


@st.cache_data(show_spinner=False)
def sample_paths(s0, r, sigma, t, n_steps, n, seed):
    return simulate_gbm_paths(s0, r, sigma, t, n_steps, n, rng=np.random.default_rng(seed))


@st.cache_data(show_spinner="Running convergence study…")
def convergence(s0, strike, t, r, sigma, kind, n_seeds, seed):
    n_grid = np.unique(np.logspace(2, 5.5, 12).astype(int))
    return convergence_study(s0, strike, t, r, sigma, n_grid, n_seeds=n_seeds, kind=kind, seed=seed)


st_T = terminal_prices(s0, r, sigma, maturity, n_paths, seed)
res = price_with_confidence(st_T, s0, strike, maturity, r, sigma, kind=kind, z=z)

# --------------------------------------------------------------------------- header + headline metrics
st.title("Monte Carlo (GBM) vs Black–Scholes")
st.caption(
    f"European {kind} · S₀={s0:g}, K={strike:g}, T={maturity:g}y, r={r:.2%}, σ={sigma:.2%} · "
    f"N={n_paths:,} risk-neutral GBM terminal prices"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Black–Scholes price", f"{res['bs_price']:.4f}")
c2.metric(
    "Monte Carlo price",
    f"{res['mc_price']:.4f}",
    delta=f"{res['mc_price'] - res['bs_price']:+.4f} vs BS",
    delta_color="off",
    delta_arrow="off",
)
c3.metric(
    "Standard error",
    f"{res['std_error']:.4f}",
    delta=f"{res['abs_diff'] / res['bs_price']:.3%} rel. error" if res["bs_price"] > 0 else None,
    delta_color="off",
    delta_arrow="off",
)
c4.metric(
    f"{confidence:.0%} CI half-width",
    f"±{z * res['std_error']:.4f}",
    delta=f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]",
    delta_color="off",
    delta_arrow="off",
)

if res["bs_in_ci"]:
    st.success(f"Black–Scholes price lies inside the MC {confidence:.0%} confidence interval (z = {res['z_score']:+.2f}).")
else:
    st.warning(f"Black–Scholes price lies outside the MC {confidence:.0%} confidence interval (z = {res['z_score']:+.2f}).")

tab_dist, tab_conv, tab_chain, tab_parity = st.tabs(
    ["Paths & distribution", "Convergence", "Strike chain", "Put–call parity"]
)

# --------------------------------------------------------------------------- tab 1: paths & terminal distribution
with tab_dist:
    left, right = st.columns(2)

    with left:
        st.subheader("Sample GBM paths")
        n_show = st.slider("Paths shown", 10, 300, 60, key="n_show")
        n_steps = st.slider("Time steps", 20, 504, 252, key="n_steps")
        paths = sample_paths(s0, r, sigma, maturity, n_steps, n_show, seed)
        grid = np.linspace(0, maturity, n_steps + 1)
        fig = go.Figure()
        for p in paths:
            fig.add_trace(go.Scatter(x=grid, y=p, mode="lines", line=dict(width=0.7), opacity=0.45, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=grid, y=s0 * np.exp(r * grid), name="E[Sₜ] = S₀e^{rt}", line=dict(color="black", width=2.5, dash="dash")))
        fig.add_hline(y=strike, line=dict(color=BS_COLOR, dash="dot"), annotation_text="Strike")
        fig.update_layout(xaxis_title="t (years)", yaxis_title="Price", height=430, margin=dict(t=10), legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig)

    with right:
        st.subheader("Terminal price distribution")
        lo, hi = np.quantile(st_T, [0.001, 0.999])
        x = np.linspace(lo, hi, 400)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=st_T[(st_T >= lo) & (st_T <= hi)], nbinsx=120, histnorm="probability density", name="Simulated S_T", marker_color=MC_COLOR, opacity=0.6))
        fig.add_trace(go.Scatter(x=x, y=terminal_lognormal(s0, r, sigma, maturity).pdf(x), name="Lognormal (BS) density", line=dict(color=BS_COLOR, width=2.5)))
        fig.add_vline(x=strike, line=dict(color="black", dash="dot"), annotation_text="K")
        itm = (st_T > strike).mean() if kind == "call" else (st_T < strike).mean()
        fig.update_layout(xaxis_title="S_T", yaxis_title="Density", height=430, margin=dict(t=10), barmode="overlay", legend=dict(x=0.6, y=0.99))
        st.plotly_chart(fig)

        d2 = (np.log(s0 / strike) + (r - 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
        itm_bs = norm.cdf(d2) if kind == "call" else norm.cdf(-d2)
        st.dataframe(
            pd.DataFrame(
                {
                    "Monte Carlo": [st_T.mean(), np.median(st_T), st_T.std(ddof=1), itm],
                    "Theory (lognormal)": [
                        s0 * exp(r * maturity),
                        s0 * exp((r - 0.5 * sigma**2) * maturity),
                        s0 * exp(r * maturity) * np.sqrt(exp(sigma**2 * maturity) - 1),
                        itm_bs,
                    ],
                },
                index=["Mean S_T", "Median S_T", "Std dev S_T", "P(in the money)"],
            ).style.format("{:.4f}"),
        )

# --------------------------------------------------------------------------- tab 2: convergence
with tab_conv:
    st.subheader("Running estimate on the current sample")
    disc_pay = exp(-r * maturity) * payoff(st_T, strike, kind)
    n_idx = np.unique(np.geomspace(100, n_paths, 300).astype(int))
    csum, csum2 = np.cumsum(disc_pay), np.cumsum(disc_pay**2)
    run_mean = csum[n_idx - 1] / n_idx
    run_sd = np.sqrt(np.maximum(csum2[n_idx - 1] / n_idx - run_mean**2, 0) * n_idx / (n_idx - 1))
    run_se = run_sd / np.sqrt(n_idx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[n_idx, n_idx[::-1]], y=np.r_[run_mean + z * run_se, (run_mean - z * run_se)[::-1]], fill="toself", fillcolor=BAND_COLOR, line=dict(width=0), name=f"{confidence:.0%} CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=n_idx, y=run_mean, name="MC estimate", line=dict(color=MC_COLOR, width=2)))
    fig.add_hline(y=res["bs_price"], line=dict(color=BS_COLOR, dash="dash", width=2), annotation_text=f"BS = {res['bs_price']:.4f}")
    fig.update_layout(xaxis_type="log", xaxis_title="Number of paths", yaxis_title=f"{kind.title()} price", height=420, margin=dict(t=10))
    st.plotly_chart(fig)

    st.subheader("Error vs N across independent seeds")
    n_seeds = st.slider("Seeds per N", 5, 50, 20, key="n_seeds")
    study = convergence(s0, strike, maturity, r, sigma, kind, n_seeds, int(seed))
    ng = study["n_grid"]
    fig = go.Figure()
    for j in range(study["errors"].shape[1]):
        fig.add_trace(go.Scatter(x=ng, y=np.abs(study["errors"][:, j]), mode="markers", marker=dict(color="gray", size=4, opacity=0.35), showlegend=(j == 0), name="|error|, single seed"))
    fig.add_trace(go.Scatter(x=ng, y=study["rmse"], name="Observed RMSE", mode="lines+markers", line=dict(color=MC_COLOR, width=2.5)))
    fig.add_trace(go.Scatter(x=ng, y=study["theoretical_rmse"], name="σ_payoff / √N", line=dict(color=BS_COLOR, dash="dash", width=2)))
    fig.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="Number of paths N", yaxis_title="|MC − BS|", height=420, margin=dict(t=10))
    st.plotly_chart(fig)
    slope = np.polyfit(np.log(ng), np.log(study["rmse"]), 1)[0]
    st.caption(f"Fitted log–log slope of RMSE: **{slope:.3f}** (theory: −0.5, i.e. error ∝ 1/√N).")

# --------------------------------------------------------------------------- tab 3: strike chain
with tab_chain:
    st.subheader("Prices across strikes (same simulated sample)")
    lo_k, hi_k = st.slider("Strike range (% of S₀)", 50, 200, (70, 130), step=5, key="krange")
    strikes = np.linspace(s0 * lo_k / 100, s0 * hi_k / 100, 25)
    chain = price_chain(st_T, s0, strikes, maturity, r, sigma, kinds=("call", "put"), z=z)

    k_fine = np.linspace(strikes[0], strikes[-1], 200)
    cols = st.columns(2)
    for col, kd in zip(cols, ("call", "put")):
        sub = chain[chain["kind"] == kd]
        bsf = black_scholes_call if kd == "call" else black_scholes_put
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_fine, y=[bsf(s0, k, maturity, r, sigma) for k in k_fine], name="Black–Scholes", line=dict(color=BS_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=sub["strike"], y=sub["mc_price"], mode="markers", name=f"MC ± {z:.2f}·SE", marker=dict(color=MC_COLOR, size=6), error_y=dict(type="data", array=z * sub["std_error"], visible=True)))
        fig.add_vline(x=s0, line=dict(color="gray", dash="dot"), annotation_text="S₀")
        fig.update_layout(title=f"{kd.title()}s", xaxis_title="Strike", yaxis_title="Price", height=380, margin=dict(t=40))
        col.plotly_chart(fig)

    fig = go.Figure()
    for kd, color in (("call", MC_COLOR), ("put", "#ff7f0e")):
        sub = chain[chain["kind"] == kd]
        fig.add_trace(go.Scatter(x=sub["strike"], y=sub["z_score"], mode="lines+markers", name=kd, line=dict(color=color)))
    fig.add_hrect(y0=-z, y1=z, fillcolor="green", opacity=0.08, line_width=0)
    fig.update_layout(title="Standardised error (MC − BS) / SE", xaxis_title="Strike", yaxis_title="z-score", height=320, margin=dict(t=40))
    st.plotly_chart(fig)

    coverage = chain["bs_in_ci"].mean()
    st.caption(f"BS inside the MC {confidence:.0%} CI for **{coverage:.0%}** of the {len(chain)} contracts. Errors are correlated across strikes because they share one sample.")
    with st.expander("Full table"):
        st.dataframe(
            chain.style.format({c: "{:.4f}" for c in ["strike", "mc_price", "bs_price", "std_error", "ci_low", "ci_high", "abs_diff", "z_score"]}),
        )
        st.download_button("Download CSV", chain.to_csv(index=False), "mc_vs_bs_chain.csv", "text/csv")

# --------------------------------------------------------------------------- tab 4: parity
with tab_parity:
    st.subheader("Put–call parity: C − P = S₀ − K·e^{−rT}")
    parity = put_call_parity_table(st_T, s0, strikes, maturity, r, sigma)
    parity["MC gap"] = parity["MC  C - P"] - parity["S0 - K e^{-rT}"]
    parity["BS gap"] = parity["BS  C - P"] - parity["S0 - K e^{-rT}"]
    gap = exp(-r * maturity) * st_T.mean() - s0
    st.markdown(
        f"Black–Scholes satisfies parity exactly. The Monte Carlo gap is the same for every strike and equals "
        f"e^(−rT)·mean(S_T) − S₀ = **{gap:+.5f}**, pure sampling noise in the simulated forward."
    )
    st.dataframe(parity.style.format("{:.5f}"))
