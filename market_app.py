"""Streamlit app: does our Monte Carlo (GBM) price match the real option market, for any stock?

This is the interactive version of ``notebooks/Market_Comparison.ipynb``:
search a ticker, download its last 5 years of prices, today's quote and today's
option chain, price every option with Monte Carlo and compare with what traders pay.

Run with:  streamlit run market_app.py
"""

from __future__ import annotations

import warnings
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import brentq

from src import black_scholes_call, black_scholes_put, price_chain, simulate_terminal_prices
from src.market_data import (
    TRADING_DAYS_PER_YEAR,
    fetch_option_chain,
    fetch_price_history,
    fetch_risk_free_rate,
    realized_volatility,
)

st.set_page_config(page_title="Monte Carlo vs the market", page_icon=":material/query_stats:", layout="wide")

# Categorical slots from a colour-blind-validated palette, always used for the same thing.
MC_COLOR, MARKET_COLOR = "#2a78d6", "#eb6834"  # model vs market
CALL_COLOR, PUT_COLOR = "#1baf7a", "#6250d6"  # calls vs puts
EXPIRY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
POPULAR = {
    "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco QQQ (Nasdaq-100)",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla, Inc.",
    "AMZN": "Amazon.com, Inc.",
}


def usd(x: float, decimals: int = 2) -> str:
    """Dollar amount escaped for st.markdown (a bare $ would start LaTeX)."""
    return f"{'−' if x < 0 else ''}\\${abs(x):,.{decimals}f}"


def explain(simple: str, detailed: str) -> None:
    """Show the plain-words and the fuller explanation of the next chart side by side."""
    left, right = st.columns(2)
    with left.container(border=True, height="stretch"):
        st.markdown("**:material/lightbulb: In plain words**")
        st.markdown(simple)
    with right.container(border=True, height="stretch"):
        st.markdown("**:material/menu_book: In more detail**")
        st.markdown(detailed)


# --------------------------------------------------------------------------- data access (cached)
@st.cache_data(ttl=3600, max_entries=200, show_spinner=False)
def search_tickers(query: str) -> dict[str, str]:
    """Yahoo Finance search -> {symbol: 'Name · Exchange'} for stocks and ETFs only."""
    try:
        quotes = yf.Search(query, max_results=12, news_count=0).quotes
    except Exception:
        return {}
    out = {}
    for q in quotes:
        if q.get("quoteType") in ("EQUITY", "ETF") and q.get("symbol"):
            name = q.get("longname") or q.get("shortname") or q["symbol"]
            out[q["symbol"]] = f"{name} · {q.get('exchDisp', '')}".rstrip(" ·")
    return out


@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def load_prices(symbol: str) -> pd.DataFrame:
    return fetch_price_history(symbol, period="5y", save=False)


@st.cache_data(ttl=3600, max_entries=5, show_spinner=False)
def load_rates() -> pd.DataFrame:
    return fetch_risk_free_rate(period="5y", save=False)


@st.cache_data(ttl=300, max_entries=50, show_spinner=False)
def load_quote(symbol: str) -> dict:
    """Today's quote: last price, previous close, 52-week range, currency."""
    fi = yf.Ticker(symbol).fast_info
    out = {}
    for key in ("last_price", "previous_close", "year_high", "year_low", "currency"):
        try:
            out[key] = fi[key]
        except Exception:
            out[key] = None
    return out


@st.cache_data(ttl=900, max_entries=50, show_spinner=False)
def load_options(symbol: str, n_expiries: int, min_days: int, max_days: int) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "market closed" warning: shown in the UI instead
        return fetch_option_chain(symbol, n_expiries, min_days, max_days, save=False)


# --------------------------------------------------------------------------- pricing (cached)
def implied_vol(price: float, s0: float, strike: float, t: float, r: float, kind: str) -> float:
    """The sigma that makes Black-Scholes equal the market price (NaN if none exists)."""
    f = black_scholes_call if kind == "call" else black_scholes_put
    try:
        return brentq(lambda s: f(s0, strike, t, r, s) - price, 1e-4, 5.0)
    except ValueError:
        return np.nan


@st.cache_data(max_entries=30, show_spinner=False)
def run_comparison(clean: pd.DataFrame, s0: float, r: float, sigma: float, n_sims: int, seed: int) -> pd.DataFrame:
    """Monte Carlo price for every contract (one simulated sample per expiry) plus implied vol."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in clean.groupby("expiry"):
        t = float(group.time_to_maturity.iloc[0])
        sims = simulate_terminal_prices(s0, r, sigma, t, n_sims, rng=rng)
        chain = price_chain(sims, s0, sorted(group.strike.unique()), t, r, sigma)
        parts.append(group.merge(chain, on=["kind", "strike"]))
    out = pd.concat(parts, ignore_index=True)
    out["error"] = out.mc_price - out.mid  # + means the model is more expensive than the market
    out["error_pct"] = out.error / out.mid
    out["implied_vol"] = [
        implied_vol(row.mid, s0, row.strike, row.time_to_maturity, r, row.kind) for row in out.itertuples()
    ]
    return out


@st.cache_data(max_entries=30, show_spinner=False)
def reprice_atm_with_implied(compare: pd.DataFrame, s0: float, r: float, n_sims: int, seed: int) -> pd.DataFrame:
    """Re-price the near-the-money options with the market's own (implied) sigma for that expiry."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    for expiry, d in compare.groupby("expiry"):
        dist = (d.moneyness - 1).abs()
        d = d[dist <= max(0.01, dist.min())]
        iv = float(d.implied_vol.median())
        if d.empty or not np.isfinite(iv):
            continue
        t = float(d.time_to_maturity.iloc[0])
        sims = simulate_terminal_prices(s0, r, iv, t, n_sims, rng=rng)
        chain = price_chain(sims, s0, sorted(d.strike.unique()), t, r, iv)
        m = d[["expiry_label", "kind", "strike", "mid", "mc_price"]].merge(
            chain[["kind", "strike", "mc_price"]], on=["kind", "strike"], suffixes=("_hist", "_implied")
        )
        m["implied_sigma"] = iv
        rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def moneyness_buckets(band: float) -> tuple[list[float], list[str]]:
    edges = sorted({round(1 - band, 4), 0.95, 0.98, 1.02, 1.05, round(1 + band, 4)})
    edges = [e for e in edges if 1 - band <= e <= 1 + band]
    labels = [f"{lo:.0%}–{hi:.0%}" + (" (ATM)" if (lo, hi) == (0.98, 1.02) else "") for lo, hi in zip(edges, edges[1:])]
    return edges, labels


# --------------------------------------------------------------------------- sidebar: pick a stock + settings
with st.sidebar:
    st.header("Stock")
    query = st.text_input(
        "Search by name or ticker",
        type="search",
        live="400ms",
        placeholder="e.g. apple, nvidia, SPY…",
        icon=":material/search:",
    )
    matches = search_tickers(query.strip()) if len(query.strip()) >= 2 else POPULAR
    if not matches:
        st.caption("No stocks or ETFs found. You can still type a ticker directly below.")
        matches = POPULAR
    symbol = st.selectbox(
        "Ticker",
        options=list(matches),
        format_func=lambda s: f"{s} — {matches[s]}" if s in matches else s,
        accept_new_options=True,
        help="Pick a search result, or type a Yahoo Finance ticker and press Enter.",
    )
    symbol = (symbol or "SPY").strip().upper()

    st.header("Model")
    sigma_source = st.segmented_control(
        "Volatility σ from",
        ["30-day", "60-day", "1-year", "5-year"],
        default="60-day",
        required=True,
        help="How much past price history is used to estimate volatility. The notebook uses 60 days.",
    )
    n_sims = st.select_slider("Simulations per expiry", options=[20_000, 50_000, 100_000, 200_000], value=200_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.header("Options to compare")
    n_expiries = st.slider("Expiry dates", 1, 6, 4)
    min_days, max_days = st.slider("Days to expiry", 7, 365, (20, 120))
    band = st.slider("Strikes within ± of today's price", 6, 30, 10, format="%d%%") / 100
    fresh_days = st.slider(
        "Last traded within (trading days)",
        1,
        5,
        1,
        help="1 = only contracts that traded on the latest trading day (as in the notebook). Raise it for less liquid stocks.",
    )


# --------------------------------------------------------------------------- header
st.title("Monte Carlo vs the real option market")
st.caption(
    "Give the simulation real-world inputs and check whether it lands on the option prices traders actually pay. "
    "Interactive version of `notebooks/Market_Comparison.ipynb`."
)

# --------------------------------------------------------------------------- 1. download
with st.spinner(f"Downloading 5 years of {symbol} prices, today's quote and option chain…"):
    try:
        prices = load_prices(symbol)
    except Exception as exc:
        st.error(f"Couldn't download price history for **{symbol}** ({exc}). Check the ticker.", icon=":material/error:")
        st.stop()
    rates = load_rates()
    quote = load_quote(symbol)
    try:
        options = load_options(symbol, n_expiries, min_days, max_days)
        options_error = None
    except Exception as exc:
        options, options_error = pd.DataFrame(), str(exc)

closes = prices.close
last_px = quote.get("last_price") or float(closes.iloc[-1])
prev_px = quote.get("previous_close") or float(closes.iloc[-2])
currency = quote.get("currency") or "USD"
log_ret = np.log(prices.adj_close).diff().dropna()
years = (prices.date.iloc[-1] - prices.date.iloc[0]).days / 365.25
total_return = prices.adj_close.iloc[-1] / prices.adj_close.iloc[0] - 1
r = float(rates.r_cont.iloc[-1])
trailing_div = prices.loc[prices.date > prices.date.iloc[-1] - pd.Timedelta(days=365), "dividends"].sum()

st.header(f"1. {symbol} today and over the last 5 years", divider="gray")
with st.container(horizontal=True):
    st.metric(
        "Last price",
        f"{last_px:,.2f} {currency}",
        delta=f"{last_px / prev_px - 1:+.2%} today",
        border=True,
        height="stretch",
    )
    st.metric(
        "52-week range",
        f"{quote.get('year_low') or closes.tail(252).min():,.2f} – {quote.get('year_high') or closes.tail(252).max():,.2f}",
        border=True,
        height="stretch",
    )
    st.metric(
        f"Total return ({years:.1f}y)",
        f"{total_return:+.0%}",
        delta=f"{(1 + total_return) ** (1 / years) - 1:+.1%} per year",
        delta_color="off",
        delta_arrow="off",
        border=True,
        height="stretch",
    )
    st.metric("Dividend yield (12m)", f"{trailing_div / last_px:.2%}", border=True, height="stretch")
    st.metric("Risk-free rate r (3m T-bill)", f"{r:.2%}", border=True, height="stretch")

if currency != "USD":
    st.info(
        f"{symbol} trades in {currency}, but the risk-free rate is the US T-bill rate. "
        "The comparison still runs, but r may not match this stock's home market.",
        icon=":material/currency_exchange:",
    )

explain(
    f"""
**What it shows:** the price of {symbol} every trading day for the last 5 years.

**Why it matters:** the simulation starts from *today's* price and needs a guess of how
much the price wobbles. We get that guess from this history, so this chart is the raw
material for everything below.
""",
    f"""
**What it shows:** the daily closing price (not adjusted for dividends) over
{len(prices):,} trading days, from {prices.date.iloc[0]:%d %b %Y} to {prices.date.iloc[-1]:%d %b %Y}.
Hover to read any day.

**Why it matters:** the Monte Carlo model is geometric Brownian motion, which assumes prices
move by random *percentage* steps of constant size. The two numbers it takes from the
market are the starting price $S_0$ (the latest close) and the volatility σ, estimated
from log-returns of the dividend-adjusted price. Look for calm stretches, sudden crashes or
jumps: those are exactly what a constant-σ model cannot reproduce.
""",
)
price_chart = (
    alt.Chart(prices, title=alt.Title(f"{symbol} daily close"))
    .mark_line(color=MC_COLOR, strokeWidth=2)
    .encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("close:Q", title=f"Price ({currency})", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("close:Q", title="Close", format=",.2f")],
    )
    .properties(height=320)
)
st.altair_chart(price_chart, alt=f"Line chart of {symbol} daily closing price over the last 5 years")

with st.expander("Download the raw data", icon=":material/download:"):
    with st.container(horizontal=True):
        st.download_button(
            f"{symbol} prices (5y)", prices.to_csv(index=False), f"{symbol}_prices_5y.csv", "text/csv",
            icon=":material/download:",
        )
        st.download_button(
            "Risk-free rate (5y)", rates.to_csv(index=False), "risk_free_rate_5y.csv", "text/csv",
            icon=":material/download:",
        )
        if not options.empty:
            st.download_button(
                f"{symbol} option chain (today)", options.to_csv(index=False),
                f"{symbol}_options_{date.today().isoformat()}.csv", "text/csv", icon=":material/download:",
            )

if options_error or options.empty:
    st.error(
        f"No option chain available for **{symbol}** with expiries {min_days}–{max_days} days out "
        f"({options_error or 'empty chain'}). Try a US-listed stock or ETF, or widen *Days to expiry*.",
        title="Can't compare without market option prices",
        icon=":material/error:",
    )
    st.stop()

# --------------------------------------------------------------------------- 2. model inputs
S0 = float(options.underlying_price.iloc[0])
vol_30 = realized_volatility(prices.adj_close, window=30)
vol_60 = realized_volatility(prices.adj_close, window=60)
sigmas = {
    "30-day": float(vol_30.iloc[-1]),
    "60-day": float(vol_60.iloc[-1]),
    "1-year": float(log_ret.tail(TRADING_DAYS_PER_YEAR).std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
    "5-year": float(log_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
}
SIGMA = sigmas[sigma_source]
SIGMA_PERIOD = {"30-day": "30 days", "60-day": "60 days", "1-year": "year", "5-year": "5 years"}

st.header("2. The model's inputs: price, interest rate, volatility", divider="gray")
with st.container(horizontal=True):
    st.metric("S₀ (spot)", f"{S0:,.2f}", border=True, height="stretch")
    st.metric("r (risk-free)", f"{r:.2%}", border=True, height="stretch")
    for name, value in sigmas.items():
        st.metric(
            f"σ {name}" + (" · used" if name == sigma_source else ""),
            f"{value:.1%}",
            border=True,
            height="stretch",
        )

explain(
    f"""
**What it shows:** how jumpy {symbol} has been, measured over the last 30 and 60 days,
at every point in the past 5 years. The dashed line is the σ we feed the model
({SIGMA:.1%}, {sigma_source}).

**Why it matters:** volatility is the one input we have to *guess*. The line is not flat:
it spikes in panics and settles in calm periods. Our model assumes one fixed number
forever, so if today happens to be unusually calm or wild, the model inherits that.
""",
    f"""
**What it shows:** rolling realised volatility, $\\sigma = \\text{{std}}(\\text{{daily log-returns}})\\times\\sqrt{{252}}$,
over 30- and 60-day windows. σ = 0.20 means a typical one-year move of about ±20%.

**Why it matters:** Monte Carlo pricing under GBM is exact *given* σ, so the pricing error
versus the market is driven almost entirely by this choice. Realised volatility is
backward-looking and clusters (volatile days follow volatile days), while option prices are
forward-looking. Comparing the dashed line with section 6's *implied* volatility tells you
whether the market expects the next few months to be calmer or wilder than the recent past.
""",
)
vol_df = pd.concat(
    [
        pd.DataFrame({"date": prices.date, "vol": vol_30, "window": "30-day"}),
        pd.DataFrame({"date": prices.date, "vol": vol_60, "window": "60-day"}),
    ]
).dropna()
vol_lines = (
    alt.Chart(vol_df)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("vol:Q", title="Annualised volatility σ", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "window:N", title=None, scale=alt.Scale(domain=["30-day", "60-day"], range=[MC_COLOR, MARKET_COLOR]),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("window:N", title="Window"),
            alt.Tooltip("vol:Q", title="σ", format=".1%"),
        ],
    )
)
sigma_rule = (
    alt.Chart(pd.DataFrame({"y": [SIGMA], "label": [f"σ used = {SIGMA:.1%}"]}))
    .mark_rule(strokeDash=[6, 4], strokeWidth=1.5, color="gray")
    .encode(y="y:Q", tooltip=alt.Tooltip("label:N", title=None))
)
st.altair_chart(
    (vol_lines + sigma_rule).properties(height=300, title=f"{symbol}'s volatility changes over time"),
    alt=f"Line chart of {symbol}'s 30-day and 60-day rolling volatility with the chosen sigma as a dashed line",
)

# --------------------------------------------------------------------------- 3. clean the option chain
st.header("3. Today's option prices, cleaned up", divider="gray")
trade_day = pd.to_datetime(options.last_trade_date, utc=True).dt.tz_convert("America/New_York").dt.normalize()
recent_days = sorted(trade_day.dropna().unique())[-fresh_days:]
moneyness = options.strike / S0
clean = options[
    moneyness.between(1 - band, 1 + band)
    & trade_day.isin(recent_days)
    & (options.volume > 0)
    & (options.mid >= 0.10)
].copy()
clean["moneyness"] = clean.strike / S0
clean["expiry_label"] = clean.expiry.dt.strftime("%Y-%m-%d") + " (" + clean.days_to_expiry.astype(str) + "d)"
market_open = options.has_quote.mean() > 0.5

st.markdown(
    f"""
A raw option chain is full of contracts that are useless for a fair comparison, so we keep only
strikes within **±{band:.0%}** of today's price, that **traded recently** (within {fresh_days} trading
day{'s' if fresh_days > 1 else ''}) with **volume > 0**, and that cost at least **{usd(0.10)}**. The market price
used is the **mid** (halfway between bid and ask), or the last traded price if no live quote exists.
"""
)
with st.container(horizontal=True):
    st.metric("Contracts downloaded", f"{len(options):,}", border=True, height="stretch")
    st.metric("Contracts kept", f"{len(clean):,}", border=True, height="stretch")
    st.metric("Expiries", f"{clean.expiry.nunique()}", border=True, height="stretch")
    st.metric("Live bid/ask quotes", f"{options.has_quote.mean():.0%}", border=True, height="stretch")
if not market_open:
    st.warning(
        "The US market looks closed, so most contracts have no live bid/ask and the app falls back to "
        "**last traded prices**, which can be stale. Numbers are most reliable during 9:30–16:00 New York time.",
        icon=":material/schedule:",
    )
if len(clean) < 10:
    st.error(
        f"Only {len(clean)} contracts survived the filters. Raise *Last traded within* or widen the strike band.",
        title="Too few clean option prices",
        icon=":material/filter_alt_off:",
    )
    st.stop()

# --------------------------------------------------------------------------- 4. price everything
with st.spinner(f"Simulating {n_sims:,} prices per expiry and pricing {len(clean):,} options…"):
    compare = run_comparison(clean, S0, r, SIGMA, n_sims, int(seed))
bucket_edges, bucket_labels = moneyness_buckets(band)
compare["bucket"] = pd.cut(compare.moneyness, bucket_edges, labels=bucket_labels, include_lowest=True)
expiry_labels = sorted(compare.expiry_label.unique())

st.header("4. Sanity check: is the simulation itself right?", divider="gray")
max_gap = float((compare.mc_price - compare.bs_price).abs().max())
coverage = float(compare.bs_in_ci.mean())
margin = float((1.96 * compare.std_error).median())
explain(
    """
**What it shows:** before blaming the model, we check the code. Black–Scholes is an exact formula
for the same model, so Monte Carlo should land on (almost) the same number.

**Why it matters:** if these agree, any gap with the market in the next sections is about the
model's *assumptions*, not a bug.
""",
    f"""
**What it shows:** the largest |MC − BS| gap across all {len(compare):,} contracts, how often the BS price
falls inside the MC 95% confidence interval, and the typical half-width of that interval.

**Why it matters:** Monte Carlo is an unbiased estimator with standard error
$e^{{-rT}}\\,\\text{{sd}}(\\text{{payoff}})/\\sqrt{{N}}$. With a correct implementation, coverage should be
close to 95% (all strikes of one expiry share a sample, so expect some clumping). Low coverage or
gaps far beyond a few standard errors would mean the engine, not the market, is the problem.
""",
)
with st.container(horizontal=True):
    st.metric("Largest |MC − BS| gap", f"\\${max_gap:.3f}", border=True, height="stretch")
    st.metric("BS inside MC 95% CI", f"{coverage:.0%}", delta="expect ~95%", delta_color="off", delta_arrow="off", border=True, height="stretch")
    st.metric("Typical MC margin of error", f"±\\${margin:.3f}", border=True, height="stretch")

# --------------------------------------------------------------------------- 5. model vs market
st.header("5. Compare: Monte Carlo vs what traders pay", divider="gray")
explain(
    f"""
**What it shows:** for one expiry date, every strike's option price according to our simulation
(blue line) and according to the market (orange dots). Calls on the left, puts on the right.

**Why it matters:** this is the main test. Where the orange dots sit **above** the blue line,
traders pay **more** than our model thinks the option is worth; below it, less.
""",
    f"""
**What it shows:** Monte Carlo prices with their 95% confidence band (often too thin to see) against
market mid prices, across strikes within ±{band:.0%} of $S_0$ = {usd(S0)} (grey dotted line). Calls fall with the
strike, puts rise with it, and both carry more time value near the money.

**Why it matters:** a systematic vertical gap means the model's single σ is off for that expiry;
a gap that grows towards one side (usually low-strike puts) is the volatility *skew*, which a
constant-σ GBM cannot produce. Pick each expiry to see whether the gap changes with time to maturity.
""",
)
picked = st.segmented_control("Expiry", expiry_labels, default=expiry_labels[0], required=True)
sel = compare[compare.expiry_label == picked]
series_scale = alt.Scale(domain=["Monte Carlo", "Market"], range=[MC_COLOR, MARKET_COLOR])
cols = st.columns(2)
for col, kind in zip(cols, ("call", "put")):
    d = sel[sel.kind == kind].sort_values("strike")
    if d.empty:
        col.info(f"No clean {kind} prices for this expiry.", icon=":material/info:")
        continue
    base = alt.Chart(d).encode(x=alt.X("strike:Q", title="Strike K", scale=alt.Scale(zero=False)))
    tooltip = [
        alt.Tooltip("strike:Q", title="Strike", format=",.2f"),
        alt.Tooltip("mid:Q", title="Market", format=",.2f"),
        alt.Tooltip("mc_price:Q", title="Monte Carlo", format=",.2f"),
        alt.Tooltip("error_pct:Q", title="Model vs market", format="+.0%"),
    ]
    band_layer = base.mark_area(opacity=0.2, color=MC_COLOR).encode(y="ci_low:Q", y2="ci_high:Q")
    series_color = alt.Color("series:N", scale=series_scale, legend=alt.Legend(title=None, orient="top"))
    line_layer = base.transform_calculate(series="'Monte Carlo'").mark_line(strokeWidth=2).encode(
        y=alt.Y("mc_price:Q", title="Option price"), color=series_color
    )
    dot_layer = base.transform_calculate(series="'Market'").mark_circle(size=70, opacity=0.9).encode(
        y="mid:Q", color=series_color, tooltip=tooltip
    )
    s0_rule = alt.Chart(pd.DataFrame({"x": [S0]})).mark_rule(strokeDash=[2, 3], color="gray").encode(x="x:Q")
    col.altair_chart(
        alt.layer(band_layer, line_layer, dot_layer, s0_rule).properties(height=340, title=f"{kind.title()}s · {picked}"),
        alt=f"{kind.title()} prices by strike for {picked}: Monte Carlo line versus market dots",
    )

st.subheader("Putting a number on it")
explain(
    """
**What it shows:** each dot is one option. Its height is how far off our price is, in percent
of the market price. Zero = spot on; below zero = our model is too cheap.

**Why it matters:** it shows *where* the model goes wrong. A \\$1 miss on a \\$30 option is small,
on a \\$2 option it is huge, so percentages make cheap and expensive options comparable.
""",
    f"""
**What it shows:** error % = (MC − market) / market against moneyness K/S₀ (1.0 = at the money),
for every expiry, plus the average dollar and percentage error per moneyness bucket.

**Why it matters:** if the only problem were the *level* of σ, calls and puts would be off by a
similar dollar amount near the money. A strong slope across moneyness — puts below S₀ much too
cheap, calls above S₀ less so — is the signature of the skew. Very large % errors at the edges come
from options worth only a dollar or two, so read the dollar column too.
""",
)
left, right = st.columns([3, 2])
err_chart = (
    alt.Chart(compare)
    .mark_circle(size=50, opacity=0.75)
    .encode(
        x=alt.X("moneyness:Q", title="Strike ÷ today's price (1.0 = at the money)", scale=alt.Scale(zero=False)),
        y=alt.Y("error_pct:Q", title="Model vs market", axis=alt.Axis(format="+.0%")),
        color=alt.Color(
            "kind:N", title=None, scale=alt.Scale(domain=["call", "put"], range=[CALL_COLOR, PUT_COLOR]),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[
            alt.Tooltip("kind:N", title="Type"),
            alt.Tooltip("expiry_label:N", title="Expiry"),
            alt.Tooltip("strike:Q", title="Strike", format=",.2f"),
            alt.Tooltip("mid:Q", title="Market", format=",.2f"),
            alt.Tooltip("mc_price:Q", title="Monte Carlo", format=",.2f"),
            alt.Tooltip("error_pct:Q", title="Error", format="+.1%"),
        ],
    )
)
zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y:Q")
left.altair_chart(
    (err_chart + zero_rule).properties(height=360, title="% error of the Monte Carlo price (below 0 = model too cheap)"),
    alt="Scatter of percentage pricing error against moneyness for calls and puts",
)
summary = (
    compare.groupby(["kind", "bucket"], observed=True)
    .agg(n=("error", "size"), avg_error=("error", "mean"), avg_error_pct=("error_pct", "mean"))
    .reset_index()
)
right.dataframe(
    summary,
    hide_index=True,
    column_config={
        "kind": "Type",
        "bucket": "Strike ÷ price",
        "n": st.column_config.NumberColumn("Options"),
        "avg_error": st.column_config.NumberColumn("Avg error ($)", format="%+.2f"),
        "avg_error_pct": st.column_config.NumberColumn("Avg error (%)", format="percent"),
    },
    alt="Average Monte Carlo pricing error by option type and moneyness bucket",
)

# --------------------------------------------------------------------------- 6. why: the smile
st.header("6. Why are we off? The volatility smile", divider="gray")
otm = compare[((compare.kind == "put") & (compare.strike <= S0)) | ((compare.kind == "call") & (compare.strike >= S0))]
otm = otm.dropna(subset=["implied_vol"])
explain(
    f"""
**What it shows:** for every option we ask "what σ would make our model match the market price
exactly?" That is the market's *implied* volatility. Each coloured line is one expiry; the dashed
flat line is the single σ our model uses ({SIGMA:.1%}).

**Why it matters:** if our model were perfect, every line would be flat and sit on the dashed line.
Wherever a line is above the dashed one, the market charges more than our model.
""",
    f"""
**What it shows:** Black–Scholes implied volatility (solved with Brent's method from the mid price)
against moneyness, using out-of-the-money options only (puts below S₀, calls above), which trade most
and carry the cleanest prices.

**Why it matters:** GBM implies a flat line. Real equity markets usually show a **skew**: higher implied
σ for low strikes, because crashes are faster and more frequent than a lognormal model allows and
investors pay up for crash insurance. The *height* of the curve at 1.0 versus the dashed line explains
the level error of section 5; its *slope* explains the pattern across strikes. Fixing both needs a
richer model (local or stochastic volatility, jumps), not more simulations.
""",
)
smile = (
    alt.Chart(otm)
    .mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2)
    .encode(
        x=alt.X("moneyness:Q", title="Strike ÷ today's price", scale=alt.Scale(zero=False)),
        y=alt.Y("implied_vol:Q", title="Implied volatility", axis=alt.Axis(format=".0%"), scale=alt.Scale(zero=False)),
        color=alt.Color(
            "expiry_label:N", title=None, sort=expiry_labels,
            scale=alt.Scale(domain=expiry_labels, range=EXPIRY_COLORS[: len(expiry_labels)]),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[
            alt.Tooltip("expiry_label:N", title="Expiry"),
            alt.Tooltip("kind:N", title="Type"),
            alt.Tooltip("strike:Q", title="Strike", format=",.2f"),
            alt.Tooltip("implied_vol:Q", title="Implied σ", format=".1%"),
        ],
    )
)
model_rule = (
    alt.Chart(pd.DataFrame({"y": [SIGMA], "label": [f"our model: σ = {SIGMA:.1%} (flat)"]}))
    .mark_rule(strokeDash=[6, 4], strokeWidth=2, color="gray")
    .encode(y="y:Q", tooltip=alt.Tooltip("label:N", title=None))
)
st.altair_chart(
    (smile + model_rule).properties(height=380, title="What σ the market is really using, strike by strike"),
    alt="Implied volatility smile by expiry against moneyness, with the model's flat sigma as a dashed line",
)


def median_iv(mask) -> float:
    vals = otm.loc[mask, "implied_vol"]
    return float(vals.median()) if len(vals) else float("nan")


atm_band = max(0.01, float((otm.moneyness - 1).abs().min())) if len(otm) else 0.01
iv_atm = median_iv((otm.moneyness - 1).abs() <= atm_band)
iv_low = median_iv(otm.moneyness < 1 - 0.8 * band)
iv_high = median_iv(otm.moneyness > 1 + 0.5 * band)
fmt = lambda v: f"{v:.1%}" if np.isfinite(v) else "n/a"  # noqa: E731
with st.container(horizontal=True):
    st.metric(f"Our σ ({sigma_source} history)", fmt(SIGMA), border=True, height="stretch")
    st.metric("Market's σ at the money", fmt(iv_atm), border=True, height="stretch")
    st.metric(f"Market's σ at low strikes (< {1 - 0.8 * band:.0%})", fmt(iv_low), border=True, height="stretch")
    st.metric(f"Market's σ at high strikes (> {1 + 0.5 * band:.0%})", fmt(iv_high), border=True, height="stretch")

st.subheader("Bonus: is only the σ guess wrong?")
atm_fix = reprice_atm_with_implied(compare, S0, r, n_sims, int(seed))
explain(
    """
**What it shows:** we re-run the simulation for the at-the-money options, but this time feed it the
market's own σ instead of our historical guess. The bars compare the average miss before and after.

**Why it matters:** if the miss shrinks a lot, the simulation engine is fine and the real question is
simply *which σ to feed it*.
""",
    """
**What it shows:** mean absolute pricing error (in dollars) per expiry for near-the-money options,
priced with historical σ versus each expiry's median at-the-money implied σ.

**Why it matters:** this separates the two sources of error. Near the money the skew is small, so
calibrating σ to the market should remove most of the gap; what remains is sampling noise, bid–ask
width, stale prices, dividends and early exercise. It is the practical argument for calibrating
models to implied rather than historical volatility.
""",
)
if atm_fix.empty:
    st.info("Not enough near-the-money options with a valid implied volatility to run this check.", icon=":material/info:")
    err_hist = err_iv = float("nan")
else:
    err_hist = float((atm_fix.mc_price_hist - atm_fix.mid).abs().mean())
    err_iv = float((atm_fix.mc_price_implied - atm_fix.mid).abs().mean())
    by_exp = (
        atm_fix.assign(
            **{
                f"Historical σ ({SIGMA:.1%})": (atm_fix.mc_price_hist - atm_fix.mid).abs(),
                "Market's implied σ": (atm_fix.mc_price_implied - atm_fix.mid).abs(),
            }
        )
        .groupby("expiry_label")[[f"Historical σ ({SIGMA:.1%})", "Market's implied σ"]]
        .mean()
        .reset_index()
        .melt("expiry_label", var_name="σ used", value_name="avg_abs_error")
    )
    sigma_names = [f"Historical σ ({SIGMA:.1%})", "Market's implied σ"]
    bars = (
        alt.Chart(by_exp)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("expiry_label:N", title=None, sort=expiry_labels, axis=alt.Axis(labelAngle=-30, labelOverlap=False)),
            xOffset=alt.XOffset("σ used:N", sort=sigma_names),
            y=alt.Y("avg_abs_error:Q", title="Average |MC − market| ($)"),
            color=alt.Color(
                "σ used:N", title=None, scale=alt.Scale(domain=sigma_names, range=[MC_COLOR, MARKET_COLOR]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("expiry_label:N", title="Expiry"),
                alt.Tooltip("σ used:N"),
                alt.Tooltip("avg_abs_error:Q", title="Avg |error| ($)", format=",.2f"),
            ],
        )
        .properties(height=320, title="At-the-money pricing error: historical vs implied σ")
    )
    left, right = st.columns([3, 1])
    left.altair_chart(bars, alt="Grouped bars of average at-the-money pricing error per expiry with historical versus implied sigma")
    right.metric("Error, historical σ", f"\\${err_hist:,.2f}", border=True)
    right.metric(
        "Error, implied σ",
        f"\\${err_iv:,.2f}",
        delta=f"{err_iv / err_hist - 1:+.0%}" if err_hist > 0 else None,
        delta_color="inverse",
        border=True,
    )

# --------------------------------------------------------------------------- 7. conclusion
st.header("7. Conclusion: how does Monte Carlo perform?", divider="gray")
atm_rows = compare[(compare.moneyness - 1).abs() <= max(0.02, atm_band)]
atm_err_pct = float(atm_rows.error_pct.median()) if len(atm_rows) else float(compare.error_pct.median())
atm_err_usd = float(atm_rows.error.median()) if len(atm_rows) else float(compare.error.median())
engine_ok = coverage >= 0.85
fix_ratio = err_iv / err_hist if np.isfinite(err_hist) and err_hist > 0 else float("nan")
skew = iv_low - iv_high if np.isfinite(iv_low) and np.isfinite(iv_high) else float("nan")

if abs(atm_err_pct) < 0.10:
    level_verdict = (
        f"**Close to the market at the money**: the typical at-the-money option is off by only "
        f"{atm_err_pct:+.0%} ({usd(atm_err_usd)}). The {sigma_source} σ of {SIGMA:.1%} happens to be close to what "
        f"traders are pricing in ({fmt(iv_atm)})."
    )
else:
    direction = "too cheap" if atm_err_pct < 0 else "too expensive"
    more_less = "more" if atm_err_pct < 0 else "less"
    level_verdict = (
        f"**{direction.capitalize()} versus the market**: at the money our price is off by about "
        f"{atm_err_pct:+.0%} ({usd(atm_err_usd)} per option). The reason is one number — the market is pricing in "
        f"{more_less} volatility ({fmt(iv_atm)}) than {symbol} showed over the last {SIGMA_PERIOD[sigma_source]} "
        f"({SIGMA:.1%})."
    )

points = [
    (
        f"**The simulation engine works.** Monte Carlo and the exact Black–Scholes formula agree to within "
        f"{usd(max_gap, 3)} on every contract, and the BS price sits inside the 95% confidence interval for "
        f"{coverage:.0%} of options (expect ~95%)."
        if engine_ok
        else f"**Check the engine.** Black–Scholes fell inside the MC 95% interval for only {coverage:.0%} of options "
        f"(largest gap {usd(max_gap, 3)}). Try more simulations or another seed."
    ),
    level_verdict,
]
if np.isfinite(skew):
    if skew > 0.02:
        points.append(
            f"**There is a volatility skew a flat σ can't fit.** Low strikes carry an implied σ of {fmt(iv_low)}, "
            f"high strikes only {fmt(iv_high)}. Crash insurance (low-strike puts) is where a constant-σ model is "
            "most wrong."
        )
    else:
        points.append(
            f"**The smile is fairly flat** (low strikes {fmt(iv_low)}, high strikes {fmt(iv_high)}), so a single σ "
            "is a reasonable description of this market today."
        )
if np.isfinite(fix_ratio):
    points.append(
        f"**Swap in the market's σ and the at-the-money error goes from {usd(err_hist)} to {usd(err_iv)}** "
        f"({fix_ratio - 1:+.0%}). "
        + (
            "The engine is fine; the open question is which σ to feed it."
            if fix_ratio < 0.5
            else "Even the market's own σ leaves a sizeable gap, so dividends, stale quotes or early exercise matter here too."
        )
    )

st.markdown("\n".join(f"{i}. {p}" for i, p in enumerate(points, 1)))

if engine_ok and abs(atm_err_pct) < 0.10:
    st.success(
        f"Monte Carlo with {sigma_source} historical volatility prices {symbol}'s at-the-money options close to the market today. "
        "Away from the money, watch the skew.",
        title="Verdict: a good fit today",
        icon=":material/check_circle:",
    )
elif engine_ok:
    st.warning(
        "The Monte Carlo method is computing the GBM price correctly, but GBM with historical σ is not how the market "
        f"prices {symbol}'s options. Use implied volatility (or a model with a skew) to get closer.",
        title="Verdict: right maths, wrong volatility",
        icon=":material/balance:",
    )
else:
    st.error(
        "The simulation and the closed-form price disagree more than sampling noise allows.",
        title="Verdict: inconclusive",
        icon=":material/error:",
    )

with st.expander("What the model ignores", icon=":material/info:"):
    st.markdown(
        f"""
- **Dividends** — {symbol}'s trailing yield is {trailing_div / last_px:.2%}. Dividends lower the stock price when paid,
  making calls a bit cheaper and puts a bit dearer than this no-dividend model says.
- **Early exercise** — US single-stock and ETF options are *American* (exercisable any day); the model is *European*.
- **Stale prices** — {'the market is open, so prices are live mids.' if market_open else 'the market looks closed, so many "market prices" are last trades, not live quotes.'}
- **"Over/undervalued" wording** — a gap means the market is pricing in a different volatility than history shows.
  Sometimes the market knows something (earnings, an event), sometimes history is right.
"""
    )
    st.download_button(
        "Download full comparison (CSV)",
        compare.drop(columns=["bucket"]).to_csv(index=False),
        f"{symbol}_mc_vs_market_{date.today().isoformat()}.csv",
        "text/csv",
        icon=":material/download:",
    )
