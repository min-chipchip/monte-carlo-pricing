"""Download, cache and load market data from Yahoo Finance (via ``yfinance``).

Three kinds of data are handled:

* daily price history of the underlying (for ``S0``, volatility and VaR backtests),
* the 13-week US T-bill rate ``^IRX`` (a proxy for the risk-free rate ``r``),
* a snapshot of today's option chain (market prices to compare MC prices against).

Everything is saved as CSV under ``data/raw/`` so notebooks and the app can
work offline and reproducibly; call the ``fetch_*`` functions again to refresh.
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
TRADING_DAYS_PER_YEAR = 252


def _save(df: pd.DataFrame, filename: str, data_dir: Path | None) -> Path:
    """Write ``df`` to ``<data_dir>/<filename>`` (default ``data/raw``) and return the path."""
    directory = Path(data_dir) if data_dir is not None else RAW_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    df.to_csv(path, index=False)
    return path


def _ticker(symbol: str):
    """Return a ``yfinance.Ticker``; imported lazily so the rest of ``src`` works without yfinance."""
    import yfinance as yf

    return yf.Ticker(symbol)


def fetch_price_history(
    symbol: str = "SPY", period: str = "10y", save: bool = True, data_dir: Path | None = None
) -> pd.DataFrame:
    """Download daily OHLCV price history for ``symbol``.

    Parameters
    ----------
    symbol : str
        Yahoo Finance ticker, e.g. ``"SPY"``.
    period : str
        Look-back window understood by yfinance (``"1y"``, ``"10y"``, ``"5y"``, ...).
    save : bool
        If True, also write ``<symbol>_prices.csv`` to ``data_dir``.
    data_dir : Path, optional
        Output folder; defaults to ``data/raw``.

    Returns
    -------
    pandas.DataFrame
        One row per trading day with columns ``date, open, high, low, close,
        adj_close, volume, dividends``. ``close`` is the raw traded price (use it
        for ``S0``); ``adj_close`` is adjusted for dividends/splits (use it for returns).
    """
    raw = _ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"No price data returned for {symbol!r}")

    df = raw.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.rename(columns={"Date": "date", "Adj Close": "adj_close"})
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df[["date", "open", "high", "low", "close", "adj_close", "volume", "dividends"]]

    if save:
        _save(df, f"{symbol}_prices.csv", data_dir)
    return df


def fetch_risk_free_rate(period: str = "10y", save: bool = True, data_dir: Path | None = None) -> pd.DataFrame:
    """Download the 13-week US T-bill yield (``^IRX``) as a daily risk-free rate.

    Yahoo quotes ``^IRX`` in percent (e.g. ``4.25`` means 4.25%). This function
    converts it to a decimal annual rate and also gives the continuously
    compounded equivalent ``r_cont = ln(1 + r)``, which is the ``r`` used by
    the GBM / Black--Scholes code in this project.

    Parameters
    ----------
    period : str
        Look-back window understood by yfinance.
    save : bool
        If True, also write ``risk_free_rate.csv`` to ``data_dir``.
    data_dir : Path, optional
        Output folder; defaults to ``data/raw``.

    Returns
    -------
    pandas.DataFrame
        Columns ``date, rate, r_cont`` (both rates as decimals, annualised).
    """
    raw = _ticker("^IRX").history(period=period, interval="1d")
    if raw.empty:
        raise RuntimeError("No data returned for ^IRX")

    df = raw.reset_index()[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.rename(columns={"Date": "date", "Close": "rate"})
    df["rate"] = df["rate"] / 100.0
    df["r_cont"] = np.log1p(df["rate"])

    if save:
        _save(df, "risk_free_rate.csv", data_dir)
    return df


def _pick_expiries(expiries: list[str], today: pd.Timestamp, n: int, min_days: int, max_days: int) -> list[str]:
    """Choose up to ``n`` expiries between ``min_days`` and ``max_days`` away, spread evenly across that range."""
    eligible = [e for e in expiries if min_days <= (pd.Timestamp(e) - today).days <= max_days]
    if len(eligible) <= n:
        return eligible
    idx = np.unique(np.linspace(0, len(eligible) - 1, n).round().astype(int))
    return [eligible[i] for i in idx]


def fetch_option_chain(
    symbol: str = "SPY",
    n_expiries: int = 4,
    min_days: int = 20,
    max_days: int = 120,
    save: bool = True,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Download a snapshot of the current option chain for ``symbol``.

    Only ``n_expiries`` expiry dates are fetched, spread evenly between
    ``min_days`` and ``max_days`` from today (SPY has daily expiries, so taking
    "the nearest N" would give only very short-dated options).

    Take snapshots during US market hours (9:30-16:00 New York time). Outside
    those hours Yahoo usually reports ``bid = ask = 0`` and a meaningless implied
    volatility; such rows get ``has_quote = False`` and ``mid`` falls back to the
    last traded price, which may be stale.

    Parameters
    ----------
    symbol : str
        Yahoo Finance ticker of the underlying.
    n_expiries : int
        Number of expiry dates to include.
    min_days, max_days : int
        Calendar-day window for expiries. The default 20-120 days covers the
        typical 1-3 month options used for model comparison.
    save : bool
        If True, write ``<symbol>_options_<YYYY-MM-DD>.csv`` to ``data_dir``.
        The snapshot date is in the filename, so repeated runs on different days
        build up a history instead of overwriting it.
    data_dir : Path, optional
        Output folder; defaults to ``data/raw``.

    Returns
    -------
    pandas.DataFrame
        One row per contract with columns including ``snapshot_date, expiry,
        days_to_expiry, time_to_maturity, kind ('call'/'put'), strike, bid, ask,
        mid, has_quote, last_price, implied_volatility, volume, open_interest,
        underlying_price``. ``time_to_maturity`` is in years (calendar days / 365).
    """
    ticker = _ticker(symbol)
    today = pd.Timestamp(date.today())

    expiries = _pick_expiries(list(ticker.options), today, n_expiries, min_days, max_days)
    if not expiries:
        raise RuntimeError(f"No option expiries found for {symbol!r}")

    spot = float(ticker.history(period="1d")["Close"].iloc[-1])

    frames = []
    for expiry in expiries:
        chain = ticker.option_chain(expiry)
        for kind, table in (("call", chain.calls), ("put", chain.puts)):
            t = table.copy()
            t["kind"] = kind
            t["expiry"] = pd.Timestamp(expiry)
            frames.append(t)

    df = pd.concat(frames, ignore_index=True).rename(
        columns={
            "contractSymbol": "contract",
            "lastPrice": "last_price",
            "impliedVolatility": "implied_volatility",
            "openInterest": "open_interest",
            "lastTradeDate": "last_trade_date",
        }
    )
    df["snapshot_date"] = today
    df["underlying_price"] = spot
    df["days_to_expiry"] = (df["expiry"] - today).dt.days
    df["time_to_maturity"] = df["days_to_expiry"] / 365.0
    # Mid price is the fairest single market price; fall back to last trade if quotes are missing.
    df["has_quote"] = (df["bid"] > 0) & (df["ask"] > 0)
    df["mid"] = np.where(df["has_quote"], (df["bid"] + df["ask"]) / 2.0, df["last_price"])
    if df["has_quote"].mean() < 0.5:
        warnings.warn(
            f"Only {df['has_quote'].sum()} of {len(df)} contracts have live bid/ask quotes - "
            "the US market is probably closed. 'mid' falls back to last traded prices.",
            stacklevel=2,
        )

    columns = [
        "snapshot_date", "contract", "expiry", "days_to_expiry", "time_to_maturity", "kind",
        "strike", "bid", "ask", "mid", "has_quote", "last_price", "implied_volatility",
        "volume", "open_interest", "in_the_money", "underlying_price", "last_trade_date",
    ]
    df = df[[c for c in columns if c in df.columns]]

    if save:
        _save(df, f"{symbol}_options_{today.date().isoformat()}.csv", data_dir)
    return df


def realized_volatility(prices: pd.Series, window: int = 30) -> pd.Series:
    """Rolling annualised volatility from daily prices.

    Computes ``std(log returns over the last `window` days) * sqrt(252)``.
    Pass adjusted closes so dividends and splits don't show up as fake returns.

    Parameters
    ----------
    prices : pandas.Series
        Daily prices, oldest first.
    window : int
        Number of daily returns per estimate (30 ~ 1.5 months, 60 ~ 3 months).

    Returns
    -------
    pandas.Series
        Annualised volatility as a decimal (0.18 = 18%); the first ``window``
        values are NaN.
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def load_price_history(symbol: str = "SPY", data_dir: Path | None = None) -> pd.DataFrame:
    """Load ``<symbol>_prices.csv`` saved by :func:`fetch_price_history`."""
    directory = Path(data_dir) if data_dir is not None else RAW_DIR
    return pd.read_csv(directory / f"{symbol}_prices.csv", parse_dates=["date"])


def load_risk_free_rate(data_dir: Path | None = None) -> pd.DataFrame:
    """Load ``risk_free_rate.csv`` saved by :func:`fetch_risk_free_rate`."""
    directory = Path(data_dir) if data_dir is not None else RAW_DIR
    return pd.read_csv(directory / "risk_free_rate.csv", parse_dates=["date"])


def load_option_chain(symbol: str = "SPY", snapshot: str | None = None, data_dir: Path | None = None) -> pd.DataFrame:
    """Load a saved option-chain snapshot.

    Parameters
    ----------
    symbol : str
        Underlying ticker.
    snapshot : str, optional
        Snapshot date as ``"YYYY-MM-DD"``. If omitted, the most recent saved
        snapshot for ``symbol`` is loaded.
    data_dir : Path, optional
        Folder to read from; defaults to ``data/raw``.
    """
    directory = Path(data_dir) if data_dir is not None else RAW_DIR
    if snapshot is None:
        files = sorted(directory.glob(f"{symbol}_options_*.csv"))
        if not files:
            raise FileNotFoundError(f"No saved option chains for {symbol!r} in {directory}")
        path = files[-1]
    else:
        path = directory / f"{symbol}_options_{snapshot}.csv"
    return pd.read_csv(path, parse_dates=["snapshot_date", "expiry"])


def download_all(symbol: str = "SPY", period: str = "10y", n_expiries: int = 4) -> dict[str, pd.DataFrame]:
    """Fetch and save price history, risk-free rate and an option-chain snapshot in one call.

    Returns a dict with keys ``"prices"``, ``"rates"`` and ``"options"``.
    """
    return {
        "prices": fetch_price_history(symbol, period),
        "rates": fetch_risk_free_rate(period),
        "options": fetch_option_chain(symbol, n_expiries),
    }
