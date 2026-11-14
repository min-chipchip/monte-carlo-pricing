"""Download and load SPY market data from Yahoo Finance (via `yfinance`).

Files are cached as CSV in data/raw/ so the notebooks run offline.
Refresh everything (ideally during US market hours) with:

    python -c "from src.market_data import download_all; download_all('SPY')"
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
TRADING_DAYS_PER_YEAR = 252


# ----------------------------------------------------------------- downloading
def fetch_price_history(symbol="SPY", period="10y"):
    """Daily prices -> data/raw/<symbol>_prices.csv (columns include close, adj_close)."""
    import yfinance as yf

    df = yf.Ticker(symbol).history(period=period, auto_adjust=False).reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.rename(columns={"Date": "date", "Adj Close": "adj_close"})
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df[["date", "open", "high", "low", "close", "adj_close", "volume", "dividends"]]
    df.to_csv(RAW_DIR / f"{symbol}_prices.csv", index=False)
    return df


def fetch_risk_free_rate(period="10y"):
    """13-week T-bill yield (^IRX) -> data/raw/risk_free_rate.csv.

    `rate` is the annual yield as a decimal; `r_cont = ln(1 + rate)` is the
    continuously compounded rate used as r in the models.
    """
    import yfinance as yf

    df = yf.Ticker("^IRX").history(period=period).reset_index()[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.rename(columns={"Date": "date", "Close": "rate"})
    df["rate"] = df["rate"] / 100.0
    df["r_cont"] = np.log1p(df["rate"])
    df.to_csv(RAW_DIR / "risk_free_rate.csv", index=False)
    return df


def fetch_option_chain(symbol="SPY", n_expiries=4, min_days=20, max_days=120):
    """Today's option prices for `n_expiries` expiries spread over 20-120 days.

    Saved as data/raw/<symbol>_options_<YYYY-MM-DD>.csv. `mid` is the
    bid/ask midpoint, or the last traded price when the market is closed.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    today = pd.Timestamp(date.today())
    eligible = [e for e in ticker.options if min_days <= (pd.Timestamp(e) - today).days <= max_days]
    picks = np.unique(np.linspace(0, len(eligible) - 1, n_expiries).round().astype(int))
    expiries = [eligible[i] for i in picks] if eligible else []
    if not expiries:
        raise RuntimeError(f"No option expiries found for {symbol!r}")

    frames = []
    for expiry in expiries:
        chain = ticker.option_chain(expiry)
        for kind, table in (("call", chain.calls), ("put", chain.puts)):
            frames.append(table.assign(kind=kind, expiry=pd.Timestamp(expiry)))

    df = pd.concat(frames, ignore_index=True).rename(columns={
        "contractSymbol": "contract",
        "lastPrice": "last_price",
        "impliedVolatility": "implied_volatility",
        "openInterest": "open_interest",
        "lastTradeDate": "last_trade_date",
    })
    df["snapshot_date"] = today
    df["underlying_price"] = float(ticker.history(period="1d")["Close"].iloc[-1])
    df["days_to_expiry"] = (df["expiry"] - today).dt.days
    df["time_to_maturity"] = df["days_to_expiry"] / 365.0
    df["has_quote"] = (df["bid"] > 0) & (df["ask"] > 0)
    df["mid"] = np.where(df["has_quote"], (df["bid"] + df["ask"]) / 2.0, df["last_price"])
    df.to_csv(RAW_DIR / f"{symbol}_options_{today.date()}.csv", index=False)
    return df


def download_all(symbol="SPY"):
    """Refresh prices, the risk-free rate and an option-chain snapshot."""
    return {
        "prices": fetch_price_history(symbol),
        "rates": fetch_risk_free_rate(),
        "options": fetch_option_chain(symbol),
    }


# ------------------------------------------------------------------- loading
def load_price_history(symbol="SPY"):
    return pd.read_csv(RAW_DIR / f"{symbol}_prices.csv", parse_dates=["date"])


def load_risk_free_rate():
    return pd.read_csv(RAW_DIR / "risk_free_rate.csv", parse_dates=["date"])


def load_option_chain(symbol="SPY"):
    """Load the most recent saved option-chain snapshot."""
    files = sorted(RAW_DIR.glob(f"{symbol}_options_*.csv"))
    if not files:
        raise FileNotFoundError(f"No saved option chains for {symbol!r} in {RAW_DIR}")
    return pd.read_csv(files[-1], parse_dates=["snapshot_date", "expiry"])


# ---------------------------------------------------------------- volatility
def realized_volatility(prices, window=60):
    """Rolling annualised volatility: sd(daily log returns over `window` days) * sqrt(252)."""
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
