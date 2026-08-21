"""
Project 1: Stock Data Explorer
--------------------------------
Goal: Learn the 3 most fundamental finance concepts by computing them yourself.

  1. PRICE      - what one share costs over time
  2. RETURNS    - the % change day-to-day (this is what investors actually care about)
  3. VOLATILITY - how "jumpy" the returns are = risk

Run it:   python3 explore.py AAPL
          python3 explore.py TSLA
"""

import sys
import yfinance as yf
import pandas as pd


def explore(ticker: str):
    print(f"\nDownloading 1 year of daily data for {ticker}...\n")

    # yfinance gives us a DataFrame: one row per trading day.
    data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)

    if data.empty:
        print(f"No data found for '{ticker}'. Is the symbol correct?")
        return

    # yfinance sometimes returns multi-level columns; flatten to just "Close".
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # ----- 1. PRICE -----
    first_price = close.iloc[0]
    last_price = close.iloc[-1]
    total_change_pct = (last_price / first_price - 1) * 100

    print("=" * 50)
    print(f"  {ticker}  —  last 12 months")
    print("=" * 50)
    print(f"  Price 1 year ago : ${first_price:,.2f}")
    print(f"  Price today      : ${last_price:,.2f}")
    print(f"  Total return     : {total_change_pct:+.1f}%")

    # ----- 2. RETURNS -----
    # Daily return = today's price / yesterday's price - 1
    # This is THE core quantity in finance. We normalize away the raw price so
    # a $3000 stock and a $30 stock can be compared fairly.
    daily_returns = close.pct_change().dropna()
    avg_daily = daily_returns.mean() * 100
    best_day = daily_returns.max() * 100
    worst_day = daily_returns.min() * 100

    print(f"\n  Average daily move : {avg_daily:+.3f}%")
    print(f"  Best single day    : {best_day:+.2f}%")
    print(f"  Worst single day   : {worst_day:+.2f}%")

    # ----- 3. VOLATILITY (risk) -----
    # Volatility = standard deviation of returns = how spread out the daily
    # moves are. Big number = wild ride = risky. We "annualize" the daily
    # number by multiplying by sqrt(252), the ~number of trading days in a year.
    daily_vol = daily_returns.std() * 100
    annual_vol = daily_vol * (252 ** 0.5)

    print(f"\n  Daily volatility   : {daily_vol:.2f}%")
    print(f"  Annualized vol     : {annual_vol:.1f}%   <- this is 'risk'")
    print("=" * 50)

    print(
        "\nWhat you just learned:\n"
        "  • Returns strip away raw price so stocks are comparable.\n"
        "  • Volatility is the standard textbook measure of RISK.\n"
        "  • Higher return usually comes with higher volatility — that\n"
        "    trade-off is the heart of investing.\n"
    )


if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    explore(ticker)
