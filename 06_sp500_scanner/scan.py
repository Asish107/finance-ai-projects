"""
Project 6: S&P 500 Scanner — screen the entire index at once
-------------------------------------------------------------
Goal: Move from analyzing ONE stock to screening 500 at once. This is what a
"stock screener" is: compute the same metrics for every company, then rank
and filter to find what you're looking for.

Data: 100% real, from Yahoo Finance, updated through the latest market close.

What it computes for each of the ~500 companies:
  • 1-year return    (how much it gained/lost)
  • annualized volatility (risk)
  • return / risk ratio  (reward per unit of risk — higher is "better")

Then it prints leaderboards and saves everything to a CSV you can open in Excel.

Run it:  python3 scan.py            # full S&P 500 (takes ~1-2 min)
         python3 scan.py 50         # quick test on first 50 names
"""

import sys
import io
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TRADING_DAYS = 252
# The "risk-free rate" = what you'd earn with zero risk (e.g. US Treasury bills).
# The Sharpe ratio only rewards return ABOVE this baseline, because you could
# always earn the risk-free rate without taking any risk. ~4.5% as of 2026.
RISK_FREE_RATE = 4.5


def get_sp500_tickers():
    """Scrape the current S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia 403s the default urllib agent, so send a browser-like header.
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()  # BRK.B -> BRK-B
    names = dict(zip(table["Symbol"].str.replace(".", "-", regex=False), table["Security"]))
    return tickers, names


def scan(limit=None):
    print("\nFetching the S&P 500 company list from Wikipedia...")
    tickers, names = get_sp500_tickers()
    if limit:
        tickers = tickers[:limit]
    print(f"Downloading 1 year of prices for {len(tickers)} companies "
          f"(this takes a minute)...\n")

    # One big batched download is far faster than 500 separate requests.
    data = yf.download(tickers, period="1y", interval="1d",
                       progress=True, auto_adjust=True)["Close"]

    rows = []
    for t in tickers:
        try:
            s = data[t].dropna()
            if len(s) < 200:            # need enough history to be meaningful
                continue
            ret = s.pct_change().dropna()
            rows.append({
                "ticker": t,
                "name": names.get(t, "")[:28],
                "price": round(float(s.iloc[-1]), 2),
                "return_1y_%": round(float(s.iloc[-1] / s.iloc[0] - 1) * 100, 1),
                "volatility_%": round(float(ret.std() * np.sqrt(TRADING_DAYS)) * 100, 1),
            })
        except (KeyError, IndexError):
            continue

    df = pd.DataFrame(rows)
    # SHARPE RATIO = (return above the risk-free rate) / volatility.
    # It answers: "how much EXTRA reward did I get per unit of risk, versus
    # just parking my money risk-free?" Higher = better risk-adjusted return.
    # This is the single most-cited number in professional investing.
    df["sharpe"] = ((df["return_1y_%"] - RISK_FREE_RATE) / df["volatility_%"]).round(2)

    out = "sp500_scan.csv"
    df.to_csv(out, index=False)

    def leaderboard(title, sort_col, ascending=False, n=10):
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)
        top = df.sort_values(sort_col, ascending=ascending).head(n)
        for _, r in top.iterrows():
            print(f"  {r['ticker']:<6} {r['name']:<28} "
                  f"{r['return_1y_%']:+7.1f}%  vol {r['volatility_%']:5.1f}%  "
                  f"sharpe {r['sharpe']:+5.2f}")

    print(f"\nScanned {len(df)} companies. Full data saved to: {out}")
    leaderboard("BIGGEST GAINERS (1 year)", "return_1y_%")
    leaderboard("BIGGEST LOSERS (1 year)", "return_1y_%", ascending=True)
    leaderboard("BEST RISK-ADJUSTED RETURN (Sharpe ratio)", "sharpe")
    leaderboard("MOST VOLATILE (riskiest)", "volatility_%")

    print(
        "\n  How to read this:\n"
        "  • 'Biggest gainers' alone is a trap — they're often the riskiest.\n"
        "  • The SHARPE RATIO rewards steady climbers over wild gamblers: it's\n"
        "    return ABOVE the risk-free rate, per unit of risk. >1 is good,\n"
        "    >2 is excellent. It's THE standard risk-adjusted-return metric.\n"
        f"  • Open {out} in Excel/Numbers to sort and filter yourself.\n"
    )


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    scan(limit)
