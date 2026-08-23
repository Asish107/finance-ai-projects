"""
Project 9: Backtester — "would my strategy actually have worked?"
------------------------------------------------------------------
Goal: Test a trading STRATEGY against history and compare it to just buying
the whole market (SPY). This is the core activity of quantitative finance:
you never trust an idea until you've backtested it against real past data.

The strategy we test (simple, to learn the mechanics):
  "A year ago, buy the N stocks that had the BEST Sharpe ratio over the
   PRIOR year, hold equally weighted, and see how they did."

Then we compare that portfolio's growth to SPY (the S&P 500 index).

THE #1 BACKTESTING TRAP — lookahead bias:
  We must pick stocks using ONLY data available AT THE START of the holding
  period. If we peeked at which stocks won DURING the period, we'd "predict"
  the past perfectly and fool ourselves. This code splits time into:
     [ FORMATION year ] -> rank stocks     [ HOLDING year ] -> measure results
  The two windows never overlap. That discipline is everything.

Run:  python3 backtest.py            # top 10 by Sharpe, vs SPY
      python3 backtest.py 5          # top 5
"""

import sys
import io
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TRADING_DAYS = 252
RISK_FREE_RATE = 4.5


def sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    return t["Symbol"].str.replace(".", "-", regex=False).tolist()


def annualized_sharpe(daily_returns):
    """Sharpe ratio from a series of daily returns."""
    ann_ret = daily_returns.mean() * TRADING_DAYS * 100
    ann_vol = daily_returns.std() * np.sqrt(TRADING_DAYS) * 100
    return (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol else np.nan


def run(top_n=10):
    tickers = sp500_tickers()
    print(f"\nDownloading 2 years of data for {len(tickers)} stocks + SPY...\n")

    # 2 years total: first year = formation, second year = holding.
    close = yf.download(tickers + ["SPY"], period="2y", interval="1d",
                        progress=True, auto_adjust=True)["Close"]
    close = close.dropna(axis=1, thresh=int(len(close) * 0.9))  # drop sparse names

    mid = len(close) // 2
    formation = close.iloc[:mid]      # <-- only this is used to PICK stocks
    holding = close.iloc[mid:]        # <-- only this is used to MEASURE results

    form_start, form_end = formation.index[0].date(), formation.index[-1].date()
    hold_start, hold_end = holding.index[0].date(), holding.index[-1].date()

    # ----- Rank stocks by Sharpe in the FORMATION window only -----
    form_ret = formation.pct_change(fill_method=None).dropna()
    sharpes = form_ret.drop(columns=["SPY"], errors="ignore").apply(annualized_sharpe)
    winners = sharpes.sort_values(ascending=False).head(top_n).index.tolist()

    print("=" * 60)
    print(f"  STRATEGY: buy top {top_n} Sharpe stocks from the formation year")
    print("=" * 60)
    print(f"  Formation (pick here): {form_start} -> {form_end}")
    print(f"  Holding   (test here): {hold_start} -> {hold_end}")
    print(f"  Picked: {', '.join(winners)}")

    # ----- Measure how the picks did in the HOLDING window -----
    hold_ret = holding.pct_change(fill_method=None).dropna()
    strat_daily = hold_ret[winners].mean(axis=1)          # equal weight
    spy_daily = hold_ret["SPY"]

    def summarize(daily, label):
        total = ((1 + daily).prod() - 1) * 100
        vol = daily.std() * np.sqrt(TRADING_DAYS) * 100
        sharpe = annualized_sharpe(daily)
        # Max drawdown = worst peak-to-trough drop (the scariest number for investors).
        curve = (1 + daily).cumprod()
        drawdown = ((curve / curve.cummax()) - 1).min() * 100
        print(f"\n  {label}")
        print(f"    Total return   : {total:+7.1f}%")
        print(f"    Volatility     : {vol:6.1f}%")
        print(f"    Sharpe ratio   : {sharpe:+6.2f}")
        print(f"    Max drawdown   : {drawdown:6.1f}%")
        return total, sharpe

    print("\n" + "=" * 60)
    print("  RESULTS (holding year — out of sample)")
    print("=" * 60)
    s_ret, s_sharpe = summarize(strat_daily, f"STRATEGY (top {top_n})")
    b_ret, b_sharpe = summarize(spy_daily, "BENCHMARK (SPY = whole market)")

    print("\n" + "=" * 60)
    verdict = "BEAT" if s_ret > b_ret else "LOST TO"
    print(f"  The strategy {verdict} the market by {s_ret - b_ret:+.1f} pts of return.")
    sharpe_verdict = "better" if s_sharpe > b_sharpe else "worse"
    print(f"  Risk-adjusted (Sharpe), it was {sharpe_verdict}.")
    print("=" * 60)

    print(
        "\n  What you learned:\n"
        "  • A backtest tests a strategy on PAST data, split so you never peek\n"
        "    at the future when choosing (no lookahead bias).\n"
        "  • Beating the market ONCE proves little — one period is luck-prone.\n"
        "    Real research tests many periods and many markets.\n"
        "  • 'Past winners keep winning' (momentum) sometimes works and sometimes\n"
        "    reverses hard. That uncertainty is why this is a research problem,\n"
        "    not a solved formula. Always compare to the humble SPY benchmark.\n"
    )


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(n)
