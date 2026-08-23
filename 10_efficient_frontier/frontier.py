"""
Project 10: The Efficient Frontier (Modern Portfolio Theory)
-------------------------------------------------------------
Goal: In Project 2 you learned diversification LOWERS risk. Now the big
question: if mixing stocks helps, what is the BEST mix? This is Modern
Portfolio Theory (Harry Markowitz, 1952, Nobel Prize 1990).

The core idea:
  Every possible way of splitting your money across N stocks gives one
  (risk, return) point. Plot thousands of them and they form a shape. The
  TOP-LEFT edge of that shape is the "efficient frontier": for any level of
  risk, it is the mix with the highest possible return. Any portfolio BELOW
  the frontier is dumb, you could get more return for the same risk.

How we find it (Monte Carlo, the intuitive way):
  Generate thousands of RANDOM portfolios, compute each one's risk and return,
  and let the winners reveal the frontier. Then we highlight two special ones:
    * MAX SHARPE  = best reward-per-risk (the "tangency" portfolio)
    * MIN VARIANCE = the lowest-risk mix possible

Run:  python3 frontier.py
      python3 frontier.py AAPL MSFT KO XOM GLD TLT
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TRADING_DAYS = 252
RISK_FREE_RATE = 0.045      # 4.5% as a decimal
N_PORTFOLIOS = 20000        # how many random mixes to simulate


def run(tickers):
    print(f"\nDownloading 3 years of data for: {', '.join(tickers)}\n")
    prices = yf.download(tickers, period="3y", interval="1d",
                         progress=False, auto_adjust=True)["Close"].dropna()
    prices = prices[[t for t in tickers if t in prices.columns]]
    tickers = list(prices.columns)

    returns = prices.pct_change(fill_method=None).dropna()

    # Expected annual return of each stock, and the covariance matrix.
    # Covariance captures BOTH each stock's volatility AND how they move
    # together, which is the whole engine of diversification.
    mean_annual = returns.mean() * TRADING_DAYS
    cov_annual = returns.cov() * TRADING_DAYS

    n = len(tickers)
    results = np.zeros((N_PORTFOLIOS, 3))     # [return, volatility, sharpe]
    all_weights = np.zeros((N_PORTFOLIOS, n))

    for i in range(N_PORTFOLIOS):
        # Random weights that sum to 1 (a valid portfolio).
        w = np.random.random(n)
        w /= w.sum()
        all_weights[i] = w

        port_return = np.dot(w, mean_annual)
        # Portfolio variance = w^T * covariance * w. This formula is the
        # mathematical heart of MPT, it rewards low-correlation mixes.
        port_vol = np.sqrt(w @ cov_annual @ w)
        sharpe = (port_return - RISK_FREE_RATE) / port_vol

        results[i] = [port_return, port_vol, sharpe]

    ret, vol, sharpe = results[:, 0], results[:, 1], results[:, 2]

    # The two special portfolios.
    max_sharpe_i = sharpe.argmax()
    min_vol_i = vol.argmin()

    def describe(i, label):
        print("=" * 60)
        print(f"  {label}")
        print("=" * 60)
        print(f"    Expected return : {ret[i]*100:6.1f}%")
        print(f"    Volatility (risk): {vol[i]*100:6.1f}%")
        print(f"    Sharpe ratio    : {sharpe[i]:6.2f}")
        print("    Weights:")
        for t, wt in sorted(zip(tickers, all_weights[i]), key=lambda x: -x[1]):
            bar = "#" * int(wt * 40)
            print(f"      {t:<6} {wt*100:5.1f}%  {bar}")
        print()

    describe(max_sharpe_i, "MAX SHARPE PORTFOLIO (best reward per unit of risk)")
    describe(min_vol_i, "MINIMUM VARIANCE PORTFOLIO (lowest possible risk)")

    # ----- Plot the frontier -----
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(vol*100, ret*100, c=sharpe, cmap="viridis", s=6, alpha=0.6)
    fig.colorbar(sc, label="Sharpe ratio")

    ax.scatter(vol[max_sharpe_i]*100, ret[max_sharpe_i]*100,
               marker="*", color="red", s=500, edgecolors="black",
               label="Max Sharpe", zorder=5)
    ax.scatter(vol[min_vol_i]*100, ret[min_vol_i]*100,
               marker="*", color="deepskyblue", s=500, edgecolors="black",
               label="Min Variance", zorder=5)

    ax.set_xlabel("Risk  (annualized volatility %)")
    ax.set_ylabel("Expected return (%)")
    ax.set_title(f"Efficient Frontier: {N_PORTFOLIOS:,} random portfolios of "
                 + ", ".join(tickers))
    ax.legend()
    ax.grid(alpha=0.3)
    out = "efficient_frontier.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"  Chart saved to: {out}")

    print(
        "\n  How to read the chart:\n"
        "  • Each dot is one possible portfolio. Left = less risk, up = more return.\n"
        "  • The upper-left EDGE of the cloud is the 'efficient frontier': the\n"
        "    best return you can get for each level of risk. Dots below it are\n"
        "    inefficient, wasted risk.\n"
        "  • The RED star (max Sharpe) is the mix with the best reward-per-risk.\n"
        "  • The BLUE star (min variance) is the safest possible mix.\n"
        "  • Notice you did NOT pick these weights, the MATH found the optimal\n"
        "    blend by exploiting how the stocks move together. That is MPT.\n"
    )


if __name__ == "__main__":
    tickers = [t.upper() for t in sys.argv[1:]] or ["AAPL", "MSFT", "KO", "XOM", "GLD", "TLT"]
    run(tickers)
