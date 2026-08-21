"""
Project 2: Portfolio & Risk Dashboard
--------------------------------------
Goal: See WHY diversification is called "the only free lunch in finance."

Big idea: A portfolio's risk is NOT just the average of its stocks' risks.
When stocks don't move in lockstep, their ups and downs partly cancel out,
so the combined risk can be LOWER than the average piece. That cancellation
is measured by CORRELATION.

We will:
  1. Download several stocks at once.
  2. Measure how correlated they are (do they move together?).
  3. Build an equal-weight portfolio and show its risk is lower than the
     average of the individual risks.
  4. Draw a dashboard: growth chart + correlation heatmap.

Run it:  python3 portfolio.py
         python3 portfolio.py AAPL MSFT KO XOM GLD
"""

import sys
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRADING_DAYS = 252


def build_dashboard(tickers):
    print(f"\nDownloading 2 years of data for: {', '.join(tickers)}\n")
    raw = yf.download(tickers, period="2y", interval="1d",
                      progress=False, auto_adjust=True)["Close"]

    # If a single ticker slips through, make sure we still have a DataFrame.
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    prices = raw.dropna()

    # ----- Daily returns for every stock -----
    returns = prices.pct_change().dropna()

    # ----- Individual annualized volatility (risk) of each stock -----
    indiv_vol = returns.std() * np.sqrt(TRADING_DAYS)

    # ----- Correlation: -1 (opposite) .. 0 (unrelated) .. +1 (lockstep) -----
    corr = returns.corr()

    # ----- Equal-weight portfolio -----
    # Put the same fraction of money in each stock.
    n = len(tickers)
    weights = np.repeat(1 / n, n)

    # Portfolio daily return = weighted sum of each stock's daily return.
    port_returns = returns.dot(weights)
    port_vol = port_returns.std() * np.sqrt(TRADING_DAYS)

    # The "naive" risk = simple average of the individual risks.
    # This is what you'd GUESS the portfolio risk is if you ignored correlation.
    naive_vol = indiv_vol.mean()

    # ----- Print the lesson -----
    print("=" * 56)
    print("  INDIVIDUAL RISK (annualized volatility)")
    print("=" * 56)
    for t in tickers:
        print(f"  {t:<6}  {indiv_vol[t] * 100:5.1f}%")
    print("-" * 56)
    print(f"  Average of the pieces      : {naive_vol * 100:5.1f}%")
    print(f"  ACTUAL portfolio risk      : {port_vol * 100:5.1f}%   <-- lower!")
    reduction = (1 - port_vol / naive_vol) * 100
    print(f"  Risk removed by diversifying: {reduction:5.1f}%")
    print("=" * 56)
    print(
        "\n  ^ The portfolio is LESS risky than the average of its stocks.\n"
        "    That gap is the 'free lunch': the stocks' independent wobbles\n"
        "    partially cancelled. Lower correlation => bigger free lunch.\n"
    )

    # ----- Chart: growth of $100 + correlation heatmap -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    growth = (1 + returns).cumprod() * 100          # $100 invested at start
    port_growth = (1 + port_returns).cumprod() * 100
    growth.plot(ax=ax1, alpha=0.55, linewidth=1)
    port_growth.plot(ax=ax1, color="black", linewidth=2.5, label="PORTFOLIO")
    ax1.set_title("Growth of $100")
    ax1.set_ylabel("Value ($)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    im = ax2.imshow(corr, cmap="RdYlGn_r", vmin=-1, vmax=1)
    ax2.set_xticks(range(n)); ax2.set_xticklabels(tickers, rotation=45)
    ax2.set_yticks(range(n)); ax2.set_yticklabels(tickers)
    ax2.set_title("Correlation (green=moves apart, red=moves together)")
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax2, fraction=0.046)

    fig.suptitle("Portfolio Dashboard  |  " + "  ".join(tickers), fontsize=13)
    fig.tight_layout()
    out = "dashboard.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"  Chart saved to: {out}\n")


if __name__ == "__main__":
    tickers = [t.upper() for t in sys.argv[1:]] or ["AAPL", "MSFT", "KO", "XOM", "GLD"]
    build_dashboard(tickers)
