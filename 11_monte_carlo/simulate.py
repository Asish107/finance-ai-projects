"""
Project 11: Monte Carlo Simulation - "where could my money end up?"
--------------------------------------------------------------------
Goal: You cannot predict the future of the market. So instead of ONE guess,
you simulate THOUSANDS of possible futures and look at the whole range. This
is a Monte Carlo simulation, named after the casino, because it uses
randomness to answer questions that have no clean formula.

The question: "If I invest $10,000 today for 10 years, where might it end up,
and what is the chance I actually LOSE money?"

How it works:
  1. Measure a stock's historical average daily return and its volatility.
  2. Simulate one future by taking random daily steps drawn from that
     distribution, compounding day by day for 10 years.
  3. Do that 5,000 times. Each run is one "possible future."
  4. The SPREAD of the 5,000 endings is your honest range of outcomes.

Key lesson: the average outcome hides enormous uncertainty. A good investment
is not one guaranteed number, it is a distribution of possibilities, and you
must respect the bad tail as much as the good one.

Run:  python3 simulate.py SPY
      python3 simulate.py AAPL 15000 20      # 15k dollars, 20 years
"""

import sys
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

TRADING_DAYS = 252
N_SIMS = 5000


def run(ticker, start_value=10000, years=10):
    print(f"\nDownloading history for {ticker}...\n")
    prices = yf.download(ticker, period="5y", interval="1d",
                         progress=False, auto_adjust=True)["Close"]
    prices = prices.iloc[:, 0] if hasattr(prices, "columns") else prices
    returns = prices.pct_change(fill_method=None).dropna()

    # The two numbers that define the stock's behavior.
    mu = returns.mean()             # average daily return
    sigma = returns.std()           # daily volatility
    days = years * TRADING_DAYS

    # ----- Run the simulation -----
    # Each column is one possible 10-year future. We draw random daily returns
    # from a normal distribution with the stock's real mean and volatility,
    # then compound them.
    rng = np.random.default_rng(42)
    random_daily = rng.normal(mu, sigma, size=(days, N_SIMS))
    paths = start_value * np.cumprod(1 + random_daily, axis=0)
    endings = paths[-1]

    # ----- Summarize the range of outcomes -----
    p10, p50, p90 = np.percentile(endings, [10, 50, 90])
    prob_loss = (endings < start_value).mean() * 100
    prob_double = (endings > 2 * start_value).mean() * 100

    print("=" * 60)
    print(f"  MONTE CARLO: ${start_value:,} in {ticker} for {years} years")
    print(f"  ({N_SIMS:,} simulated futures)")
    print("=" * 60)
    print(f"    Historical avg return : {mu*TRADING_DAYS*100:5.1f}% / year")
    print(f"    Historical volatility : {sigma*np.sqrt(TRADING_DAYS)*100:5.1f}% / year")
    print("    -----------------------------------------------")
    print(f"    Pessimistic (10th pct): ${p10:12,.0f}")
    print(f"    Median      (50th pct): ${p50:12,.0f}")
    print(f"    Optimistic  (90th pct): ${p90:12,.0f}")
    print("    -----------------------------------------------")
    print(f"    Chance of LOSING money: {prob_loss:5.1f}%")
    print(f"    Chance of DOUBLING+   : {prob_double:5.1f}%")
    print("=" * 60)

    # ----- Plot: sample of paths + histogram of endings -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [2, 1]})

    # Show 200 sample paths so the "fan of futures" is visible.
    time_axis = np.arange(days) / TRADING_DAYS
    ax1.plot(time_axis, paths[:, :200], color="steelblue", alpha=0.06)
    ax1.axhline(start_value, color="black", linestyle="--", linewidth=1, label="Start")
    ax1.plot(time_axis, np.median(paths, axis=1), color="red", linewidth=2, label="Median path")
    ax1.set_xlabel("Years"); ax1.set_ylabel("Portfolio value ($)")
    ax1.set_title(f"{N_SIMS:,} possible futures for ${start_value:,} in {ticker}")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.hist(endings, bins=80, color="steelblue", orientation="horizontal")
    ax2.axhline(start_value, color="black", linestyle="--", linewidth=1)
    ax2.axhline(p50, color="red", linewidth=2)
    ax2.set_xlabel("Frequency"); ax2.set_title("Distribution of endings")
    ax2.set_ylim(ax1.get_ylim())

    fig.tight_layout()
    out = "monte_carlo.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\n  Chart saved to: {out}")

    print(
        "\n  How to read this:\n"
        "  • Each faint line is one randomly-simulated future. Together they\n"
        "    form a 'fan' that widens over time, uncertainty compounds.\n"
        "  • The gap between the 10th and 90th percentile is HUGE. That gap\n"
        "    IS the risk, made visual. The median is not a promise.\n"
        "  • 'Chance of losing money' is a far more honest risk number than a\n"
        "    single expected return. Real financial planning uses exactly this.\n"
        "  • Caveat: this assumes returns are normal and the future resembles\n"
        "    the past. Real markets have fatter tails (crashes are more common\n"
        "    than a normal curve predicts). A model is a tool, not a crystal ball.\n"
    )


if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 10000
    yrs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    run(ticker, start, yrs)
