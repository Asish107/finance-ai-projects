"""
Project 3: Your First ML Model — "Will the stock go UP tomorrow?"
------------------------------------------------------------------
Goal: Build a real machine-learning classifier AND learn why markets are
hard to beat. The honest result matters more than a flashy accuracy number.

Machine learning in one sentence:
  Show the model many examples of (INPUTS -> ANSWER), and it learns a rule
  that maps inputs to the answer, hopefully on data it has never seen.

Here:
  INPUTS  (called "features") = a few numbers describing recent price action.
  ANSWER  (called "label")    = did the price go UP the NEXT day? (1 or 0)

Two traps this code is built to avoid:
  * LOOKAHEAD BIAS: never let a feature "peek" at future info. Every feature
    below uses only data available at the close of that day.
  * CHEATING ON THE TEST: we split time-wise — train on the PAST, test on the
    FUTURE. Shuffling would let the model see the future. Never shuffle time series.

Run it:  python3 predict.py SPY
         python3 predict.py AAPL
"""

import sys
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def make_features(prices: pd.Series) -> pd.DataFrame:
    """Turn a price series into a table of features + the label."""
    df = pd.DataFrame(index=prices.index)
    ret = prices.pct_change()

    # --- FEATURES: all use only PAST/CURRENT info ---
    df["return_1d"] = ret                          # yesterday->today move
    df["return_5d"] = prices.pct_change(5)         # 1-week momentum
    df["volatility_10d"] = ret.rolling(10).std()   # recent choppiness
    # Price relative to its own 20-day average (>1 = above trend).
    df["above_avg_20d"] = prices / prices.rolling(20).mean()
    # Momentum: how far above/below the 10-day average.
    df["momentum_10d"] = prices / prices.rolling(10).mean() - 1

    # --- LABEL: did price go UP the NEXT day? (this is what we predict) ---
    # shift(-1) pulls TOMORROW's return back to today's row = the answer key.
    df["target"] = (ret.shift(-1) > 0).astype(int)

    return df.dropna()


def run(ticker: str):
    print(f"\nDownloading 5 years of data for {ticker}...\n")
    raw = yf.download(ticker, period="5y", interval="1d",
                      progress=False, auto_adjust=True)["Close"]
    prices = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw

    df = make_features(prices)
    feature_cols = ["return_1d", "return_5d", "volatility_10d",
                    "above_avg_20d", "momentum_10d"]
    X = df[feature_cols]
    y = df["target"]

    # --- TIME-BASED SPLIT: train on first 80%, test on last 20% (the future) ---
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # --- Train the model ---
    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # --- Predict on the unseen future ---
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # --- The all-important BASELINE ---
    # If a stock goes up ~53% of days, a dumb "always predict UP" scores 53%.
    # Our model only impresses if it clearly beats this.
    baseline = max(y_test.mean(), 1 - y_test.mean())

    print("=" * 56)
    print(f"  {ticker}  —  predicting next-day direction")
    print("=" * 56)
    print(f"  Test period       : {X_test.index[0].date()} -> {X_test.index[-1].date()}")
    print(f"  Days tested       : {len(y_test)}")
    print(f"  Model accuracy    : {acc * 100:5.1f}%")
    print(f"  Dumb baseline     : {baseline * 100:5.1f}%   (always guess the common direction)")
    edge = (acc - baseline) * 100
    print(f"  Edge over baseline: {edge:+5.1f} percentage points")
    print("=" * 56)

    # --- Which features mattered most? ---
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n  Feature importance (what the model leaned on):")
    for name, val in importances.items():
        bar = "#" * int(val * 40)
        print(f"    {name:<16} {val:4.2f}  {bar}")

    print(
        "\n  READ THIS CAREFULLY:\n"
        "  A tiny or negative edge is the CORRECT, HONEST result. Daily price\n"
        "  direction is close to a coin flip — markets are highly efficient,\n"
        "  meaning obvious patterns get traded away. You just experienced the\n"
        "  Efficient Market Hypothesis firsthand, not as a slogan but as data.\n"
        "  Real edge in finance comes from better DATA and longer HORIZONS,\n"
        "  not from a fancier model on price alone. That motivates Project 4.\n"
    )


if __name__ == "__main__":
    run(sys.argv[1].upper() if len(sys.argv) > 1 else "SPY")
