"""
Project 4: News Sentiment Analyzer — AI reads the news
-------------------------------------------------------
Goal: Use NLP (Natural Language Processing) to turn HEADLINES into NUMBERS,
then see whether the mood of the news lines up with how the stock is moving.

Why this matters in finance:
  In Project 3 we learned price-alone barely predicts price. The reason is
  that prices move on INFORMATION, and information arrives as TEXT — earnings
  reports, headlines, tweets. If we can quantify text, we get a signal that
  isn't already baked into past prices. This is the real edge in modern
  AI-driven finance.

How the AI works here (VADER):
  VADER is a sentiment model. It reads a sentence and returns a "compound"
  score from -1 (very negative) to +1 (very positive). It knows that words
  like "surges", "beats", "record" are positive and "warns", "plunges",
  "lawsuit" are negative, and it understands intensifiers and negation.
  (It's a lightweight, rule-based cousin of the big LLMs — same idea, turning
   language into a number, just smaller and instant.)

Run it:  python3 sentiment.py NVDA
         python3 sentiment.py AAPL
"""

import sys
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def label(score: float) -> str:
    if score >= 0.25:
        return "POSITIVE"
    if score <= -0.25:
        return "NEGATIVE"
    return "neutral "


def run(ticker: str):
    print(f"\nFetching recent news for {ticker}...\n")
    tk = yf.Ticker(ticker)
    news = tk.news or []

    if not news:
        print("No news returned. Try a large, well-covered ticker like AAPL or NVDA.")
        return

    analyzer = SentimentIntensityAnalyzer()

    print("=" * 70)
    print(f"  HEADLINE SENTIMENT  —  {ticker}")
    print("=" * 70)

    scores = []
    for item in news:
        c = item.get("content", item)
        title = (c.get("title") or "").strip()
        summary = (c.get("summary") or c.get("description") or "").strip()
        if not title:
            continue

        # We score title + summary together for a fuller read of the story.
        text = f"{title}. {summary}"
        score = analyzer.polarity_scores(text)["compound"]
        scores.append(score)

        print(f"\n  [{label(score)}]  {score:+.2f}")
        print(f"    {title[:66]}")

    if not scores:
        print("Headlines had no usable text.")
        return

    avg = sum(scores) / len(scores)
    n_pos = sum(1 for s in scores if s >= 0.25)
    n_neg = sum(1 for s in scores if s <= -0.25)

    # How has the price actually moved over the last week? (reality check)
    hist = tk.history(period="5d")["Close"]
    price_move = (hist.iloc[-1] / hist.iloc[0] - 1) * 100 if len(hist) > 1 else float("nan")

    print("\n" + "=" * 70)
    print(f"  Headlines scored : {len(scores)}")
    print(f"  Positive / Negative : {n_pos} / {n_neg}")
    print(f"  AVERAGE SENTIMENT : {avg:+.2f}   ({label(avg).strip()})")
    print(f"  Actual 5-day price move : {price_move:+.1f}%")
    print("=" * 70)

    # Does the mood match the move?
    agree = (avg > 0 and price_move > 0) or (avg < 0 and price_move < 0)
    print(
        f"\n  News mood and recent price move {'AGREE' if agree else 'DISAGREE'}.\n"
        "  Note: sentiment often LEADS or LAGS price and is noisy on small samples.\n"
        "  Traders combine it with many other signals — you've just built one\n"
        "  raw ingredient of an AI trading system.\n"
    )

    print(
        "  What you learned:\n"
        "  • NLP turns unstructured TEXT into a structured NUMBER you can model.\n"
        "  • This is a NEW signal, independent of price history (Project 3's gap).\n"
        "  • VADER is generic; finance-specific models (e.g. FinBERT) and full\n"
        "    LLMs like Claude read nuance far better — that's Project 5.\n"
    )


if __name__ == "__main__":
    run(sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA")
