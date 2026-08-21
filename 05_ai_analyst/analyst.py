"""
Project 5: AI Financial Analyst — DeepSeek reads real data and explains it
---------------------------------------------------------------------------
Goal: Combine EVERYTHING from Projects 1-4 into one conversational tool.
A large language model (DeepSeek, via OpenRouter) answers plain-English
questions about a stock — but grounded in REAL numbers your code computes,
not made up.

The key architecture idea (this is how pro AI tools work):
  1. TOOLS/DATA layer  : your Python computes hard facts (price, risk, mood).
  2. LLM layer         : the model REASONS over those facts in natural language.
  Keeping them separate stops the AI from hallucinating numbers. The LLM's job
  is judgment and explanation; your code's job is truth.

Setup (one time), get a key at openrouter.ai then in your terminal:
    export OPENROUTER_API_KEY="sk-or-..."

Run it:  python3 analyst.py NVDA "Is this a risky stock right now?"
         python3 analyst.py AAPL "Summarize the mood and how it's doing."
         python3 analyst.py                 # -> interactive chat
"""

import os
import sys
import numpy as np
import yfinance as yf
from openai import OpenAI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL = "deepseek/deepseek-chat"
TRADING_DAYS = 252


def gather_facts(ticker: str) -> dict:
    """Project 1-4 logic condensed into one factual snapshot for the LLM."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1y")["Close"]
    if hist.empty:
        return {"ticker": ticker, "error": "no price data"}

    ret = hist.pct_change().dropna()
    facts = {
        "ticker": ticker,
        "current_price": round(float(hist.iloc[-1]), 2),
        "return_1y_pct": round(float(hist.iloc[-1] / hist.iloc[0] - 1) * 100, 1),
        "return_1m_pct": round(float(hist.iloc[-1] / hist.iloc[-21] - 1) * 100, 1) if len(hist) > 21 else None,
        "annualized_volatility_pct": round(float(ret.std() * np.sqrt(TRADING_DAYS)) * 100, 1),
        "worst_day_pct": round(float(ret.min()) * 100, 1),
        "best_day_pct": round(float(ret.max()) * 100, 1),
    }

    # Sentiment from recent headlines (Project 4).
    news = tk.news or []
    analyzer = SentimentIntensityAnalyzer()
    scores, headlines = [], []
    for item in news[:8]:
        c = item.get("content", item)
        title = (c.get("title") or "").strip()
        if not title:
            continue
        scores.append(analyzer.polarity_scores(title)["compound"])
        headlines.append(title)
    facts["avg_news_sentiment"] = round(sum(scores) / len(scores), 2) if scores else None
    facts["recent_headlines"] = headlines
    return facts


def ask_llm(client: OpenAI, facts: dict, question: str) -> str:
    system = (
        "You are a careful financial analyst assistant for a beginner who is "
        "learning finance and AI. You are given a JSON snapshot of REAL market "
        "data computed from live prices and news. Rules:\n"
        "- Base every number you state ONLY on the provided data. Never invent figures.\n"
        "- Explain finance terms (volatility, return) in plain language as you go.\n"
        "- Be balanced and note uncertainty. This is education, NOT investment advice.\n"
        "- Keep it concise and friendly."
    )
    user = f"DATA SNAPSHOT:\n{facts}\n\nUSER QUESTION: {question}"

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.4,
    )
    return resp.choices[0].message.content


def main():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("\n  Missing API key. Run this first:\n"
              '    export OPENROUTER_API_KEY="sk-or-..."\n'
              "  Get one at https://openrouter.ai/keys\n")
        return

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else input("Ticker: ").upper()
    print(f"\nGathering real data for {ticker} (price, risk, news mood)...")
    facts = gather_facts(ticker)
    if facts.get("error"):
        print(f"  Could not load {ticker}: {facts['error']}")
        return

    print("  Facts computed. Handing them to DeepSeek to interpret.\n")
    print("-" * 68)
    print(f"  SNAPSHOT: {ticker} @ ${facts['current_price']} | "
          f"1y {facts['return_1y_pct']:+}% | vol {facts['annualized_volatility_pct']}% | "
          f"news mood {facts['avg_news_sentiment']}")
    print("-" * 68)

    if len(sys.argv) > 2:
        # One-shot question mode.
        answer = ask_llm(client, facts, " ".join(sys.argv[2:]))
        print("\n" + answer + "\n")
    else:
        # Interactive chat mode.
        print("Ask anything (type 'quit' to exit). e.g. 'Is this risky?'\n")
        while True:
            q = input("You: ").strip()
            if q.lower() in {"quit", "exit", "q", ""}:
                break
            print("\nAnalyst: " + ask_llm(client, facts, q) + "\n")


if __name__ == "__main__":
    main()
