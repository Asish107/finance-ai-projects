"""
Project 7: Weekly Market Recap — a day-by-day story of the past week
--------------------------------------------------------------------
Goal: Tie everything together. Walk through each trading day of the past week
(Mon -> Fri), compute what actually happened across the whole S&P 500 that day,
then have DeepSeek write a plain-English narrative recap — like a market
newsletter, generated from your own data.

Pipeline (this is the whole course in one script):
  1. DATA  : download the last ~2 weeks of prices for all S&P 500 companies.
  2. FACTS : for each day, compute the index move + biggest winners/losers.
  3. AI    : hand the day-by-day facts to DeepSeek to write the recap.

The facts are REAL and computed by your code; the LLM only narrates them.
That grounding is what keeps it from making things up.

Setup:  export OPENROUTER_API_KEY="sk-or-..."
Run:    python3 recap.py            # full S&P 500 recap of the past week
        python3 recap.py --no-ai    # just the raw facts, no API key needed
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import requests
import yfinance as yf

MODEL = "deepseek/deepseek-chat"
# A single-day move bigger than this is almost never a real price change — it's
# usually a STOCK SPLIT or a bad data tick (e.g. a 2-for-1 split halves the price
# overnight but shareholders lose nothing). We flag these instead of reporting
# them as genuine crashes/spikes. Real market crashes rarely exceed ~25% in a day.
SPLIT_THRESHOLD = 40.0  # percent


def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
    return table["Symbol"].tolist(), dict(zip(table["Symbol"], table["Security"]))


def build_daily_facts(n_days=5):
    """Return a list of per-day fact dicts for the last n trading days."""
    tickers, names = get_sp500()
    print(f"Downloading recent prices for {len(tickers)} companies...\n")
    close = yf.download(tickers, period="1mo", interval="1d",
                        progress=True, auto_adjust=True)["Close"]

    # Daily % change for every stock, every day.
    daily = close.pct_change() * 100
    # Equal-weight "market" move = average across all stocks that day.
    market = daily.mean(axis=1)

    recent_days = daily.index[-n_days:]
    facts = []
    for day in recent_days:
        row = daily.loc[day].dropna()

        # Split guard: set aside implausibly large moves as likely data artifacts
        # so they don't pollute the "top movers" or mislead the AI narrative.
        flagged = row[row.abs() > SPLIT_THRESHOLD]
        clean = row[row.abs() <= SPLIT_THRESHOLD]

        gainers = clean.sort_values(ascending=False).head(5)
        losers = clean.sort_values().head(5)
        facts.append({
            "date": str(day.date()),
            "weekday": day.strftime("%A"),
            "market_move_pct": round(float(market.loc[day]), 2),
            "advancers": int((row > 0).sum()),   # how many stocks rose
            "decliners": int((row < 0).sum()),   # how many fell
            "top_gainers": [(t, names.get(t, t)[:22], round(float(v), 1)) for t, v in gainers.items()],
            "top_losers": [(t, names.get(t, t)[:22], round(float(v), 1)) for t, v in losers.items()],
            # Excluded as probable stock splits / bad data — reported honestly.
            "excluded_as_split_or_artifact": [(t, names.get(t, t)[:22], round(float(v), 1))
                                              for t, v in flagged.items()],
        })
    return facts


def print_facts(facts):
    for d in facts:
        arrow = "UP  " if d["market_move_pct"] >= 0 else "DOWN"
        print("=" * 64)
        print(f"  {d['weekday']}, {d['date']}   MARKET {arrow} {d['market_move_pct']:+.2f}%"
              f"   ({d['advancers']} up / {d['decliners']} down)")
        print("=" * 64)
        print("  Top gainers:")
        for t, name, v in d["top_gainers"]:
            print(f"    {t:<6} {name:<22} {v:+6.1f}%")
        print("  Top losers:")
        for t, name, v in d["top_losers"]:
            print(f"    {t:<6} {name:<22} {v:+6.1f}%")
        if d.get("excluded_as_split_or_artifact"):
            print("  ⚠ Excluded (probable stock split / data artifact, NOT a real move):")
            for t, name, v in d["excluded_as_split_or_artifact"]:
                print(f"    {t:<6} {name:<22} {v:+6.1f}%")
        print()


def narrate(facts):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("\n  (No OPENROUTER_API_KEY set — showing raw facts only.)")
        print('  To get the AI narrative:  export OPENROUTER_API_KEY="sk-or-..."\n')
        return

    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    system = (
        "You are a financial newsletter writer for a beginner learning markets. "
        "You are given REAL day-by-day S&P 500 data. Write an engaging weekly "
        "recap that goes day by day (by weekday), explaining what the market did "
        "each day and calling out notable movers. Rules:\n"
        "- Use ONLY the numbers provided; never invent figures or news events.\n"
        "- If you speculate on WHY (e.g. a sector theme), clearly mark it as a guess.\n"
        "- Explain terms simply. End with a 2-sentence 'big picture' takeaway.\n"
        "- This is education, not investment advice."
    )
    user = f"Here is the day-by-day S&P 500 data for the past week:\n{facts}\n\nWrite the weekly recap."

    print("\nAsking DeepSeek to write the weekly recap...\n")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.5,
    )
    print("=" * 64)
    print("  AI WEEKLY MARKET RECAP")
    print("=" * 64 + "\n")
    print(resp.choices[0].message.content + "\n")


def main():
    no_ai = "--no-ai" in sys.argv
    facts = build_daily_facts(n_days=5)
    print_facts(facts)
    if not no_ai:
        narrate(facts)


if __name__ == "__main__":
    main()
