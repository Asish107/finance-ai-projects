# 📈 Finance + AI — Learn by Building

A hands-on journey through **finance and AI**, built as 8 progressively richer
projects that all use **real, live market data** from Yahoo Finance. Each project
teaches one finance concept and one AI/data concept — and it all comes together
in an interactive web dashboard.

> ⚠️ **Educational only. Not investment advice.** These tools are for learning,
> not for making real money decisions.

---

## What's inside

| # | Project | Finance concept | AI / tech concept |
|---|---------|-----------------|-------------------|
| 1 | [Stock Explorer](01_stock_explorer/) | price, returns, volatility (risk) | pulling live data, pandas |
| 2 | [Portfolio Dashboard](02_portfolio_dashboard/) | diversification, correlation | data viz |
| 3 | [Price Predictor](03_price_predictor/) | efficient market hypothesis | ML: train/test split, baselines |
| 4 | [News Sentiment](04_news_sentiment/) | how information moves prices | NLP sentiment scoring |
| 5 | [AI Analyst](05_ai_analyst/) | reading financials | LLM (DeepSeek) grounded on real data |
| 6 | [S&P 500 Scanner](06_sp500_scanner/) | stock screening, Sharpe ratio | batch data, ranking |
| 7 | [Weekly Recap](07_weekly_recap/) | market breadth, daily moves | LLM-generated newsletter |
| 8 | [**Dashboard**](08_dashboard/) | all of the above | Streamlit web app |

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run any single project
python3 01_stock_explorer/explore.py AAPL
python3 06_sp500_scanner/scan.py

# 3. Or launch the full interactive dashboard
streamlit run 08_dashboard/app.py
```

## Using the AI features (Projects 5, 7, and the dashboard)

The AI features call **DeepSeek** through [OpenRouter](https://openrouter.ai).
Get a free API key, then set it as an environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-your-key-here"
```

The key is **never stored in the code** — it's read from the environment (or
typed into a password box in the dashboard). Never commit your key.

---

## Key lessons this repo demonstrates

- **Returns & volatility** — why % returns matter more than raw price, and how
  volatility *is* the standard measure of risk.
- **Diversification is the only "free lunch"** — combining uncorrelated stocks
  cuts risk without cutting return (Project 2 shows ~48% risk removed).
- **Markets are hard to beat** — a price-only ML model barely beats a coin flip;
  Project 3 makes the efficient market hypothesis tangible, and shows how to
  avoid lookahead bias and fooling yourself with a missing baseline.
- **Text is a signal** — NLP turns headlines into numbers that price history
  alone can't provide.
- **Ground your AI** — the LLM only *narrates* numbers your code computes; it
  never invents figures. That separation prevents hallucinated finance.
- **Real data has artifacts** — stock splits can look like a 50% crash; good
  analysts learn to smell and handle them (see the split guard in Project 7).

## Ideas for next steps

- Backtest a strategy ("what if I'd bought the top return/risk stocks a year ago?")
- Swap generic VADER sentiment for a finance-tuned model (FinBERT)
- Deploy the dashboard to Streamlit Community Cloud for a shareable link

---

Built while learning. Data via [yfinance](https://github.com/ranaroussi/yfinance).
Licensed under [MIT](LICENSE).
