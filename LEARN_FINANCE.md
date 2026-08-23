# Finance Concepts: A Beginner's Notes

> Written by a beginner, for beginners. Every concept here is one I learned by
> building the projects in this repo, not from a textbook. Each one is tied to a
> real number my own code produced, so it sticks. If you are new to finance and
> want to actually understand it (not just memorize), start here.

These concepts build on each other, so read them in order.

---

## 1. Price vs. Return: the foundation

**Price** is what one share costs. But investors do not really care about price.
They care about **return**, the percent change:

```
return = (price today / price yesterday) - 1
```

**Why it matters:** a $3,000 stock and a $30 stock cannot be compared by price.
But if both rise 2%, that is the same result for your money. Returns strip away
the raw price so everything is comparable.

> Apple went from $230 to $304 over a year: a +32% return.

---

## 2. Volatility is Risk

**Volatility** is how much daily returns swing around (mathematically, the
standard deviation of returns). A calm stock has low volatility. A jumpy stock
has high volatility.

**The key insight:** in finance, "risk" is not vague. It has a number, and that
number is volatility.

> Apple's 25% annualized volatility means a typical yearly swing is about 25%.

**Annualizing (the square root of 252):** daily volatility is tiny, so we scale
it to a yearly figure by multiplying by the square root of 252 (about 252
trading days per year). We use the square root because risk grows with the
square root of time, not linearly, which is a quirk of how random movements
accumulate.

---

## 3. The Risk and Return Tradeoff

Higher returns almost always come bundled with higher risk.

- **Coca-Cola (KO):** low return, low volatility. Boring and safe.
- **Tesla (TSLA):** high potential return, wild volatility.

**There is no free high-return, low-risk stock.** If there were, everyone would
buy it, its price would jump, and the future return would vanish. This tradeoff
is the central bargain of investing.

---

## 4. Correlation and Diversification: the only free lunch

**Correlation** measures whether two stocks move together (+1), oppositely (-1),
or independently (0).

**Diversification** is the magic that follows. Combine stocks that do not move
together, and their random ups and downs partly cancel out. So a portfolio's
risk is lower than the average risk of its parts, and you gave up no return to
get it. That is why it is called the only free lunch in finance.

> Five stocks each carrying 18 to 29% risk (averaging 24.8%) combined into a
> portfolio with just 12.9% risk. Nearly half the risk vanished, for free,
> because gold (GLD) moves differently from tech stocks.

---

## 5. The Efficient Market Hypothesis (EMH)

The idea that all known information is already baked into prices almost
instantly, because millions of people are trading on it. Any obvious pattern
gets traded away before you can profit from it.

> An ML model predicting tomorrow's direction hit 55%, exactly tying a dumb
> "always guess up" baseline. That is a +0.0 edge. Short-term prediction from
> past prices alone is essentially a coin flip, because the market is efficient.

**Beginner trap this teaches:** never celebrate an accuracy number without a
**baseline**. "55% accuracy" is meaningless until you ask "55% versus what?"

---

## 6. Information Moves Prices (and text is data)

If past prices cannot predict future prices, what can? **New information.**
Prices move when something new is learned, and news arrives as text.

> Scoring NVDA headlines gave +0.39 sentiment (positive) while the stock was up
> +3.7%. Mood and price agreed.

This is a signal that is not already in the price history, which is exactly why
professionals hunt for it.

---

## 7. The Sharpe Ratio: risk-adjusted return

Raw returns are misleading (a 3,000% gain with insane risk is not "good"). The
**Sharpe ratio** fixes this:

```
Sharpe = (return - risk-free rate) / volatility
```

It asks: how much reward did I earn per unit of risk, above what I would get
risk-free? The **risk-free rate** (about 4.5%) is what you would earn with zero
risk (government T-bills). You only get credit for beating that.

**Rule of thumb:** above 1 is good, above 2 is excellent.

---

## 8. Max Drawdown

The worst peak-to-trough drop. In plain terms: how much would I have lost if I
bought at the top and panic-sold at the bottom? It captures the emotional pain
of an investment that pure volatility misses.

> A momentum strategy suffered a 14.4% drawdown vs. the market's gentler 8.9%.

---

## 9. Backtesting, Lookahead Bias, and the Benchmark

**Backtesting** is testing a strategy on historical data before risking real
money.

**Lookahead bias** is the cardinal sin: accidentally using future information
when making a past decision. It makes any strategy look brilliant and then fail
live. Avoid it by splitting time into a **formation window** (pick stocks) and a
separate **holding window** (measure results) that never overlap.

**The benchmark (SPY):** always compare your strategy to "just buy the whole
market."

> A "buy last year's winners" strategy earned +15.3% with 25.8% risk, and lost
> to just buying the index (+20% at half the risk). This is why most
> professional fund managers underperform simple index funds.

---

## 10. Data Artifacts: trust, but verify

Real data has traps. A **stock split** (for example 2-for-1) halves the price
overnight, but you own twice the shares, so you lost nothing. Yet a naive
calculation reads it as a 50% crash.

> A "49.6% crash" in Monster Beverage was really a stock split, not a loss.

**The lesson:** a number that looks insane is usually a data quirk, not reality.
Investigate before believing it.

---

## 11. Modern Portfolio Theory and the Efficient Frontier

Diversification lowers risk (concept 4). Modern Portfolio Theory (MPT) asks the
next question: what is the single BEST mix? Every possible split of your money
across N stocks gives one (risk, return) point. Plot thousands of them and the
best ones form a curve called the **efficient frontier**: for each level of
risk, the mix with the highest possible return. Any portfolio below the frontier
is wasting risk.

Two special points on it matter most:

- **Max Sharpe portfolio:** the best reward per unit of risk.
- **Minimum variance portfolio:** the lowest possible risk.

The engine is the **covariance matrix**, which captures not just each stock's
volatility but how every pair moves together. Markowitz's Nobel-winning insight
was that risk is about relationships between assets, not just individual stocks,
so you can mathematically engineer the optimal portfolio instead of guessing.

> Optimizing 6 assets found a max-Sharpe mix of 24.7% return at 12% risk (Sharpe
> 1.69), and it did it by loading up on the diversifiers (gold and Coca-Cola),
> not the flashy tech stocks. The math rediscovered the free lunch and pushed it
> to the optimum.

---

## The whole thing in one sentence

> Return is your reward, volatility is your risk, they come as a pair,
> diversification is the one way to improve the deal for free, markets are
> efficient so beating them is genuinely hard, and you should never trust a
> number (from a model, a strategy, or a data feed) without checking it against
> a baseline and against reality.

That is the conceptual core of an intro quantitative-finance course, learned by
building, not memorizing. Now go run the projects and see it for yourself.

---

*Educational only. Nothing here is investment advice.*
