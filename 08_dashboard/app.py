"""
Project 8: The Dashboard — all 7 projects in one interactive web app
---------------------------------------------------------------------
Streamlit turns a plain Python script into a web UI. Every st.something()
call draws a widget. Run it and it opens in your browser.

Run:  streamlit run app.py

Tabs:
  1. Stock Explorer   (Project 1)  - price, return, volatility + chart
  2. Portfolio        (Project 2)  - diversification & correlation
  3. S&P 500 Scanner  (Project 6)  - screen the whole index
  4. AI Analyst       (Project 5)  - DeepSeek explains a stock (needs API key)
"""

import io
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

TRADING_DAYS = 252
RISK_FREE_RATE = 4.5      # ~risk-free return (T-bills), used in the Sharpe ratio
SPLIT_THRESHOLD = 40.0    # daily moves bigger than this are probable splits/artifacts
st.set_page_config(page_title="Finance + AI Dashboard", layout="wide", page_icon="📈")


# ---------- cached data helpers (so we don't re-download on every click) ----------
@st.cache_data(ttl=1800)
def load_prices(tickers, period="1y"):
    data = yf.download(tickers, period=period, interval="1d",
                       progress=False, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers if isinstance(tickers, str) else tickers[0])
    return data.dropna(how="all")


@st.cache_data(ttl=3600)
def sp500_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    t["Symbol"] = t["Symbol"].str.replace(".", "-", regex=False)
    return t["Symbol"].tolist(), dict(zip(t["Symbol"], t["Security"]))


# =================================  HEADER  ==================================
st.title("📈 Finance + AI Dashboard")
st.caption("Live market data from Yahoo Finance. Educational only — not investment advice.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["🔍 Stock Explorer", "🧺 Portfolio", "📊 S&P 500 Scanner",
     "🤖 AI Analyst", "🗞️ Weekly Recap",
     "🎯 Best Mix", "🎲 Future Simulator"])


# =============================  TAB 1: EXPLORER  =============================
with tab1:
    st.subheader("Single-stock explorer")
    ticker = st.text_input("Ticker (symbol, e.g. AAPL not Apple)", "AAPL",
                           key="explorer").upper().strip()
    if ticker:
        prices = load_prices(ticker)
        s = prices.iloc[:, 0].dropna() if not prices.empty else prices

        if prices.empty or len(s) < 2:
            st.warning(f"Couldn't find market data for '{ticker}'. "
                       "Make sure you're using the stock's ticker symbol "
                       "(for example AAPL for Apple, TSLA for Tesla, MSFT for Microsoft).")
        else:
            ret = s.pct_change().dropna()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${s.iloc[-1]:,.2f}")
            c2.metric("1-Year Return", f"{(s.iloc[-1]/s.iloc[0]-1)*100:+.1f}%")
            c3.metric("Annualized Volatility", f"{ret.std()*np.sqrt(TRADING_DAYS)*100:.1f}%")
            c4.metric("Worst Day", f"{ret.min()*100:.1f}%")

            fig = px.line(s, title=f"{ticker} — 1 year price")
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Volatility = how much daily returns swing = the standard "
                       "measure of risk. Higher return usually means higher volatility.")


# ============================  TAB 2: PORTFOLIO  ============================
with tab2:
    st.subheader("Portfolio & diversification")
    raw = st.text_input("Tickers (space-separated)", "AAPL MSFT KO XOM GLD", key="port")
    tickers = [t.upper() for t in raw.split() if t.strip()]

    if len(tickers) >= 2:
        prices = load_prices(tickers, period="2y")
        prices = prices[[t for t in tickers if t in prices.columns]].dropna()

    if len(tickers) >= 2 and prices.shape[1] >= 2 and len(prices) > 1:
        returns = prices.pct_change().dropna()

        indiv_vol = returns.std() * np.sqrt(TRADING_DAYS)
        weights = np.repeat(1/len(prices.columns), len(prices.columns))
        port_ret = returns.dot(weights)
        port_vol = port_ret.std() * np.sqrt(TRADING_DAYS)
        naive_vol = indiv_vol.mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg of individual risks", f"{naive_vol*100:.1f}%")
        c2.metric("Actual portfolio risk", f"{port_vol*100:.1f}%",
                  delta=f"{-(1-port_vol/naive_vol)*100:.1f}% lower", delta_color="inverse")
        c3.metric("Risk removed by diversifying", f"{(1-port_vol/naive_vol)*100:.1f}%")

        col_a, col_b = st.columns(2)
        with col_a:
            growth = (1+returns).cumprod()*100
            pg = (1+port_ret).cumprod()*100
            fig = go.Figure()
            for t in growth.columns:
                fig.add_trace(go.Scatter(x=growth.index, y=growth[t], name=t, opacity=0.5))
            fig.add_trace(go.Scatter(x=pg.index, y=pg, name="PORTFOLIO",
                                     line=dict(color="black", width=3)))
            fig.update_layout(title="Growth of $100", height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            corr = returns.corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn_r",
                            zmin=-1, zmax=1, title="Correlation (green = diversifiers)")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        st.caption("The portfolio's risk is LOWER than the average of its parts — "
                   "the 'free lunch' of diversification. Green cells = stocks that "
                   "move differently, giving the biggest benefit.")
    else:
        st.info("Enter at least 2 valid ticker symbols (e.g. AAPL MSFT KO). "
                "Use symbols, not company names.")


# ==========================  TAB 3: S&P 500 SCANNER  ========================
with tab3:
    st.subheader("Screen the entire S&P 500")
    st.caption("Downloads ~500 companies. First run takes a minute, then it's cached.")
    if st.button("Run scan", type="primary"):
        with st.spinner("Downloading S&P 500..."):
            tickers, names = sp500_list()
            close = load_prices(tickers, period="1y")
            rows = []
            for t in tickers:
                if t not in close.columns:
                    continue
                srs = close[t].dropna()
                if len(srs) < 200:
                    continue
                r = srs.pct_change().dropna()
                rows.append({
                    "Ticker": t, "Name": names.get(t, "")[:30],
                    "Price": round(float(srs.iloc[-1]), 2),
                    "Return 1Y %": round(float(srs.iloc[-1]/srs.iloc[0]-1)*100, 1),
                    "Volatility %": round(float(r.std()*np.sqrt(TRADING_DAYS))*100, 1),
                })
            df = pd.DataFrame(rows)
            # Sharpe ratio = (return above risk-free rate) / volatility.
            df["Sharpe"] = ((df["Return 1Y %"] - RISK_FREE_RATE)/df["Volatility %"]).round(2)
            st.session_state["scan_df"] = df

    if "scan_df" in st.session_state:
        df = st.session_state["scan_df"]
        st.success(f"Scanned {len(df)} companies (real data, latest close).")
        sort_by = st.selectbox("Sort by", ["Return 1Y %", "Volatility %", "Sharpe"])
        st.dataframe(df.sort_values(sort_by, ascending=False),
                     use_container_width=True, height=430, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False),
                           "sp500_scan.csv", "text/csv")


# ============================  TAB 4: AI ANALYST  ===========================
with tab4:
    st.subheader("AI analyst (DeepSeek via OpenRouter)")
    key = st.text_input("OpenRouter API key", type="password",
                        value=os.environ.get("OPENROUTER_API_KEY", ""),
                        help="Kept only in this session; used to call the model.")
    a_ticker = st.text_input("Ticker", "NVDA", key="ai").upper().strip()
    question = st.text_area("Your question",
                            "How is this stock doing and how risky is it right now?")

    if st.button("Ask the analyst", type="primary"):
        if not key:
            st.error("Enter your OpenRouter API key above.")
        else:
            with st.spinner("Gathering real data + asking DeepSeek..."):
                tk = yf.Ticker(a_ticker)
                s = tk.history(period="1y")["Close"].dropna()
                if len(s) < 2:
                    st.warning(f"Couldn't find data for '{a_ticker}'. Use a ticker "
                               "symbol like AAPL, TSLA, or MSFT.")
                    st.stop()
                r = s.pct_change().dropna()
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                an = SentimentIntensityAnalyzer()
                news = tk.news or []
                sc = [an.polarity_scores((i.get("content", i).get("title") or ""))["compound"]
                      for i in news[:8] if (i.get("content", i).get("title"))]
                facts = {
                    "ticker": a_ticker,
                    "price": round(float(s.iloc[-1]), 2),
                    "return_1y_pct": round(float(s.iloc[-1]/s.iloc[0]-1)*100, 1),
                    "volatility_pct": round(float(r.std()*np.sqrt(TRADING_DAYS))*100, 1),
                    "avg_news_sentiment": round(sum(sc)/len(sc), 2) if sc else None,
                }
                st.info(f"Real snapshot → {facts}")

                from openai import OpenAI
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
                system = ("You are a careful financial analyst for a beginner. Use ONLY "
                          "the provided real data; never invent numbers. Explain terms "
                          "simply. Educational, not investment advice.")
                resp = client.chat.completions.create(
                    model="deepseek/deepseek-chat", temperature=0.4,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": f"DATA: {facts}\n\nQUESTION: {question}"}])
                st.markdown(resp.choices[0].message.content)


# ==========================  TAB 5: WEEKLY RECAP  ===========================
@st.cache_data(ttl=1800)
def weekly_facts(n_days=5):
    """Per-day S&P 500 facts for the last n trading days (Project 7)."""
    tickers, names = sp500_list()
    close = load_prices(tickers, period="1mo")
    daily = close.pct_change() * 100
    market = daily.mean(axis=1)
    out = []
    for day in daily.index[-n_days:]:
        row = daily.loc[day].dropna()
        # Split guard: drop implausible moves (likely splits/bad data) from movers.
        clean = row[row.abs() <= SPLIT_THRESHOLD]
        flagged = row[row.abs() > SPLIT_THRESHOLD]
        out.append({
            "date": str(day.date()),
            "weekday": day.strftime("%A"),
            "market_move_pct": round(float(market.loc[day]), 2),
            "advancers": int((row > 0).sum()),
            "decliners": int((row < 0).sum()),
            "top_gainers": [(t, names.get(t, t)[:22], round(float(v), 1))
                            for t, v in clean.sort_values(ascending=False).head(5).items()],
            "top_losers": [(t, names.get(t, t)[:22], round(float(v), 1))
                           for t, v in clean.sort_values().head(5).items()],
            "excluded_as_split_or_artifact": [(t, names.get(t, t)[:22], round(float(v), 1))
                                              for t, v in flagged.items()],
        })
    return out


with tab5:
    st.subheader("Weekly market recap — day by day")
    st.caption("What the whole S&P 500 did each trading day this past week, "
               "then an AI-written newsletter narrating it.")

    if st.button("Load the week's data", type="primary"):
        with st.spinner("Crunching the past week across ~500 stocks..."):
            st.session_state["week"] = weekly_facts()

    if "week" in st.session_state:
        facts = st.session_state["week"]

        # Market-move-per-day bar chart.
        mv = pd.DataFrame([{"day": f"{d['weekday'][:3]} {d['date'][5:]}",
                            "move": d["market_move_pct"]} for d in facts])
        fig = px.bar(mv, x="day", y="move", title="Daily market move (equal-weight %)",
                     color="move", color_continuous_scale="RdYlGn", range_color=[-1, 1])
        fig.update_layout(height=300, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # Per-day movers.
        for d in facts:
            arrow = "🟢▲" if d["market_move_pct"] >= 0 else "🔴▼"
            with st.expander(f"{arrow} {d['weekday']} {d['date']} — "
                             f"market {d['market_move_pct']:+.2f}%  "
                             f"({d['advancers']} up / {d['decliners']} down)"):
                cga, clo = st.columns(2)
                cga.write("**Top gainers**")
                cga.dataframe(pd.DataFrame(d["top_gainers"],
                              columns=["Ticker", "Name", "%"]), hide_index=True)
                clo.write("**Top losers**")
                clo.dataframe(pd.DataFrame(d["top_losers"],
                              columns=["Ticker", "Name", "%"]), hide_index=True)
                if d.get("excluded_as_split_or_artifact"):
                    st.warning("⚠ Excluded as probable stock split / data artifact "
                               "(not a real move): " +
                               ", ".join(f"{t} ({v:+.0f}%)"
                                         for t, _, v in d["excluded_as_split_or_artifact"]))

        st.divider()
        key2 = st.text_input("OpenRouter API key (for the AI recap)", type="password",
                             value=os.environ.get("OPENROUTER_API_KEY", ""), key="recap_key")
        if st.button("Write the AI recap"):
            if not key2:
                st.error("Enter your OpenRouter API key.")
            else:
                with st.spinner("DeepSeek is writing your weekly newsletter..."):
                    from openai import OpenAI
                    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key2)
                    system = ("You are a financial newsletter writer for a beginner. "
                              "Use ONLY the real day-by-day numbers provided; never invent "
                              "figures or news. Go day by day, call out notable movers, and "
                              "if you guess WHY a sector moved, mark it clearly as a guess. "
                              "End with a 2-sentence big-picture takeaway. Not investment advice.")
                    resp = client.chat.completions.create(
                        model="deepseek/deepseek-chat", temperature=0.5,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": f"Day-by-day S&P 500 data:\n{facts}\n\nWrite the weekly recap."}])
                    st.markdown(resp.choices[0].message.content)


# ==================  TAB 6: BEST MIX (Efficient Frontier)  ==================
with tab6:
    st.subheader("🎯 Best Mix — how should I split my money?")
    st.caption("Tries thousands of random ways to divide your money across these "
               "stocks, and points to the smartest split (most reward for least risk).")
    raw6 = st.text_input("Stocks to mix (space-separated)",
                         "AAPL MSFT KO XOM GLD TLT", key="frontier")
    tks = [t.upper() for t in raw6.split() if t.strip()]

    if st.button("Find the best mix", type="primary") and len(tks) >= 2:
        with st.spinner("Trying 20,000 random splits..."):
            px_ = load_prices(tks, period="3y")
            px_ = px_[[t for t in tks if t in px_.columns]].dropna()
            tks = list(px_.columns)
            if len(tks) < 2 or len(px_) < 2:
                st.warning("Need at least 2 valid ticker symbols (e.g. AAPL MSFT KO). "
                           "Use symbols, not company names.")
            else:
                rets = px_.pct_change().dropna()
                mean_a = rets.mean() * TRADING_DAYS
                cov_a = rets.cov() * TRADING_DAYS
                N = 20000
                res = np.zeros((N, 3)); W = np.zeros((N, len(tks)))
                for i in range(N):
                    w = np.random.random(len(tks)); w /= w.sum(); W[i] = w
                    r = w @ mean_a; v = np.sqrt(w @ cov_a @ w)
                    res[i] = [r, v, (r - RISK_FREE_RATE/100) / v]
                st.session_state["frontier_data"] = (res, W, tks)

    if "frontier_data" in st.session_state:
        res, W, tks = st.session_state["frontier_data"]
        r, v, sh = res[:, 0]*100, res[:, 1]*100, res[:, 2]
        best = sh.argmax(); safe = res[:, 1].argmin()

        c1, c2 = st.columns(2)
        c1.metric("🌟 Best mix return", f"{r[best]:.1f}%", f"risk {v[best]:.1f}%")
        c2.metric("🛡️ Safest mix return", f"{r[safe]:.1f}%", f"risk {v[safe]:.1f}%")

        fig = px.scatter(x=v, y=r, color=sh, color_continuous_scale="viridis",
                         labels={"x": "Risk (volatility %)", "y": "Return %", "color": "Reward/risk"},
                         title="Every dot = one possible mix. Top-left is best.")
        fig.add_scatter(x=[v[best]], y=[r[best]], mode="markers",
                        marker=dict(size=22, color="red", symbol="star"), name="Best mix")
        fig.add_scatter(x=[v[safe]], y=[r[safe]], mode="markers",
                        marker=dict(size=22, color="deepskyblue", symbol="star"), name="Safest mix")
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

        st.write("**🌟 The best mix puts your money here:**")
        best_w = pd.DataFrame({"Stock": tks, "Put this %": (W[best]*100).round(1)}
                              ).sort_values("Put this %", ascending=False)
        st.dataframe(best_w, hide_index=True, use_container_width=True)
        st.caption("Notice the math usually favors calm diversifiers (like gold/bonds) "
                   "that steady the ride — not just the flashiest stocks.")


# ==================  TAB 7: FUTURE SIMULATOR (Monte Carlo)  =================
with tab7:
    st.subheader("🎲 Future Simulator — where could my money end up?")
    st.caption("Rolls the dice on thousands of possible futures for your investment, "
               "so you see the whole RANGE of outcomes, not one fake number.")
    c1, c2, c3 = st.columns(3)
    mc_ticker = c1.text_input("Invest in", "SPY", key="mc").upper().strip()
    mc_amount = c2.number_input("Amount ($)", 1000, 1_000_000, 10000, step=1000)
    mc_years = c3.slider("Years", 1, 30, 10)

    if st.button("Simulate my future", type="primary"):
        with st.spinner("Rolling the dice 5,000 times..."):
            _px = load_prices(mc_ticker, period="5y")
            s = _px.iloc[:, 0].dropna() if not _px.empty else _px
            if _px.empty or len(s) < 2:
                st.warning(f"Couldn't find data for '{mc_ticker}'. Use a ticker "
                           "symbol like SPY, AAPL, or TSLA.")
                st.stop()
            rets = s.pct_change().dropna()
            mu, sigma = rets.mean(), rets.std()
            days = mc_years * TRADING_DAYS
            rng = np.random.default_rng(42)
            paths = mc_amount * np.cumprod(1 + rng.normal(mu, sigma, (days, 5000)), axis=0)
            endings = paths[-1]
            p10, p50, p90 = np.percentile(endings, [10, 50, 90])

            c1, c2, c3 = st.columns(3)
            c1.metric("😟 Unlucky (bottom 10%)", f"${p10:,.0f}")
            c2.metric("😐 Most likely (middle)", f"${p50:,.0f}")
            c3.metric("😄 Lucky (top 10%)", f"${p90:,.0f}")
            c1, c2 = st.columns(2)
            c1.metric("Chance of LOSING money", f"{(endings < mc_amount).mean()*100:.1f}%")
            c2.metric("Chance of DOUBLING+", f"{(endings > 2*mc_amount).mean()*100:.1f}%")

            t_axis = np.arange(days) / TRADING_DAYS
            fig = go.Figure()
            for i in range(0, 300):   # show 300 sample futures
                fig.add_scatter(x=t_axis, y=paths[:, i], mode="lines",
                                line=dict(color="steelblue", width=0.5),
                                opacity=0.08, showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=t_axis, y=np.median(paths, axis=1), mode="lines",
                            line=dict(color="red", width=3), name="Median")
            fig.update_layout(title=f"5,000 possible futures for ${mc_amount:,} in {mc_ticker}",
                              xaxis_title="Years", yaxis_title="Value ($)", height=430)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The 'fan' widens over time — the further out, the more uncertain. "
                       "That spread IS the risk, made visible.")
