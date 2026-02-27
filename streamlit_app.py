import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(layout="wide", page_title="Crypto Signal Dashboard")
st.title("🟠 Crypto Signal Dashboard")

COINS = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Solana (SOL)": "solana",
    "XRP": "ripple",
    "BNB": "binancecoin",
}

st.sidebar.header("⚙️ Settings")
coin_label = st.sidebar.selectbox("Select Coin", list(COINS.keys()))
coin_id = COINS[coin_label]
timeframe = st.sidebar.radio("Timeframe", ["Daily", "Weekly"])
days = 365 if timeframe == "Weekly" else 180

@st.cache_data(ttl=300)
def get_data(coin_id, days):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        params = {"vs_currency": "usd", "days": str(days)}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()

def compute_indicators(df):
    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # Stochastic RSI
    stoch = ta.momentum.StochRSIIndicator(df["close"], window=14)
    df["StochRSI_k"] = stoch.stochrsi_k() * 100
    df["StochRSI_d"] = stoch.stochrsi_d() * 100

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_mid"] = bb.bollinger_mavg()

    # EMAs
    df["EMA_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["EMA_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()

    return df

def composite_score(row):
    score = 0
    reasons = []

    # RSI
    if row["RSI"] < 30:
        score += 2
        reasons.append("✅ RSI strongly oversold (<30)")
    elif row["RSI"] < 40:
        score += 1
        reasons.append("✅ RSI oversold (<40)")
    elif row["RSI"] > 70:
        score -= 2
        reasons.append("🔴 RSI overbought (>70)")

    # Stochastic RSI
    if row["StochRSI_k"] < 20:
        score += 2
        reasons.append("✅ Stoch RSI oversold (<20)")
    elif row["StochRSI_k"] > 80:
        score -= 2
        reasons.append("🔴 Stoch RSI overbought (>80)")

    # MACD
    if row["MACD"] > row["MACD_signal"]:
        score += 1
        reasons.append("✅ MACD bullish crossover")
    else:
        score -= 1
        reasons.append("🔴 MACD bearish")

    # Bollinger Bands
    if row["close"] < row["BB_lower"]:
        score += 2
        reasons.append("✅ Price below lower Bollinger Band")
    elif row["close"] > row["BB_upper"]:
        score -= 2
        reasons.append("🔴 Price above upper Bollinger Band")

    # EMA trend
    if pd.notna(row["EMA_200"]):
        if row["close"] > row["EMA_200"]:
            score += 1
            reasons.append("✅ Price above EMA 200 (uptrend)")
        else:
            score -= 1
            reasons.append("🔴 Price below EMA 200 (downtrend)")

    if pd.notna(row["EMA_50"]) and pd.notna(row["EMA_200"]):
        if row["EMA_50"] > row["EMA_200"]:
            score += 1
            reasons.append("✅ EMA 50 > EMA 200 (golden cross zone)")
        else:
            score -= 1
            reasons.append("🔴 EMA 50 < EMA 200 (death cross zone)")

    return score, reasons

def signal_label(score):
    if score >= 6:
        return "🟢 STRONG BUY", "green"
    elif score >= 3:
        return "🟡 DCA BUY ZONE", "yellow"
    elif score >= 0:
        return "⚪ HOLD / WATCH", "gray"
    elif score >= -3:
        return "🟠 CAUTION / REDUCE", "orange"
    else:
        return "🔴 STRONG SELL", "red"

# --- Alert log ---
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# Load data
df = get_data(coin_id, days)
if df.empty or len(df) < 20:
    st.error("Not enough data. Try refreshing.")
    st.stop()

if timeframe == "Weekly":
    df = df.set_index("time").resample("W").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()

df = compute_indicators(df)
df = df.dropna(subset=["RSI"])

latest = df.iloc[-1]
price = latest["close"]
score, reasons = composite_score(latest)
label, color = signal_label(score)

# Log alert
ts = datetime.now().strftime("%Y-%m-%d %H:%M")
st.session_state.alerts.insert(0, f"{ts} | {coin_label} ({timeframe}) | Score: {score}/10 | {label}")
st.session_state.alerts = st.session_state.alerts[:30]

# --- Header metrics ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("💰 Price", f"${price:,.2f}")
col2.metric("📊 RSI (14)", round(latest["RSI"], 1))
col3.metric("⚡ Stoch RSI", round(latest["StochRSI_k"], 1))
col4.metric("📉 MACD Hist", round(latest["MACD_hist"], 2))
col5.metric("🎯 Signal Score", f"{score}/10")

st.markdown(f"## Signal: {label}")

# --- Composite score breakdown ---
with st.expander("📋 Signal Breakdown (click to expand)"):
    for r in reasons:
        st.markdown(f"- {r}")

st.divider()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 RSI + Stoch RSI", "💵 Price + Bollinger + EMA", "📉 MACD", "🔔 Alert History"
])

with tab1:
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.5, 0.5],
                         subplot_titles=("RSI (14)", "Stochastic RSI"))
    fig1.add_trace(go.Scatter(x=df["time"], y=df["RSI"],
                              name="RSI", line=dict(color="#F59E0B", width=2)), row=1, col=1)
    fig1.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig1.add_hline(y=40, line_dash="dash", line_color="green", row=1, col=1)
    fig1.add_hline(y=30, line_dash="dash", line_color="lime", row=1, col=1)
    fig1.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.05, row=1, col=1)
    fig1.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.05, row=1, col=1)

    fig1.add_trace(go.Scatter(x=df["time"], y=df["StochRSI_k"],
                              name="Stoch K", line=dict(color="#60A5FA", width=2)), row=2, col=1)
    fig1.add_trace(go.Scatter(x=df["time"], y=df["StochRSI_d"],
                              name="Stoch D", line=dict(color="#F472B6", width=1, dash="dot")), row=2, col=1)
    fig1.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
    fig1.add_hline(y=20, line_dash="dash", line_color="lime", row=2, col=1)
    fig1.update_layout(height=600)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price"
    ))
    fig2.add_trace(go.Scatter(x=df["time"], y=df["BB_upper"],
                              name="BB Upper", line=dict(color="gray", dash="dot", width=1)))
    fig2.add_trace(go.Scatter(x=df["time"], y=df["BB_lower"],
                              name="BB Lower", line=dict(color="gray", dash="dot", width=1),
                              fill="tonexty", fillcolor="rgba(128,128,128,0.1)"))
    fig2.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"],
                              name="BB Mid", line=dict(color="gray", width=1)))
    fig2.add_trace(go.Scatter(x=df["time"], y=df["EMA_50"],
                              name="EMA 50", line=dict(color="#34D399", width=1.5)))
    fig2.add_trace(go.Scatter(x=df["time"], y=df["EMA_200"],
                              name="EMA 200", line=dict(color="#F87171", width=1.5)))
    fig2.update_layout(
        title=f"{coin_label} Price + Bollinger Bands + EMAs — {timeframe}",
        height=600, xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.5, 0.5],
                         subplot_titles=("Price", "MACD"))
    fig3.add_trace(go.Scatter(x=df["time"], y=df["close"],
                              name="Price", line=dict(color="#F59E0B")), row=1, col=1)
    fig3.add_trace(go.Scatter(x=df["time"], y=df["MACD"],
                              name="MACD", line=dict(color="#60A5FA", width=2)), row=2, col=1)
    fig3.add_trace(go.Scatter(x=df["time"], y=df["MACD_signal"],
                              name="Signal", line=dict(color="#F472B6", width=1.5)), row=2, col=1)
    colors = ["green" if v >= 0 else "red" for v in df["MACD_hist"]]
    fig3.add_trace(go.Bar(x=df["time"], y=df["MACD_hist"],
                          name="Histogram", marker_color=colors), row=2, col=1)
    fig3.update_layout(height=600)
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("🔔 Signal History (this session)")
    if st.session_state.alerts:
        for alert in st.session_state.alerts:
            st.markdown(f"- {alert}")
    else:
        st.info("No alerts yet.")

st.caption("Data: CoinGecko · Auto-refreshes every 5 minutes")
