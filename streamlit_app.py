import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(layout="wide", page_title="Crypto RSI Dashboard")
st.title("🟠 Crypto RSI Dashboard")

COINS = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Solana (SOL)": "solana",
    "XRP": "ripple",
    "BNB": "binancecoin",
}

# --- Sidebar controls ---
st.sidebar.header("⚙️ Settings")
coin_label = st.sidebar.selectbox("Select Coin", list(COINS.keys()))
coin_id = COINS[coin_label]
timeframe = st.sidebar.radio("Timeframe", ["Daily", "Weekly"])
days = 365 if timeframe == "Weekly" else 180

# --- Data fetch ---
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
        df["close"] = df["close"].astype(float)
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        return df
    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()

# --- Alert log ---
if "alerts" not in st.session_state:
    st.session_state.alerts = []

def check_alert(coin, rsi, timeframe):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    if rsi < 30:
        st.session_state.alerts.insert(0, f"{ts} | {coin} ({timeframe}) RSI = {rsi} 🟢 STRONG BUY")
    elif rsi < 40:
        st.session_state.alerts.insert(0, f"{ts} | {coin} ({timeframe}) RSI = {rsi} 🟡 BUY ZONE")
    elif rsi > 70:
        st.session_state.alerts.insert(0, f"{ts} | {coin} ({timeframe}) RSI = {rsi} 🔴 SELL ZONE")
    # Keep last 20
    st.session_state.alerts = st.session_state.alerts[:20]

df = get_data(coin_id, days)

if df.empty or len(df) < 14:
    st.error("No data loaded. Try refreshing.")
    st.stop()

# --- Resample to weekly if needed ---
if timeframe == "Weekly":
    df = df.set_index("time").resample("W").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

price = df["close"].iloc[-1]
rsi_now = round(df["RSI"].iloc[-1], 2)

check_alert(coin_label, rsi_now, timeframe)

# --- Signal ---
if rsi_now < 30:
    signal = "🟢 **STRONG BUY — RSI below 30**"
    signal_color = "green"
elif rsi_now < 40:
    signal = "🟡 **DCA BUY ZONE — RSI below 40**"
    signal_color = "yellow"
elif rsi_now > 70:
    signal = "🔴 **SELL 25% — RSI above 70**"
    signal_color = "red"
else:
    signal = "⚪ **HOLD / WATCH**"
    signal_color = "gray"

# --- Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric(f"💰 {coin_label} Price", f"${price:,.2f}")
col2.metric(f"📊 RSI (14) — {timeframe}", rsi_now)
col3.markdown(f"### Signal\n{signal}")

st.divider()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 RSI Chart", "💵 Price Chart", "🔔 Alert History"])

with tab1:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=df["time"], y=df["RSI"],
        name="RSI", line=dict(color="#F59E0B", width=2)
    ))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sell (70)")
    fig_rsi.add_hline(y=40, line_dash="dash", line_color="green", annotation_text="Buy (40)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="lime", annotation_text="Strong Buy (30)")
    fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.05)
    fig_rsi.add_hrect(y0=30, y1=40, fillcolor="yellow", opacity=0.05)
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.05)
    fig_rsi.update_layout(
        title=f"{coin_label} RSI (14) — {timeframe}",
        height=500, yaxis_range=[0, 100]
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

with tab2:
    fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.7, 0.3],
                               subplot_titles=(f"{coin_label} Price (USD)", "RSI (14)"))
    fig_price.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price"
    ), row=1, col=1)
    fig_price.add_trace(go.Scatter(
        x=df["time"], y=df["RSI"],
        name="RSI", line=dict(color="#F59E0B")
    ), row=2, col=1)
    fig_price.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_price.add_hline(y=30, line_dash="dash", line_color="lime", row=2, col=1)
    fig_price.update_layout(height=700, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_price, use_container_width=True)

with tab3:
    st.subheader("🔔 RSI Alert History (this session)")
    if st.session_state.alerts:
        for alert in st.session_state.alerts:
            st.markdown(f"- {alert}")
    else:
        st.info("No alerts triggered yet. Alerts fire when RSI crosses 30, 40, or 70.")

st.caption("Data: CoinGecko · Auto-refreshes every 5 minutes")
