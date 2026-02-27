import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="BTC RSI Dashboard")
st.title("🟠 BTC RSI Dashboard")

@st.cache_data(ttl=300)
def get_btc_data(days=180):
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
        params = {"vs_currency": "usd", "days": str(days)}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['close'] = df['close'].astype(float)
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        return df
    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()

df = get_btc_data(180)

if df.empty or len(df) < 14:
    st.error("No data loaded. Try refreshing in 1 minute.")
    st.stop()

price = df['close'].iloc[-1]
rsi_now = round(df['RSI'].iloc[-1], 2)

col1, col2 = st.columns(2)
col1.metric("🔸 BTC Price", f"${price:,.0f}")
col2.metric("📊 RSI (14)", rsi_now)

if rsi_now < 40:
    signal = "🟢 **DCA BUY ZONE**"
elif rsi_now > 70:
    signal = "🔴 **SELL 25% ZONE**"
else:
    signal = "⚪ **HOLD / WATCH**"

st.markdown(f"### Signal: {signal}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['time'], y=df['RSI'],
                         name='RSI', line=dict(color='#F59E0B')))
fig.add_hline(y=70, line_dash="dash", line_color="red")
fig.add_hline(y=40, line_dash="dash", line_color="green")
fig.add_hline(y=30, line_dash="dash", line_color="lime")
fig.update_layout(title="BTC RSI (14) - 180 Days", height=500)
st.plotly_chart(fig, use_container_width=True)

st.caption("Data from CoinGecko · Auto-refreshes every 5 minutes")
