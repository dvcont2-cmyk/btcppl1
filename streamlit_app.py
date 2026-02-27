import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="BTC RSI Dashboard")
st.title("🟠 BTC RSI Dashboard")

@st.cache_data(ttl=300)
def get_btc_data(interval="1w"):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": interval, "limit": 100}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            'time','open','high','low','close','vol',
            'close_time','qav','trades','tbbav','tbqav','ignore'
        ])
        df['close'] = df['close'].astype(float)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        return df
    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()

df_weekly = get_btc_data("1w")
df_daily = get_btc_data("1d")

if df_weekly.empty or len(df_weekly) < 14:
    st.error("No data loaded. Try refreshing in 1 minute.")
    st.stop()

price = df_weekly['close'].iloc[-1]
rsi_weekly = round(df_weekly['RSI'].iloc[-1], 2)
rsi_daily = round(df_daily['RSI'].iloc[-1], 2) if not df_daily.empty else "N/A"

col1, col2, col3 = st.columns(3)
col1.metric("🔸 BTC Price", f"${price:,.0f}")
col2.metric("📊 Weekly RSI", rsi_weekly)
col3.metric("📈 Daily RSI", rsi_daily)

if rsi_weekly < 40:
    signal = "🟢 **DCA BUY ZONE**"
elif rsi_weekly > 70:
    signal = "🔴 **SELL 25% ZONE**"
else:
    signal = "⚪ **HOLD / WATCH**"

st.markdown(f"### Signal: {signal}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_weekly['time'], y=df_weekly['RSI'],
                         name='Weekly RSI', line=dict(color='#F59E0B')))
fig.add_hline(y=70, line_dash="dash", line_color="red")
fig.add_hline(y=40, line_dash="dash", line_color="green")
fig.add_hline(y=30, line_dash="dash", line_color="lime")
fig.update_layout(title="BTC Weekly RSI (14)", height=500)
st.plotly_chart(fig, use_container_width=True)

st.caption("Auto-refreshes every 5 minutes")
