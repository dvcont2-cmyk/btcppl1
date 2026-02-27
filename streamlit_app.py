import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="BTC RSI Dashboard")

st.title("🟠 BTC RSI Dashboard")

@st.cache_data(ttl=300)
def get_btc_data(interval="1w"):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": 100}
    data = requests.get(url, params=params).json()
    
    df = pd.DataFrame(data, columns=[
        'time','open','high','low','close','vol',
        'close_time','qav','trades','tbbav','tbqav','ignore'
    ])
    df['close'] = df['close'].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    return df

# Get data
df_weekly = get_btc_data("1w")
df_daily = get_btc_data("1d")

# Current values
price = df_weekly['close'].iloc[-1]
rsi_weekly = round(df_weekly['RSI'].iloc[-1], 2)
rsi_daily = round(df_daily['RSI'].iloc[-1], 2)

col1, col2, col3 = st.columns(3)
col1.metric("🔸 BTC Price", f"${price:,.0f}")
col2.metric("📊 Weekly RSI", rsi_weekly)
col3.metric("📈 Daily RSI", rsi_daily)

# Signal logic
if rsi_weekly < 40:
    signal = "🟢 **DCA BUY ZONE**"
elif rsi_weekly > 70:
    signal = "🔴 **SELL 25% ZONE**"
else:
    signal = "⚪ **HOLD / WATCH**"

st.markdown(f"### Signal: {signal}")
st.caption("Weekly RSI < 40 = Buy zone, RSI > 70 = Sell zone")

# Weekly RSI chart
fig_weekly = go.Figure()
fig_weekly.add_trace(go.Scatter(x=df_weekly['time'], y=df_weekly['RSI'], 
                               name='RSI (14)', line=dict(color='#F59E0B')))
fig_weekly.add_hline(y=70, line_dash="dash", line_color="#EF4444", 
                    annotation_text="Sell", annotation_position="right")
fig_weekly.add_hline(y=40, line_dash="dash", line_color="#10B981", 
                    annotation_text="Buy", annotation_position="right")
fig_weekly.add_hline(y=30, line_dash="dash", line_color="#059669", 
                    annotation_text="Strong Buy", annotation_position="right")
fig_weekly.update_layout(title="BTC Weekly RSI (14)", height=500, showlegend=False)
st.plotly_chart(fig_weekly, use_container_width=True)

# Quick action buttons
st.markdown("---")
st.subheader("🚀 Quick Actions")
col1, col2, col3 = st.columns(3)
if col1.button("💰 KuCoin BTC"):
    st.balloons()
    st.success("Open KuCoin and buy BTC!")
if col2.button("📱 Koinly"):
    st.success("Connect KuCoin API to Koinly!")
if col3.button("📊 TradingView"):
    st.markdown("[BTC Weekly Chart](https://www.tradingview.com/chart/?symbol=BINANCE:BTCUSDT&interval=W)")

st.caption("Auto-refreshes every 5 minutes | Data from Binance API")
