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
            st.error(f"No data for {interval}")
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

# Get data
df_weekly = get_btc_data("1w")
df_daily = get_btc_data("1d")

if df_weekly.empty or len(df_weekly) < 14:
    st.error("Not enough data. Refresh in 1 minute.")
    st.stop()

# Values
price = df_weekly['close'].iloc[-1]
rsi_weekly = round(df_weekly['RSI'].iloc[-1],
