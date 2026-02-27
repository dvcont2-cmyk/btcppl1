import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(layout="wide", page_title="Crypto Signal Dashboard")

st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 1rem !important;
    font-weight: 600;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
}
[data-testid="stMetric"] {
    padding: 4px 0px !important;
}
.block-container {
    padding-top: 2rem !important;
}
[data-testid="stAppViewBlockContainer"] {
    padding-top: 1.5rem !important;
}
</style>
""", unsafe_allow_html=True)

COINS = {
    "Bitcoin (BTC)": ("bitcoin", "BTC", "https://assets.coingecko.com/coins/images/1/large/bitcoin.png"),
    "Ethereum (ETH)": ("ethereum", "ETH", "https://assets.coingecko.com/coins/images/279/large/ethereum.png"),
    "Solana (SOL)": ("solana", "SOL", "https://assets.coingecko.com/coins/images/4128/large/solana.png"),
    "XRP": ("ripple", "XRP", "https://assets.coingecko.com/coins/images/44/large/xrp-symbol-white-128.png"),
    "BNB": ("binancecoin", "BNB", "https://assets.coingecko.com/coins/images/825/large/bnb-icon2_2x.png"),
}

st.sidebar.header("⚙️ Settings")
coin_label = st.sidebar.selectbox("Select Coin", list(COINS.keys()))
coin_id, coin_ticker, coin_logo = COINS[coin_label]
timeframe = st.sidebar.radio("Timeframe", ["Daily", "Weekly"])
days = 365 if timeframe == "Weekly" else 180

@st.cache_data(ttl=300)
def get_market_data(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        params = {"localization": "false", "tickers": "false", "community_data": "false", "developer_data": "false"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        market = data.get("market_data", {})
        return {
            "price": market.get("current_price", {}).get("usd", 0),
            "change_1h": market.get("price_change_percentage_1h_in_currency", {}).get("usd", None),
            "change_24h": market.get("price_change_percentage_24h", None),
            "change_7d": market.get("price_change_percentage_7d", None),
            "change_30d": market.get("price_change_percentage_30d", None),
            "market_cap": market.get("market_cap", {}).get("usd", None),
            "volume_24h": market.get("total_volume", {}).get("usd", None),
            "ath": market.get("ath", {}).get("usd", None),
            "ath_change": market.get("ath_change_percentage", {}).get("usd", None),
        }
    except Exception as e:
        st.error(f"Market data error: {e}")
        return {}

@st.cache_data(ttl=300)
def get_ohlc_data(coin_id, days):
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
    df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    stoch = ta.momentum.StochRSIIndicator(df["close"], window=14)
    df["StochRSI_k"] = stoch.stochrsi_k() * 100
    df["StochRSI_d"] = stoch.stochrsi_d() * 100
    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_mid"] = bb.bollinger_mavg()
    df["BB_pct"] = bb.bollinger_pband()
    df["EMA_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["EMA_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["ADX"] = adx.adx()
    df["ADX_pos"] = adx.adx_pos()
    df["ADX_neg"] = adx.adx_neg()
    df["CCI"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
    df["WilliamsR"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=14).williams_r()
    df["ROC"] = ta.momentum.ROCIndicator(df["close"], window=12).roc()
    return df

def composite_score(row):
    indicators = []
    close = row["close"]

    rsi = row["RSI"]
    if pd.isna(rsi):
        indicators.append(("RSI (14)", 0, "⚪", "Neutral — not enough data"))
    elif rsi < 30:
        indicators.append(("RSI (14)", 1, "✅", f"Strongly oversold at {rsi:.1f} — Strong Buy signal (<30)"))
    elif rsi < 40:
        indicators.append(("RSI (14)", 1, "✅", f"Oversold at {rsi:.1f} — Buy zone (<40)"))
    elif rsi > 70:
        indicators.append(("RSI (14)", -1, "🔴", f"Overbought at {rsi:.1f} — Sell signal (>70)"))
    else:
        indicators.append(("RSI (14)", 0, "⚪", f"Neutral at {rsi:.1f} — No clear signal (40–70)"))

    k = row["StochRSI_k"]
    if pd.isna(k):
        indicators.append(("Stoch RSI", 0, "⚪", "Neutral — not enough data"))
    elif k < 20:
        indicators.append(("Stoch RSI", 1, "✅", f"Oversold at {k:.1f} — reversal up likely (<20)"))
    elif k > 80:
        indicators.append(("Stoch RSI", -1, "🔴", f"Overbought at {k:.1f} — reversal down likely (>80)"))
    else:
        indicators.append(("Stoch RSI", 0, "⚪", f"Neutral at {k:.1f} — between 20 and 80"))

    macd_val, macd_sig = row["MACD"], row["MACD_signal"]
    if pd.isna(macd_val) or pd.isna(macd_sig):
        indicators.append(("MACD", 0, "⚪", "Neutral — not enough data"))
    elif macd_val > macd_sig:
        indicators.append(("MACD", 1, "✅", f"Bullish — MACD ({macd_val:.2f}) above signal ({macd_sig:.2f})"))
    else:
        indicators.append(("MACD", -1, "🔴", f"Bearish — MACD ({macd_val:.2f}) below signal ({macd_sig:.2f})"))

    bb_pct, bb_lower, bb_upper = row["BB_pct"], row["BB_lower"], row["BB_upper"]
    if pd.isna(bb_pct):
        indicators.append(("Bollinger Bands", 0, "⚪", "Neutral — not enough data"))
    elif close < bb_lower:
        indicators.append(("Bollinger Bands", 1, "✅", f"Price (${close:,.2f}) below lower band — oversold Buy signal"))
    elif close > bb_upper:
        indicators.append(("Bollinger Bands", -1, "🔴", f"Price (${close:,.2f}) above upper band — overbought Sell signal"))
    elif bb_pct < 0.2:
        indicators.append(("Bollinger Bands", 1, "✅", f"Price near lower band ({bb_pct*100:.0f}% of band) — leaning oversold"))
    elif bb_pct > 0.8:
        indicators.append(("Bollinger Bands", -1, "🔴", f"Price near upper band ({bb_pct*100:.0f}% of band) — leaning overbought"))
    else:
        indicators.append(("Bollinger Bands", 0, "⚪", f"Price mid-band ({bb_pct*100:.0f}% of band) — neutral"))

    ema200 = row["EMA_200"]
    if pd.isna(ema200):
        indicators.append(("EMA 200 Trend", 0, "⚪", "Neutral — not enough data for 200-period EMA"))
    elif close > ema200:
        indicators.append(("EMA 200 Trend", 1, "✅", f"Price (${close:,.2f}) above EMA 200 (${ema200:,.2f}) — long-term uptrend"))
    else:
        indicators.append(("EMA 200 Trend", -1, "🔴", f"Price (${close:,.2f}) below EMA 200 (${ema200:,.2f}) — long-term downtrend"))

    ema50 = row["EMA_50"]
    if pd.isna(ema50) or pd.isna(ema200):
        indicators.append(("EMA 50/200 Cross", 0, "⚪", "Neutral — not enough data"))
    elif ema50 > ema200:
        indicators.append(("EMA 50/200 Cross", 1, "✅", "Golden Cross — EMA 50 above EMA 200 — bullish long-term"))
    else:
        indicators.append(("EMA 50/200 Cross", -1, "🔴", "Death Cross — EMA 50 below EMA 200 — bearish long-term"))

    adx, adx_pos, adx_neg = row["ADX"], row["ADX_pos"], row["ADX_neg"]
    if pd.isna(adx):
        indicators.append(("ADX Strength", 0, "⚪", "Neutral — not enough data"))
    elif adx < 20:
        indicators.append(("ADX Strength", 0, "⚪", f"ADX {adx:.1f} — weak trend, signals less reliable"))
    elif adx_pos > adx_neg:
        indicators.append(("ADX Strength", 1, "✅", f"ADX {adx:.1f} — strong trend, bullish direction (+DI > -DI)"))
    else:
        indicators.append(("ADX Strength", -1, "🔴", f"ADX {adx:.1f} — strong trend, bearish direction (-DI > +DI)"))

    cci = row["CCI"]
    if pd.isna(cci):
        indicators.append(("CCI (20)", 0, "⚪", "Neutral — not enough data"))
    elif cci < -100:
        indicators.append(("CCI (20)", 1, "✅", f"CCI {cci:.0f} — oversold below -100, potential Buy signal"))
    elif cci > 100:
        indicators.append(("CCI (20)", -1, "🔴", f"CCI {cci:.0f} — overbought above +100, potential Sell signal"))
    else:
        indicators.append(("CCI (20)", 0, "⚪", f"CCI {cci:.0f} — neutral between -100 and +100"))

    wr = row["WilliamsR"]
    if pd.isna(wr):
        indicators.append(("Williams %R", 0, "⚪", "Neutral — not enough data"))
    elif wr < -80:
        indicators.append(("Williams %R", 1, "✅", f"Williams %R {wr:.1f} — oversold below -80, Buy zone"))
    elif wr > -20:
        indicators.append(("Williams %R", -1, "🔴", f"Williams %R {wr:.1f} — overbought above -20, Sell zone"))
    else:
        indicators.append(("Williams %R", 0, "⚪", f"Williams %R {wr:.1f} — neutral between -80 and -20"))

    roc = row["ROC"]
    if pd.isna(roc):
        indicators.append(("ROC (12)", 0, "⚪", "Neutral — not enough data"))
    elif roc > 5:
        indicators.append(("ROC (12)", 1, "✅", f"ROC {roc:.1f}% — strong positive momentum"))
    elif roc < -5:
        indicators.append(("ROC (12)", -1, "🔴", f"ROC {roc:.1f}% — strong negative momentum"))
    else:
        indicators.append(("ROC (12)", 0, "⚪", f"ROC {roc:.1f}% — momentum flat (-5% to +5%)"))

    bullish_count = sum(1 for _, s, _, _ in indicators if s > 0)
    bearish_count = sum(1 for _, s, _, _ in indicators if s < 0)
    net_score = bullish_count - bearish_count
    return net_score, bullish_count, bearish_count, indicators

def signal_label(score):
    if score >= 5:
        return "🟢 STRONG BUY"
    elif score >= 2:
        return "🟡 DCA BUY ZONE"
    elif score >= -1:
        return "⚪ HOLD / WATCH"
    elif score >= -3:
        return "🟠 CAUTION / REDUCE"
    else:
        return "🔴 STRONG SELL"

def delta_metric(col, label, val):
    if val is None:
        col.metric(label, "N/A")
    else:
        col.metric(label, f"{val:+.2f}%", delta=f"{val:.2f}%")

if "alerts" not in st.session_state:
    st.session_state.alerts = []

market = get_market_data(coin_id)
df = get_ohlc_data(coin_id, days)

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
price = market.get("price", latest["close"])
score, bullish_count, bearish_count, indicators = composite_score(latest)
neutral_count = 10 - bullish_count - bearish_count
label = signal_label(score)

ts = datetime.now().strftime("%Y-%m-%d %H:%M")
st.session_state.alerts.insert(0, f"{ts} | {coin_label} ({timeframe}) | {bullish_count}/10 bullish | {label}")
st.session_state.alerts = st.session_state.alerts[:30]

# --- Coin header ---
logo_col, title_col = st.columns([1, 11])
with logo_col:
    st.image(coin_logo, width=48)
with title_col:
    st.markdown(f"## {coin_ticker} · ${price:,.2f}  &nbsp; {label}")

st.divider()

# --- Top metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 RSI (14)", round(latest["RSI"], 1) if pd.notna(latest["RSI"]) else "N/A")
col2.metric("⚡ Stoch RSI", round(latest["StochRSI_k"], 1) if pd.notna(latest["StochRSI_k"]) else "N/A")
col3.metric("📉 MACD Hist", round(latest["MACD_hist"], 4) if pd.notna(latest["MACD_hist"]) else "N/A")
col4.metric("🎯 Signal Score", f"{bullish_count}/10 bullish")

# --- Price Performance ---
st.subheader("📊 Price Performance")
p1, p2, p3, p4, p5 = st.columns(5)
delta_metric(p1, "1 Hour", market.get("change_1h"))
delta_metric(p2, "24 Hours", market.get("change_24h"))
delta_metric(p3, "7 Days", market.get("change_7d"))
delta_metric(p4, "30 Days", market.get("change_30d"))
ath = market.get("ath")
ath_change = market.get("ath_change")
if ath:
    p5.metric("📈 ATH", f"${ath:,.0f}", delta=f"{ath_change:.1f}% from ATH" if ath_change else None)

v1, v2 = st.columns(2)
vol = market.get("volume_24h")
mcap = market.get("market_cap")
if vol:
    v1.metric("💹 24h Volume", f"${vol/1e9:.2f}B")
if mcap:
    v2.metric("🏦 Market Cap", f"${mcap/1e9:.1f}B")

st.divider()

# --- Signal breakdown ---
with st.expander("📋 All 10 Indicators — Signal Breakdown (click to expand)", expanded=True):
    bullish_list = [(n, d) for n, s, e, d in indicators if s > 0]
    neutral_list = [(n, d) for n, s, e, d in indicators if s == 0]
    bearish_list = [(n, d) for n, s, e, d in indicators if s < 0]
    col_b, col_n, col_r = st.columns(3)
    with col_b:
        st.markdown(f"**✅ Bullish ({bullish_count})**")
        for name, desc in bullish_list:
            st.markdown(f"- **{name}**: {desc}")
    with col_n:
        st.markdown(f"**⚪ Neutral ({neutral_count})**")
        for name, desc in neutral_list:
            st.markdown(f"- **{name}**: {desc}")
    with col_r:
        st.markdown(f"**🔴 Bearish ({bearish_count})**")
        for name, desc in bearish_list:
            st.markdown(f"- **{name}**: {desc}")

st.divider()

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 RSI + Stoch RSI", "💵 Price + Bollinger + EMA", "📉 MACD", "📡 ADX + CCI + WilliamsR + ROC", "🔔 Alert History"
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
    st.caption("📊 RSI: Measures momentum 0–100. Below 30 = strongly oversold (🟢 Strong Buy); below 40 = buy zone; above 70 = overbought (🔴 Sell).")
    st.caption("⚡ Stochastic RSI: Faster/more sensitive than RSI. K line below 20 = potential reversal up (🟢 Buy); above 80 = potential reversal down (🔴 Sell).")

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
        title=f"{coin_ticker} Price + Bollinger Bands + EMAs — {timeframe}",
        height=600, xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("📉 Bollinger Bands (gray): Price below lower band = oversold (🟢 Buy); above upper band = overbought (🔴 Sell). Band width shows volatility — wider = more volatile.")
    st.caption("📈 EMA 50 (green) / EMA 200 (red): Golden Cross = EMA 50 crosses above EMA 200 (🟢 bullish). Death Cross = EMA 50 crosses below EMA 200 (🔴 bearish). Price above EMA 200 = long-term uptrend.")

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
    colors = ["green" if v >= 0 else "red" for v in df["MACD_hist"].fillna(0)]
    fig3.add_trace(go.Bar(x=df["time"], y=df["MACD_hist"],
                          name="Histogram", marker_color=colors), row=2, col=1)
    fig3.update_layout(height=600)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("📉 MACD (blue line) vs Signal (pink line): Blue crossing above pink = bullish momentum (🟢 Buy). Blue crossing below pink = bearish momentum (🔴 Sell).")
    st.caption("📊 Histogram bars show the gap between MACD and signal line. Growing green bars = strengthening upward momentum. Growing red bars = strengthening downward momentum.")

with tab4:
    fig_adx = go.Figure()
    fig_adx.add_trace(go.Scatter(x=df["time"], y=df["ADX"], name="ADX", line=dict(color="#A78BFA", width=2)))
    fig_adx.add_trace(go.Scatter(x=df["time"], y=df["ADX_pos"], name="+DI", line=dict(color="#34D399", width=1.5)))
    fig_adx.add_trace(go.Scatter(x=df["time"], y=df["ADX_neg"], name="-DI", line=dict(color="#F87171", width=1.5)))
    fig_adx.add_hline(y=20, line_dash="dash", line_color="gray")
    fig_adx.add_hline(y=40, line_dash="dash", line_color="white")
    fig_adx.update_layout(title="ADX — Trend Strength", height=350)
    st.plotly_chart(fig_adx, use_container_width=True)
    st.caption("📡 ADX (purple): Measures trend strength, not direction. Above 20 = trend forming; above 40 = strong trend. Green +DI above red -DI = bullish direction (🟢). Red -DI above green +DI = bearish direction (🔴). Below 20 = choppy/no trend — other signals less reliable.")

    fig_cci = go.Figure()
    fig_cci.add_trace(go.Scatter(x=df["time"], y=df["CCI"], name="CCI", line=dict(color="#F59E0B", width=2)))
    fig_cci.add_hline(y=100, line_dash="dash", line_color="red")
    fig_cci.add_hline(y=-100, line_dash="dash", line_color="lime")
    fig_cci.add_hrect(y0=-300, y1=-100, fillcolor="green", opacity=0.05)
    fig_cci.add_hrect(y0=100, y1=300, fillcolor="red", opacity=0.05)
    fig_cci.update_layout(title="CCI — Commodity Channel Index (20)", height=350)
    st.plotly_chart(fig_cci, use_container_width=True)
    st.caption("📊 CCI: Measures how far price is from its average. Below -100 = oversold (🟢 Buy signal). Above +100 = overbought (🔴 Sell signal). Between -100 and +100 = neutral, no clear signal.")

    fig_wr = go.Figure()
    fig_wr.add_trace(go.Scatter(x=df["time"], y=df["WilliamsR"], name="Williams %R", line=dict(color="#60A5FA", width=2)))
    fig_wr.add_hline(y=-20, line_dash="dash", line_color="red")
    fig_wr.add_hline(y=-80, line_dash="dash", line_color="lime")
    fig_wr.add_hrect(y0=-100, y1=-80, fillcolor="green", opacity=0.05)
    fig_wr.add_hrect(y0=-20, y1=0, fillcolor="red", opacity=0.05)
    fig_wr.update_layout(title="Williams %R (14)", height=350)
    st.plotly_chart(fig_wr, use_container_width=True)
    st.caption("📉 Williams %R: Ranges from -100 to 0. Below -80 = oversold, price likely to bounce up (🟢 Buy zone). Above -20 = overbought, price likely to pull back (🔴 Sell zone). Mid-range = neutral.")

    colors_roc = ["green" if v >= 0 else "red" for v in df["ROC"].fillna(0)]
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Bar(x=df["time"], y=df["ROC"], name="ROC", marker_color=colors_roc))
    fig_roc.add_hline(y=5, line_dash="dash", line_color="lime")
    fig_roc.add_hline(y=-5, line_dash="dash", line_color="red")
    fig_roc.add_hline(y=0, line_color="gray")
    fig_roc.update_layout(title="ROC — Rate of Change (12)", height=350)
    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption("📈 ROC: Measures % price change over the last 12 periods. Above +5% = strong positive momentum (🟢 bullish). Below -5% = strong negative momentum (🔴 bearish). Near 0 = momentum stalling, potential reversal point.")

with tab5:
    st.subheader("🔔 Signal History (this session)")
    if st.session_state.alerts:
        for alert in st.session_state.alerts:
            st.markdown(f"- {alert}")
    else:
        st.info("No alerts yet.")
    st.caption("🔔 Logged each refresh. Shows bullish count out of 10 indicators and overall signal. Resets on app restart.")

st.caption("Data: CoinGecko · Auto-refreshes every 5 minutes")
