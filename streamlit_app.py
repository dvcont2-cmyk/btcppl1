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
    "DOT": "dot",
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
    df["BB_pct"] = bb.bollinger_pband()  # 0=lower band, 1=upper band

    # EMAs
    df["EMA_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["EMA_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()

    # ADX - trend strength
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["ADX"] = adx.adx()
    df["ADX_pos"] = adx.adx_pos()
    df["ADX_neg"] = adx.adx_neg()

    # CCI - Commodity Channel Index
    df["CCI"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()

    # Williams %R
    df["WilliamsR"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=14).williams_r()

    # ROC - Rate of Change
    df["ROC"] = ta.momentum.ROCIndicator(df["close"], window=12).roc()

    return df

def composite_score(row):
    indicators = []

    # 1. RSI
    rsi = row["RSI"]
    if pd.isna(rsi):
        indicators.append(("RSI (14)", 0, "⚪", "Neutral — not enough data"))
    elif rsi < 30:
        indicators.append(("RSI (14)", 2, "✅", f"Strongly oversold at {rsi:.1f} — Strong Buy signal (<30)"))
    elif rsi < 40:
        indicators.append(("RSI (14)", 1, "✅", f"Oversold at {rsi:.1f} — Buy zone (<40)"))
    elif rsi > 70:
        indicators.append(("RSI (14)", -2, "🔴", f"Overbought at {rsi:.1f} — Sell signal (>70)"))
    else:
        indicators.append(("RSI (14)", 0, "⚪", f"Neutral at {rsi:.1f} — No clear signal (40–70)"))

    # 2. Stochastic RSI
    k = row["StochRSI_k"]
    if pd.isna(k):
        indicators.append(("Stoch RSI", 0, "⚪", "Neutral — not enough data"))
    elif k < 20:
        indicators.append(("Stoch RSI", 2, "✅", f"Oversold at {k:.1f} — reversal up likely (<20)"))
    elif k > 80:
        indicators.append(("Stoch RSI", -2, "🔴", f"Overbought at {k:.1f} — reversal down likely (>80)"))
    else:
        indicators.append(("Stoch RSI", 0, "⚪", f"Neutral at {k:.1f} — between 20 and 80"))

    # 3. MACD
    macd_val = row["MACD"]
    macd_sig = row["MACD_signal"]
    macd_hist = row["MACD_hist"]
    if pd.isna(macd_val) or pd.isna(macd_sig):
        indicators.append(("MACD", 0, "⚪", "Neutral — not enough data"))
    elif macd_val > macd_sig:
        indicators.append(("MACD", 1, "✅", f"Bullish — MACD ({macd_val:.1f}) above signal ({macd_sig:.1f})"))
    else:
        indicators.append(("MACD", -1, "🔴", f"Bearish — MACD ({macd_val:.1f}) below signal ({macd_sig:.1f})"))

    # 4. Bollinger Bands
    bb_pct = row["BB_pct"]
    close = row["close"]
    bb_lower = row["BB_lower"]
    bb_upper = row["BB_upper"]
    if pd.isna(bb_pct):
        indicators.append(("Bollinger Bands", 0, "⚪", "Neutral — not enough data"))
    elif close < bb_lower:
        indicators.append(("Bollinger Bands", 2, "✅", f"Price (${close:,.0f}) below lower band (${bb_lower:,.0f}) — oversold Buy signal"))
    elif close > bb_upper:
        indicators.append(("Bollinger Bands", -2, "🔴", f"Price (${close:,.0f}) above upper band (${bb_upper:,.0f}) — overbought Sell signal"))
    elif bb_pct < 0.2:
        indicators.append(("Bollinger Bands", 1, "✅", f"Price near lower band ({bb_pct*100:.0f}% of band) — leaning oversold"))
    elif bb_pct > 0.8:
        indicators.append(("Bollinger Bands", -1, "🔴", f"Price near upper band ({bb_pct*100:.0f}% of band) — leaning overbought"))
    else:
        indicators.append(("Bollinger Bands", 0, "⚪", f"Price mid-band ({bb_pct*100:.0f}% of band) — neutral"))

    # 5. EMA 200 trend
    ema200 = row["EMA_200"]
    if pd.isna(ema200):
        indicators.append(("EMA 200 Trend", 0, "⚪", "Neutral — not enough data for 200-period EMA"))
    elif close > ema200:
        indicators.append(("EMA 200 Trend", 1, "✅", f"Price (${close:,.0f}) above EMA 200 (${ema200:,.0f}) — long-term uptrend"))
    else:
        indicators.append(("EMA 200 Trend", -1, "🔴", f"Price (${close:,.0f}) below EMA 200 (${ema200:,.0f}) — long-term downtrend"))

    # 6. EMA 50/200 cross
    ema50 = row["EMA_50"]
    if pd.isna(ema50) or pd.isna(ema200):
        indicators.append(("EMA 50/200 Cross", 0, "⚪", "Neutral — not enough data"))
    elif ema50 > ema200:
        indicators.append(("EMA 50/200 Cross", 1, "✅", f"Golden Cross — EMA 50 (${ema50:,.0f}) above EMA 200 (${ema200:,.0f}) — bullish"))
    else:
        indicators.append(("EMA 50/200 Cross", -1, "🔴", f"Death Cross — EMA 50 (${ema50:,.0f}) below EMA 200 (${ema200:,.0f}) — bearish"))

    # 7. ADX trend strength
    adx = row["ADX"]
    adx_pos = row["ADX_pos"]
    adx_neg = row["ADX_neg"]
    if pd.isna(adx):
        indicators.append(("ADX Strength", 0, "⚪", "Neutral — not enough data"))
    elif adx < 20:
        indicators.append(("ADX Strength", 0, "⚪", f"ADX {adx:.1f} — weak/no trend, signals less reliable"))
    elif adx_pos > adx_neg:
        indicators.append(("ADX Strength", 1, "✅", f"ADX {adx:.1f} — strong trend with bullish direction (+DI > -DI)"))
    else:
        indicators.append(("ADX Strength", -1, "🔴", f"ADX {adx:.1f} — strong trend with bearish direction (-DI > +DI)"))

    # 8. CCI
    cci = row["CCI"]
    if pd.isna(cci):
        indicators.append(("CCI (20)", 0, "⚪", "Neutral — not enough data"))
    elif cci < -100:
        indicators.append(("CCI (20)", 1, "✅", f"CCI {cci:.0f} — oversold below -100, potential Buy signal"))
    elif cci > 100:
        indicators.append(("CCI (20)", -1, "🔴", f"CCI {cci:.0f} — overbought above +100, potential Sell signal"))
    else:
        indicators.append(("CCI (20)", 0, "⚪", f"CCI {cci:.0f} — neutral between -100 and +100"))

    # 9. Williams %R
    wr = row["WilliamsR"]
    if pd.isna(wr):
        indicators.append(("Williams %R", 0, "⚪", "Neutral — not enough data"))
    elif wr < -80:
        indicators.append(("Williams %R", 1, "✅", f"Williams %R {wr:.1f} — oversold below -80, Buy zone"))
    elif wr > -20:
        indicators.append(("Williams %R", -1, "🔴", f"Williams %R {wr:.1f} — overbought above -20, Sell zone"))
    else:
        indicators.append(("Williams %R", 0, "⚪", f"Williams %R {wr:.1f} — neutral between -80 and -20"))

    # 10. ROC
    roc = row["ROC"]
    if pd.isna(roc):
        indicators.append(("ROC (12)", 0, "⚪", "Neutral — not enough data"))
    elif roc > 5:
        indicators.append(("ROC (12)", 1, "✅", f"ROC {roc:.1f}% — strong positive momentum"))
    elif roc < -5:
        indicators.append(("ROC (12)", -1, "🔴", f"ROC {roc:.1f}% — strong negative momentum"))
    else:
        indicators.append(("ROC (12)", 0, "⚪", f"ROC {roc:.1f}% — momentum is flat (-5% to +5%)"))

    total_score = sum(i[1] for i in indicators)
    return total_score, indicators

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

if "alerts" not in st.session_state:
    st.session_state.alerts = []

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
score, indicators = composite_score(latest)
label, color = signal_label(score)

ts = datetime.now().strftime("%Y-%m-%d %H:%M")
st.session_state.alerts.insert(0, f"{ts} | {coin_label} ({timeframe}) | Score: {score}/10 | {label}")
st.session_state.alerts = st.session_state.alerts[:30]

# --- Header metrics ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("💰 Price", f"${price:,.2f}")
col2.metric("📊 RSI (14)", round(latest["RSI"], 1) if pd.notna(latest["RSI"]) else "N/A")
col3.metric("⚡ Stoch RSI", round(latest["StochRSI_k"], 1) if pd.notna(latest["StochRSI_k"]) else "N/A")
col4.metric("📉 MACD Hist", round(latest["MACD_hist"], 2) if pd.notna(latest["MACD_hist"]) else "N/A")
col5.metric("🎯 Signal Score", f"{score}/10")

st.markdown(f"## Signal: {label}")

# --- Full 10-indicator breakdown ---
with st.expander("📋 All 10 Indicators — Signal Breakdown (click to expand)", expanded=True):
    bullish = [(n, d) for n, s, e, d in indicators if s > 0]
    neutral = [(n, d) for n, s, e, d in indicators if s == 0]
    bearish = [(n, d) for n, s, e, d in indicators if s < 0]

    col_b, col_n, col_r = st.columns(3)
    with col_b:
        st.markdown(f"**✅ Bullish ({len(bullish)})**")
        for name, desc in bullish:
            st.markdown(f"- **{name}**: {desc}")
    with col_n:
        st.markdown(f"**⚪ Neutral ({len(neutral)})**")
        for name, desc in neutral:
            st.markdown(f"- **{name}**: {desc}")
    with col_r:
        st.markdown(f"**🔴 Bearish ({len(bearish)})**")
        for name, desc in bearish:
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
    st.caption(
        "📊 RSI (Relative Strength Index): Measures momentum on a 0–100 scale. "
        "Below 30 = strongly oversold (🟢 Strong Buy), below 40 = buy zone, above 70 = overbought (🔴 Sell). "
        "Stochastic RSI is a faster, more sensitive version — K line below 20 signals a potential reversal up; above 80 signals a potential reversal down."
    )

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
    st.caption(
        "📉 Bollinger Bands: The shaded band shows the normal price range (20-period MA ± 2 std deviations). "
        "Price below the lower band = oversold (🟢 Buy); above the upper band = overbought (🔴 Sell). "
        "EMA 50 (green) and EMA 200 (red): Golden Cross (EMA 50 > EMA 200) = bullish; Death Cross = bearish. "
        "Price above EMA 200 confirms a long-term uptrend."
    )

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
    st.caption(
        "📈 MACD: Tracks momentum using the difference between 12 and 26-period EMAs. "
        "Blue MACD line crossing above the pink signal line = bullish (🟢 Buy); crossing below = bearish (🔴 Sell). "
        "Green histogram bars = growing upward momentum; red bars = growing downward momentum."
    )

with tab4:
    fig4 = make_subplots(rows=2, cols=2, shared_xaxes=False,
                         subplot_titles=("ADX (Trend Strength)", "CCI (20)", "Williams %R (14)", "ROC (12)"))

    fig4.add_trace(go.Scatter(x=df["time"], y=df["ADX"],
                              name="ADX", line=dict(color="#A78BFA", width=2)), row=1, col=1)
    fig4.add_trace(go.Scatter(x=df["time"], y=df["ADX_pos"],
                              name="+DI", line=dict(color="#34D399", width=1)), row=1, col=1)
    fig4.add_trace(go.Scatter(x=df["time"], y=df["ADX_neg"],
                              name="-DI", line=dict(color="#F87171", width=1)), row=1, col=1)
    fig4.add_hline(y=20, line_dash="dash", line_color="gray", row=1, col=1)
    fig4.add_hline(y=40, line_dash="dash", line_color="white", row=1, col=1)

    fig4.add_trace(go.Scatter(x=df["time"], y=df["CCI"],
                              name="CCI", line=dict(color="#F59E0B", width=2)), row=1, col=2)
    fig4.add_hline(y=100, line_dash="dash", line_color="red", row=1, col=2)
    fig4.add_hline(y=-100, line_dash="dash", line_color="lime", row=1, col=2)

    fig4.add_trace(go.Scatter(x=df["time"], y=df["WilliamsR"],
                              name="Williams %R", line=dict(color="#60A5FA", width=2)), row=2, col=1)
    fig4.add_hline(y=-20, line_dash="dash", line_color="red", row=2, col=1)
    fig4.add_hline(y=-80, line_dash="dash", line_color="lime", row=2, col=1)

    colors_roc = ["green" if v >= 0 else "red" for v in df["ROC"].fillna(0)]
    fig4.add_trace(go.Bar(x=df["time"], y=df["ROC"],
                          name="ROC", marker_color=colors_roc), row=2, col=2)
    fig4.add_hline(y=0, line_color="gray", row=2, col=2)

    fig4.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "📡 ADX: Measures trend strength (not direction). Above 20 = trend is forming; above 40 = strong trend. "
        "Green +DI above red -DI = bullish direction. "
        "CCI: Above +100 = overbought (🔴 Sell); below -100 = oversold (🟢 Buy). "
        "Williams %R: Ranges -100 to 0. Below -80 = oversold Buy zone; above -20 = overbought Sell zone. "
        "ROC: Measures % price change over 12 periods. Above +5% = strong momentum up; below -5% = strong momentum down."
    )

with tab5:
    st.subheader("🔔 Signal History (this session)")
    if st.session_state.alerts:
        for alert in st.session_state.alerts:
            st.markdown(f"- {alert}")
    else:
        st.info("No alerts yet.")
    st.caption(
        "🔔 Alerts are logged each time you load or refresh the dashboard. "
        "They record the composite signal score across all 10 indicators at that moment in time. "
        "History resets when the app is restarted or redeployed."
    )

st.caption("Data: CoinGecko · Auto-refreshes every 5 minutes")
