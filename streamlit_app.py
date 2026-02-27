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
    padding-top: 2.75rem !important;
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

# ── DATA FETCHERS ──────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=30", timeout=10)
        r.raise_for_status()
        data = r.json()["data"]
        df = pd.DataFrame(data)
        df["value"] = df["value"].astype(int)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df = df.sort_values("timestamp")
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_btc_dominance():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        r.raise_for_status()
        data = r.json()["data"]
        btc_dom = data["market_cap_percentage"].get("btc", None)
        total_mcap = data.get("total_market_cap", {}).get("usd", None)
        return btc_dom, total_mcap
    except:
        return None, None

@st.cache_data(ttl=3600)
def get_altcoin_index():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        r.raise_for_status()
        data = r.json()["data"]
        btc_dom = data["market_cap_percentage"].get("btc", 0)
        eth_dom = data["market_cap_percentage"].get("eth", 0)
        alt_index = round(100 - btc_dom - eth_dom, 1)
        return alt_index
    except:
        return None

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
        st.error(f"OHLC error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_daily_btc_long():
    # 365 days daily for Pi Cycle + 200w MA
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
        params = {"vs_currency": "usd", "days": "365"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["close"] = df["close"].astype(float)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_btc_weekly_long():
    # Max weekly data for 200-week MA (need 200+ weeks = ~4 years)
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": "1460", "interval": "weekly"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["time", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["close"] = df["close"].astype(float)
        df["MA_200w"] = df["close"].rolling(200).mean()
        return df
    except:
        return pd.DataFrame()

# ── INDICATORS ─────────────────────────────────────────────────

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
    df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    return df

def compute_pi_cycle(df):
    df = df.copy()
    df["MA_111"] = df["close"].rolling(111).mean()
    df["MA_350x2"] = df["close"].rolling(350).mean() * 2
    return df

def detect_support_resistance(df, window=5, num_levels=5):
    highs = df["high"] if "high" in df.columns else df["close"]
    lows = df["low"] if "low" in df.columns else df["close"]
    resistance = []
    support = []
    for i in range(window, len(df) - window):
        if all(highs.iloc[i] >= highs.iloc[i-window:i]) and all(highs.iloc[i] >= highs.iloc[i+1:i+window+1]):
            resistance.append(highs.iloc[i])
        if all(lows.iloc[i] <= lows.iloc[i-window:i]) and all(lows.iloc[i] <= lows.iloc[i+1:i+window+1]):
            support.append(lows.iloc[i])
    resistance = sorted(set([round(r, -int(len(str(int(r)))-2)) for r in resistance]), reverse=True)[:num_levels]
    support = sorted(set([round(s, -int(len(str(int(s)))-2)) for s in support]), reverse=False)[-num_levels:]
    return support, resistance

def detect_trend_structure(df, window=10):
    closes = df["close"].values
    highs = []
    lows = []
    for i in range(window, len(closes) - window):
        if closes[i] == max(closes[i-window:i+window]):
            highs.append((df["time"].iloc[i], closes[i]))
        if closes[i] == min(closes[i-window:i+window]):
            lows.append((df["time"].iloc[i], closes[i]))
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1][1] > highs[-2][1]
        hl = lows[-1][1] > lows[-2][1]
        lh = highs[-1][1] < highs[-2][1]
        ll = lows[-1][1] < lows[-2][1]
        if hh and hl:
            return "🟢 Uptrend — Higher Highs & Higher Lows", "green"
        elif lh and ll:
            return "🔴 Downtrend — Lower Highs & Lower Lows", "red"
        else:
            return "⚪ Choppy — No clear structure", "gray"
    return "⚪ Not enough data", "gray"

def compute_squeeze(df):
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    kc_mid = df["close"].rolling(20).mean()
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=20).average_true_range()
    kc_upper = kc_mid + 1.5 * atr
    kc_lower = kc_mid - 1.5 * atr
    df["squeeze"] = (bb.bollinger_hband() < kc_upper) & (bb.bollinger_lband() > kc_lower)
    delta = df["close"] - ((df["high"].rolling(20).max() + df["low"].rolling(20).min()) / 2 + df["close"].rolling(20).mean()) / 2
    df["squeeze_hist"] = delta.rolling(20).mean()
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
        indicators.append(("Bollinger Bands", 1, "✅", f"Price below lower band — oversold Buy signal"))
    elif close > bb_upper:
        indicators.append(("Bollinger Bands", -1, "🔴", f"Price above upper band — overbought Sell signal"))
    elif bb_pct < 0.2:
        indicators.append(("Bollinger Bands", 1, "✅", f"Price near lower band ({bb_pct*100:.0f}%) — leaning oversold"))
    elif bb_pct > 0.8:
        indicators.append(("Bollinger Bands", -1, "🔴", f"Price near upper band ({bb_pct*100:.0f}%) — leaning overbought"))
    else:
        indicators.append(("Bollinger Bands", 0, "⚪", f"Price mid-band ({bb_pct*100:.0f}%) — neutral"))

    ema200 = row["EMA_200"]
    if pd.isna(ema200):
        indicators.append(("EMA 200 Trend", 0, "⚪", "Neutral — not enough data for 200-period EMA"))
    elif close > ema200:
        indicators.append(("EMA 200 Trend", 1, "✅", f"Price above EMA 200 (${ema200:,.2f}) — long-term uptrend"))
    else:
        indicators.append(("EMA 200 Trend", -1, "🔴", f"Price below EMA 200 (${ema200:,.2f}) — long-term downtrend"))

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

# ── LOAD ALL DATA ──────────────────────────────────────────────

market = get_market_data(coin_id)
df = get_ohlc_data(coin_id, days)
fg_df = get_fear_greed()
btc_dom, total_mcap = get_btc_dominance()
alt_index = get_altcoin_index()
df_btc_daily = get_daily_btc_long()
df_btc_weekly = get_btc_weekly_long()

if df.empty or len(df) < 20:
    st.error("Not enough data. Try refreshing.")
    st.stop()

if timeframe == "Weekly":
    df = df.set_index("time").resample("W").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()

df = compute_indicators(df)
df = compute_squeeze(df)
df = df.dropna(subset=["RSI"])

latest = df.iloc[-1]
price = market.get("price", latest["close"])
score, bullish_count, bearish_count, indicators = composite_score(latest)
neutral_count = 10 - bullish_count - bearish_count
label = signal_label(score)

ts = datetime.now().strftime("%Y-%m-%d %H:%M")
st.session_state.alerts.insert(0, f"{ts} | {coin_label} ({timeframe}) | {bullish_count}/10 bullish | {label}")
st.session_state.alerts = st.session_state.alerts[:30]

# ── MARKET SENTIMENT BAR (always BTC/global) ──────────────────

st.markdown("#### 🌐 Market Sentiment")
ms1, ms2, ms3, ms4 = st.columns(4)

# Fear & Greed
if not fg_df.empty:
    fg_val = fg_df.iloc[-1]["value"]
    fg_class = fg_df.iloc[-1]["value_classification"]
    if fg_val <= 25:
        fg_emoji = "😱"
    elif fg_val <= 45:
        fg_emoji = "😨"
    elif fg_val <= 55:
        fg_emoji = "😐"
    elif fg_val <= 75:
        fg_emoji = "😊"
    else:
        fg_emoji = "🤑"
    ms1.metric(f"😨 Fear & Greed", f"{fg_val} — {fg_class} {fg_emoji}")
else:
    ms1.metric("😨 Fear & Greed", "N/A")

# BTC Dominance
if btc_dom:
    dom_signal = "🟢 BTC leading" if btc_dom > 55 else ("🔴 Alts leading" if btc_dom < 45 else "⚪ Balanced")
    ms2.metric("🟠 BTC Dominance", f"{btc_dom:.1f}%", delta=dom_signal)
else:
    ms2.metric("🟠 BTC Dominance", "N/A")

# Altcoin Index
if alt_index:
    alt_signal = "🔥 Alt season" if alt_index > 40 else "🥶 BTC season"
    ms3.metric("🪙 Altcoin Market %", f"{alt_index:.1f}%", delta=alt_signal)
else:
    ms3.metric("🪙 Altcoin Market %", "N/A")

# Total Market Cap
if total_mcap:
    ms4.metric("🏦 Total Market Cap", f"${total_mcap/1e12:.2f}T")
else:
    ms4.metric("🏦 Total Market Cap", "N/A")

st.divider()

# ── COIN HEADER ───────────────────────────────────────────────

logo_col, title_col = st.columns([1, 11])
with logo_col:
    st.image(coin_logo, width=48)
with title_col:
    st.markdown(f"## {coin_ticker} · ${price:,.2f}  &nbsp; {label}")

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 RSI (14)", round(latest["RSI"], 1) if pd.notna(latest["RSI"]) else "N/A")
col2.metric("⚡ Stoch RSI", round(latest["StochRSI_k"], 1) if pd.notna(latest["StochRSI_k"]) else "N/A")
col3.metric("📉 MACD Hist", round(latest["MACD_hist"], 4) if pd.notna(latest["MACD_hist"]) else "N/A")
col4.metric("🎯 Signal Score", f"{bullish_count}/10 bullish")

# Price Performance
st.markdown("#### 📊 Price Performance")
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

# Signal breakdown
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

# ── MAIN INDICATOR GRAPHS ─────────────────────────────────────

st.markdown("#### 📈 RSI + Stochastic RSI")
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

st.divider()

st.markdown("#### 💵 Price + Bollinger Bands + EMA")
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
# Support & Resistance
support, resistance = detect_support_resistance(df)
for s in support:
    fig2.add_hline(y=s, line_dash="dot", line_color="lime", opacity=0.5,
                   annotation_text=f"S ${s:,.0f}", annotation_position="left")
for r in resistance:
    fig2.add_hline(y=r, line_dash="dot", line_color="red", opacity=0.5,
                   annotation_text=f"R ${r:,.0f}", annotation_position="left")
fig2.update_layout(
    title=f"{coin_ticker} Price + Bollinger Bands + EMAs + S/R — {timeframe}",
    height=650, xaxis_rangeslider_visible=False
)
st.plotly_chart(fig2, use_container_width=True)
st.caption("📉 Bollinger Bands (gray): Price below lower band = oversold (🟢 Buy); above upper band = overbought (🔴 Sell). Band width shows volatility — wider = more volatile.")
st.caption("📈 EMA 50 (green) / EMA 200 (red): Golden Cross = EMA 50 crosses above EMA 200 (🟢 bullish). Death Cross = EMA 50 crosses below EMA 200 (🔴 bearish).")
st.caption("🟢 Support levels (green dotted) = price floors where buyers historically step in. 🔴 Resistance levels (red dotted) = price ceilings where sellers historically appear. Auto-detected from recent price structure.")

st.divider()

st.markdown("#### 📉 MACD")
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
st.caption("📉 MACD (blue) vs Signal (pink): Blue crossing above pink = bullish momentum (🟢 Buy). Blue crossing below pink = bearish momentum (🔴 Sell).")
st.caption("📊 Histogram: Growing green bars = strengthening upward momentum. Growing red bars = strengthening downward momentum.")

st.divider()

st.markdown("#### 📡 ADX — Trend Strength")
fig_adx = go.Figure()
fig_adx.add_trace(go.Scatter(x=df["time"], y=df["ADX"], name="ADX", line=dict(color="#A78BFA", width=2)))
fig_adx.add_trace(go.Scatter(x=df["time"], y=df["ADX_pos"], name="+DI", line=dict(color="#34D399", width=1.5)))
fig_adx.add_trace(go.Scatter(x=df["time"], y=df["ADX_neg"], name="-DI", line=dict(color="#F87171", width=1.5)))
fig_adx.add_hline(y=20, line_dash="dash", line_color="gray")
fig_adx.add_hline(y=40, line_dash="dash", line_color="white")
fig_adx.update_layout(height=350)
st.plotly_chart(fig_adx, use_container_width=True)
st.caption("📡 ADX (purple): Measures trend strength, not direction. Above 20 = trend forming; above 40 = strong trend. Green +DI above red -DI = bullish (🟢). Red -DI above green = bearish (🔴). Below 20 = choppy market.")

st.divider()

st.markdown("#### 📊 CCI — Commodity Channel Index")
fig_cci = go.Figure()
fig_cci.add_trace(go.Scatter(x=df["time"], y=df["CCI"], name="CCI", line=dict(color="#F59E0B", width=2)))
fig_cci.add_hline(y=100, line_dash="dash", line_color="red")
fig_cci.add_hline(y=-100, line_dash="dash", line_color="lime")
fig_cci.add_hrect(y0=-300, y1=-100, fillcolor="green", opacity=0.05)
fig_cci.add_hrect(y0=100, y1=300, fillcolor="red", opacity=0.05)
fig_cci.update_layout(height=350)
st.plotly_chart(fig_cci, use_container_width=True)
st.caption("📊 CCI: Measures how far price is from its average. Below -100 = oversold (🟢 Buy). Above +100 = overbought (🔴 Sell). Between -100 and +100 = neutral.")

st.divider()

st.markdown("#### 📉 Williams %R")
fig_wr = go.Figure()
fig_wr.add_trace(go.Scatter(x=df["time"], y=df["WilliamsR"], name="Williams %R", line=dict(color="#60A5FA", width=2)))
fig_wr.add_hline(y=-20, line_dash="dash", line_color="red")
fig_wr.add_hline(y=-80, line_dash="dash", line_color="lime")
fig_wr.add_hrect(y0=-100, y1=-80, fillcolor="green", opacity=0.05)
fig_wr.add_hrect(y0=-20, y1=0, fillcolor="red", opacity=0.05)
fig_wr.update_layout(height=350)
st.plotly_chart(fig_wr, use_container_width=True)
st.caption("📉 Williams %R: Ranges -100 to 0. Below -80 = oversold, bounce likely (🟢 Buy). Above -20 = overbought, pullback likely (🔴 Sell). Mid-range = neutral.")

st.divider()

st.markdown("#### 📈 ROC — Rate of Change")
colors_roc = ["green" if v >= 0 else "red" for v in df["ROC"].fillna(0)]
fig_roc = go.Figure()
fig_roc.add_trace(go.Bar(x=df["time"], y=df["ROC"], name="ROC", marker_color=colors_roc))
fig_roc.add_hline(y=5, line_dash="dash", line_color="lime")
fig_roc.add_hline(y=-5, line_dash="dash", line_color="red")
fig_roc.add_hline(y=0, line_color="gray")
fig_roc.update_layout(height=350)
st.plotly_chart(fig_roc, use_container_width=True)
st.caption("📈 ROC: % price change over 12 periods. Above +5% = strong positive momentum (🟢). Below -5% = strong negative momentum (🔴). Near 0 = momentum stalling.")

st.divider()

# ── ADVANCED TAB ──────────────────────────────────────────────

with st.expander("🔬 Advanced Analysis (click to expand)", expanded=False):

    # Trend Structure
    trend_label, trend_color = detect_trend_structure(df)
    st.markdown(f"#### 📐 Market Structure: {trend_label}")
    st.caption("Detects whether BTC is making Higher Highs & Higher Lows (uptrend), Lower Highs & Lower Lows (downtrend), or no clear structure (choppy). Based on recent swing points.")

    st.divider()

    # ATR
    st.markdown("#### 📏 ATR — Average True Range (Volatility)")
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=df["time"], y=df["ATR"], name="ATR", line=dict(color="#C084FC", width=2)))
    fig_atr.update_layout(height=300)
    st.plotly_chart(fig_atr, use_container_width=True)
    st.caption("📏 ATR: Measures average price movement per period (volatility). Rising ATR = increasing volatility (big moves incoming). Falling ATR = market calming down. Use for stop-loss sizing — place stops 1.5–2x ATR away from entry.")

    st.divider()

    # TTM Squeeze
    st.markdown("#### 🔫 TTM Squeeze — Momentum Buildup")
    squeeze_colors = ["green" if v >= 0 else "red" for v in df["squeeze_hist"].fillna(0)]
    dot_colors = ["black" if s else "lime" for s in df["squeeze"].fillna(False)]
    fig_sq = go.Figure()
    fig_sq.add_trace(go.Bar(x=df["time"], y=df["squeeze_hist"], name="Momentum", marker_color=squeeze_colors))
    fig_sq.add_trace(go.Scatter(x=df["time"], y=[0]*len(df), mode="markers",
                                marker=dict(color=dot_colors, size=6, symbol="circle"),
                                name="Squeeze (black=on, lime=off)"))
    fig_sq.update_layout(height=350)
    st.plotly_chart(fig_sq, use_container_width=True)
    st.caption("🔫 TTM Squeeze: Black dots = squeeze ON (Bollinger Bands inside Keltner Channels) — price is coiling for a big move. Lime dots = squeeze OFF — the move has been released. Green bars = bullish momentum; red bars = bearish momentum.")

    st.divider()

    # Pi Cycle Top
    st.markdown("#### 🔄 Pi Cycle Top Indicator (BTC Daily)")
    if not df_btc_daily.empty:
        df_pi = compute_pi_cycle(df_btc_daily)
        df_pi = df_pi.dropna(subset=["MA_111"])
        fig_pi = go.Figure()
        fig_pi.add_trace(go.Scatter(x=df_pi["time"], y=df_pi["close"], name="BTC Price", line=dict(color="#F59E0B", width=1)))
        fig_pi.add_trace(go.Scatter(x=df_pi["time"], y=df_pi["MA_111"], name="111-day MA", line=dict(color="#60A5FA", width=2)))
        fig_pi.add_trace(go.Scatter(x=df_pi["time"], y=df_pi["MA_350x2"], name="350-day MA x2", line=dict(color="#F87171", width=2)))
        fig_pi.update_layout(height=400, title="Pi Cycle Top — BTC Daily (365 days)")
        st.plotly_chart(fig_pi, use_container_width=True)
        last_pi = df_pi.dropna().iloc[-1]
        gap_pct = ((last_pi["MA_111"] - last_pi["MA_350x2"]) / last_pi["MA_350x2"]) * 100
        if last_pi["MA_111"] >= last_pi["MA_350x2"]:
            st.markdown(f"⚠️ **Pi Cycle TOP signal ACTIVE** — 111-day MA has crossed above 2x 350-day MA. Historically signals cycle top.")
        else:
            st.markdown(f"✅ **No Pi Cycle top signal.** 111-day MA is {abs(gap_pct):.1f}% below the 2x 350-day MA. Gap needs to close to signal a top.")
    else:
        st.info("Pi Cycle data unavailable.")
    st.caption("🔄 Pi Cycle Top: When the 111-day MA crosses above the 2x 350-day MA, it has historically marked BTC cycle tops within days. Currently only useful as a SELL/top signal — not a buy signal.")

    st.divider()

    # 200-week MA
    st.markdown("#### 📅 200-Week Moving Average (BTC Bear Market Floor)")
    if not df_btc_weekly.empty:
        df_w = df_btc_weekly.dropna(subset=["MA_200w"])
        fig_200w = go.Figure()
        fig_200w.add_trace(go.Scatter(x=df_w["time"], y=df_w["close"], name="BTC Price", line=dict(color="#F59E0B", width=1)))
        fig_200w.add_trace(go.Scatter(x=df_w["time"], y=df_w["MA_200w"], name="200-week MA", line=dict(color="#34D399", width=2.5)))
        fig_200w.update_layout(height=400, title="BTC Price vs 200-Week MA")
        st.plotly_chart(fig_200w, use_container_width=True)
        current_200w = df_w["MA_200w"].iloc[-1]
        current_price = df_w["close"].iloc[-1]
        pct_above = ((current_price - current_200w) / current_200w) * 100
        if pct_above < 0:
            st.markdown(f"🟢 **BTC is {abs(pct_above):.1f}% BELOW the 200-week MA (${current_200w:,.0f})** — historically a generational buy zone.")
        else:
            st.markdown(f"📊 BTC is **{pct_above:.1f}% above** the 200-week MA (${current_200w:,.0f}).")
    else:
        st.info("200-week MA data unavailable.")
    st.caption("📅 200-Week MA (green): The legendary BTC bear market floor. Every major bear market bottom in BTC history has been at or near this line. Price below it = rare generational buy opportunity.")

    st.divider()

    # 30-day Fear & Greed Chart
    st.markdown("#### 😨 Fear & Greed Index — 30 Day History")
    if not fg_df.empty:
        fg_colors = ["red" if v > 75 else ("orange" if v > 55 else ("gray" if v > 45 else ("lightblue" if v > 25 else "lime"))) for v in fg_df["value"]]
        fig_fg = go.Figure()
        fig_fg.add_trace(go.Bar(x=fg_df["timestamp"], y=fg_df["value"],
                                marker_color=fg_colors, name="Fear & Greed"))
        fig_fg.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Extreme Greed")
        fig_fg.add_hline(y=25, line_dash="dash", line_color="lime", annotation_text="Extreme Fear")
        fig_fg.update_layout(height=350, yaxis_range=[0, 100])
        st.plotly_chart(fig_fg, use_container_width=True)
    else:
        st.info("Fear & Greed data unavailable.")
    st.caption("😨 Fear & Greed Index: 0 = Extreme Fear (market panic = historically good buy), 100 = Extreme Greed (market euphoria = historically good sell). Lime = fear zones, red = greed zones.")

# ── ALERT HISTORY ─────────────────────────────────────────────

st.divider()
st.markdown("#### 🔔 Alert History (this session)")
if st.session_state.alerts:
    for alert in st.session_state.alerts:
        st.markdown(f"- {alert}")
else:
    st.info("No alerts yet.")
st.caption("🔔 Logged each refresh. Resets on app restart.")

st.caption("Data: CoinGecko · alternative.me · Auto-refreshes every 5 minutes")
