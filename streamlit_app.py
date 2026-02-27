import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide", page_title="Crypto Signal Dashboard")

st_autorefresh(interval=60_000, key="price_refresh")

CG_KEY = st.secrets["COINGECKO_API_KEY"]

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
days = 1460 if timeframe == "Weekly" else 220   # 4 years for weekly EMA 200

if st.sidebar.button("🔄 Force Refresh"):
    st.cache_data.clear()
    st.rerun()

# ── API WRAPPERS ───────────────────────────────────────────────

def coingecko_get(url, params=None, retries=3, backoff=8):
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": CG_KEY
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 429:
                time.sleep(backoff * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError:
            if attempt == retries - 1:
                return None
            time.sleep(backoff)
    return None

def coingecko_get_fast(url, params=None, retries=2, backoff=3):
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": CG_KEY
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code == 429:
                time.sleep(backoff * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError:
            if attempt == retries - 1:
                return None
            time.sleep(backoff)
    return None

# ── DATA FETCHERS ──────────────────────────────────────────────

@st.cache_data(ttl=3600)
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

@st.cache_data(ttl=1800)
def get_global_data():
    try:
        r = coingecko_get_fast("https://api.coingecko.com/api/v3/global")
        if r is None:
            return None, None, None
        data       = r.json()["data"]
        btc_dom    = data["market_cap_percentage"].get("btc", None)
        eth_dom    = data["market_cap_percentage"].get("eth", 0)
        total_mcap = data.get("total_market_cap", {}).get("usd", None)
        alt_index  = round(100 - (btc_dom or 0) - eth_dom, 1)
        return btc_dom, total_mcap, alt_index
    except:
        return None, None, None

@st.cache_data(ttl=60)
def get_market_data(coin_id):
    try:
        r = coingecko_get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}",
            params={"localization": "false", "tickers": "false",
                    "community_data": "false", "developer_data": "false"}
        )
        if r is None:
            return {}
        data   = r.json()
        market = data.get("market_data", {})
        return {
            "price":      market.get("current_price", {}).get("usd", 0),
            "change_1h":  market.get("price_change_percentage_1h_in_currency", {}).get("usd", None),
            "change_24h": market.get("price_change_percentage_24h", None),
            "change_7d":  market.get("price_change_percentage_7d", None),
            "change_30d": market.get("price_change_percentage_30d", None),
            "market_cap": market.get("market_cap", {}).get("usd", None),
            "volume_24h": market.get("total_volume", {}).get("usd", None),
            "ath":        market.get("ath", {}).get("usd", None),
            "ath_change": market.get("ath_change_percentage", {}).get("usd", None),
        }
    except Exception as e:
        st.error(f"Market data error: {e}")
        return {}

@st.cache_data(ttl=600)
def get_ohlc_data(coin_id, days):
    try:
        # For weekly (days=1460) use market_chart and resample to OHLC
        if days > 365:
            r = coingecko_get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                params={"vs_currency": "usd", "days": "max"}
            )
            if r is None:
                return pd.DataFrame()
            prices = r.json()["prices"]
            df = pd.DataFrame(prices, columns=["time", "close"])
            df["time"]  = pd.to_datetime(df["time"], unit="ms")
            df["close"] = df["close"].astype(float)
            # Resample daily prices into weekly OHLC
            df = df.set_index("time").resample("W").agg(
                open=("close", "first"),
                high=("close", "max"),
                low=("close", "min"),
                close=("close", "last")
            ).dropna().reset_index()
            return df
        else:
            # Daily — use standard OHLC endpoint
            r = coingecko_get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
                params={"vs_currency": "usd", "days": str(days)}
            )
            if r is None:
                return pd.DataFrame()
            data = r.json()
            df   = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            return df
    except Exception as e:
        st.error(f"OHLC error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_btc_long_daily():
    try:
        r = coingecko_get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": "max"}
        )
        if r is None:
            return pd.DataFrame()
        prices = r.json()["prices"]
        df = pd.DataFrame(prices, columns=["time", "close"])
        df["time"]  = pd.to_datetime(df["time"], unit="ms")
        df["close"] = df["close"].astype(float)
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"BTC long daily error: {e}")
        return pd.DataFrame()

# ── INDICATORS ─────────────────────────────────────────────────

def compute_indicators(df):
    df["RSI"]         = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    stoch             = ta.momentum.StochRSIIndicator(df["close"], window=14)
    df["StochRSI_k"]  = stoch.stochrsi_k() * 100
    df["StochRSI_d"]  = stoch.stochrsi_d() * 100
    macd              = ta.trend.MACD(df["close"])
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"]   = macd.macd_diff()
    bb                = ta.volatility.BollingerBands(df["close"], window=20)
    df["BB_upper"]    = bb.bollinger_hband()
    df["BB_lower"]    = bb.bollinger_lband()
    df["BB_mid"]      = bb.bollinger_mavg()
    df["BB_pct"]      = bb.bollinger_pband()
    df["EMA_50"]      = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["EMA_200"]     = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    adx               = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["ADX"]         = adx.adx()
    df["ADX_pos"]     = adx.adx_pos()
    df["ADX_neg"]     = adx.adx_neg()
    df["CCI"]         = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
    df["WilliamsR"]   = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=14).williams_r()
    df["ROC"]         = ta.momentum.ROCIndicator(df["close"], window=12).roc()
    df["ATR"]         = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    return df

def compute_squeeze(df):
    bb       = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    kc_mid   = df["close"].rolling(20).mean()
    atr      = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=20).average_true_range()
    kc_upper = kc_mid + 1.5 * atr
    kc_lower = kc_mid - 1.5 * atr
    df["squeeze"] = (bb.bollinger_hband() < kc_upper) & (bb.bollinger_lband() > kc_lower)
    delta = df["close"] - (
        (df["high"].rolling(20).max() + df["low"].rolling(20).min()) / 2
        + df["close"].rolling(20).mean()
    ) / 2
    df["squeeze_hist"] = delta.rolling(20).mean()
    return df

def compute_pi_cycle(df):
    df = df.copy()
    df["MA_111"]   = df["close"].rolling(111).mean()
    df["MA_350x2"] = df["close"].rolling(350).mean() * 2
    return df.dropna(subset=["MA_111"]).reset_index(drop=True)

def compute_200w_ma(df_daily):
    df = df_daily.copy()
    df = df.set_index("time").resample("W").last().reset_index()
    df["MA_200w"] = df["close"].rolling(200).mean()
    return df

def detect_support_resistance(df, window=3, num_levels=4):
    highs = df["high"] if "high" in df.columns else df["close"]
    lows  = df["low"]  if "low"  in df.columns else df["close"]
    resistance, support = [], []
    for i in range(window, len(df) - window):
        if all(highs.iloc[i] >= highs.iloc[i-window:i]) and \
           all(highs.iloc[i] >= highs.iloc[i+1:i+window+1]):
            resistance.append(float(highs.iloc[i]))
        if all(lows.iloc[i] <= lows.iloc[i-window:i]) and \
           all(lows.iloc[i] <= lows.iloc[i+1:i+window+1]):
            support.append(float(lows.iloc[i]))

    def cluster(levels, pct=0.02):
        levels = sorted(set(levels))
        clustered = []
        for l in levels:
            if not clustered or abs(l - clustered[-1]) / clustered[-1] > pct:
                clustered.append(l)
        return clustered

    return cluster(support)[:num_levels], cluster(resistance)[-num_levels:]

def detect_trend_structure(df, window=5):
    closes    = df["close"].values
    highs_pts = []
    lows_pts  = []
    for i in range(window, len(closes) - window):
        if closes[i] == max(closes[max(0, i-window):i+window+1]):
            highs_pts.append(closes[i])
        if closes[i] == min(closes[max(0, i-window):i+window+1]):
            lows_pts.append(closes[i])
    if len(highs_pts) >= 2 and len(lows_pts) >= 2:
        hh = highs_pts[-1] > highs_pts[-2]
        hl = lows_pts[-1]  > lows_pts[-2]
        lh = highs_pts[-1] < highs_pts[-2]
        ll = lows_pts[-1]  < lows_pts[-2]
        if hh and hl:
            return "🟢 Uptrend — Higher Highs & Higher Lows", "green"
        elif lh and ll:
            return "🔴 Downtrend — Lower Highs & Lower Lows", "red"
        else:
            return "⚪ Choppy — No clear structure", "gray"
    return "⚪ Not enough swing points yet", "gray"

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

    bb_pct   = row["BB_pct"]
    bb_lower = row["BB_lower"]
    bb_upper = row["BB_upper"]
    if pd.isna(bb_pct):
        indicators.append(("Bollinger Bands", 0, "⚪", "Neutral — not enough data"))
    elif close < bb_lower:
        indicators.append(("Bollinger Bands", 1, "✅", "Price below lower band — oversold Buy signal"))
    elif close > bb_upper:
        indicators.append(("Bollinger Bands", -1, "🔴", "Price above upper band — overbought Sell signal"))
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

    adx     = row["ADX"]
    adx_pos = row["ADX_pos"]
    adx_neg = row["ADX_neg"]
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
    net_score     = bullish_count - bearish_count
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

def val_span(value, color):
    return f"<span style='font-size:0.85rem; color:{color}'>{value}</span>"

if "alerts" not in st.session_state:
    st.session_state.alerts = []

# ── LOAD ALL DATA ──────────────────────────────────────────────

with st.spinner("Loading price data..."):
    market = get_market_data(coin_id)
    df     = get_ohlc_data(coin_id, days)

with st.spinner("Loading global sentiment..."):
    fg_df                          = get_fear_greed()
    btc_dom, total_mcap, alt_index = get_global_data()

st.write(f"DEBUG: days={days}, timeframe={timeframe}, df rows={len(df)}, df empty={df.empty}")  # ← ADD THIS
st.write(f"DEBUG: df columns={df.columns.tolist()}")  # ← ADD THIS

if df.empty or len(df) < 20:
    st.error("Not enough OHLC data. Try refreshing.")
    st.stop()

if timeframe == "Weekly":
    df = df.set_index("time").resample("W").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()

df = compute_indicators(df)
df = compute_squeeze(df)
df = df.dropna(subset=["RSI"])

latest        = df.iloc[-1]
price         = market.get("price", latest["close"])
score, bullish_count, bearish_count, indicators = composite_score(latest)
neutral_count = 10 - bullish_count - bearish_count
label         = signal_label(score)

ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.session_state.alerts.insert(0,
    f"{ts} | {coin_label} ({timeframe}) | {bullish_count}/10 bullish | {label}")
st.session_state.alerts = st.session_state.alerts[:30]

# ── PRECOMPUTE DISPLAY VALUES ──────────────────────────────────

rsi_val    = f"{latest['RSI']:.1f}"          if pd.notna(latest['RSI'])        else "N/A"
stoch_val  = f"{latest['StochRSI_k']:.1f}"   if pd.notna(latest['StochRSI_k']) else "N/A"
stochd_val = f"{latest['StochRSI_d']:.1f}"   if pd.notna(latest['StochRSI_d']) else "N/A"
macd_val   = f"{latest['MACD']:.2f}"         if pd.notna(latest['MACD'])       else "N/A"
msig_val   = f"{latest['MACD_signal']:.2f}"  if pd.notna(latest['MACD_signal'])else "N/A"
mhst_val   = f"{latest['MACD_hist']:.2f}"    if pd.notna(latest['MACD_hist'])  else "N/A"
bb_pct_val = f"{latest['BB_pct']*100:.0f}%"  if pd.notna(latest['BB_pct'])     else "N/A"
ema50_val  = f"${latest['EMA_50']:,.0f}"      if pd.notna(latest['EMA_50'])     else "N/A"
ema200_val = f"${latest['EMA_200']:,.0f}"     if pd.notna(latest['EMA_200'])    else "N/A"
adx_val    = f"{latest['ADX']:.1f}"          if pd.notna(latest['ADX'])        else "N/A"
dip_val    = f"{latest['ADX_pos']:.1f}"      if pd.notna(latest['ADX_pos'])    else "N/A"
din_val    = f"{latest['ADX_neg']:.1f}"      if pd.notna(latest['ADX_neg'])    else "N/A"
cci_val    = f"{latest['CCI']:.1f}"          if pd.notna(latest['CCI'])        else "N/A"
wr_val     = f"{latest['WilliamsR']:.1f}"    if pd.notna(latest['WilliamsR'])  else "N/A"
roc_val    = f"{latest['ROC']:.2f}%"         if pd.notna(latest['ROC'])        else "N/A"
atr_val    = f"{latest['ATR']:.2f}"          if pd.notna(latest['ATR'])        else "N/A"

# ── MARKET SENTIMENT BAR ──────────────────────────────────────

st.markdown("#### 🌐 Market Sentiment")
ms1, ms2, ms3, ms4 = st.columns(4)

if not fg_df.empty:
    fg_val   = fg_df.iloc[-1]["value"]
    fg_class = fg_df.iloc[-1]["value_classification"]
    fg_emoji = ("😱" if fg_val <= 25 else "😨" if fg_val <= 45 else
                "😐" if fg_val <= 55 else "😊" if fg_val <= 75 else "🤑")
    ms1.metric("😨 Fear & Greed", f"{fg_val} — {fg_class} {fg_emoji}")
else:
    ms1.metric("😨 Fear & Greed", "N/A")

if btc_dom:
    dom_signal = ("🟢 BTC leading" if btc_dom > 55 else
                  "🔴 Alts leading" if btc_dom < 45 else "⚪ Balanced")
    ms2.metric("🟠 BTC Dominance", f"{btc_dom:.1f}%", delta=dom_signal)
else:
    ms2.metric("🟠 BTC Dominance", "N/A")

if alt_index:
    alt_signal = "🔥 Alt season" if alt_index > 40 else "🥶 BTC season"
    ms3.metric("🪙 Altcoin Market %", f"{alt_index:.1f}%", delta=alt_signal)
else:
    ms3.metric("🪙 Altcoin Market %", "N/A")

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

col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 RSI (14)",     rsi_val)
col2.metric("⚡ Stoch RSI",    stoch_val)
col3.metric("📉 MACD Hist",    mhst_val)
col4.metric("🎯 Signal Score", f"{bullish_count}/10 bullish")

st.markdown("#### 📊 Price Performance")
p1, p2, p3, p4, p5 = st.columns(5)
delta_metric(p1, "1 Hour",   market.get("change_1h"))
delta_metric(p2, "24 Hours", market.get("change_24h"))
delta_metric(p3, "7 Days",   market.get("change_7d"))
delta_metric(p4, "30 Days",  market.get("change_30d"))
ath        = market.get("ath")
ath_change = market.get("ath_change")
if ath:
    p5.metric("📈 ATH", f"${ath:,.0f}",
              delta=f"{ath_change:.1f}% from ATH" if ath_change else None)

v1, v2 = st.columns(2)
vol  = market.get("volume_24h")
mcap = market.get("market_cap")
if vol:  v1.metric("💹 24h Volume", f"${vol/1e9:.2f}B")
if mcap: v2.metric("🏦 Market Cap", f"${mcap/1e9:.1f}B")

st.caption(f"⏱ Price last updated: {ts} · Indicators refresh every 10 min · Advanced data refreshes daily")

st.divider()

# ── SIGNAL BREAKDOWN ──────────────────────────────────────────

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

# ── MAIN GRAPHS ───────────────────────────────────────────────

st.markdown(
    f"#### 📈 RSI + Stochastic RSI &nbsp;&nbsp; "
    f"{val_span('RSI: ' + rsi_val, '#F59E0B')} &nbsp; "
    f"{val_span('Stoch K: ' + stoch_val, '#60A5FA')} &nbsp; "
    f"{val_span('Stoch D: ' + stochd_val, '#F472B6')}",
    unsafe_allow_html=True)

fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     row_heights=[0.5, 0.5],
                     subplot_titles=("RSI (14)", "Stochastic RSI"))
fig1.add_trace(go.Scatter(x=df["time"], y=df["RSI"],
    name="RSI", line=dict(color="#F59E0B", width=2)), row=1, col=1)
fig1.add_hline(y=70, line_dash="dash", line_color="red",   row=1, col=1)
fig1.add_hline(y=40, line_dash="dash", line_color="green", row=1, col=1)
fig1.add_hline(y=30, line_dash="dash", line_color="lime",  row=1, col=1)
fig1.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.05, row=1, col=1)
fig1.add_hrect(y0=70, y1=100, fillcolor="red",   opacity=0.05, row=1, col=1)
fig1.add_trace(go.Scatter(x=df["time"], y=df["StochRSI_k"],
    name="Stoch K", line=dict(color="#60A5FA", width=2)), row=2, col=1)
fig1.add_trace(go.Scatter(x=df["time"], y=df["StochRSI_d"],
    name="Stoch D", line=dict(color="#F472B6", width=1, dash="dot")), row=2, col=1)
fig1.add_hline(y=80, line_dash="dash", line_color="red",  row=2, col=1)
fig1.add_hline(y=20, line_dash="dash", line_color="lime", row=2, col=1)
fig1.update_layout(height=600)
st.plotly_chart(fig1, use_container_width=True)
st.caption("📊 RSI: Below 30 = strongly oversold (🟢 Strong Buy); below 40 = buy zone; above 70 = overbought (🔴 Sell).")
st.caption("⚡ Stoch RSI: K line below 20 = reversal up likely (🟢); above 80 = reversal down likely (🔴).")

st.divider()

st.markdown(
    f"#### 💵 Price + Bollinger Bands + EMA + S/R &nbsp;&nbsp; "
    f"{val_span('BB: ' + bb_pct_val + ' of band', 'gray')} &nbsp; "
    f"{val_span('EMA50: ' + ema50_val, '#34D399')} &nbsp; "
    f"{val_span('EMA200: ' + ema200_val, '#F87171')}",
    unsafe_allow_html=True)

fig2 = go.Figure()
fig2.add_trace(go.Candlestick(
    x=df["time"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"], name="Price"))
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
support, resistance = detect_support_resistance(df)
for s in support:
    fig2.add_hline(y=s, line_dash="dot", line_color="lime", opacity=0.6,
                   annotation_text=f"S ${s:,.0f}", annotation_position="bottom left")
for r in resistance:
    fig2.add_hline(y=r, line_dash="dot", line_color="tomato", opacity=0.6,
                   annotation_text=f"R ${r:,.0f}", annotation_position="top left")
fig2.update_layout(
    title=f"{coin_ticker} Price + BB + EMA + S/R — {timeframe}",
    height=650, xaxis_rangeslider_visible=False)
st.plotly_chart(fig2, use_container_width=True)
st.caption("📉 Bollinger Bands: Price below lower = oversold (🟢); above upper = overbought (🔴).")
st.caption("📈 EMA 50 (green) / EMA 200 (red): Golden Cross = bullish (🟢). Death Cross = bearish (🔴).")
st.caption("🟢 Support (green dotted) = price floors. 🔴 Resistance (red dotted) = price ceilings.")

st.divider()

st.markdown(
    f"#### 📉 MACD &nbsp;&nbsp; "
    f"{val_span('MACD: ' + macd_val, '#60A5FA')} &nbsp; "
    f"{val_span('Signal: ' + msig_val, '#F472B6')} &nbsp; "
    f"{val_span('Hist: ' + mhst_val, 'gray')}",
    unsafe_allow_html=True)

fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     row_heights=[0.5, 0.5],
                     subplot_titles=("Price", "MACD"))
fig3.add_trace(go.Scatter(x=df["time"], y=df["close"],
    name="Price", line=dict(color="#F59E0B")), row=1, col=1)
fig3.add_trace(go.Scatter(x=df["time"], y=df["MACD"],
    name="MACD", line=dict(color="#60A5FA", width=2)), row=2, col=1)
fig3.add_trace(go.Scatter(x=df["time"], y=df["MACD_signal"],
    name="Signal", line=dict(color="#F472B6", width=1.5)), row=2, col=1)
colors_macd = ["green" if v >= 0 else "red" for v in df["MACD_hist"].fillna(0)]
fig3.add_trace(go.Bar(x=df["time"], y=df["MACD_hist"],
    name="Histogram", marker_color=colors_macd), row=2, col=1)
fig3.update_layout(height=600)
st.plotly_chart(fig3, use_container_width=True)
st.caption("📉 MACD (blue) vs Signal (pink): Blue above pink = bullish (🟢). Blue below pink = bearish (🔴).")
st.caption("📊 Histogram: Growing green bars = strengthening upward momentum. Growing red = downward momentum.")

st.divider()

st.markdown(
    f"#### 📡 ADX — Trend Strength &nbsp;&nbsp; "
    f"{val_span('ADX: ' + adx_val, '#A78BFA')} &nbsp; "
    f"{val_span('+DI: ' + dip_val, '#34D399')} &nbsp; "
    f"{val_span('-DI: ' + din_val, '#F87171')}",
    unsafe_allow_html=True)

fig_adx = go.Figure()
fig_adx.add_trace(go.Scatter(x=df["time"], y=df["ADX"],
    name="ADX", line=dict(color="#A78BFA", width=2)))
fig_adx.add_trace(go.Scatter(x=df["time"], y=df["ADX_pos"],
    name="+DI", line=dict(color="#34D399", width=1.5)))
fig_adx.add_trace(go.Scatter(x=df["time"], y=df["ADX_neg"],
    name="-DI", line=dict(color="#F87171", width=1.5)))
fig_adx.add_hline(y=20, line_dash="dash", line_color="gray")
fig_adx.add_hline(y=40, line_dash="dash", line_color="white")
fig_adx.update_layout(height=350)
st.plotly_chart(fig_adx, use_container_width=True)
st.caption("📡 ADX (purple): >20 = trend forming; >40 = strong. +DI above -DI = bullish (🟢). -DI above +DI = bearish (🔴). <20 = choppy.")

st.divider()

st.markdown(
    f"#### 📊 CCI — Commodity Channel Index &nbsp;&nbsp; "
    f"{val_span('CCI: ' + cci_val, '#F59E0B')}",
    unsafe_allow_html=True)

fig_cci = go.Figure()
fig_cci.add_trace(go.Scatter(x=df["time"], y=df["CCI"],
    name="CCI", line=dict(color="#F59E0B", width=2)))
fig_cci.add_hline(y=100,  line_dash="dash", line_color="red")
fig_cci.add_hline(y=-100, line_dash="dash", line_color="lime")
fig_cci.add_hrect(y0=-300, y1=-100, fillcolor="green", opacity=0.05)
fig_cci.add_hrect(y0=100,  y1=300,  fillcolor="red",   opacity=0.05)
fig_cci.update_layout(height=350)
st.plotly_chart(fig_cci, use_container_width=True)
st.caption("📊 CCI: Below -100 = oversold (🟢). Above +100 = overbought (🔴). Between = neutral.")

st.divider()

st.markdown(
    f"#### 📉 Williams %R &nbsp;&nbsp; "
    f"{val_span('%R: ' + wr_val, '#60A5FA')}",
    unsafe_allow_html=True)

fig_wr = go.Figure()
fig_wr.add_trace(go.Scatter(x=df["time"], y=df["WilliamsR"],
    name="Williams %R", line=dict(color="#60A5FA", width=2)))
fig_wr.add_hline(y=-20, line_dash="dash", line_color="red")
fig_wr.add_hline(y=-80, line_dash="dash", line_color="lime")
fig_wr.add_hrect(y0=-100, y1=-80, fillcolor="green", opacity=0.05)
fig_wr.add_hrect(y0=-20,  y1=0,   fillcolor="red",   opacity=0.05)
fig_wr.update_layout(height=350)
st.plotly_chart(fig_wr, use_container_width=True)
st.caption("📉 Williams %R: Below -80 = oversold (🟢). Above -20 = overbought (🔴). Mid-range = neutral.")

st.divider()

st.markdown(
    f"#### 📈 ROC — Rate of Change &nbsp;&nbsp; "
    f"{val_span('ROC: ' + roc_val, '#F59E0B')}",
    unsafe_allow_html=True)

colors_roc = ["green" if v >= 0 else "red" for v in df["ROC"].fillna(0)]
fig_roc = go.Figure()
fig_roc.add_trace(go.Bar(x=df["time"], y=df["ROC"],
    name="ROC", marker_color=colors_roc))
fig_roc.add_hline(y=5,  line_dash="dash", line_color="lime")
fig_roc.add_hline(y=-5, line_dash="dash", line_color="red")
fig_roc.add_hline(y=0,  line_color="gray")
fig_roc.update_layout(height=350)
st.plotly_chart(fig_roc, use_container_width=True)
st.caption("📈 ROC: Above +5% = strong positive momentum (🟢). Below -5% = strong negative momentum (🔴). Near 0 = stalling.")

st.divider()

# ── ADVANCED ANALYSIS ─────────────────────────────────────────

with st.expander("🔬 Advanced Analysis (click to expand)", expanded=False):

    load_advanced = st.button("📡 Load Advanced Data (Pi Cycle, 200w MA)")
    if "advanced_loaded" not in st.session_state:
        st.session_state.advanced_loaded = False
    if load_advanced:
        st.session_state.advanced_loaded = True

    trend_label, trend_color = detect_trend_structure(df)
    st.markdown(f"#### 📐 Market Structure: {trend_label}")
    st.caption("Detects Higher Highs & Higher Lows (uptrend), Lower Highs & Lower Lows (downtrend), or no clear structure.")

    st.divider()

    st.markdown(
        f"#### 📏 ATR — Average True Range &nbsp;&nbsp; "
        f"{val_span('ATR: ' + atr_val, '#C084FC')}",
        unsafe_allow_html=True)
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=df["time"], y=df["ATR"],
        name="ATR", line=dict(color="#C084FC", width=2)))
    fig_atr.update_layout(height=300)
    st.plotly_chart(fig_atr, use_container_width=True)
    st.caption("📏 ATR: Rising = increasing volatility. Falling = calming. Use for stop-loss sizing — place stops 1.5–2× ATR from entry.")

    st.divider()

    sq_val = "ON 🔒" if latest["squeeze"] else "OFF 🔓"
    sq_color = "red" if latest["squeeze"] else "lime"
    st.markdown(
        f"#### 🔫 TTM Squeeze — Momentum Buildup &nbsp;&nbsp; "
        f"{val_span('Squeeze: ' + sq_val, sq_color)}",
        unsafe_allow_html=True)
    squeeze_colors = ["green" if v >= 0 else "red" for v in df["squeeze_hist"].fillna(0)]
    dot_colors     = ["black" if s else "lime" for s in df["squeeze"].fillna(False)]
    fig_sq = go.Figure()
    fig_sq.add_trace(go.Bar(x=df["time"], y=df["squeeze_hist"],
        name="Momentum", marker_color=squeeze_colors))
    fig_sq.add_trace(go.Scatter(x=df["time"], y=[0]*len(df), mode="markers",
        marker=dict(color=dot_colors, size=6, symbol="circle"),
        name="Squeeze (black=on, lime=off)"))
    fig_sq.update_layout(height=350)
    st.plotly_chart(fig_sq, use_container_width=True)
    st.caption("🔫 TTM Squeeze: Black dots = coiling for big move. Lime = move released. Green bars = bullish momentum; red = bearish.")

    st.divider()

    if st.session_state.advanced_loaded:
        with st.spinner("Loading BTC long-term data..."):
            df_btc_long = get_btc_long_daily()

        st.markdown("#### 🔄 Pi Cycle Top Indicator (BTC Daily — Max History)")
        if not df_btc_long.empty and len(df_btc_long) >= 111:
            df_pi = compute_pi_cycle(df_btc_long)
            if not df_pi.empty:
                fig_pi = go.Figure()
                fig_pi.add_trace(go.Scatter(x=df_pi["time"], y=df_pi["close"],
                    name="BTC Price", line=dict(color="#F59E0B", width=1)))
                fig_pi.add_trace(go.Scatter(x=df_pi["time"], y=df_pi["MA_111"],
                    name="111-day MA", line=dict(color="#60A5FA", width=2)))
                df_pi_350 = df_pi.dropna(subset=["MA_350x2"])
                if not df_pi_350.empty:
                    fig_pi.add_trace(go.Scatter(x=df_pi_350["time"], y=df_pi_350["MA_350x2"],
                        name="350-day MA ×2", line=dict(color="#F87171", width=2)))
                fig_pi.update_layout(height=400, title="Pi Cycle Top — BTC Max Daily History")
                st.plotly_chart(fig_pi, use_container_width=True)
                last_pi = df_pi.iloc[-1]
                if pd.notna(last_pi.get("MA_350x2")):
                    gap_pct = ((last_pi["MA_111"] - last_pi["MA_350x2"]) / last_pi["MA_350x2"]) * 100
                    if last_pi["MA_111"] >= last_pi["MA_350x2"]:
                        st.markdown("⚠️ **Pi Cycle TOP signal ACTIVE** — 111-day MA crossed above 2× 350-day MA.")
                    else:
                        st.markdown(f"✅ **No Pi Cycle top signal.** 111-day MA is {abs(gap_pct):.1f}% below the 2× 350-day MA.")
                else:
                    st.markdown("⏳ 350-day MA still building.")
        else:
            st.info("Not enough data for Pi Cycle.")
        st.caption("🔄 Pi Cycle Top: When 111-day MA crosses above 2× 350-day MA it has historically marked BTC cycle tops. SELL signal only.")

        st.divider()

        st.markdown("#### 📅 200-Week Moving Average (BTC Bear Market Floor)")
        if not df_btc_long.empty and len(df_btc_long) >= 200:
            df_w       = compute_200w_ma(df_btc_long)
            df_w_valid = df_w.dropna(subset=["MA_200w"])
            if not df_w_valid.empty:
                current_200w = df_w_valid["MA_200w"].iloc[-1]
                current_btc  = df_w["close"].iloc[-1]
                pct_above    = ((current_btc - current_200w) / current_200w) * 100
                pct_color    = "lime" if pct_above < 0 else "#F59E0B"
                pct_str      = f"{abs(pct_above):.1f}% {'below' if pct_above < 0 else 'above'} 200w MA"
                st.markdown(
                    f"#### 📅 200-Week MA &nbsp;&nbsp; "
                    f"{val_span('$' + f'{current_200w:,.0f}', '#34D399')} &nbsp; "
                    f"{val_span(pct_str, pct_color)}",
                    unsafe_allow_html=True)
                fig_200w = go.Figure()
                fig_200w.add_trace(go.Scatter(x=df_w["time"], y=df_w["close"],
                    name="BTC Price", line=dict(color="#F59E0B", width=1)))
                fig_200w.add_trace(go.Scatter(x=df_w_valid["time"], y=df_w_valid["MA_200w"],
                    name="200-week MA", line=dict(color="#34D399", width=2.5)))
                fig_200w.update_layout(height=400, title="BTC Price vs 200-Week MA")
                st.plotly_chart(fig_200w, use_container_width=True)
                if pct_above < 0:
                    st.markdown(f"🟢 **BTC is {abs(pct_above):.1f}% BELOW the 200-week MA (${current_200w:,.0f})** — historically a generational buy zone.")
                else:
                    st.markdown(f"📊 BTC is **{pct_above:.1f}% above** the 200-week MA (${current_200w:,.0f}).")
            else:
                st.info("200-week MA still computing.")
        else:
            st.info("Not enough data for 200-week MA.")
        st.caption("📅 200-Week MA: Every major BTC bear market bottom has been at or near this line. Price below = generational buy opportunity.")

    else:
        st.info("👆 Click 'Load Advanced Data' above to render Pi Cycle and 200-Week MA charts.")

    st.divider()

    st.markdown("#### 😨 Fear & Greed Index — 30 Day History")
    if not fg_df.empty:
        fg_colors = [
            "red"       if v > 75 else
            "orange"    if v > 55 else
            "gray"      if v > 45 else
            "lightblue" if v > 25 else
            "lime"
            for v in fg_df["value"]
        ]
        fig_fg = go.Figure()
        fig_fg.add_trace(go.Bar(x=fg_df["timestamp"], y=fg_df["value"],
            marker_color=fg_colors, name="Fear & Greed"))
        fig_fg.add_hline(y=75, line_dash="dash", line_color="red",  annotation_text="Extreme Greed")
        fig_fg.add_hline(y=25, line_dash="dash", line_color="lime", annotation_text="Extreme Fear")
        fig_fg.update_layout(height=350, yaxis_range=[0, 100])
        st.plotly_chart(fig_fg, use_container_width=True)
    else:
        st.info("Fear & Greed data unavailable.")
    st.caption("😨 0 = Extreme Fear (historically good buy), 100 = Extreme Greed (historically good sell).")

# ── ALERT HISTORY ─────────────────────────────────────────────

st.divider()
st.markdown("#### 🔔 Alert History (this session)")
if st.session_state.alerts:
    for alert in st.session_state.alerts:
        st.markdown(f"- {alert}")
else:
    st.info("No alerts yet.")
st.caption("🔔 Logged each refresh. Resets on app restart.")

st.caption(f"Data: CoinGecko · alternative.me · Price auto-refreshes every 60s · Last run: {ts}")
