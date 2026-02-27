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
[data-testid="stMetricValue"] { font-size: 1rem !important; font-weight: 600; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
[data-testid="stMetric"]      { padding: 4px 0px !important; }
.block-container              { padding-top: 2.75rem !important; }
[data-testid="stAppViewBlockContainer"] { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

COINS = {
    "Bitcoin (BTC)":   ("bitcoin",          "BTC",  "https://assets.coingecko.com/coins/images/1/large/bitcoin.png"),
    "Ethereum (ETH)":  ("ethereum",         "ETH",  "https://assets.coingecko.com/coins/images/279/large/ethereum.png"),
    "Solana (SOL)":    ("solana",           "SOL",  "https://assets.coingecko.com/coins/images/4128/large/solana.png"),
    "XRP":             ("ripple",           "XRP",  "https://assets.coingecko.com/coins/images/44/large/xrp-symbol-white-128.png"),
    "BNB":             ("binancecoin",      "BNB",  "https://assets.coingecko.com/coins/images/825/large/bnb-icon2_2x.png"),
    "Polkadot (DOT)":  ("polkadot",         "DOT",  "https://assets.coingecko.com/coins/images/12171/large/polkadot.png"),
    "Chainlink (LINK)":("chainlink",        "LINK", "https://assets.coingecko.com/coins/images/877/large/chainlink-new-logo.png"),
    "Sui (SUI)":       ("sui",              "SUI",  "https://assets.coingecko.com/coins/images/26375/large/sui_asset.jpeg"),
    "Hbar (HBAR)":     ("hedera-hashgraph", "HBAR", "https://assets.coingecko.com/coins/images/3688/large/hbar.png"),
}

CRYPTOCOMPARE_SYMBOLS = {
    "bitcoin":          "BTC",
    "ethereum":         "ETH",
    "solana":           "SOL",
    "ripple":           "XRP",
    "binancecoin":      "BNB",
    "polkadot":         "DOT",
    "chainlink":        "LINK",
    "sui":              "SUI",
    "hedera-hashgraph": "HBAR",
}

st.sidebar.header("⚙️ Settings")
coin_label = st.sidebar.selectbox("Select Coin", list(COINS.keys()))
coin_id, coin_ticker, coin_logo = COINS[coin_label]
timeframe = st.sidebar.radio("Timeframe", ["Daily", "Weekly"])
days = 365 if timeframe == "Weekly" else 180

if st.sidebar.button("🔄 Force Refresh"):
    st.cache_data.clear()
    st.rerun()

# ── API WRAPPERS ───────────────────────────────────────────────

def coingecko_get(url, params=None, retries=3, backoff=8):
    headers = {"accept": "application/json", "x-cg-demo-api-key": CG_KEY}
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
    headers = {"accept": "application/json", "x-cg-demo-api-key": CG_KEY}
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
        df["value"]     = df["value"].astype(int)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        return df.sort_values("timestamp")
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
        market = r.json().get("market_data", {})
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
        valid_days = [1, 7, 14, 30, 90, 180, 365]
        closest    = min(valid_days, key=lambda x: abs(x - days))
        r = coingecko_get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": str(closest)}
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
def get_long_daily(coin_id):
    try:
        symbol = CRYPTOCOMPARE_SYMBOLS.get(coin_id, "BTC")
        r = requests.get(
            "https://min-api.cryptocompare.com/data/v2/histoday",
            params={"fsym": symbol, "tsym": "USD", "limit": 2000},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()["Data"]["Data"]
        df   = pd.DataFrame(data)
        df["time"]  = pd.to_datetime(df["time"], unit="s")
        df["close"] = df["close"].astype(float)
        df["open"]  = df["open"].astype(float)
        df["high"]  = df["high"].astype(float)
        df["low"]   = df["low"].astype(float)
        df = df[df["close"] > 0].sort_values("time").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Long daily fetch error: {e}")
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
    df["squeeze"]      = (bb.bollinger_hband() < kc_upper) & (bb.bollinger_lband() > kc_lower)
    delta              = df["close"] - ((df["high"].rolling(20).max() + df["low"].rolling(20).min()) / 2 + df["close"].rolling(20).mean()) / 2
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

def inject_long_ema(df, df_long, timeframe):
    if df_long.empty or len(df_long) < 200:
        return df
    df_ema = df_long[["time", "close"]].copy()
    df_ema["EMA_50"]  = ta.trend.EMAIndicator(df_ema["close"], window=50).ema_indicator()
    df_ema["EMA_200"] = ta.trend.EMAIndicator(df_ema["close"], window=200).ema_indicator()
    if timeframe == "Weekly":
        df_ema = df_ema.set_index("time").resample("W").last().reset_index()
    df     = df.sort_values("time").reset_index(drop=True)
    df_ema = df_ema.sort_values("time").reset_index(drop=True)
    df     = pd.merge_asof(df, df_ema[["time", "EMA_50", "EMA_200"]],
                           on="time", direction="nearest", suffixes=("_short", "_long"))
    for col in ["EMA_50", "EMA_200"]:
        if f"{col}_long" in df.columns:
            df[col] = df[f"{col}_long"]
            df.drop(columns=[f"{col}_long"], inplace=True)
        if f"{col}_short" in df.columns:
            df.drop(columns=[f"{col}_short"], inplace=True)
    return df

def detect_support_resistance(df, window=3, num_levels=5):
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

def detect_major_sr_zones(df_long, price, num_levels=6):
    if df_long.empty:
        return [], []
    highs  = df_long["high"]
    lows   = df_long["low"]
    window = 20
    resistance, support = [], []
    for i in range(window, len(df_long) - window):
        if all(highs.iloc[i] >= highs.iloc[i-window:i]) and \
           all(highs.iloc[i] >= highs.iloc[i+1:i+window+1]):
            resistance.append(float(highs.iloc[i]))
        if all(lows.iloc[i] <= lows.iloc[i-window:i]) and \
           all(lows.iloc[i] <= lows.iloc[i+1:i+window+1]):
            support.append(float(lows.iloc[i]))
    def cluster(levels, pct=0.03):
        levels = sorted(set(levels))
        clustered = []
        for l in levels:
            if not clustered or abs(l - clustered[-1]) / clustered[-1] > pct:
                clustered.append(l)
        return clustered
    sup_levels = cluster(support)
    res_levels = cluster(resistance)
    sup_levels = [s for s in sup_levels if s < price and s > price * 0.5]
    res_levels = [r for r in res_levels if r > price and r < price * 1.5]
    magnitude  = 10 ** (len(str(int(price))) - 2)
    for mult in range(int(price * 0.7 / magnitude), int(price * 1.3 / magnitude) + 1):
        round_level = mult * magnitude
        if round_level < price * 0.98:
            sup_levels.append(round_level)
        elif round_level > price * 1.02:
            res_levels.append(round_level)
    sup_levels = sorted(cluster(sup_levels))[-num_levels:]
    res_levels = sorted(cluster(res_levels))[:num_levels]
    return sup_levels, res_levels

def detect_trend_structure(df, window=5):
    closes    = df["close"].values
    highs_pts, lows_pts = [], []
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
        if hh and hl:   return "🟢 Uptrend — Higher Highs & Higher Lows", "green"
        elif lh and ll: return "🔴 Downtrend — Lower Highs & Lower Lows", "red"
        else:           return "⚪ Choppy — No clear structure", "gray"
    return "⚪ Not enough swing points yet", "gray"

# ── SCORING ────────────────────────────────────────────────────

def composite_score(row):
    indicators = []
    close = row["close"]

    rsi = row["RSI"]
    if pd.isna(rsi):
        indicators.append(("RSI (14)", 0, "⚪", "Neutral — not enough data"))
    elif rsi < 30:
        indicators.append(("RSI (14)", 2, "✅", f"Strongly oversold at {rsi:.1f} — Strong Buy signal (<30)"))
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
        indicators.append(("EMA 200 Trend", 2, "✅", f"Price above EMA 200 (${ema200:,.2f}) — long-term uptrend"))
    else:
        indicators.append(("EMA 200 Trend", -2, "🔴", f"Price below EMA 200 (${ema200:,.2f}) — long-term downtrend"))

    ema50 = row["EMA_50"]
    if pd.isna(ema50) or pd.isna(ema200):
        indicators.append(("EMA 50/200 Cross", 0, "⚪", "Neutral — not enough data"))
    elif ema50 > ema200:
        indicators.append(("EMA 50/200 Cross", 2, "✅", "Golden Cross — EMA 50 above EMA 200 — bullish long-term"))
    else:
        indicators.append(("EMA 50/200 Cross", -2, "🔴", "Death Cross — EMA 50 below EMA 200 — bearish long-term"))

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
    net_score     = sum(s for _, s, _, _ in indicators)
    return net_score, bullish_count, bearish_count, indicators

def signal_label(score):
    if score >= 7:    return "🟢 STRONG BUY"
    elif score >= 3:  return "🟡 DCA BUY ZONE"
    elif score >= -2: return "⚪ HOLD / WATCH"
    elif score >= -5: return "🟠 CAUTION / REDUCE"
    else:             return "🔴 STRONG SELL"

# ── DCA COMMENTARY ─────────────────────────────────────────────

def format_entry_block(entries, stop, target):
    rows = ""
    for label, px in entries:
        rows += f"<tr><td style='padding:3px 14px 3px 0; color:#86efac'>🟢 {label}</td><td style='color:#f1f5f9; font-weight:600'>${px:,.2f}</td></tr>"
    rows += f"<tr><td style='padding:3px 14px 3px 0; color:#f87171'>🛑 Stop Reference</td><td style='color:#f1f5f9; font-weight:600'>${stop:,.2f}</td></tr>"
    rows += f"<tr><td style='padding:3px 14px 3px 0; color:#fbbf24'>🎯 First Target</td><td style='color:#f1f5f9; font-weight:600'>${target:,.2f}</td></tr>"
    return f"<table style='margin-top:10px; font-size:0.85rem; border-collapse:collapse'>{rows}</table>"

def format_sr_block(support_levels, resistance_levels, price):
    rows = ""
    for r in sorted(resistance_levels, reverse=True):
        pct = ((r - price) / price) * 100
        rows += f"<tr><td style='padding:3px 14px 3px 0; color:#f87171'>🔴 Resistance</td><td style='color:#f1f5f9; font-weight:600'>${r:,.2f}</td><td style='color:#9ca3af; font-size:0.8rem'>+{pct:.1f}%</td></tr>"
    rows += f"<tr><td style='padding:3px 14px 3px 0; color:#60a5fa'>── Current Price</td><td style='color:#60a5fa; font-weight:600'>${price:,.2f}</td><td></td></tr>"
    for s in sorted(support_levels, reverse=True):
        pct = ((s - price) / price) * 100
        rows += f"<tr><td style='padding:3px 14px 3px 0; color:#86efac'>🟢 Support</td><td style='color:#f1f5f9; font-weight:600'>${s:,.2f}</td><td style='color:#9ca3af; font-size:0.8rem'>{pct:.1f}%</td></tr>"
    return f"<table style='margin-top:6px; font-size:0.85rem; border-collapse:collapse'>{rows}</table>"

def generate_commentary(row, df_btc_long, df_long_coin, price):
    signals   = []
    rsi       = row["RSI"]
    wr        = row["WilliamsR"]
    ema50     = row["EMA_50"]
    ema200    = row["EMA_200"]
    stoch_k   = row["StochRSI_k"]
    macd_hist = row["MACD_hist"]
    atr       = row["ATR"] if pd.notna(row["ATR"]) else price * 0.03

    death_cross  = pd.notna(ema50) and pd.notna(ema200) and ema50 < ema200
    golden_cross = pd.notna(ema50) and pd.notna(ema200) and ema50 > ema200
    rsi_extreme  = pd.notna(rsi) and rsi < 30
    rsi_oversold = pd.notna(rsi) and rsi < 40
    wr_extreme   = pd.notna(wr)  and wr < -80
    stoch_low    = pd.notna(stoch_k) and stoch_k < 20
    macd_turning = pd.notna(macd_hist) and macd_hist > 0

    near_200w_ma  = False
    below_200w_ma = False
    ma_200w_val   = None
    pct_from_200w = None
    if not df_btc_long.empty and len(df_btc_long) >= 200:
        df_w = df_btc_long.set_index("time").resample("W").last().reset_index()
        df_w["MA_200w"] = df_w["close"].rolling(200).mean()
        df_w_valid = df_w.dropna(subset=["MA_200w"])
        if not df_w_valid.empty:
            ma_200w_val   = df_w_valid["MA_200w"].iloc[-1]
            btc_now       = df_w["close"].iloc[-1]
            pct_from_200w = ((btc_now - ma_200w_val) / ma_200w_val) * 100
            near_200w_ma  = abs(pct_from_200w) <= 15
            below_200w_ma = pct_from_200w < 0

    pi_gap_pct     = None
    pi_bottom_zone = False
    if not df_btc_long.empty and len(df_btc_long) >= 350:
        df_pi = df_btc_long.copy()
        df_pi["MA_111"]   = df_pi["close"].rolling(111).mean()
        df_pi["MA_350x2"] = df_pi["close"].rolling(350).mean() * 2
        df_pi = df_pi.dropna(subset=["MA_111", "MA_350x2"])
        if not df_pi.empty:
            last           = df_pi.iloc[-1]
            pi_gap_pct     = ((last["MA_350x2"] - last["MA_111"]) / last["MA_350x2"]) * 100
            pi_bottom_zone = pi_gap_pct > 30

    major_support, major_resistance = detect_major_sr_zones(df_long_coin, price)
    nearest_support    = max([s for s in major_support    if s < price], default=price * 0.85)
    nearest_resistance = min([r for r in major_resistance if r > price], default=price * 1.15)

    confluence = 0
    if rsi_extreme:    confluence += 2
    elif rsi_oversold: confluence += 1
    if wr_extreme:     confluence += 1
    if stoch_low:      confluence += 1
    if near_200w_ma:   confluence += 2
    if pi_bottom_zone: confluence += 2
    if macd_turning:   confluence += 1

    if golden_cross:
        stop_gc    = ema200 if pd.notna(ema200) else price * 0.88
        entries_gc = [
            ("Entry — Pullback to EMA 50",  ema50 if pd.notna(ema50) else price),
            ("Entry — EMA 50 −3%",          (ema50 if pd.notna(ema50) else price) * 0.97),
        ]
        signals.append(("bullish", "📈",
            "Golden Cross Active — Macro Uptrend Confirmed",
            f"EMA 50 is above EMA 200. Long-term trend is bullish. "
            f"Buy dips toward EMA 50 (${ema50:,.2f}) rather than chasing highs."
            + format_entry_block(entries_gc, stop_gc, nearest_resistance)))
    elif death_cross:
        signals.append(("bearish", "📉",
            "Death Cross Active — Macro Downtrend in Progress",
            f"EMA 50 (${ema50:,.2f}) is below EMA 200 (${ema200:,.2f}). "
            f"Use smaller position sizes. Look for confluence with oversold signals below before acting."))

    if death_cross and rsi_extreme and wr_extreme:
        stop_t1 = price - (1.5 * atr)
        entries_t1 = [
            ("Entry 1 — Now (market)",  price),
            ("Entry 2 — Limit −5%",     price * 0.95),
            ("Entry 3 — Limit −10%",    price * 0.90),
        ]
        signals.append(("dca_small", "🟡",
            "Tier 1 DCA — Deeply Oversold in Downtrend",
            f"RSI ({rsi:.1f}) below 30 AND Williams %R ({wr:.1f}) below -80. "
            f"Historically produces 10–25% relief bounces. "
            f"<b>Deploy 10–15% of intended position.</b> Ladder in — do not lump sum."
            + format_entry_block(entries_t1, stop_t1, ema50 if pd.notna(ema50) else nearest_resistance)))
    elif death_cross and rsi_oversold:
        signals.append(("dca_small", "🟡",
            "Tier 1 Watch — Approaching Oversold in Downtrend",
            f"RSI ({rsi:.1f}) oversold but not extreme. "
            f"Watch for RSI < 30 AND Williams %R < -80 simultaneously before triggering entry."))

    if near_200w_ma and ma_200w_val:
        if below_200w_ma:
            entries_t2 = [
                ("Entry 1 — Now (below 200w MA)", price),
                ("Entry 2 — At 200w MA",          ma_200w_val),
                ("Entry 3 — 200w MA −5%",         ma_200w_val * 0.95),
            ]
            signals.append(("dca_medium", "🟢",
                f"Tier 2 DCA — BTC Below 200-Week MA (${ma_200w_val:,.0f})",
                f"BTC is {abs(pct_from_200w):.1f}% BELOW the 200-week MA — "
                f"a zone that has marked every major bear market bottom (2015, 2018, 2022). "
                f"<b>Deploy 25–35% of intended position.</b>"
                + format_entry_block(entries_t2, ma_200w_val * 0.85, ema200 if pd.notna(ema200) else price * 1.20)))
        else:
            signals.append(("dca_medium", "🟡",
                f"Tier 2 Watch — BTC Within 15% of 200-Week MA (${ma_200w_val:,.0f})",
                f"BTC is {abs(pct_from_200w):.1f}% above the 200-week MA. "
                f"If BTC falls to ${ma_200w_val:,.0f}, prepare to deploy 25–35% of intended position."))

    if pi_bottom_zone and pi_gap_pct is not None:
        entries_t3 = [
            ("Tranche 1 — Now",        price),
            ("Tranche 2 — In 2 weeks", price * 0.93),
            ("Tranche 3 — In 4 weeks", price * 0.86),
        ]
        signals.append(("dca_large", "🟢",
            f"Tier 3 DCA — Pi Cycle Macro Bottom Zone ({pi_gap_pct:.1f}% gap)",
            f"111-day MA is {pi_gap_pct:.1f}% below 2× 350-day MA — deep macro bottom territory. "
            f"<b>Deploy 30–40% of intended position across 3 tranches over 4 weeks.</b>"
            + format_entry_block(entries_t3, price * 0.80, price * 2.0)))

    if macd_turning and rsi_oversold:
        signals.append(("bullish", "🔄",
            "Momentum Confirmation — MACD Histogram Turning Positive",
            f"MACD histogram just turned positive while RSI ({rsi:.1f}) remains oversold. "
            f"Selling pressure is easing. Adds confidence to DCA signals above."))

    if stoch_low and rsi_oversold:
        signals.append(("bullish", "⚡",
            "Stochastic RSI Confirmation — Both Oscillators Oversold",
            f"Stoch RSI K ({stoch_k:.1f}) below 20 alongside RSI ({rsi:.1f}) below 40. "
            f"Dual oversold alignment increases probability of a near-term bounce."))

    if major_support or major_resistance:
        signals.append(("neutral", "📊",
            "Major Support & Resistance Zones",
            "Key levels from long-term price history. Support = DCA targets. Resistance = take-profit levels."
            + format_sr_block(major_support, major_resistance, price)))

    active_dca = [s for s in signals if s[0] in ("dca_small", "dca_medium", "dca_large")]
    if not active_dca:
        if golden_cross:
            signals.append(("neutral", "⚪",
                "Uptrend Active — No Oversold Entry Yet",
                "Macro trend is bullish but no short-term oversold conditions. "
                "Wait for a dip toward EMA 50 or RSI < 45 before adding."))
        else:
            signals.append(("neutral", "⚪",
                "No High-Conviction Entry Signal",
                "Market is in a downtrend without sufficient oversold confluence. "
                "Wait for RSI < 35, Williams %R < -80, or proximity to 200-week MA. "
                "Patience is a position."))

    if confluence >= 6:
        conviction = "🔥 **HIGH CONVICTION DCA ZONE** — Multiple macro + momentum signals aligned. Rare setup historically associated with major lows."
    elif confluence >= 4:
        conviction = "🟡 **MODERATE CONVICTION** — Several signals aligned. Suitable for incremental DCA with smaller position sizes."
    elif confluence >= 2:
        conviction = "⚠️ **LOW CONVICTION** — Some signals present but insufficient confluence for high confidence. Small starter position only."
    else:
        conviction = "⏳ **WAIT** — No meaningful signal confluence. Preserve capital and monitor for developing setups."

    return signals, conviction, confluence

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

with st.spinner("Loading long-term data..."):
    df_long     = get_long_daily(coin_id)
    df_btc_long = get_long_daily("bitcoin")

if df.empty or len(df) < 20:
    st.error("Not enough OHLC data. Try refreshing.")
    st.stop()

if timeframe == "Weekly":
    df = df.set_index("time").resample("W").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()

df = compute_indicators(df)
df = compute_squeeze(df)
df = inject_long_ema(df, df_long, timeframe)
df = df.dropna(subset=["RSI"])

latest        = df.iloc[-1]
price         = market.get("price", latest["close"])
score, bullish_count, bearish_count, indicators = composite_score(latest)
neutral_count = 10 - bullish_count - bearish_count
label         = signal_label(score)

commentary_signals, conviction_summary, confluence_score = generate_commentary(
    latest, df_btc_long, df_long, price)

display_label = label
if confluence_score >= 6 and score < 3:
    display_label = "🟡 DCA BUY ZONE ⚠️"
elif confluence_score >= 4 and score < 0:
    display_label = "🟡 DCA BUY ZONE (Moderate)"

ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.session_state.alerts.insert(0,
    f"{ts} | {coin_label} ({timeframe}) | Score: {score} | Confluence: {confluence_score} | {display_label}")
st.session_state.alerts = st.session_state.alerts[:30]

# ── PRECOMPUTE DISPLAY VALUES ──────────────────────────────────

rsi_val    = f"{latest['RSI']:.1f}"          if pd.notna(latest['RSI'])         else "N/A"
stoch_val  = f"{latest['StochRSI_k']:.1f}"   if pd.notna(latest['StochRSI_k'])  else "N/A"
stochd_val = f"{latest['StochRSI_d']:.1f}"   if pd.notna(latest['StochRSI_d'])  else "N/A"
macd_val   = f"{latest['MACD']:.2f}"         if pd.notna(latest['MACD'])        else "N/A"
msig_val   = f"{latest['MACD_signal']:.2f}"  if pd.notna(latest['MACD_signal']) else "N/A"
mhst_val   = f"{latest['MACD_hist']:.2f}"    if pd.notna(latest['MACD_hist'])   else "N/A"
bb_pct_val = f"{latest['BB_pct']*100:.0f}%"  if pd.notna(latest['BB_pct'])      else "N/A"
ema50_val  = f"${latest['EMA_50']:,.2f}"      if pd.notna(latest['EMA_50'])      else "N/A"
ema200_val = f"${latest['EMA_200']:,.2f}"     if pd.notna(latest['EMA_200'])     else "N/A"
adx_val    = f"{latest['ADX']:.1f}"          if pd.notna(latest['ADX'])         else "N/A"
dip_val    = f"{latest['ADX_pos']:.1f}"      if pd.notna(latest['ADX_pos'])     else "N/A"
din_val    = f"{latest['ADX_neg']:.1f}"      if pd.notna(latest['ADX_neg'])     else "N/A"
cci_val    = f"{latest['CCI']:.1f}"          if pd.notna(latest['CCI'])         else "N/A"
wr_val     = f"{latest['WilliamsR']:.1f}"    if pd.notna(latest['WilliamsR'])   else "N/A"
roc_val    = f"{latest['ROC']:.2f}%"         if pd.notna(latest['ROC'])         else "N/A"
atr_val    = f"{latest['ATR']:.2f}"          if pd.notna(latest['ATR'])         else "N/A"
sq_val     = "ON 🔒" if latest["squeeze"] else "OFF 🔓"
sq_color   = "red"   if latest["squeeze"] else "lime"

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
    ms2.metric("🟠 BTC Dominance", f"{btc_dom:.1f}%",
               delta="🟢 BTC leading" if btc_dom > 55 else "🔴 Alts leading" if btc_dom < 45 else "⚪ Balanced")
else:
    ms2.metric("🟠 BTC Dominance", "N/A")
if alt_index:
    ms3.metric("🪙 Altcoin Market %", f"{alt_index:.1f}%",
               delta="🔥 Alt season" if alt_index > 40 else "🥶 BTC season")
else:
    ms3.metric("🪙 Altcoin Market %", "N/A")
if total_mcap:
    ms4.metric("🏦 Total Market Cap", f"${total_mcap/1e12:.2f}T")
else:
    ms4.metric("🏦 Total Market Cap", "N/A")

st.divider()

# ── COIN HEADER ───────────────────────────────────────────────

#logo_col, title_col = st.columns([1, 11])
logo_col, title_col = st.columns([0.4, 11])
with logo_col:
    st.image(coin_logo, width=48)
with title_col:
    st.markdown(f"#### {coin_ticker} · ${price:,.2f}  &nbsp; {display_label}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 RSI (14)",     rsi_val)
col2.metric("⚡ Stoch RSI",    stoch_val)
col3.metric("📉 MACD Hist",    mhst_val)
col4.metric("🎯 Signal Score", f"{score} pts · Confluence: {confluence_score}")

if confluence_score >= 6 and score < 3:
    st.info(
        "ℹ️ **Why DCA BUY ZONE despite a low indicator score?** "
        "The Death Cross and price below EMA 200 are suppressing the score. "
        "However high macro confluence has been detected — extreme oversold readings, "
        "proximity to the 200-week MA, and/or Pi Cycle bottom territory. "
        "These macro signals historically matter more at major market bottoms. "
        "See DCA Signal Analysis below for full detail."
    )
elif confluence_score >= 4 and score < 0:
    st.info(
        "ℹ️ **Moderate DCA conviction detected despite bearish indicator score.** "
        "Several macro/momentum signals are aligning. Keep position sizing conservative. "
        "See DCA Signal Analysis below."
    )

st.markdown("#### 📊 Price Performance")
p1, p2, p3, p4, p5 = st.columns(5)
delta_metric(p1, "1 Hour",   market.get("change_1h"))
delta_metric(p2, "24 Hours", market.get("change_24h"))
delta_metric(p3, "7 Days",   market.get("change_7d"))
delta_metric(p4, "30 Days",  market.get("change_30d"))
ath        = market.get("ath")
ath_change = market.get("ath_change")
if ath:
    p5.metric("📈 ATH", f"${ath:,.2f}",
              delta=f"{ath_change:.1f}% from ATH" if ath_change else None)
v1, v2 = st.columns(2)
if market.get("volume_24h"): v1.metric("💹 24h Volume", f"${market['volume_24h']/1e9:.2f}B")
if market.get("market_cap"): v2.metric("🏦 Market Cap", f"${market['market_cap']/1e9:.1f}B")
st.caption(f"⏱ Last updated: {ts} · Indicators: 10 min · Long-term data: daily")

st.divider()

# ── DCA COMMENTARY ────────────────────────────────────────────

st.markdown("### 🧠 DCA Signal Analysis")
st.markdown(conviction_summary)
st.markdown("")

color_map = {
    "dca_large":  "#22c55e",
    "dca_medium": "#86efac",
    "dca_small":  "#fbbf24",
    "bullish":    "#60a5fa",
    "bearish":    "#f87171",
    "neutral":    "#9ca3af",
}
for level, emoji, title, detail in commentary_signals:
    color = color_map.get(level, "#9ca3af")
    st.markdown(
        f"""<div style='border-left: 4px solid {color}; padding: 10px 16px;
        margin-bottom: 12px; background: rgba(255,255,255,0.03); border-radius: 4px;'>
        <strong style='color:{color}'>{emoji} {title}</strong><br>
        <span style='font-size:0.9rem; color:#d1d5db'>{detail}</span>
        </div>""",
        unsafe_allow_html=True)
st.caption("⚠️ Not financial advice. All signals are algorithmic and based on historical patterns. Always do your own research.")

st.divider()

# ── SIGNAL BREAKDOWN ──────────────────────────────────────────

with st.expander("📋 All 10 Indicators — Signal Breakdown", expanded=True):
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

# ── MARKET STRUCTURE ──────────────────────────────────────────

#trend_label, trend_color = detect_trend_structure(df)
if not df_long.empty and len(df_long) >= 20:
    df_struct = df_long.copy()
    trend_label, trend_color = detect_trend_structure(df_struct, window=5)
else:
    trend_label, trend_color = detect_trend_structure(df, window=3)

st.markdown(f"#### 📐 Market Structure: {trend_label}")
st.caption("Detects Higher Highs & Higher Lows (uptrend), Lower Highs & Lower Lows (downtrend), or no clear structure.")

st.divider()

# ── GRAPHS — ROW 1: RSI + Stoch RSI  |  Price + BB + EMA ─────

row1_l, row1_r = st.columns(2)

with row1_l:
    st.markdown(
        f"#### 📈 RSI + Stoch RSI &nbsp; "
        f"{val_span('RSI: ' + rsi_val, '#F59E0B')} &nbsp; "
        f"{val_span('K: ' + stoch_val, '#60A5FA')} &nbsp; "
        f"{val_span('D: ' + stochd_val, '#F472B6')}",
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
    fig1.update_layout(height=500, margin=dict(t=30, b=10))
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("RSI <30 = strongly oversold 🟢; >70 = overbought 🔴. Stoch K <20 = reversal up likely; >80 = reversal down.")

with row1_r:
    st.markdown(
        f"<h4>💵 Price + BB + EMA &nbsp; "
        f"{val_span('BB: ' + bb_pct_val, 'gray')} &nbsp; "
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
    support_short, resistance_short = detect_support_resistance(df)
    for s in support_short:
        fig2.add_hline(y=s, line_dash="dot", line_color="lime", opacity=0.5,
                       annotation_text=f"S ${s:,.2f}", annotation_position="bottom left")
    for r in resistance_short:
        fig2.add_hline(y=r, line_dash="dot", line_color="tomato", opacity=0.5,
                       annotation_text=f"R ${r:,.2f}", annotation_position="top left")
    major_sup, major_res = detect_major_sr_zones(df_long, price)
    for s in major_sup:
        fig2.add_hline(y=s, line_dash="dash", line_color="#22c55e", opacity=0.7,
                       annotation_text=f"Maj S ${s:,.2f}", annotation_position="bottom right")
    for r in major_res:
        fig2.add_hline(y=r, line_dash="dash", line_color="#f87171", opacity=0.7,
                       annotation_text=f"Maj R ${r:,.2f}", annotation_position="top right")
    fig2.update_layout(height=500, margin=dict(t=30, b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("BB below lower = oversold 🟢; above upper = overbought 🔴. EMA50 (green) / EMA200 (red). Dotted = short-term S/R; dashed = major S/R.")

st.divider()

# ── GRAPHS — ROW 2: MACD  |  ADX ─────────────────────────────

row2_l, row2_r = st.columns(2)

with row2_l:
    st.markdown(
        f"#### 📉 MACD &nbsp; "
        f"{val_span('MACD: ' + macd_val, '#60A5FA')} &nbsp; "
        f"{val_span('Sig: ' + msig_val, '#F472B6')} &nbsp; "
        f"{val_span('Hist: ' + mhst_val, 'gray')}",
        unsafe_allow_html=True)
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.45, 0.55],
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
    fig3.update_layout(height=500, margin=dict(t=30, b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("MACD (blue) above signal (pink) = bullish 🟢. Below = bearish 🔴. Green histogram bars = strengthening momentum.")

with row2_r:
    st.markdown(
        f"#### 📡 ADX — Trend Strength &nbsp; "
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
    fig_adx.update_layout(height=500, margin=dict(t=30, b=10))
    st.plotly_chart(fig_adx, use_container_width=True)
    st.caption("ADX >20 = trend forming; >40 = strong trend. +DI above -DI = bullish 🟢. -DI above +DI = bearish 🔴. <20 = choppy market.")

st.divider()

# ── GRAPHS — ROW 3: CCI  |  Williams %R ──────────────────────

row3_l, row3_r = st.columns(2)

with row3_l:
    st.markdown(
        f"#### 📊 CCI &nbsp; {val_span('CCI: ' + cci_val, '#F59E0B')}",
        unsafe_allow_html=True)
    fig_cci = go.Figure()
    fig_cci.add_trace(go.Scatter(x=df["time"], y=df["CCI"],
        name="CCI", line=dict(color="#F59E0B", width=2)))
    fig_cci.add_hline(y=100,  line_dash="dash", line_color="red")
    fig_cci.add_hline(y=-100, line_dash="dash", line_color="lime")
    fig_cci.add_hrect(y0=-300, y1=-100, fillcolor="green", opacity=0.05)
    fig_cci.add_hrect(y0=100,  y1=300,  fillcolor="red",   opacity=0.05)
    fig_cci.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_cci, use_container_width=True)
    st.caption("CCI below -100 = oversold 🟢. Above +100 = overbought 🔴. Between = neutral.")

with row3_r:
    st.markdown(
        f"#### 📉 Williams %R &nbsp; {val_span('%R: ' + wr_val, '#60A5FA')}",
        unsafe_allow_html=True)
    fig_wr = go.Figure()
    fig_wr.add_trace(go.Scatter(x=df["time"], y=df["WilliamsR"],
        name="Williams %R", line=dict(color="#60A5FA", width=2)))
    fig_wr.add_hline(y=-20, line_dash="dash", line_color="red")
    fig_wr.add_hline(y=-80, line_dash="dash", line_color="lime")
    fig_wr.add_hrect(y0=-100, y1=-80, fillcolor="green", opacity=0.05)
    fig_wr.add_hrect(y0=-20,  y1=0,   fillcolor="red",   opacity=0.05)
    fig_wr.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_wr, use_container_width=True)
    st.caption("Williams %R below -80 = oversold 🟢. Above -20 = overbought 🔴. Mid-range = neutral.")

st.divider()

# ── GRAPHS — ROW 4: ROC  |  ATR ───────────────────────────────

row4_l, row4_r = st.columns(2)

with row4_l:
    st.markdown(
        f"#### 📈 ROC — Rate of Change &nbsp; {val_span('ROC: ' + roc_val, '#F59E0B')}",
        unsafe_allow_html=True)
    colors_roc = ["green" if v >= 0 else "red" for v in df["ROC"].fillna(0)]
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Bar(x=df["time"], y=df["ROC"],
        name="ROC", marker_color=colors_roc))
    fig_roc.add_hline(y=5,  line_dash="dash", line_color="lime")
    fig_roc.add_hline(y=-5, line_dash="dash", line_color="red")
    fig_roc.add_hline(y=0,  line_color="gray")
    fig_roc.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption("ROC above +5% = strong positive momentum 🟢. Below -5% = strong negative momentum 🔴. Near 0 = stalling.")

with row4_r:
    st.markdown(
        f"#### 📏 ATR — Volatility &nbsp; {val_span('ATR: ' + atr_val, '#C084FC')}",
        unsafe_allow_html=True)
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=df["time"], y=df["ATR"],
        name="ATR", line=dict(color="#C084FC", width=2)))
    fig_atr.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_atr, use_container_width=True)
    st.caption("ATR rising = increasing volatility. Use for stop-loss sizing — place stops 1.5–2× ATR from entry.")

st.divider()

# ── GRAPHS — ROW 5: TTM Squeeze  |  Fear & Greed ─────────────

row5_l, row5_r = st.columns(2)

with row5_l:
    st.markdown(
        f"#### 🔫 TTM Squeeze &nbsp; {val_span('Squeeze: ' + sq_val, sq_color)}",
        unsafe_allow_html=True)
    squeeze_colors = ["green" if v >= 0 else "red" for v in df["squeeze_hist"].fillna(0)]
    dot_colors     = ["black" if s else "lime" for s in df["squeeze"].fillna(False)]
    fig_sq = go.Figure()
    fig_sq.add_trace(go.Bar(x=df["time"], y=df["squeeze_hist"],
        name="Momentum", marker_color=squeeze_colors))
    fig_sq.add_trace(go.Scatter(x=df["time"], y=[0]*len(df), mode="markers",
        marker=dict(color=dot_colors, size=6, symbol="circle"),
        name="Squeeze (black=on, lime=off)"))
    fig_sq.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_sq, use_container_width=True)
    st.caption("Black dots = market coiling for big move. Lime = move released. Green bars = bullish momentum; red = bearish.")

with row5_r:
    st.markdown("#### 😨 Fear & Greed — 30 Day History")
    if not fg_df.empty:
        fg_colors = [
            "red" if v > 75 else "orange" if v > 55 else
            "gray" if v > 45 else "lightblue" if v > 25 else "lime"
            for v in fg_df["value"]
        ]
        fig_fg = go.Figure()
        fig_fg.add_trace(go.Bar(x=fg_df["timestamp"], y=fg_df["value"],
            marker_color=fg_colors, name="Fear & Greed"))
        fig_fg.add_hline(y=75, line_dash="dash", line_color="red",  annotation_text="Extreme Greed")
        fig_fg.add_hline(y=25, line_dash="dash", line_color="lime", annotation_text="Extreme Fear")
        fig_fg.update_layout(height=400, margin=dict(t=30, b=10), yaxis_range=[0, 100])
        st.plotly_chart(fig_fg, use_container_width=True)
    else:
        st.info("Fear & Greed data unavailable.")
    st.caption("0 = Extreme Fear (historically good buy) 🟢. 100 = Extreme Greed (historically good sell) 🔴.")

st.divider()

# ── GRAPHS — ROW 6: Pi Cycle  |  200-Week MA ─────────────────

row6_l, row6_r = st.columns(2)

with row6_l:
    st.markdown("#### 🔄 Pi Cycle Top Indicator (BTC)")
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
            fig_pi.update_layout(height=400, margin=dict(t=30, b=10))
            st.plotly_chart(fig_pi, use_container_width=True)
            last_pi = df_pi.iloc[-1]
            if pd.notna(last_pi.get("MA_350x2")):
                gap_pct = ((last_pi["MA_350x2"] - last_pi["MA_111"]) / last_pi["MA_350x2"]) * 100
                if last_pi["MA_111"] >= last_pi["MA_350x2"]:
                    st.markdown("⚠️ **Pi Cycle TOP signal ACTIVE** — historically marks BTC cycle tops. Consider reducing.")
                else:
                    st.markdown(f"✅ **No Pi Cycle top.** 111-day MA is {gap_pct:.1f}% below 2× 350-day MA.")
            else:
                st.markdown("⏳ 350-day MA still building history.")
    else:
        st.info("Not enough BTC history for Pi Cycle.")
    st.caption("When 111-day MA crosses above 2× 350-day MA = historical BTC cycle top. SELL signal only.")

with row6_r:
    st.markdown("#### 📅 200-Week MA (BTC Bear Market Floor)")
    if not df_btc_long.empty and len(df_btc_long) >= 200:
        df_w       = compute_200w_ma(df_btc_long)
        df_w_valid = df_w.dropna(subset=["MA_200w"])
        if not df_w_valid.empty:
            current_200w = df_w_valid["MA_200w"].iloc[-1]
            current_btc  = df_w["close"].iloc[-1]
            pct_above    = ((current_btc - current_200w) / current_200w) * 100
            pct_color    = "lime" if pct_above < 0 else "#F59E0B"
            pct_str      = f"{abs(pct_above):.1f}% {'below ⬇️' if pct_above < 0 else 'above ⬆️'} 200w MA"
            st.markdown(
                f"**200w MA:** {val_span('$' + f'{current_200w:,.0f}', '#34D399')} &nbsp; "
                f"{val_span(pct_str, pct_color)}",
                unsafe_allow_html=True)
            fig_200w = go.Figure()
            fig_200w.add_trace(go.Scatter(x=df_w["time"], y=df_w["close"],
                name="BTC Price", line=dict(color="#F59E0B", width=1)))
            fig_200w.add_trace(go.Scatter(x=df_w_valid["time"], y=df_w_valid["MA_200w"],
                name="200-week MA", line=dict(color="#34D399", width=2.5)))
            fig_200w.update_layout(height=400, margin=dict(t=30, b=10))
            st.plotly_chart(fig_200w, use_container_width=True)
            if pct_above < 0:
                st.markdown(f"🟢 **BTC is {abs(pct_above):.1f}% BELOW the 200-week MA** — generational buy zone historically.")
            else:
                st.markdown(f"📊 BTC is **{pct_above:.1f}% above** the 200-week MA (${current_200w:,.0f}).")
    else:
        st.info("Not enough BTC history for 200-week MA.")
    st.caption("Every major BTC bear market bottom has been at or near the 200-week MA. Price below = rare buy opportunity.")

st.divider()

# ── ALERT HISTORY ─────────────────────────────────────────────

st.markdown("#### 🔔 Alert History (this session)")
for alert in st.session_state.alerts:
    st.markdown(f"- {alert}")
if not st.session_state.alerts:
    st.info("No alerts yet.")
st.caption("🔔 Logged each refresh. Resets on app restart.")

st.caption(f"Data: CoinGecko · CryptoCompare · alternative.me · Auto-refreshes every 60s · Last run: {ts}")
