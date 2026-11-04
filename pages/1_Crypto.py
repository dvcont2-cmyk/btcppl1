from dashboard_core import completed_bars, clean_ohlc, support_resistance, render_supertrend
from dashboard_core import compute_indicators as core_indicators, triple_supertrend_signals as core_supertrend
import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh


st_autorefresh(interval=60_000, key="price_refresh")

try:
    CG_KEY = st.secrets.get("COINGECKO_API_KEY", "")
    CC_KEY = st.secrets.get("CRYPTOCOMPARE_API_KEY", "")
except (FileNotFoundError, KeyError):
    CG_KEY = ""
    CC_KEY = ""

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
view_mode = st.sidebar.radio("View", ["Dashboard", "Multi-timeframe – Crypto"], index=0)
days = 365 if timeframe == "Weekly" else 180

if st.sidebar.button("🔄 Force Refresh"):
    st.cache_data.clear()
    st.rerun()

# ── API WRAPPERS ───────────────────────────────────────────────

def coingecko_get(url, params=None, retries=3, backoff=8):
    headers = {"accept": "application/json"}
    if CG_KEY:
        headers["x-cg-demo-api-key"] = CG_KEY
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 429:
                time.sleep(backoff * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                return None
            time.sleep(backoff)
    return None

def coingecko_get_fast(url, params=None, retries=2, backoff=3):
    headers = {"accept": "application/json"}
    if CG_KEY:
        headers["x-cg-demo-api-key"] = CG_KEY
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code == 429:
                time.sleep(backoff * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException:
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

def get_ohlc_data(coin_id, days):
    # Legacy-compatible name; never use CoinGecko's automatic four-day OHLC.
    return get_long_daily(coin_id)

@st.cache_data(ttl=3600)
def get_long_daily(coin_id):
    try:
        symbol = CRYPTOCOMPARE_SYMBOLS[coin_id]
        response = requests.get("https://min-api.cryptocompare.com/data/v2/histoday",
            params={"fsym":symbol, "tsym":"USD", "limit":2000},
            headers={"authorization": f"Apikey {CC_KEY}"} if CC_KEY else {}, timeout=20)
        if response.status_code in (401, 403):
            st.error("CryptoCompare requires an authorised API key. Set CRYPTOCOMPARE_API_KEY in Streamlit secrets, then use Force Refresh.")
            return pd.DataFrame()
        response.raise_for_status()
        payload = response.json()
        if payload.get("Response") == "Error":
            raise ValueError(payload.get("Message", "Provider error"))
        raw = pd.DataFrame(payload.get("Data", {}).get("Data", []))
        if raw.empty:
            return pd.DataFrame()
        raw['time'] = pd.to_datetime(raw['time'], unit='s', utc=True).dt.tz_localize(None)
        if 'volumefrom' in raw:
            raw['volume'] = raw['volumefrom']
        cutoff = pd.Timestamp.now(tz='UTC').tz_localize(None).normalize()
        daily = completed_bars(raw, 'Daily', cutoff)
        daily.attrs['asof'] = cutoff
        return daily
    except Exception as e:
        st.error(f"Daily OHLC fetch error: {e}")
        return pd.DataFrame()

# ── INDICATORS ─────────────────────────────────────────────────

def compute_indicators(df):
    return core_indicators(df)

def compute_squeeze(df):
    return df  # Already calculated once in core_indicators.

def compute_pi_cycle(df):
    df = df.copy()
    df["MA_111"]   = df["close"].rolling(111).mean()
    df["MA_350x2"] = df["close"].rolling(350).mean() * 2
    return df.dropna(subset=["MA_111"]).reset_index(drop=True)

def compute_200w_ma(df_daily):
    df = completed_bars(df_daily, 'Weekly', df_daily.attrs.get('asof'))
    df['MA_200w'] = df.close.rolling(200).mean()
    return df

def inject_long_ema(df, df_long, timeframe):
    # EMAs already calculated on this timeframe's full OHLC history.
    return df
    
def detect_support_resistance(df, window=3, num_levels=5):
    return support_resistance(df, window, num_levels)

def detect_major_sr_zones(df_long, price, num_levels=6):
    return support_resistance(df_long, window=20, num_levels=num_levels, price=price)

def detect_trend_structure(df, window=5):
    closes    = df["close"].values
    highs_pts, lows_pts = [], []
    for i in range(window, len(closes) - window):
        if df["high"].iloc[i] == df["high"].iloc[i-window:i+window+1].max():
            highs_pts.append(df["high"].iloc[i])
        if df["low"].iloc[i] == df["low"].iloc[i-window:i+window+1].min():
            lows_pts.append(df["low"].iloc[i])
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
        indicators.append(("RSI (14)", 0, "⚪", "Unavailable — not enough data"))
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
        indicators.append(("Stoch RSI", 0, "⚪", "Unavailable — not enough data"))
    elif k < 20:
        indicators.append(("Stoch RSI", 1, "✅", f"Oversold at {k:.1f} — oversold reading (<20)"))
    elif k > 80:
        indicators.append(("Stoch RSI", -1, "🔴", f"Overbought at {k:.1f} — overbought reading (>80)"))
    else:
        indicators.append(("Stoch RSI", 0, "⚪", f"Neutral at {k:.1f} — between 20 and 80"))

    macd_val, macd_sig = row["MACD"], row["MACD_signal"]
    if pd.isna(macd_val) or pd.isna(macd_sig):
        indicators.append(("MACD", 0, "⚪", "Unavailable — not enough data"))
    elif macd_val > macd_sig:
        indicators.append(("MACD", 1, "✅", f"Bullish — MACD ({macd_val:.2f}) above signal ({macd_sig:.2f})"))
    else:
        indicators.append(("MACD", -1, "🔴", f"Bearish — MACD ({macd_val:.2f}) below signal ({macd_sig:.2f})"))

    bb_pct   = row["BB_pct"]
    bb_lower = row["BB_lower"]
    bb_upper = row["BB_upper"]
    if pd.isna(bb_pct):
        indicators.append(("Bollinger Bands", 0, "⚪", "Unavailable — not enough data"))
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
        indicators.append(("EMA 200 Trend", 0, "⚪", "Unavailable — not enough data for 200-period EMA"))
    elif close > ema200:
        indicators.append(("EMA 200 Trend", 2, "✅", f"Price above EMA 200 (${ema200:,.2f}) — long-term uptrend"))
    else:
        indicators.append(("EMA 200 Trend", -2, "🔴", f"Price below EMA 200 (${ema200:,.2f}) — long-term downtrend"))

    ema50 = row["EMA_50"]
    if pd.isna(ema50) or pd.isna(ema200):
        indicators.append(("EMA 50/200 Cross", 0, "⚪", "Unavailable — not enough data"))
    elif ema50 > ema200:
        indicators.append(("EMA 50/200 Cross", 2, "✅", "Golden Cross — EMA 50 above EMA 200 — bullish long-term"))
    else:
        indicators.append(("EMA 50/200 Cross", -2, "🔴", "Death Cross — EMA 50 below EMA 200 — bearish long-term"))

    adx, adx_pos, adx_neg = row["ADX"], row["ADX_pos"], row["ADX_neg"]
    if pd.isna(adx):
        indicators.append(("ADX Strength", 0, "⚪", "Unavailable — not enough data"))
    elif adx < 20:
        indicators.append(("ADX Strength", 0, "⚪", f"ADX {adx:.1f} — weak trend, signals less reliable"))
    elif adx_pos > adx_neg:
        indicators.append(("ADX Strength", 1, "✅", f"ADX {adx:.1f} — strong trend, bullish direction (+DI > -DI)"))
    else:
        indicators.append(("ADX Strength", -1, "🔴", f"ADX {adx:.1f} — strong trend, bearish direction (-DI > +DI)"))

    cci = row["CCI"]
    if pd.isna(cci):
        indicators.append(("CCI (20)", 0, "⚪", "Unavailable — not enough data"))
    elif cci < -100:
        indicators.append(("CCI (20)", 1, "✅", f"CCI {cci:.0f} — oversold below -100, potential Buy signal"))
    elif cci > 100:
        indicators.append(("CCI (20)", -1, "🔴", f"CCI {cci:.0f} — overbought above +100, potential Sell signal"))
    else:
        indicators.append(("CCI (20)", 0, "⚪", f"CCI {cci:.0f} — neutral between -100 and +100"))

    wr = row["WilliamsR"]
    if pd.isna(wr):
        indicators.append(("Williams %R", 0, "⚪", "Unavailable — not enough data"))
    elif wr < -80:
        indicators.append(("Williams %R", 1, "✅", f"Williams %R {wr:.1f} — oversold below -80, Buy zone"))
    elif wr > -20:
        indicators.append(("Williams %R", -1, "🔴", f"Williams %R {wr:.1f} — overbought above -20, Sell zone"))
    else:
        indicators.append(("Williams %R", 0, "⚪", f"Williams %R {wr:.1f} — neutral between -80 and -20"))

    roc = row["ROC"]
    if pd.isna(roc):
        indicators.append(("ROC (12)", 0, "⚪", "Unavailable — not enough data"))
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
    if score >= 7: return "Strong bullish heuristic score"
    if score >= 3: return "Bullish heuristic score"
    if score >= -2: return "Mixed heuristic score"
    if score >= -5: return "Bearish heuristic score"
    return "Strong bearish heuristic score"

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
    signals = []
    for name, col, lower, upper in [('RSI','RSI',30,70), ('Stoch RSI','StochRSI_k',20,80), ('Williams %R','WilliamsR',-80,-20)]:
        value = row[col]
        if pd.notna(value):
            state = 'oversold' if value < lower else 'overbought' if value > upper else 'mid-range'
            signals.append(('neutral','📊',name,f'{value:.1f}: {state}. This is a descriptive reading, not a probability of reversal.'))
    if pd.notna(row['EMA_200']):
        state = 'above' if price > row['EMA_200'] else 'below'
        signals.append(('neutral','📈','Selected asset trend',f'Completed close is {state} its 200-period EMA.'))
    sup,res = detect_major_sr_zones(df_long_coin,price)
    if sup or res:
        signals.append(('neutral','📊','Confirmed historical pivots',format_sr_block(sup,res,price)))
    return signals, '**Descriptive indicators only.** The Supertrend strategy is tested separately below. Bitcoin macro charts do not set entry or stop prices for other coins.', 0

def delta_metric(col, label, val):
    if val is None:
        col.metric(label, "N/A")
    else:
        col.metric(label, f"{val:+.2f}%", delta=f"{val:.2f}%")

def val_span(value, color):
    return f"<span style='font-size:0.85rem; color:{color}'>{value}</span>"

if "alerts" not in st.session_state:
    st.session_state.alerts = []


# ── MULTI-TIMEFRAME (CRYPTO) ──────────────────────────────────

import math as _math

def _fmt(val, decimals=1):
    if val is None or (isinstance(val, float) and (_math.isnan(val) or _math.isinf(val))):
        return "N/A"
    try:
        return f"{val:.{decimals}f}"
    except Exception:
        return "N/A"

def _build_multi_tf_crypto(df_daily: pd.DataFrame):
    """Return (df_table, numeric_vals, ctx) for Daily / Weekly / Monthly timeframes."""
    if df_daily is None or df_daily.empty:
        return pd.DataFrame(), {}, {}

    frames: dict = {}

    df_d = compute_indicators(df_daily.copy())
    df_d = df_d.dropna(subset=["RSI"])
    if not df_d.empty:
        frames["Daily"] = df_d

    df_w = completed_bars(df_daily, "Weekly", df_daily.attrs.get("asof"))
    if not df_w.empty:
        df_w = compute_indicators(df_w)
        df_w = df_w.dropna(subset=["RSI"])
        if not df_w.empty:
            frames["Weekly"] = df_w

    df_m = completed_bars(df_daily, "Monthly", df_daily.attrs.get("asof"))
    if not df_m.empty and len(df_m) >= 6:
        try:
            df_m = compute_indicators(df_m)
        except Exception:
            df_m = pd.DataFrame()
        if not df_m.empty:
            df_m = df_m.dropna(subset=["RSI"])
            if not df_m.empty:
                frames["Monthly"] = df_m

    if not frames:
        return pd.DataFrame(), {}, {}

    latest_rows = {name: f.iloc[-1] for name, f in frames.items()}
    row_labels = ["RSI (14)", "Stoch RSI %K", "MACD", "BB %B",
                  "EMA 50", "EMA 200", "ADX (14)", "CCI (20)", "Williams %R", "ROC (12)"]
    table_display = {lbl: {"Daily": "N/A", "Weekly": "N/A", "Monthly": "N/A"} for lbl in row_labels}
    numeric_vals  = {lbl: {"Daily": None,  "Weekly": None,  "Monthly": None}  for lbl in row_labels}
    ctx: dict = {}

    for tf, row in latest_rows.items():
        sc, bc, berc, inds = composite_score(row)
        ctx[tf] = {"score": sc, "label": signal_label(sc)}

        rsi   = float(row["RSI"])        if pd.notna(row.get("RSI"))        else None
        stoch = float(row["StochRSI_k"]) if pd.notna(row.get("StochRSI_k")) else None
        macd  = float(row["MACD"])       if pd.notna(row.get("MACD"))       else None
        bbp   = float(row["BB_pct"])     if pd.notna(row.get("BB_pct"))     else None
        e50   = float(row["EMA_50"])     if pd.notna(row.get("EMA_50"))     else None
        e200  = float(row["EMA_200"])    if pd.notna(row.get("EMA_200"))    else None
        adx   = float(row["ADX"])        if pd.notna(row.get("ADX"))        else None
        cci   = float(row["CCI"])        if pd.notna(row.get("CCI"))        else None
        wr    = float(row["WilliamsR"])  if pd.notna(row.get("WilliamsR"))  else None
        roc   = float(row["ROC"])        if pd.notna(row.get("ROC"))        else None

        numeric_vals["RSI (14)"][tf]     = rsi
        numeric_vals["Stoch RSI %K"][tf] = stoch
        numeric_vals["MACD"][tf]         = macd
        numeric_vals["BB %B"][tf]        = bbp * 100 if bbp is not None else None
        numeric_vals["EMA 50"][tf]       = e50
        numeric_vals["EMA 200"][tf]      = e200
        numeric_vals["ADX (14)"][tf]     = adx
        numeric_vals["CCI (20)"][tf]     = cci
        numeric_vals["Williams %R"][tf]  = wr
        numeric_vals["ROC (12)"][tf]     = roc

        table_display["RSI (14)"][tf]     = _fmt(rsi, 1)
        table_display["Stoch RSI %K"][tf] = _fmt(stoch, 1)
        table_display["MACD"][tf]         = _fmt(macd, 2)
        table_display["BB %B"][tf]        = _fmt(bbp * 100, 1) if bbp is not None else "N/A"
        table_display["EMA 50"][tf]       = _fmt(e50, 2)
        table_display["EMA 200"][tf]      = _fmt(e200, 2)
        table_display["ADX (14)"][tf]     = _fmt(adx, 1)
        table_display["CCI (20)"][tf]     = _fmt(cci, 0)
        table_display["Williams %R"][tf]  = _fmt(wr, 1)
        table_display["ROC (12)"][tf]     = _fmt(roc, 1)

    df_table = pd.DataFrame.from_dict(table_display, orient="index")
    for col in ["Daily", "Weekly", "Monthly"]:
        if col not in df_table.columns:
            df_table[col] = "N/A"
    df_table = df_table[["Daily", "Weekly", "Monthly"]]
    df_table.index.name = "Indicator"
    return df_table, numeric_vals, ctx


def _ind_color(indicator: str, value) -> str:
    if value is None or (isinstance(value, float) and _math.isnan(value)):
        return "color: #9ca3af"
    rules = {
        "RSI (14)":     (30, 70),
        "Stoch RSI %K": (20, 80),
        "BB %B":        (20, 80),
        "ADX (14)":     (15, 35),
        "CCI (20)":     (-100, 100),
        "Williams %R":  (-80, -20),
        "ROC (12)":     (-5, 5),
    }
    if indicator in rules:
        lo, hi = rules[indicator]
        if value <= lo: return "color: #22c55e"
        if value >= hi: return "color: #ef4444"
        return "color: #9ca3af"
    if indicator == "MACD":
        if value <= -0.5: return "color: #22c55e"
        if value >= 0.5:  return "color: #ef4444"
        return "color: #9ca3af"
    return "color: #9ca3af"


def _describe_bias(score: int) -> str:
    if score >= 3:  return "bullish"
    if score <= -3: return "bearish"
    return "neutral"


def render_multi_tf_crypto(coin_id: str, coin_ticker: str, coin_logo: str, price: float):
    with st.spinner("Loading long-term data for multi-timeframe view..."):
        df_daily = get_long_daily(coin_id)

    if df_daily.empty or len(df_daily) < 60:
        st.info("Not enough daily history for multi-timeframe view.")
        return

    price = float(df_daily.close.iloc[-1])
    df_table, numeric_vals, ctx = _build_multi_tf_crypto(df_daily)
    if df_table.empty or not ctx:
        st.info("Not enough indicator data for multi-timeframe view yet.")
        return

    logo_col, title_col = st.columns([0.4, 11])
    with logo_col:
        st.image(coin_logo, width=48)
    with title_col:
        st.markdown(f"#### {coin_ticker}  —  ${price:,.2f}")

    st.markdown("##### 📊 Multi-timeframe Technical Snapshot")
    st.caption("Daily / Weekly / Monthly indicator values from CryptoCompare long-term history.")

    def _style_col(col: pd.Series):
        tf = col.name
        styles = []
        for ind in df_table.index:
            val = numeric_vals.get(ind, {}).get(tf)
            styles.append(_ind_color(ind, val))
        return styles

    styled = df_table.style.apply(_style_col, axis=0)
    st.table(styled)

    st.markdown("##### 🧠 Swing Bias by Timeframe")
    d = ctx.get("Daily")
    w = ctx.get("Weekly")
    m = ctx.get("Monthly")
    if d: st.markdown(f"**Daily:** {d['label']}")
    if w: st.markdown(f"**Weekly:** {w['label']}")
    if m: st.markdown(f"**Monthly:** {m['label']}")

    if d and w and m:
        db = _describe_bias(d["score"])
        wb = _describe_bias(w["score"])
        mb = _describe_bias(m["score"])
        if db == wb == "bullish" and mb != "bearish":
            st.markdown("- **Bias: trend-following longs** — Daily and Weekly aligned bullish, Monthly at least neutral.")
            st.markdown("- Setup: Look for pullbacks toward EMA 50 or RSI < 45 for DCA entries.")
        elif db == "bullish" and wb != "bullish" and mb == "bearish":
            st.markdown("- **Bias: counter-trend long** — Daily strength against weak higher timeframes.")
            st.markdown("- Setup: Shorter holding periods; take profits into Weekly resistance.")
        elif db == wb == "bearish":
            st.markdown("- **Bias: avoid longs / reduce exposure** — Daily and Weekly both bearish.")
            st.markdown("- Setup: Wait for extreme oversold confluence (RSI < 30, Williams %R < -80) before any DCA.")
        else:
            st.markdown("- **Bias: mixed** — timeframes not aligned. Reduce position size and wait for confluence.")

    st.caption("Based on CryptoCompare daily OHLC history, resampled to Weekly and Monthly.")


# ── LOAD ALL DATA ──────────────────────────────────────────────

with st.spinner("Loading price data..."):
    market = get_market_data(coin_id)
    df     = get_long_daily(coin_id)

with st.spinner("Loading global sentiment..."):
    fg_df                          = get_fear_greed()
    btc_dom, total_mcap, alt_index = get_global_data()

with st.spinner("Loading long-term data..."):
    df_long     = get_long_daily(coin_id)
    df_btc_long = get_long_daily("bitcoin")

# ── MULTI-TIMEFRAME VIEW ───────────────────────────────────────
if view_mode == "Multi-timeframe – Crypto":
    price_preview = market.get("price", 0) or 0
    render_multi_tf_crypto(coin_id, coin_ticker, coin_logo, price_preview)
    st.stop()

if df.empty:
    st.error("No valid completed daily OHLC data is available.")
    st.stop()

cutoff = df.attrs.get('asof')
df = completed_bars(df, timeframe, cutoff)
df = core_supertrend(compute_indicators(df))
# Full history feeds indicators; only the visible/evaluation window is cropped.
window_start = pd.Timestamp.now(tz='UTC').tz_localize(None).normalize()-pd.Timedelta(days=days)
df = df[df.time >= window_start].reset_index(drop=True)
if len(df) < 2:
    st.info("Not enough completed candles in this window.")
    st.stop()
latest = df.iloc[-1]
price = float(latest['close'])
live_price = market.get('price')
st.caption(f"Analysis: completed {timeframe.lower()} candle dated {latest['time']:%d %b %Y}; CryptoCompare OHLC, USD. Current UTC day/week excluded.")
if live_price:
    st.caption(f"Live CoinGecko quote (separate feed): USD {live_price:,.4f}. Market statistics below are live.")
score, bullish_count, bearish_count, indicators = composite_score(latest)
neutral_count = 10 - bullish_count - bearish_count
label         = signal_label(score)

commentary_signals, conviction_summary, confluence_score = generate_commentary(
    latest, df_btc_long, df_long, price)

display_label = label
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
alert_key = (coin_id, timeframe, str(latest['time']), score)
if st.session_state.get('last_alert_key') != alert_key:
    st.session_state.alerts.insert(0, f"{ts} | {coin_label} ({timeframe}) | Candle {latest['time']:%Y-%m-%d} | Score: {score} | {display_label}")
    st.session_state.last_alert_key = alert_key
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
col4.metric("🎯 Signal Score", f"{score} pts (heuristic)")


st.divider()

# ── DCA COMMENTARY ────────────────────────────────────────────

st.markdown("### Indicator Context")
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
st.caption("Heuristic readings are not calibrated probabilities. Only the explicit Supertrend rules are included in the backtest.")

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
    trend_label, trend_color = detect_trend_structure(df, window=5)
else:
    trend_label, trend_color = detect_trend_structure(df, window=3)

st.markdown(f"#### 📐 Market Structure: {trend_label}")
st.caption("Detects Higher Highs & Higher Lows (uptrend), Lower Highs & Lower Lows (downtrend), or no clear structure.")

st.divider()

render_supertrend(df, "crypto", "USD")
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
    st.plotly_chart(fig1, width="stretch")
    st.caption("RSI <30 = strongly oversold 🟢; >70 = overbought 🔴. Stoch K <20 = oversold reading; >80 = overbought reading.")

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
    st.plotly_chart(fig2, width="stretch")
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
    st.plotly_chart(fig3, width="stretch")
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
    st.plotly_chart(fig_adx, width="stretch")
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
    st.plotly_chart(fig_cci, width="stretch")
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
    st.plotly_chart(fig_wr, width="stretch")
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
    st.plotly_chart(fig_roc, width="stretch")
    st.caption("ROC above +5% = strong positive momentum 🟢. Below -5% = strong negative momentum 🔴. Near 0 = stalling.")

with row4_r:
    st.markdown(
        f"#### 📏 ATR — Volatility &nbsp; {val_span('ATR: ' + atr_val, '#C084FC')}",
        unsafe_allow_html=True)
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=df["time"], y=df["ATR"],
        name="ATR", line=dict(color="#C084FC", width=2)))
    fig_atr.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_atr, width="stretch")
    st.caption("ATR rising = increasing volatility. Use for stop-loss sizing — place stops 1.5–2× ATR from entry.")

st.divider()

# ── GRAPHS — ROW 5: Squeeze (approximation)  |  Fear & Greed ─────────────

row5_l, row5_r = st.columns(2)

with row5_l:
    st.markdown(
        f"#### 🔫 Squeeze (approximation) &nbsp; {val_span('Squeeze: ' + sq_val, sq_color)}",
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
    st.plotly_chart(fig_sq, width="stretch")
    st.caption("Approximate compression and smoothed momentum. Black = compression on; lime = off. Only the on-to-off transition is a new release.")

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
        st.plotly_chart(fig_fg, width="stretch")
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
            st.plotly_chart(fig_pi, width="stretch")
            last_pi = df_pi.iloc[-1]
            if pd.notna(last_pi.get("MA_350x2")):
                gap_pct = ((last_pi["MA_350x2"] - last_pi["MA_111"]) / last_pi["MA_350x2"]) * 100
                if last_pi["MA_111"] >= last_pi["MA_350x2"]:
                    st.markdown("⚠️ **Pi Cycle TOP signal ACTIVE** — moving-average condition met; not a validated exit rule in this app.")
                else:
                    st.markdown(f"✅ **No Pi Cycle top.** 111-day MA is {gap_pct:.1f}% below 2× 350-day MA.")
            else:
                st.markdown("⏳ 350-day MA still building history.")
    else:
        st.info("Not enough BTC history for Pi Cycle.")
    st.caption("BTC 111-day and 2×350-day moving averages; not part of the Supertrend strategy.")

    st.markdown("#### 🧱 Bear Market Support / Resistance Band (BTC)")
    if not df_btc_long.empty and len(df_btc_long) >= 200:
        df_w_band = completed_bars(df_btc_long, "Weekly", df_btc_long.attrs.get("asof"))
        df_w_band["SMA_20w"] = df_w_band["close"].rolling(20).mean()
        df_w_band["EMA_21w"] = df_w_band["close"].ewm(span=21, adjust=False).mean()
        df_w_band_valid = df_w_band.dropna(subset=["SMA_20w", "EMA_21w"])
        if not df_w_band_valid.empty:
            fig_band = go.Figure()
            fig_band.add_trace(go.Candlestick(
                x=df_w_band["time"],
                open=df_w_band["open"], high=df_w_band["high"],
                low=df_w_band["low"], close=df_w_band["close"],
                name="BTC Weekly"
            ))
            fig_band.add_trace(go.Scatter(
                x=df_w_band_valid["time"], y=df_w_band_valid["SMA_20w"],
                name="20W SMA",
                line=dict(color="#60A5FA", width=2)
            ))
            fig_band.add_trace(go.Scatter(
                x=df_w_band_valid["time"], y=df_w_band_valid["EMA_21w"],
                name="21W EMA",
                line=dict(color="#F97316", width=2),
                fill="tonexty",
                fillcolor="rgba(96,165,250,0.18)"
            ))
            fig_band.update_layout(
                height=400,
                margin=dict(t=30, b=10),
                xaxis_rangeslider_visible=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_band, width="stretch")
            last_row = df_w_band_valid.iloc[-1]
            band_mid = (last_row["SMA_20w"] + last_row["EMA_21w"]) / 2
            pct_off  = ((last_row["close"] - band_mid) / band_mid) * 100
            dir_txt  = "above" if pct_off > 0 else "below"
            st.markdown(
                f"BTC weekly close is **{abs(pct_off):.1f}% {dir_txt}** the 20W/21W band midpoint. "
                "Above = band tends to act as support in bull markets; below = resistance in bear markets."
            )
    else:
        st.info("Not enough BTC history for 20W/21W support band.")

with row6_r:
    st.markdown("#### 📅 200-Week MA (BTC Long-term Reference)")
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
            st.plotly_chart(fig_200w, width="stretch")
            if pct_above < 0:
                st.markdown(f"🟢 **BTC is {abs(pct_above):.1f}% BELOW the 200-week MA** — historical reference level; not a guaranteed floor.")
            else:
                st.markdown(f"📊 BTC is **{pct_above:.1f}% above** the 200-week MA (${current_200w:,.0f}).")
    else:
        st.info("Not enough BTC history for 200-week MA.")
    st.caption("200-week mean of completed BTC weekly closes. A descriptive reference, not a guaranteed floor or entry signal.")

st.divider()

# ── ALERT HISTORY ─────────────────────────────────────────────

st.markdown("#### 🔔 Alert History (this session)")
for alert in st.session_state.alerts:
    st.markdown(f"- {alert}")
if not st.session_state.alerts:
    st.info("No alerts yet.")
st.caption("🔔 Logged each refresh. Resets on app restart.")

st.caption(f"Data: CoinGecko · CryptoCompare · alternative.me · Auto-refreshes every 60s · Last run: {ts}")
