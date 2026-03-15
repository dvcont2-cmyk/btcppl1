import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide", page_title="ASX Stock Dashboard")

st_autorefresh(interval=300_000, key="asx_refresh")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1rem !important; font-weight: 600; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
[data-testid="stMetric"]      { padding: 4px 0px !important; }
.block-container              { padding-top: 2.75rem !important; }
</style>
""", unsafe_allow_html=True)

ASX_STOCKS = {
    "Paladin Energy (PDN)":     "PDN.AX",
    "Core Lithium (CXO)":       "CXO.AX",
    "Eclipse Metals (EPM)":     "EPM.AX",
    "Northern Star (NST)":      "NST.AX",
    "Pilbara Minerals (PLS)":   "PLS.AX",
    "Vanguard AU Shares (VAS)": "VAS.AX",
    "Vanguard Intl (VGS)":      "VGS.AX",
    "Vanguard Div HG (VDHG)":   "VDHG.AX",
    "Betashares Div HG (DHHF)": "DHHF.AX",
    "Vanguard EM (VGE)":        "VGE.AX",
}

STOCK_NOTES = {
    "PDN.AX":  "Uranium miner — high beta, sentiment-driven. Watch uranium spot price as leading signal.",
    "CXO.AX":  "Lithium explorer — speculative, thin liquidity. EV demand & lithium price are key macro drivers.",
    "EPM.AX":  "Junior explorer — very low liquidity. TA signals less reliable; treat as speculative only.",
    "NST.AX":  "Mid-cap gold miner — tracks gold price closely. Safe-haven flows & AUD/USD matter.",
    "PLS.AX":  "Lithium producer — more liquid than CXO. Tied to lithium carbonate spot & EV demand.",
    "VAS.AX":  "ASX 200 ETF — broad market exposure. TA best used for market timing, not stock picking.",
    "VGS.AX":  "Global ex-AU ETF — tracks MSCI World. USD/AUD FX has significant impact on returns.",
    "VDHG.AX": "Diversified high growth ETF — slow mover, best for long-term DCA. TA less meaningful.",
    "DHHF.AX": "100% equities diversified ETF — similar to VDHG. Swing trading not ideal; DCA vehicle.",
    "VGE.AX":  "Emerging markets ETF — higher volatility than VGS. Geopolitical risk & USD are key.",
}

SWING_TRADE_SUITABLE = {"PDN.AX", "CXO.AX", "NST.AX", "PLS.AX", "EPM.AX"}

st.sidebar.header("Settings")
stock_label = st.sidebar.selectbox("Select Stock / ETF", list(ASX_STOCKS.keys()))
ticker      = ASX_STOCKS[stock_label]
timeframe   = st.sidebar.radio("Timeframe", ["Daily", "Weekly"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Chart Range**")
range_option = st.sidebar.selectbox(
    "Quick Select",
    ["3 Months", "6 Months", "1 Year", "2 Years", "3 Years", "5 Years", "Custom"],
    index=3
)

range_map = {
    "3 Months": 90,
    "6 Months": 180,
    "1 Year":   365,
    "2 Years":  730,
    "3 Years":  1095,
    "5 Years":  1825,
}

if range_option == "Custom":
    col_s, col_e = st.sidebar.columns(2)
    date_start = col_s.date_input("From", value=date.today() - timedelta(days=730))
    date_end   = col_e.date_input("To",   value=date.today())
else:
    date_end   = date.today()
    date_start = date_end - timedelta(days=range_map[range_option])

if st.sidebar.button("Force Refresh"):
    st.cache_data.clear()
    st.rerun()


def val_span(value, color):
    return f"<span style='font-size:0.85rem; color:{color}'>{value}</span>"


def status_line(message, sentiment):
    colors = {
        "bullish":        ("#86efac", "🟢"),
        "bearish":        ("#f87171", "🔴"),
        "neutral":        ("#9ca3af", "⚪"),
        "caution":        ("#fbbf24", "🟡"),
        "strong_bullish": ("#22c55e", "🟢"),
        "strong_bearish": ("#ef4444", "🔴"),
    }
    color, emoji = colors.get(sentiment, ("#9ca3af", "⚪"))
    st.markdown(
        f"<p style='font-size:0.82rem; font-style:italic; color:{color}; margin-top:-8px'>"
        f"{emoji} {message}</p>",
        unsafe_allow_html=True)


@st.cache_data(ttl=300)
def get_stock_data(ticker, start_date, end_date, interval):
    try:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False
        )
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        df = df.rename(columns={"date": "time", "datetime": "time"})
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_stock_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "price":       info.get("currentPrice") or info.get("regularMarketPrice"),
            "change_1d":   info.get("regularMarketChangePercent"),
            "week52_high": info.get("fiftyTwoWeekHigh"),
            "week52_low":  info.get("fiftyTwoWeekLow"),
            "market_cap":  info.get("marketCap"),
            "volume":      info.get("regularMarketVolume"),
            "avg_volume":  info.get("averageVolume"),
            "pe_ratio":    info.get("trailingPE"),
            "sector":      info.get("sector", ""),
            "industry":    info.get("industry", ""),
            "name":        info.get("longName", ticker),
        }
    except:
        return {}


def compute_indicators(df):
    df = df.copy()
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
    kc_mid            = df["close"].rolling(20).mean()
    atr20             = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=20).average_true_range()
    kc_upper          = kc_mid + 1.5 * atr20
    kc_lower          = kc_mid - 1.5 * atr20
    df["squeeze"]     = (bb.bollinger_hband() < kc_upper) & (bb.bollinger_lband() > kc_lower)
    delta             = df["close"] - ((df["high"].rolling(20).max() + df["low"].rolling(20).min()) / 2 + df["close"].rolling(20).mean()) / 2
    df["squeeze_hist"] = delta.rolling(20).mean()
    return df


def compute_supertrend(df, period, multiplier):
    atr    = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range()
    hl_avg = (df["high"] + df["low"]) / 2
    upper_band = (hl_avg + multiplier * atr).copy()
    lower_band = (hl_avg - multiplier * atr).copy()

    supertrend = [float("nan")] * len(df)
    direction  = [0] * len(df)

    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i]):
            continue
        if lower_band.iloc[i] < lower_band.iloc[i-1]:
            lower_band.iloc[i] = lower_band.iloc[i-1]
        if upper_band.iloc[i] > upper_band.iloc[i-1]:
            upper_band.iloc[i] = upper_band.iloc[i-1]
        if df["close"].iloc[i] > upper_band.iloc[i-1]:
            direction[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        supertrend[i] = lower_band.iloc[i] if direction[i] == 1 else upper_band.iloc[i]

    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)


def triple_supertrend_signals(df):
    df = df.copy()
    df["st1"], df["dir1"] = compute_supertrend(df, period=7,  multiplier=3.0)
    df["st2"], df["dir2"] = compute_supertrend(df, period=14, multiplier=2.0)
    df["st3"], df["dir3"] = compute_supertrend(df, period=21, multiplier=1.0)

    df["buy_signal"]  = None
    df["sell_signal"] = None

    for i in range(1, len(df)):
        c  = df["close"].iloc[i]
        pc = df["close"].iloc[i-1]
        cross_up = (
            (pc <= df["st1"].iloc[i-1] and c > df["st1"].iloc[i]) or
            (pc <= df["st2"].iloc[i-1] and c > df["st2"].iloc[i]) or
            (pc <= df["st3"].iloc[i-1] and c > df["st3"].iloc[i])
        )
        cross_dn = (
            (pc >= df["st1"].iloc[i-1] and c < df["st1"].iloc[i]) or
            (pc >= df["st2"].iloc[i-1] and c < df["st2"].iloc[i]) or
            (pc >= df["st3"].iloc[i-1] and c < df["st3"].iloc[i])
        )
        if cross_up:
            df.at[df.index[i], "buy_signal"]  = c
        if cross_dn:
            df.at[df.index[i], "sell_signal"] = c

    return df


def detect_trend_structure(df, window=3):
    closes = df["close"].values
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
        if hh and hl:
            return "🟢 Uptrend - Higher Highs & Higher Lows", "green"
        elif lh and ll:
            return "🔴 Downtrend - Lower Highs & Lower Lows", "red"
        else:
            return "⚪ Choppy - No clear structure", "gray"
    return "⚪ Not enough swing points yet", "gray"


def detect_support_resistance(df, window=3, num_levels=4):
    highs = df["high"] if "high" in df.columns else df["close"]
    lows  = df["low"]  if "low"  in df.columns else df["close"]
    resistance, support = [], []
    for i in range(window, len(df) - window):
        if all(highs.iloc[i] >= highs.iloc[i-window:i]) and            all(highs.iloc[i] >= highs.iloc[i+1:i+window+1]):
            resistance.append(float(highs.iloc[i]))
        if all(lows.iloc[i] <= lows.iloc[i-window:i]) and            all(lows.iloc[i] <= lows.iloc[i+1:i+window+1]):
            support.append(float(lows.iloc[i]))
    def cluster(levels, pct=0.02):
        levels = sorted(l for l in set(levels) if l > 0)
        clustered = []
        for l in levels:
            if not clustered or abs(l - clustered[-1]) / clustered[-1] > pct:
                clustered.append(l)
        return clustered
    return cluster(support)[:num_levels], cluster(resistance)[-num_levels:]


def composite_score(row):
    indicators = []
    close = row["close"]

    rsi = row["RSI"]
    if pd.isna(rsi):
        indicators.append(("RSI (14)", 0, "N", "Neutral - not enough data"))
    elif rsi < 30:
        indicators.append(("RSI (14)", 2, "B", f"Strongly oversold at {rsi:.1f}"))
    elif rsi < 40:
        indicators.append(("RSI (14)", 1, "B", f"Oversold at {rsi:.1f}"))
    elif rsi > 70:
        indicators.append(("RSI (14)", -1, "S", f"Overbought at {rsi:.1f}"))
    else:
        indicators.append(("RSI (14)", 0, "N", f"Neutral at {rsi:.1f}"))

    k = row["StochRSI_k"]
    if pd.isna(k):
        indicators.append(("Stoch RSI", 0, "N", "Neutral - not enough data"))
    elif k < 20:
        indicators.append(("Stoch RSI", 1, "B", f"Oversold at {k:.1f}"))
    elif k > 80:
        indicators.append(("Stoch RSI", -1, "S", f"Overbought at {k:.1f}"))
    else:
        indicators.append(("Stoch RSI", 0, "N", f"Neutral at {k:.1f}"))

    macd_val, macd_sig = row["MACD"], row["MACD_signal"]
    if pd.isna(macd_val) or pd.isna(macd_sig):
        indicators.append(("MACD", 0, "N", "Neutral - not enough data"))
    elif macd_val > macd_sig:
        indicators.append(("MACD", 1, "B", f"Bullish - MACD ({macd_val:.4f}) above signal ({macd_sig:.4f})"))
    else:
        indicators.append(("MACD", -1, "S", f"Bearish - MACD ({macd_val:.4f}) below signal ({macd_sig:.4f})"))

    bb_pct   = row["BB_pct"]
    bb_lower = row["BB_lower"]
    bb_upper = row["BB_upper"]
    if pd.isna(bb_pct):
        indicators.append(("Bollinger Bands", 0, "N", "Neutral - not enough data"))
    elif close < bb_lower:
        indicators.append(("Bollinger Bands", 1, "B", "Price below lower band - oversold"))
    elif close > bb_upper:
        indicators.append(("Bollinger Bands", -1, "S", "Price above upper band - overbought"))
    elif bb_pct < 0.2:
        indicators.append(("Bollinger Bands", 1, "B", f"Near lower band ({bb_pct*100:.0f}%)"))
    elif bb_pct > 0.8:
        indicators.append(("Bollinger Bands", -1, "S", f"Near upper band ({bb_pct*100:.0f}%)"))
    else:
        indicators.append(("Bollinger Bands", 0, "N", f"Mid-band ({bb_pct*100:.0f}%)"))

    ema200 = row["EMA_200"]
    if pd.isna(ema200):
        indicators.append(("EMA 200 Trend", 0, "N", "Neutral - not enough data"))
    elif close > ema200:
        indicators.append(("EMA 200 Trend", 2, "B", f"Price above EMA 200 (${ema200:,.4f})"))
    else:
        indicators.append(("EMA 200 Trend", -2, "S", f"Price below EMA 200 (${ema200:,.4f})"))

    ema50 = row["EMA_50"]
    if pd.isna(ema50) or pd.isna(ema200):
        indicators.append(("EMA 50/200 Cross", 0, "N", "Neutral - not enough data"))
    elif ema50 > ema200:
        indicators.append(("EMA 50/200 Cross", 2, "B", "Golden Cross - EMA 50 above EMA 200"))
    else:
        indicators.append(("EMA 50/200 Cross", -2, "S", "Death Cross - EMA 50 below EMA 200"))

    adx, adx_pos, adx_neg = row["ADX"], row["ADX_pos"], row["ADX_neg"]
    if pd.isna(adx):
        indicators.append(("ADX Strength", 0, "N", "Neutral - not enough data"))
    elif adx < 20:
        indicators.append(("ADX Strength", 0, "N", f"ADX {adx:.1f} - weak trend"))
    elif adx_pos > adx_neg:
        indicators.append(("ADX Strength", 1, "B", f"ADX {adx:.1f} - bullish direction"))
    else:
        indicators.append(("ADX Strength", -1, "S", f"ADX {adx:.1f} - bearish direction"))

    cci = row["CCI"]
    if pd.isna(cci):
        indicators.append(("CCI (20)", 0, "N", "Neutral - not enough data"))
    elif cci < -100:
        indicators.append(("CCI (20)", 1, "B", f"CCI {cci:.0f} - oversold"))
    elif cci > 100:
        indicators.append(("CCI (20)", -1, "S", f"CCI {cci:.0f} - overbought"))
    else:
        indicators.append(("CCI (20)", 0, "N", f"CCI {cci:.0f} - neutral"))

    wr = row["WilliamsR"]
    if pd.isna(wr):
        indicators.append(("Williams %R", 0, "N", "Neutral - not enough data"))
    elif wr < -80:
        indicators.append(("Williams %R", 1, "B", f"Williams %R {wr:.1f} - oversold"))
    elif wr > -20:
        indicators.append(("Williams %R", -1, "S", f"Williams %R {wr:.1f} - overbought"))
    else:
        indicators.append(("Williams %R", 0, "N", f"Williams %R {wr:.1f} - neutral"))

    roc = row["ROC"]
    if pd.isna(roc):
        indicators.append(("ROC (12)", 0, "N", "Neutral - not enough data"))
    elif roc > 5:
        indicators.append(("ROC (12)", 1, "B", f"ROC {roc:.1f}% - strong positive momentum"))
    elif roc < -5:
        indicators.append(("ROC (12)", -1, "S", f"ROC {roc:.1f}% - strong negative momentum"))
    else:
        indicators.append(("ROC (12)", 0, "N", f"ROC {roc:.1f}% - momentum flat"))

    bullish_count = sum(1 for _, s, _, _ in indicators if s > 0)
    bearish_count = sum(1 for _, s, _, _ in indicators if s < 0)
    net_score     = sum(s for _, s, _, _ in indicators)
    return net_score, bullish_count, bearish_count, indicators


def signal_label(score):
    if score >= 7:
        return "🟢 STRONG BUY"
    elif score >= 3:
        return "🟡 WATCH / ACCUMULATE"
    elif score >= -2:
        return "⚪ HOLD / NEUTRAL"
    elif score >= -5:
        return "🟠 CAUTION / REDUCE"
    else:
        return "🔴 STRONG SELL / AVOID"


def swing_trade_commentary(row, df, ticker, price, score, info):
    atr       = row["ATR"] if pd.notna(row["ATR"]) else price * 0.03
    ema50     = row["EMA_50"]
    ema200    = row["EMA_200"]
    rsi       = row["RSI"]
    macd_hist = row["MACD_hist"]
    stoch_k   = row["StochRSI_k"]
    wr        = row["WilliamsR"]
    adx       = row["ADX"]
    adx_pos   = row["ADX_pos"]
    adx_neg   = row["ADX_neg"]
    squeeze   = row["squeeze"]
    week52_high = info.get("week52_high")
    week52_low  = info.get("week52_low")

    support_levels, resistance_levels = detect_support_resistance(df)
    nearest_support    = max([s for s in support_levels if s < price], default=price * 0.92)
    nearest_resistance = min([r for r in resistance_levels if r > price], default=price * 1.10)

    signals = []
    setup   = "none"

    golden_cross   = pd.notna(ema50) and pd.notna(ema200) and ema50 > ema200
    death_cross    = pd.notna(ema50) and pd.notna(ema200) and ema50 < ema200
    above_ema200   = pd.notna(ema200) and price > ema200
    rsi_oversold   = pd.notna(rsi) and rsi < 40
    rsi_extreme    = pd.notna(rsi) and rsi < 30
    rsi_overbought = pd.notna(rsi) and rsi > 70
    macd_bullish   = pd.notna(macd_hist) and macd_hist > 0
    stoch_low      = pd.notna(stoch_k) and stoch_k < 20
    stoch_high     = pd.notna(stoch_k) and stoch_k > 80
    wr_oversold    = pd.notna(wr) and wr < -80
    strong_trend   = pd.notna(adx) and adx > 25
    trending_up    = strong_trend and pd.notna(adx_pos) and pd.notna(adx_neg) and adx_pos > adx_neg

    if golden_cross and above_ema200 and rsi_oversold and macd_bullish:
        stop   = nearest_support - atr * 0.5
        target = nearest_resistance
        rr     = (target - price) / (price - stop) if (price - stop) > 0 else 0
        setup  = "pullback_buy"
        signals.append((
            "🟢 SWING BUY - Pullback in Uptrend",
            f"Golden Cross active + price above EMA 200. RSI ({rsi:.1f}) has pulled back to oversold "
            f"while MACD histogram turned positive - classic dip-buy in a healthy uptrend.",
            f"Entry: ${price:,.4f} | Stop: ${stop:,.4f} (-{((price-stop)/price)*100:.1f}%) "
            f"| Target: ${target:,.4f} (+{((target-price)/price)*100:.1f}%) | R:R {rr:.1f}x",
            "green"
        ))
    elif rsi_extreme and wr_oversold and stoch_low:
        stop   = price - atr * 1.5
        target = ema50 if pd.notna(ema50) and ema50 > price else nearest_resistance
        rr     = (target - price) / (price - stop) if (price - stop) > 0 else 0
        setup  = "oversold_reversal"
        signals.append((
            "🟡 SWING WATCH - Extreme Oversold Reversal",
            f"Triple oversold confluence: RSI ({rsi:.1f}) below 30, Williams %R ({wr:.1f}) below -80, "
            f"Stoch RSI K ({stoch_k:.1f}) below 20. High probability of a short-term bounce. "
            f"{'Caution - Death Cross in effect, keep size small.' if death_cross else 'No Death Cross - more reliable signal.'}",
            f"Entry: ${price:,.4f} | Stop: ${stop:,.4f} (-{((price-stop)/price)*100:.1f}%) "
            f"| Target: ${target:,.4f} (+{((target-price)/price)*100:.1f}%) | R:R {rr:.1f}x",
            "yellow"
        ))
    elif trending_up and macd_bullish and not rsi_overbought and squeeze == False:
        stop   = ema50 if pd.notna(ema50) else price - atr * 2
        target = nearest_resistance
        rr     = (target - price) / (price - stop) if (price - stop) > 0 else 0
        setup  = "momentum_breakout"
        signals.append((
            "🟢 SWING BUY - Momentum Breakout",
            f"TTM Squeeze released. ADX ({adx:.1f}) confirms strong trend with +DI ({adx_pos:.1f}) "
            f"above -DI ({adx_neg:.1f}). MACD confirming. Trail stop rather than fixed target.",
            f"Entry: ${price:,.4f} | Stop: ${stop:,.4f} (-{((price-stop)/price)*100:.1f}%) "
            f"| First Target: ${target:,.4f} (+{((target-price)/price)*100:.1f}%) | R:R {rr:.1f}x",
            "green"
        ))
    elif death_cross and rsi_overbought and stoch_high:
        setup = "overbought_short"
        signals.append((
            "🔴 CAUTION - Overbought in Downtrend",
            f"Death Cross active + RSI ({rsi:.1f}) overbought + Stoch RSI K ({stoch_k:.1f}) above 80. "
            f"High risk of reversal down. Not a long entry. Consider reducing existing position.",
            "Watch for RSI to fall below 65 and MACD to cross bearish before reassessing.",
            "red"
        ))
    else:
        setup = "none"
        if score >= 0:
            signals.append((
                "⚪ NO CLEAR SWING SETUP - Leaning Neutral/Bullish",
                f"Indicators mixed or consolidating. No high-conviction entry signal. "
                f"{'Golden Cross backdrop - wait for dip to EMA 50 or RSI < 45.' if golden_cross else 'Watch for Golden Cross or oversold bounce to form a setup.'}",
                "Monitor for RSI dip below 40 or TTM Squeeze firing to trigger entry.",
                "gray"
            ))
        else:
            signals.append((
                "🟠 NO CLEAR SWING SETUP - Leaning Bearish",
                f"More bearish signals than bullish but no clean reversal setup yet. "
                f"{'Death Cross active - downtrend bias.' if death_cross else 'Trend weakening.'} "
                f"Avoid new longs until RSI < 30 and MACD turns positive.",
                "Wait for extreme oversold confluence before considering entry.",
                "orange"
            ))

    if week52_high and week52_low and week52_high != week52_low:
        pct_from_high = ((price - week52_high) / week52_high) * 100
        pct_from_low  = ((price - week52_low)  / week52_low)  * 100
        range_pos     = ((price - week52_low) / (week52_high - week52_low)) * 100
        signals.append((
            "📅 52-Week Range Context",
            f"52w High: ${week52_high:,.4f} | 52w Low: ${week52_low:,.4f} | Position in range: {range_pos:.0f}%",
            f"Price is {abs(pct_from_high):.1f}% below 52w high and {abs(pct_from_low):.1f}% above 52w low. "
            f"{'Near lows - potential value zone.' if range_pos < 30 else 'Near highs - higher risk for new entries.' if range_pos > 70 else 'Mid-range - no particular edge from price position alone.'}",
            "blue"
        ))

    signals.append((
        "📏 ATR Volatility - Position Sizing",
        f"ATR (14): ${atr:,.4f} | Stop at 1.5x ATR: ${atr*1.5:,.4f} | Stop at 2x ATR: ${atr*2:,.4f}",
        f"For 1% portfolio risk at 1.5x ATR stop: position size approx {(0.01 / ((atr * 1.5) / price)) * 100:.1f}% of capital.",
        "purple"
    ))

    return signals, setup


interval = "1d" if timeframe == "Daily" else "1wk"

with st.spinner(f"Loading {ticker} data..."):
    df   = get_stock_data(ticker, date_start, date_end, interval)
    info = get_stock_info(ticker)

if df.empty or len(df) < 30:
    st.error(f"Not enough data for {ticker}. Try a wider date range or force refresh.")
    st.stop()

df     = compute_indicators(df)
df     = df.dropna(subset=["RSI"]).reset_index(drop=True)
df     = triple_supertrend_signals(df)
latest = df.iloc[-1]

price         = info.get("price") or float(latest["close"])
score, bullish_count, bearish_count, indicators = composite_score(latest)
neutral_count = 10 - bullish_count - bearish_count
label         = signal_label(score)
ts            = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

swing_signals, setup_type = swing_trade_commentary(latest, df, ticker, price, score, info)
trend_label, trend_color  = detect_trend_structure(df, window=3)

rsi_val    = f"{latest['RSI']:.1f}"          if pd.notna(latest['RSI'])         else "N/A"
stoch_val  = f"{latest['StochRSI_k']:.1f}"   if pd.notna(latest['StochRSI_k'])  else "N/A"
stochd_val = f"{latest['StochRSI_d']:.1f}"   if pd.notna(latest['StochRSI_d'])  else "N/A"
macd_val   = f"{latest['MACD']:.4f}"         if pd.notna(latest['MACD'])        else "N/A"
msig_val   = f"{latest['MACD_signal']:.4f}"  if pd.notna(latest['MACD_signal']) else "N/A"
mhst_val   = f"{latest['MACD_hist']:.4f}"    if pd.notna(latest['MACD_hist'])   else "N/A"
bb_pct_val = f"{latest['BB_pct']*100:.0f}%"  if pd.notna(latest['BB_pct'])      else "N/A"
ema50_val  = f"${latest['EMA_50']:,.4f}"      if pd.notna(latest['EMA_50'])      else "N/A"
ema200_val = f"${latest['EMA_200']:,.4f}"     if pd.notna(latest['EMA_200'])     else "N/A"
adx_val    = f"{latest['ADX']:.1f}"          if pd.notna(latest['ADX'])         else "N/A"
dip_val    = f"{latest['ADX_pos']:.1f}"      if pd.notna(latest['ADX_pos'])     else "N/A"
din_val    = f"{latest['ADX_neg']:.1f}"      if pd.notna(latest['ADX_neg'])     else "N/A"
cci_val    = f"{latest['CCI']:.1f}"          if pd.notna(latest['CCI'])         else "N/A"
wr_val     = f"{latest['WilliamsR']:.1f}"    if pd.notna(latest['WilliamsR'])   else "N/A"
roc_val    = f"{latest['ROC']:.2f}%"         if pd.notna(latest['ROC'])         else "N/A"
atr_val    = f"${latest['ATR']:,.4f}"         if pd.notna(latest['ATR'])         else "N/A"
sq_val     = "ON" if latest["squeeze"] else "OFF"
sq_color   = "red" if latest["squeeze"] else "lime"

is_swing_suitable = ticker in SWING_TRADE_SUITABLE

# ── HEADER ─────────────────────────────────────────────────────

st.markdown("#### ASX Stock Technical Dashboard")
st.caption(f"Swing trade focused analysis | {timeframe} | {date_start.strftime('%d %b %Y')} to {date_end.strftime('%d %b %Y')}")
st.divider()

st.markdown(
    f"<div style='display:flex; align-items:center; gap:10px; margin-bottom:4px'>"
    f"<h4 style='margin:0'>{info.get('name', stock_label)} ({ticker}) &nbsp; "
    f"${price:,.4f} &nbsp; {label}</h4></div>",
    unsafe_allow_html=True)

if info.get("sector"):
    st.caption(f"Sector: {info.get('sector')} | {info.get('industry')}")

if is_swing_suitable:
    st.success("Swing Trade Suitable - sufficient liquidity and volatility for active swing trading.")
else:
    st.info("ETF / Low-liquidity stock - better suited to DCA / long-term holding than active swing trading.")

note = STOCK_NOTES.get(ticker, "")
if note:
    st.markdown(f"> {note}")

st.markdown("#### Key Metrics")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Price",     f"${price:,.4f}")
m2.metric("RSI (14)",  rsi_val)
m3.metric("MACD Hist", mhst_val)
m4.metric("Score",     f"{score} pts")
if info.get("week52_high"): m5.metric("52w High", f"${info['week52_high']:,.4f}")
if info.get("week52_low"):  m6.metric("52w Low",  f"${info['week52_low']:,.4f}")

p1, p2, p3, p4 = st.columns(4)
if info.get("market_cap"):  p1.metric("Market Cap", f"${info['market_cap']/1e6:.0f}M")
if info.get("volume"):      p2.metric("Volume",     f"{info['volume']:,.0f}")
if info.get("avg_volume"):  p3.metric("Avg Volume", f"{info['avg_volume']:,.0f}")
if info.get("pe_ratio"):    p4.metric("P/E Ratio",  f"{info['pe_ratio']:.1f}x")

st.divider()
st.markdown(f"#### Market Structure: {trend_label}")
st.caption("Based on recent swing highs and lows.")
st.divider()

# ── SWING TRADE ANALYSIS ───────────────────────────────────────

st.markdown("### Swing Trade Analysis")

signal_color_map = {
    "green":  "#22c55e",
    "yellow": "#fbbf24",
    "red":    "#f87171",
    "gray":   "#9ca3af",
    "orange": "#f97316",
    "blue":   "#60a5fa",
    "purple": "#c084fc",
}

for title, detail, levels, color_key in swing_signals:
    color = signal_color_map.get(color_key, "#9ca3af")
    st.markdown(
        f"<div style='border-left: 4px solid {color}; padding: 10px 16px; "
        f"margin-bottom: 12px; background: rgba(255,255,255,0.03); border-radius: 4px;'>"
        f"<strong style='color:{color}'>{title}</strong><br>"
        f"<span style='font-size:0.9rem; color:#d1d5db'>{detail}</span><br>"
        f"<span style='font-size:0.82rem; color:#9ca3af; margin-top:4px; display:block'>{levels}</span>"
        f"</div>",
        unsafe_allow_html=True)

st.caption("Not financial advice. All signals are algorithmic. Confirm setups with volume and broader market context.")
st.divider()

with st.expander("All 10 Indicators - Signal Breakdown", expanded=False):
    bullish_list = [(n, d) for n, s, e, d in indicators if s > 0]
    neutral_list = [(n, d) for n, s, e, d in indicators if s == 0]
    bearish_list = [(n, d) for n, s, e, d in indicators if s < 0]
    col_b, col_n, col_r = st.columns(3)
    with col_b:
        st.markdown(f"**Bullish ({bullish_count})**")
        for name, desc in bullish_list:
            st.markdown(f"- **{name}**: {desc}")
    with col_n:
        st.markdown(f"**Neutral ({neutral_count})**")
        for name, desc in neutral_list:
            st.markdown(f"- **{name}**: {desc}")
    with col_r:
        st.markdown(f"**Bearish ({bearish_count})**")
        for name, desc in bearish_list:
            st.markdown(f"- **{name}**: {desc}")

st.divider()

# ── TRIPLE SUPERTREND ──────────────────────────────────────────

st.markdown("### 📊 Triple Supertrend Crossover")

bulls = sum(1 for d in [latest["dir1"], latest["dir2"], latest["dir3"]] if d == 1)
bears = sum(1 for d in [latest["dir1"], latest["dir2"], latest["dir3"]] if d == -1)

def st_badge(direction):
    if direction == 1:
        return val_span("▲ Bull", "#22c55e")
    else:
        return val_span("▼ Bear", "#ef4444")

st.markdown(
    f"<p style='font-size:0.9rem; margin-bottom:2px'>"
    f"ST1 (7, 3.0): {st_badge(latest['dir1'])} &nbsp;&nbsp; "
    f"ST2 (14, 2.0): {st_badge(latest['dir2'])} &nbsp;&nbsp; "
    f"ST3 (21, 1.0): {st_badge(latest['dir3'])}"
    f"</p>",
    unsafe_allow_html=True)

if bulls == 3:
    status_line("All 3 Supertrends bullish — strong trend alignment across all timeframes, high-conviction long setup.", "strong_bullish")
elif bulls == 2:
    status_line(f"2 of 3 Supertrends bullish — trend leaning up but not fully aligned. Wait for ST3 (21,1) to confirm.", "bullish")
elif bears == 3:
    status_line("All 3 Supertrends bearish — strong downtrend alignment across all timeframes, avoid new longs.", "strong_bearish")
elif bears == 2:
    status_line(f"2 of 3 Supertrends bearish — trend leaning down, caution on long entries until alignment improves.", "bearish")
else:
    status_line("Supertrends conflicted (1 bull, 1 bear, 1 neutral) — no directional edge, stay out until clearer.", "neutral")

fig_tst = go.Figure()

fig_tst.add_trace(go.Candlestick(
    x=df["time"], open=df["open"], high=df["high"],
    low=df["low"], close=df["close"],
    name="Price",
    increasing_line_color="#22c55e",
    decreasing_line_color="#ef4444",
    increasing_fillcolor="#22c55e",
    decreasing_fillcolor="#ef4444"))

st_configs = [
    ("st1", "dir1", "ST1 (7, 3.0)",  "#34d399", "#f87171"),
    ("st2", "dir2", "ST2 (14, 2.0)", "#60a5fa", "#fb923c"),
    ("st3", "dir3", "ST3 (21, 1.0)", "#c084fc", "#f472b6"),
]

for st_col, dir_col, label, bull_color, bear_color in st_configs:
    bull_idx = df[dir_col] == 1
    bear_idx = df[dir_col] == -1

    fig_tst.add_trace(go.Scatter(
        x=df["time"][bull_idx], y=df[st_col][bull_idx],
        name=f"{label} Bull", mode="lines",
        line=dict(color=bull_color, width=2),
        connectgaps=False))

    fig_tst.add_trace(go.Scatter(
        x=df["time"][bear_idx], y=df[st_col][bear_idx],
        name=f"{label} Bear", mode="lines",
        line=dict(color=bear_color, width=2, dash="dot"),
        connectgaps=False))

buy_df  = df[df["buy_signal"].notna()]
sell_df = df[df["sell_signal"].notna()]

fig_tst.add_trace(go.Scatter(
    x=buy_df["time"],
    y=buy_df["buy_signal"].astype(float) * 0.995,
    mode="markers",
    name="Buy Signal",
    marker=dict(symbol="triangle-up", size=14, color="#22c55e",
                line=dict(color="white", width=1))))

fig_tst.add_trace(go.Scatter(
    x=sell_df["time"],
    y=sell_df["sell_signal"].astype(float) * 1.005,
    mode="markers",
    name="Sell Signal",
    marker=dict(symbol="triangle-down", size=14, color="#ef4444",
                line=dict(color="white", width=1))))

fig_tst.update_layout(
    height=600,
    margin=dict(t=10, b=10),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)")

st.plotly_chart(fig_tst, use_container_width=True)
st.caption(
    "Solid lines = bullish supertrend (dynamic support). Dotted lines = bearish supertrend (dynamic resistance). "
    "Green triangles = buy signal (price crossed any ST line upward). Red triangles = sell signal (price crossed any ST line downward). "
    "Strongest signals occur when all 3 lines agree on direction.")

st.divider()

# ── ROW 1: RSI + Stoch | Price + BB + EMA ──────────────────────

row1_l, row1_r = st.columns(2)

with row1_l:
    st.markdown(
        f"<h4>RSI + Stoch RSI &nbsp;"
        f"{val_span('RSI: ' + rsi_val, '#F59E0B')} &nbsp;"
        f"{val_span('K: ' + stoch_val, '#60A5FA')} &nbsp;"
        f"{val_span('D: ' + stochd_val, '#F472B6')}</h4>",
        unsafe_allow_html=True)
    rsi_f = latest["RSI"]
    k_f   = latest["StochRSI_k"]
    if pd.notna(rsi_f) and pd.notna(k_f):
        if rsi_f < 30 and k_f < 20:
            status_line(f"RSI {rsi_f:.1f} & Stoch K {k_f:.1f} — both deeply oversold, high probability bounce setup.", "strong_bullish")
        elif rsi_f < 40:
            status_line(f"RSI {rsi_f:.1f} — oversold territory, watching for Stoch K to confirm reversal.", "bullish")
        elif rsi_f > 70 and k_f > 80:
            status_line(f"RSI {rsi_f:.1f} & Stoch K {k_f:.1f} — both overbought, elevated risk of pullback.", "strong_bearish")
        elif rsi_f > 70:
            status_line(f"RSI {rsi_f:.1f} — overbought, momentum may be fading.", "bearish")
        else:
            status_line(f"RSI {rsi_f:.1f} & Stoch K {k_f:.1f} — neutral range, no strong momentum signal.", "neutral")
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
    st.caption("RSI <30 = strongly oversold; >70 = overbought. Stoch K <20 = reversal up likely.")

with row1_r:
    st.markdown(
        f"<h4>Price + BB + EMA &nbsp;"
        f"{val_span('BB: ' + bb_pct_val, 'gray')} &nbsp;"
        f"{val_span('EMA50: ' + ema50_val, '#34D399')} &nbsp;"
        f"{val_span('EMA200: ' + ema200_val, '#F87171')}</h4>",
        unsafe_allow_html=True)
    bb_f = latest["BB_pct"]
    e50  = latest["EMA_50"]
    e200 = latest["EMA_200"]
    if pd.notna(bb_f) and pd.notna(e200):
        cross      = "Golden Cross" if (pd.notna(e50) and e50 > e200) else "Death Cross"
        cross_sent = "bullish" if "Golden" in cross else "bearish"
        if price < latest["BB_lower"]:
            status_line(f"Price below lower BB — oversold. {cross} active on EMAs.", cross_sent)
        elif price > latest["BB_upper"]:
            status_line(f"Price above upper BB — overbought. {cross} active on EMAs.", "bearish")
        elif bb_f < 0.2:
            status_line(f"Price near lower BB ({bb_f*100:.0f}%) — leaning oversold. {cross} on EMAs.", cross_sent)
        elif bb_f > 0.8:
            status_line(f"Price near upper BB ({bb_f*100:.0f}%) — extended. {cross} on EMAs.", "caution")
        else:
            status_line(f"Price mid-band ({bb_f*100:.0f}%), {cross} active — no edge from BB position.", cross_sent)
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
    support_s, resistance_s = detect_support_resistance(df)
    for s in support_s:
        fig2.add_hline(y=s, line_dash="dot", line_color="lime", opacity=0.5,
                       annotation_text=f"S ${s:,.4f}", annotation_position="bottom left")
    for r in resistance_s:
        fig2.add_hline(y=r, line_dash="dot", line_color="tomato", opacity=0.5,
                       annotation_text=f"R ${r:,.4f}", annotation_position="top left")
    fig2.update_layout(height=500, margin=dict(t=30, b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("BB below lower = oversold; above upper = overbought. EMA50 (green) / EMA200 (red).")

st.divider()

# ── ROW 2: MACD | ADX ─────────────────────────────────────────

row2_l, row2_r = st.columns(2)

with row2_l:
    st.markdown(
        f"<h4>MACD &nbsp;"
        f"{val_span('MACD: ' + macd_val, '#60A5FA')} &nbsp;"
        f"{val_span('Sig: ' + msig_val, '#F472B6')} &nbsp;"
        f"{val_span('Hist: ' + mhst_val, 'gray')}</h4>",
        unsafe_allow_html=True)
    mh = latest["MACD_hist"]
    mv = latest["MACD"]
    ms = latest["MACD_signal"]
    if pd.notna(mh) and pd.notna(mv) and pd.notna(ms):
        if mv > ms and mh > 0:
            status_line(f"MACD {mv:.4f} above signal, histogram positive — bullish momentum confirmed.", "bullish")
        elif mv > ms and mh < 0:
            status_line(f"MACD above signal but histogram shrinking — momentum weakening, watch closely.", "caution")
        elif mv < ms and mh < 0:
            status_line(f"MACD {mv:.4f} below signal, histogram negative — bearish momentum in control.", "bearish")
        else:
            status_line(f"MACD crossing signal zone — potential momentum shift, wait for confirmation.", "caution")
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
    st.caption("MACD above signal = bullish. Histogram turning green = momentum shift - key swing entry trigger.")

with row2_r:
    st.markdown(
        f"<h4>ADX - Trend Strength &nbsp;"
        f"{val_span('ADX: ' + adx_val, '#A78BFA')} &nbsp;"
        f"{val_span('+DI: ' + dip_val, '#34D399')} &nbsp;"
        f"{val_span('-DI: ' + din_val, '#F87171')}</h4>",
        unsafe_allow_html=True)
    adx_f = latest["ADX"]
    dp    = latest["ADX_pos"]
    dn    = latest["ADX_neg"]
    if pd.notna(adx_f) and pd.notna(dp) and pd.notna(dn):
        if adx_f < 20:
            status_line(f"ADX {adx_f:.1f} — weak/no trend, choppy conditions. TA signals less reliable.", "neutral")
        elif adx_f >= 25 and dp > dn:
            status_line(f"ADX {adx_f:.1f} — strong trend confirmed, bullish direction (+DI {dp:.1f} > -DI {dn:.1f}).", "bullish")
        elif adx_f >= 25 and dn > dp:
            status_line(f"ADX {adx_f:.1f} — strong trend confirmed, bearish direction (-DI {dn:.1f} > +DI {dp:.1f}).", "bearish")
        else:
            status_line(f"ADX {adx_f:.1f} — trend forming, direction not yet decisive.", "caution")
    fig_adx = go.Figure()
    fig_adx.add_trace(go.Scatter(x=df["time"], y=df["ADX"],
        name="ADX", line=dict(color="#A78BFA", width=2)))
    fig_adx.add_trace(go.Scatter(x=df["time"], y=df["ADX_pos"],
        name="+DI", line=dict(color="#34D399", width=1.5)))
    fig_adx.add_trace(go.Scatter(x=df["time"], y=df["ADX_neg"],
        name="-DI", line=dict(color="#F87171", width=1.5)))
    fig_adx.add_hline(y=20, line_dash="dash", line_color="gray")
    fig_adx.add_hline(y=25, line_dash="dash", line_color="white",
                      annotation_text="Trend threshold")
    fig_adx.update_layout(height=500, margin=dict(t=30, b=10))
    st.plotly_chart(fig_adx, use_container_width=True)
    st.caption("ADX >25 = trending market. <20 = choppy - avoid breakout entries.")

st.divider()

# ── ROW 3: CCI | Williams %R ───────────────────────────────────

row3_l, row3_r = st.columns(2)

with row3_l:
    st.markdown(
        f"<h4>CCI &nbsp;{val_span('CCI: ' + cci_val, '#F59E0B')}</h4>",
        unsafe_allow_html=True)
    cci_f = latest["CCI"]
    if pd.notna(cci_f):
        if cci_f < -100:
            status_line(f"CCI {cci_f:.0f} — oversold below -100, potential swing buy zone.", "bullish")
        elif cci_f > 100:
            status_line(f"CCI {cci_f:.0f} — overbought above +100, consider tightening stops.", "bearish")
        elif cci_f < 0:
            status_line(f"CCI {cci_f:.0f} — mildly negative, below centreline but not extreme.", "neutral")
        else:
            status_line(f"CCI {cci_f:.0f} — mildly positive, above centreline but not overbought.", "neutral")
    fig_cci = go.Figure()
    fig_cci.add_trace(go.Scatter(x=df["time"], y=df["CCI"],
        name="CCI", line=dict(color="#F59E0B", width=2)))
    fig_cci.add_hline(y=100,  line_dash="dash", line_color="red")
    fig_cci.add_hline(y=-100, line_dash="dash", line_color="lime")
    fig_cci.add_hrect(y0=-300, y1=-100, fillcolor="green", opacity=0.05)
    fig_cci.add_hrect(y0=100,  y1=300,  fillcolor="red",   opacity=0.05)
    fig_cci.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_cci, use_container_width=True)
    st.caption("CCI below -100 = oversold. Above +100 = overbought. Use for swing entry/exit timing.")

with row3_r:
    st.markdown(
        f"<h4>Williams %R &nbsp;{val_span('%R: ' + wr_val, '#60A5FA')}</h4>",
        unsafe_allow_html=True)
    wr_f = latest["WilliamsR"]
    if pd.notna(wr_f):
        if wr_f < -80:
            status_line(f"Williams %R {wr_f:.1f} — oversold below -80, look for bounce confirmation.", "bullish")
        elif wr_f > -20:
            status_line(f"Williams %R {wr_f:.1f} — overbought above -20, elevated reversal risk.", "bearish")
        else:
            status_line(f"Williams %R {wr_f:.1f} — neutral mid-range, no extreme reading.", "neutral")
    fig_wr = go.Figure()
    fig_wr.add_trace(go.Scatter(x=df["time"], y=df["WilliamsR"],
        name="Williams %R", line=dict(color="#60A5FA", width=2)))
    fig_wr.add_hline(y=-20, line_dash="dash", line_color="red")
    fig_wr.add_hline(y=-80, line_dash="dash", line_color="lime")
    fig_wr.add_hrect(y0=-100, y1=-80, fillcolor="green", opacity=0.05)
    fig_wr.add_hrect(y0=-20,  y1=0,   fillcolor="red",   opacity=0.05)
    fig_wr.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_wr, use_container_width=True)
    st.caption("Williams %R below -80 = oversold. Above -20 = overbought. Fast-reacting - useful for swing timing.")

st.divider()

# ── ROW 4: ROC | ATR ──────────────────────────────────────────

row4_l, row4_r = st.columns(2)

with row4_l:
    st.markdown(
        f"<h4>ROC - Rate of Change &nbsp;{val_span('ROC: ' + roc_val, '#F59E0B')}</h4>",
        unsafe_allow_html=True)
    roc_f = latest["ROC"]
    if pd.notna(roc_f):
        if roc_f > 10:
            status_line(f"ROC {roc_f:.1f}% — strong positive momentum, trend has real energy.", "strong_bullish")
        elif roc_f > 5:
            status_line(f"ROC {roc_f:.1f}% — positive momentum building above +5% threshold.", "bullish")
        elif roc_f < -10:
            status_line(f"ROC {roc_f:.1f}% — strong negative momentum, selling pressure dominant.", "strong_bearish")
        elif roc_f < -5:
            status_line(f"ROC {roc_f:.1f}% — negative momentum below -5%, caution on new longs.", "bearish")
        else:
            status_line(f"ROC {roc_f:.1f}% — momentum flat near zero, no directional conviction.", "neutral")
    colors_roc = ["green" if v >= 0 else "red" for v in df["ROC"].fillna(0)]
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Bar(x=df["time"], y=df["ROC"],
        name="ROC", marker_color=colors_roc))
    fig_roc.add_hline(y=5,  line_dash="dash", line_color="lime")
    fig_roc.add_hline(y=-5, line_dash="dash", line_color="red")
    fig_roc.add_hline(y=0,  line_color="gray")
    fig_roc.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption("ROC above +5% = strong positive momentum. Below -5% = strong negative momentum.")

with row4_r:
    st.markdown(
        f"<h4>ATR - Volatility &nbsp;{val_span('ATR: ' + atr_val, '#C084FC')}</h4>",
        unsafe_allow_html=True)
    atr_f = latest["ATR"]
    if pd.notna(atr_f):
        atr_pct    = (atr_f / price) * 100
        recent_atr = df["ATR"].tail(20).mean()
        if atr_f > recent_atr * 1.2:
            status_line(f"ATR ${atr_f:.4f} ({atr_pct:.1f}% of price) — volatility expanding, widen stops.", "caution")
        elif atr_f < recent_atr * 0.8:
            status_line(f"ATR ${atr_f:.4f} ({atr_pct:.1f}% of price) — volatility contracting, breakout may be near.", "caution")
        else:
            status_line(f"ATR ${atr_f:.4f} ({atr_pct:.1f}% of price) — normal volatility, use 1.5x ATR for stops.", "neutral")
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=df["time"], y=df["ATR"],
        name="ATR", line=dict(color="#C084FC", width=2)))
    fig_atr.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_atr, use_container_width=True)
    st.caption("ATR rising = increasing volatility. Size stops at 1.5-2x ATR. Expanding ATR on breakouts = confirmation.")

st.divider()

# ── ROW 5: TTM Squeeze | Volume ────────────────────────────────

row5_l, row5_r = st.columns(2)

with row5_l:
    st.markdown(
        f"<h4>TTM Squeeze &nbsp;{val_span('Squeeze: ' + sq_val, sq_color)}</h4>",
        unsafe_allow_html=True)
    sq_f = latest["squeeze"]
    sh_f = latest["squeeze_hist"]
    if pd.notna(sq_f) and pd.notna(sh_f):
        if sq_f and sh_f > 0:
            status_line("Squeeze ON — market coiling with bullish momentum building. Watch for release.", "caution")
        elif sq_f and sh_f < 0:
            status_line("Squeeze ON — market coiling but bearish momentum. Release likely to be downside.", "caution")
        elif not sq_f and sh_f > 0:
            status_line("Squeeze FIRED — momentum released to the upside. Breakout signal active.", "strong_bullish")
        elif not sq_f and sh_f < 0:
            status_line("Squeeze FIRED — momentum released to the downside. Breakdown signal active.", "strong_bearish")
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
    st.caption("Black dots = coiling for breakout. Lime = released. Green bars after squeeze = bullish breakout signal.")

with row5_r:
    st.markdown("<h4>Volume</h4>", unsafe_allow_html=True)
    if "volume" in df.columns and len(df) > 1:
        last_vol   = df["volume"].iloc[-1]
        avg_vol_20 = df["volume"].tail(20).mean()
        vol_ratio  = last_vol / avg_vol_20 if avg_vol_20 > 0 else 1
        up_day     = df["close"].iloc[-1] >= df["close"].iloc[-2]
        if vol_ratio > 1.5 and up_day:
            status_line(f"Volume {vol_ratio:.1f}x above average on an up day — strong buying conviction.", "bullish")
        elif vol_ratio > 1.5 and not up_day:
            status_line(f"Volume {vol_ratio:.1f}x above average on a down day — strong selling pressure.", "bearish")
        elif vol_ratio < 0.6:
            status_line(f"Volume {vol_ratio:.1f}x below average — low conviction, be cautious of false moves.", "neutral")
        else:
            status_line(f"Volume {vol_ratio:.1f}x average — normal activity, no unusual conviction detected.", "neutral")
        vol_colors = ["green" if df["close"].iloc[i] >= df["open"].iloc[i] else "red"
                      for i in range(len(df))]
        avg_vol = df["volume"].rolling(20).mean()
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df["time"], y=df["volume"],
            name="Volume", marker_color=vol_colors, opacity=0.7))
        fig_vol.add_trace(go.Scatter(x=df["time"], y=avg_vol,
            name="20-period Avg", line=dict(color="#F59E0B", width=1.5, dash="dot")))
        fig_vol.update_layout(height=400, margin=dict(t=30, b=10))
        st.plotly_chart(fig_vol, use_container_width=True)
        st.caption("Green = up day volume; red = down day volume. Volume above average on breakouts = confirmation.")
    else:
        st.info("Volume data not available.")

st.caption(f"Data: Yahoo Finance | Auto-refreshes every 5 min | Last run: {ts} | {timeframe} | {date_start.strftime('%d %b %Y')} to {date_end.strftime('%d %b %Y')}")
