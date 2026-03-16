import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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
timeframe   = st.sidebar.radio("Timeframe", ["Hourly", "Daily", "Weekly", "Monthly"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Chart Range**")
range_option = st.sidebar.selectbox(
    "Quick Select",
    ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year",
     "2 Years", "3 Years", "5 Years", "Custom"],
    index=4          # default: 6 Months
)
range_map = {
    "1 Day":    1,   "1 Week":  7,   "1 Month":  30,
    "3 Months": 90,  "6 Months": 180, "1 Year":  365,
    "2 Years":  730, "3 Years": 1095, "5 Years": 1825,
}

if range_option == "Custom":
    col_s, col_e = st.sidebar.columns(2)
    date_start = col_s.date_input("From", value=date.today() - timedelta(days=365))
    date_end   = col_e.date_input("To",   value=date.today())
    # yfinance end is exclusive — add 1 day so the selected date is included
    fetch_end = date_end + timedelta(days=1)
else:
    date_end   = date.today()
    date_start = date_end - timedelta(days=range_map[range_option])
    # yfinance end is exclusive — use tomorrow so today's bars are included
    fetch_end  = date_end + timedelta(days=1)

# ── Timeframe → yfinance interval mapping ──────────────────────
INTERVAL_MAP = {
    "Hourly":  "1h",
    "Daily":   "1d",
    "Weekly":  "1wk",
    "Monthly": "1mo",
}

# yfinance only serves intraday data for the last 730 days max;
# hourly data is limited to the last 60 days.
RANGE_LIMITS = {
    "Hourly":  60,    # max days yfinance will return 1h bars
    "Daily":   None,  # no practical limit
    "Weekly":  None,
    "Monthly": None,
}

# Minimum bars needed for meaningful indicator calculation
MIN_BARS = {
    "Hourly":  30,
    "Daily":   60,
    "Weekly":  15,
    "Monthly": 6,
}

# Clamp hourly range silently and warn the user
range_days = (date_end - date_start).days
if timeframe == "Hourly" and range_days > 60:
    st.sidebar.warning("⚠️ Hourly data is limited to 60 days by Yahoo Finance. Range clamped to 60 days.")
    date_start = date_end - timedelta(days=60)
    range_days = 60
    fetch_end  = date_end + timedelta(days=1)

# Monthly needs at least 6 months of data to be useful
if timeframe == "Monthly" and range_days < 180:
    st.sidebar.warning("⚠️ Monthly timeframe needs at least 6 months of range. Extending to 2 years.")
    date_start = date_end - timedelta(days=730)
    range_days = 730
    fetch_end  = date_end + timedelta(days=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Chart 3 — ATR Trail Stop**")
atr_trail_mult = st.sidebar.slider(
    "Multiplier (× ATR 14)",
    min_value=1.0, max_value=3.0, value=1.5, step=0.1,
    help="Trail stop = highest close since entry minus this × ATR(14). "
         "Lower = tighter/faster exit. Higher = more room to breathe."
)

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
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
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
    df   = df.copy()
    n    = len(df)

    def safe(fn, cols):
        """Run fn(); on any error fill listed cols with NaN."""
        try:
            fn()
        except Exception:
            for c in cols:
                df[c] = np.nan

    # RSI
    safe(lambda: df.__setitem__("RSI",
            ta.momentum.RSIIndicator(df["close"], window=14).rsi()),
         ["RSI"])

    # Stoch RSI — needs 3× window bars (14*3=42); safe() handles short data
    def _stochrsi():
        s = ta.momentum.StochRSIIndicator(df["close"], window=14)
        df["StochRSI_k"] = s.stochrsi_k() * 100
        df["StochRSI_d"] = s.stochrsi_d() * 100
    safe(_stochrsi, ["StochRSI_k", "StochRSI_d"])

    # MACD — needs 26+ bars
    def _macd():
        m = ta.trend.MACD(df["close"])
        df["MACD"]        = m.macd()
        df["MACD_signal"] = m.macd_signal()
        df["MACD_hist"]   = m.macd_diff()
    safe(_macd, ["MACD", "MACD_signal", "MACD_hist"])

    # Bollinger Bands
    def _bb():
        b = ta.volatility.BollingerBands(df["close"], window=20)
        df["BB_upper"] = b.bollinger_hband()
        df["BB_lower"] = b.bollinger_lband()
        df["BB_mid"]   = b.bollinger_mavg()
        df["BB_pct"]   = b.bollinger_pband()
        return b
    try:
        _bb_obj = ta.volatility.BollingerBands(df["close"], window=20)
        df["BB_upper"] = _bb_obj.bollinger_hband()
        df["BB_lower"] = _bb_obj.bollinger_lband()
        df["BB_mid"]   = _bb_obj.bollinger_mavg()
        df["BB_pct"]   = _bb_obj.bollinger_pband()
    except Exception:
        for c in ["BB_upper", "BB_lower", "BB_mid", "BB_pct"]:
            df[c] = np.nan
        _bb_obj = None

    # EMAs
    safe(lambda: df.__setitem__("EMA_50",
            ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()),
         ["EMA_50"])
    safe(lambda: df.__setitem__("EMA_200",
            ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()),
         ["EMA_200"])

    # ADX — needs 2×window+1 bars minimum; window=14 → 29 bars
    def _adx():
        a = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
        df["ADX"]     = a.adx()
        df["ADX_pos"] = a.adx_pos()
        df["ADX_neg"] = a.adx_neg()
    safe(_adx, ["ADX", "ADX_pos", "ADX_neg"])

    # CCI
    safe(lambda: df.__setitem__("CCI",
            ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()),
         ["CCI"])

    # Williams %R
    safe(lambda: df.__setitem__("WilliamsR",
            ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=14).williams_r()),
         ["WilliamsR"])

    # ROC
    safe(lambda: df.__setitem__("ROC",
            ta.momentum.ROCIndicator(df["close"], window=12).roc()),
         ["ROC"])

    # ATR
    safe(lambda: df.__setitem__("ATR",
            ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()),
         ["ATR"])

    # TTM Squeeze
    def _squeeze():
        kc_mid  = df["close"].rolling(20).mean()
        atr20   = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=20).average_true_range()
        kc_upper = kc_mid + 1.5 * atr20
        kc_lower = kc_mid - 1.5 * atr20
        if _bb_obj is not None:
            df["squeeze"] = (_bb_obj.bollinger_hband() < kc_upper) & (_bb_obj.bollinger_lband() > kc_lower)
        else:
            df["squeeze"] = False
        delta = df["close"] - ((df["high"].rolling(20).max() + df["low"].rolling(20).min()) / 2
                                + df["close"].rolling(20).mean()) / 2
        df["squeeze_hist"] = delta.rolling(20).mean()
    safe(_squeeze, ["squeeze", "squeeze_hist"])

    # Ensure all expected columns exist even if entirely NaN
    for col in ["RSI", "StochRSI_k", "StochRSI_d", "MACD", "MACD_signal", "MACD_hist",
                "BB_upper", "BB_lower", "BB_mid", "BB_pct", "EMA_50", "EMA_200",
                "ADX", "ADX_pos", "ADX_neg", "CCI", "WilliamsR", "ROC", "ATR",
                "squeeze", "squeeze_hist"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


def compute_supertrend(df, period, multiplier):
    nan_st  = pd.Series(np.nan,  index=df.index)
    zero_dir = pd.Series(0,      index=df.index, dtype=int)
    if len(df) <= period:          # not enough bars — return silent NaN
        return nan_st, zero_dir
    try:
        atr = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=period
        ).average_true_range()
    except Exception:
        return nan_st, zero_dir

    hl_avg      = (df["high"] + df["low"]) / 2
    basic_upper = (hl_avg + multiplier * atr).values.copy()
    basic_lower = (hl_avg - multiplier * atr).values.copy()
    close       = df["close"].values
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend  = np.full(len(df), np.nan)
    direction   = np.zeros(len(df), dtype=int)

    for i in range(1, len(df)):
        if np.isnan(atr.iloc[i]):
            continue
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
        if direction[i-1] == 1:
            direction[i] = -1 if close[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close[i] > final_upper[i] else -1
        supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)


def triple_supertrend_signals(df):
    df = df.copy()
    df["st1"], df["dir1"] = compute_supertrend(df, period=7,  multiplier=3.0)
    df["st2"], df["dir2"] = compute_supertrend(df, period=14, multiplier=2.0)
    df["st3"], df["dir3"] = compute_supertrend(df, period=21, multiplier=1.0)
    df["buy_signal"]  = None
    df["sell_signal"] = None
    for i in range(1, len(df)):
        d1, d2, d3   = df["dir1"].iloc[i], df["dir2"].iloc[i], df["dir3"].iloc[i]
        pd1, pd2, pd3 = df["dir1"].iloc[i-1], df["dir2"].iloc[i-1], df["dir3"].iloc[i-1]
        c = df["close"].iloc[i]
        cross_up = (d1==1 and pd1==-1) or (d2==1 and pd2==-1) or (d3==1 and pd3==-1)
        cross_dn = (d1==-1 and pd1==1) or (d2==-1 and pd2==1) or (d3==-1 and pd3==1)
        if cross_up:
            df.at[df.index[i], "buy_signal"]  = c
        if cross_dn:
            df.at[df.index[i], "sell_signal"] = c
    return df


# ── CHART HELPERS ─────────────────────────────────────────────
# All ST lines use the same two colours: green (bullish, below price as support)
# and red (bearish, above price as resistance) — matching the Nordman visual style.
# No per-ST colour variation.

def add_st_line(fig, df, st_col, dir_col, name, bull_color="#00c853", bear_color="#ff1744", width=2):
    """Draw one ST as two traces: green when bullish, red when bearish.
    NaN breaks split cleanly at direction changes."""
    x = df["time"].values
    y = df[st_col].values.astype(float)
    d = df[dir_col].values

    bull_y = y.copy()
    bear_y = y.copy()
    bull_y[d != 1]  = np.nan
    bear_y[d != -1] = np.nan

    # Extend one extra point at each transition so lines connect visually
    for i in range(1, len(d)):
        if d[i] != d[i-1]:
            if d[i] == 1:
                bull_y[i-1] = y[i-1]
            else:
                bear_y[i-1] = y[i-1]

    fig.add_trace(go.Scatter(
        x=x, y=bull_y, mode="lines",
        name=f"{name} Bull", line=dict(color=bull_color, width=width),
        connectgaps=False,
        hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=bear_y, mode="lines",
        name=f"{name} Bear", line=dict(color=bear_color, width=width),
        connectgaps=False,
        hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>"))


def build_chart1(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e",  decreasing_fillcolor="#ef4444"))

    add_st_line(fig, df, "st1", "dir1", "ST1 (7, 3.0)")
    add_st_line(fig, df, "st2", "dir2", "ST2 (14, 2.0)")
    add_st_line(fig, df, "st3", "dir3", "ST3 (21, 1.0)")

    buy_df  = df[df["buy_signal"].notna()]
    sell_df = df[df["sell_signal"].notna()]

    fig.add_trace(go.Scatter(
        x=buy_df["time"], y=buy_df["buy_signal"].astype(float) * 0.995,
        mode="markers", name="Buy Signal",
        marker=dict(symbol="triangle-up", size=14, color="#0077BB",
                    line=dict(color="white", width=1.5))))
    fig.add_trace(go.Scatter(
        x=sell_df["time"], y=sell_df["sell_signal"].astype(float) * 1.005,
        mode="markers", name="Sell Signal",
        marker=dict(symbol="triangle-down", size=14, color="#EE7733",
                    line=dict(color="white", width=1.5))))

    fig.update_layout(
        title=None,
        height=600,
        margin=dict(t=50, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── CHART 2: Nordman-style ─────────────────────────────────────
# Green line below price (support) when bullish, red line above (resistance) when bearish
# Candles coloured by how many STs are bullish
# EMA 200 shown
# Big arrows only when ALL 3 align AND Stoch RSI confirms

def build_chart2(df):
    fig = go.Figure()

    # Background shading: green zone when all 3 bull, red zone when all 3 bear
    all_bull = (df["dir1"] == 1) & (df["dir2"] == 1) & (df["dir3"] == 1)
    all_bear = (df["dir1"] == -1) & (df["dir2"] == -1) & (df["dir3"] == -1)

    def add_bg_zones(fig, df, mask, color):
        in_zone = False
        start   = None
        for i, val in enumerate(mask):
            if val and not in_zone:
                in_zone = True
                start   = df["time"].iloc[i]
            elif not val and in_zone:
                in_zone = False
                fig.add_vrect(x0=start, x1=df["time"].iloc[i-1],
                              fillcolor=color, opacity=0.08,
                              layer="below", line_width=0)
        if in_zone:
            fig.add_vrect(x0=start, x1=df["time"].iloc[-1],
                          fillcolor=color, opacity=0.08,
                          layer="below", line_width=0)

    add_bg_zones(fig, df, all_bull, "#22c55e")
    add_bg_zones(fig, df, all_bear, "#ef4444")




    # Single Candlestick trace (avoids Plotly "undefined" legend group bug from multiple Candlestick traces).
    # Candle body tinting by ST bull count is achieved via coloured scatter bar overlays underneath.
    df["_bull_count"] = df.apply(
        lambda r: sum(1 for d in [r["dir1"], r["dir2"], r["dir3"]] if d == 1), axis=1)

    BULL_COUNT_COLORS = {3: "#22c55e", 2: "#86efac", 1: "#fca5a5", 0: "#ef4444"}
    BULL_COUNT_LABELS = {3: "3/3 Bull", 2: "2/3 Bull", 1: "1/3 Bull", 0: "0/3 Bull"}

    # Draw coloured body bars (open→close range) as thin bar traces — one per bull-count group
    for cnt, color in BULL_COUNT_COLORS.items():
        sub = df[df["_bull_count"] == cnt]
        if sub.empty:
            continue
        body_base = sub[["open", "close"]].min(axis=1)
        body_top  = sub[["open", "close"]].max(axis=1)
        fig.add_trace(go.Bar(
            x=sub["time"],
            y=(body_top - body_base).values,
            base=body_base.values,
            name=BULL_COUNT_LABELS[cnt],
            marker_color=color,
            marker_opacity=0.6,
            width=0.6 * (1 if len(df) < 100 else 0.5),
            showlegend=True))

    # Single standard Candlestick on top — wicks visible, body transparent
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color="rgba(34,197,94,0.5)",
        decreasing_line_color="rgba(239,68,68,0.5)",
        increasing_fillcolor="rgba(0,0,0,0)",
        decreasing_fillcolor="rgba(0,0,0,0)",
        showlegend=False))

    # ST lines — all green when bull, all red when bear (Nordman style)
    add_st_line(fig, df, "st1", "dir1", "ST1 (7, 3.0)")
    add_st_line(fig, df, "st2", "dir2", "ST2 (14, 2.0)")
    add_st_line(fig, df, "st3", "dir3", "ST3 (21, 1.0)")

    # EMA 200
    if df["EMA_200"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["EMA_200"],
            name="EMA 200",
            line=dict(color="#fbbf24", width=2, dash="dot")))

    # ── Nordman arrow conditions (per nordman-algorithms.com spec) ──
    # BULLISH: all 3 STs green + price above EMA 200 + StochRSI K < 20
    # BEARISH: all 3 STs red   + price below EMA 200 + StochRSI K > 80
    # Arrow fires on the FIRST bar that enters the alignment (no repeat until it exits and re-enters)
    # EMA/StochRSI filters are skipped gracefully if data not available
    bull_align_starts = []
    bear_align_starts = []
    prev_bulls = prev_bears = False
    for i, row in df.iterrows():
        all3_bull = (row["dir1"] == 1  and row["dir2"] == 1  and row["dir3"] == 1)
        all3_bear = (row["dir1"] == -1 and row["dir2"] == -1 and row["dir3"] == -1)

        # EMA 200 filter (skip if NaN)
        ema200 = row.get("EMA_200", float("nan"))
        if pd.notna(ema200):
            all3_bull = all3_bull and (row["close"] > ema200)
            all3_bear = all3_bear and (row["close"] < ema200)

        # StochRSI K filter (skip if NaN)
        stoch_k = row.get("StochRSI_k", float("nan"))
        if pd.notna(stoch_k):
            all3_bull = all3_bull and (stoch_k < 20)
            all3_bear = all3_bear and (stoch_k > 80)

        if all3_bull and not prev_bulls:
            bull_align_starts.append(i)
        if all3_bear and not prev_bears:
            bear_align_starts.append(i)
        prev_bulls = all3_bull
        prev_bears = all3_bear

    bull_signal_df = df.loc[bull_align_starts]
    bear_signal_df = df.loc[bear_align_starts]

    fig.add_trace(go.Scatter(
        x=bull_signal_df["time"],
        y=bull_signal_df["low"].astype(float) * 0.990,
        mode="markers+text",
        name="All 3 Bullish",
        text=["▲"] * len(bull_signal_df),
        textfont=dict(size=22, color="#0077BB"),
        marker=dict(symbol="triangle-up", size=18, color="#0077BB",
                    line=dict(color="white", width=1.5)),
        textposition="bottom center"))

    fig.add_trace(go.Scatter(
        x=bear_signal_df["time"],
        y=bear_signal_df["high"].astype(float) * 1.010,
        mode="markers+text",
        name="All 3 Bearish",
        text=["▼"] * len(bear_signal_df),
        textfont=dict(size=22, color="#EE7733"),
        marker=dict(symbol="triangle-down", size=18, color="#EE7733",
                    line=dict(color="white", width=1.5)),
        textposition="top center"))

    fig.update_layout(
        title=None,
        height=650,
        margin=dict(t=50, b=10),
        xaxis_rangeslider_visible=False,
        xaxis=dict(range=[df["time"].iloc[0], df["time"].iloc[-1]]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#0e1117",
        paper_bgcolor="rgba(0,0,0,0)")
    return fig# ── CHART 3: Triple SuperTrend + RSI(7) + Fibonacci Bollinger Bands ──────────
# Based on TradingView script by TRW_meir (jw3P5XtU)
# ST params: (10,1.0), (11,2.0), (12,3.0) — tighter ATR periods than Chart 1 & 2
# BUY:  all 3 STs bullish AND RSI(7) > 50 — first bar of full alignment only
# SELL: all 3 STs bearish AND RSI(7) < 50 — first bar of full alignment only
# EXIT: any single ST flips direction OR price touches Fibonacci BB band (200, 2.618)

# ── CHART 3: Long-only. BUY when all 3 STs bull + RSI(7)>50.
# EXIT when in a long and: any ST flips red OR price touches UPPER Fib BB
#           OR price closes below ATR trailing stop.
# ATR trailing stop trails at highest_close - (atr_mult × ATR14), ratcheting up only.
# No sell signals — short trades not used.

def compute_fib_bb(close_series, length=200, mult=2.618):
    basis = close_series.rolling(length).mean()
    dev   = close_series.rolling(length).std(ddof=0)
    return basis, basis + mult * dev, basis - mult * dev


def compute_chart3_signals(df, atr_trail_mult=1.5):
    df = df.copy()

    df["c3_st1"], df["c3_dir1"] = compute_supertrend(df, period=10, multiplier=1.0)
    df["c3_st2"], df["c3_dir2"] = compute_supertrend(df, period=11, multiplier=2.0)
    df["c3_st3"], df["c3_dir3"] = compute_supertrend(df, period=12, multiplier=3.0)

    try:
        df["c3_rsi7"] = ta.momentum.RSIIndicator(df["close"], window=7).rsi()
    except Exception:
        df["c3_rsi7"] = np.nan

    try:
        df["c3_fib_basis"], df["c3_fib_upper"], df["c3_fib_lower"] = compute_fib_bb(df["close"])
    except Exception:
        df["c3_fib_basis"] = df["c3_fib_upper"] = df["c3_fib_lower"] = np.nan

    # ATR(14) for trailing stop — reuse existing ATR if available, else compute fresh
    try:
        atr14 = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=14
        ).average_true_range()
    except Exception:
        atr14 = pd.Series(np.nan, index=df.index)

    df["c3_buy"]       = None
    df["c3_exit"]      = None
    df["c3_trail_stop"] = np.nan   # ATR trailing stop level — NaN when not in a trade

    in_long       = False
    prev_all_bull = False
    peak_close    = np.nan   # highest close since entry, used to ratchet stop up

    for i in range(1, len(df)):
        d1  = df["c3_dir1"].iloc[i];  pd1 = df["c3_dir1"].iloc[i-1]
        d2  = df["c3_dir2"].iloc[i];  pd2 = df["c3_dir2"].iloc[i-1]
        d3  = df["c3_dir3"].iloc[i];  pd3 = df["c3_dir3"].iloc[i-1]
        rsi7 = df["c3_rsi7"].iloc[i]
        cls  = df["close"].iloc[i]
        fub  = df["c3_fib_upper"].iloc[i]
        atr  = atr14.iloc[i]

        all_bull = (d1 == 1 and d2 == 1 and d3 == 1)

        if in_long:
            # Ratchet peak close upward only
            if pd.notna(cls):
                peak_close = max(peak_close, cls)

            # Compute trailing stop for this bar
            if pd.notna(atr) and pd.notna(peak_close):
                trail_stop = peak_close - atr_trail_mult * atr
                df.at[df.index[i], "c3_trail_stop"] = trail_stop
            else:
                trail_stop = np.nan

            # Exit conditions
            any_st_flipped_red = (
                (d1 == -1 and pd1 == 1) or
                (d2 == -1 and pd2 == 1) or
                (d3 == -1 and pd3 == 1)
            )
            upper_fib_touch  = pd.notna(fub) and cls >= fub
            trail_stop_hit   = pd.notna(trail_stop) and cls <= trail_stop

            if any_st_flipped_red or upper_fib_touch or trail_stop_hit:
                df.at[df.index[i], "c3_exit"] = cls
                in_long    = False
                peak_close = np.nan

        else:
            # Enter long on first bar where all 3 STs are bull + RSI(7) > 50
            if all_bull and not prev_all_bull and pd.notna(rsi7) and rsi7 > 50:
                df.at[df.index[i], "c3_buy"] = cls
                in_long    = True
                peak_close = cls   # initialise peak at entry close

        prev_all_bull = all_bull

    return df


def build_chart3(df):
    fig = go.Figure()

    # Background shading
    all_bull = (df["c3_dir1"] == 1) & (df["c3_dir2"] == 1) & (df["c3_dir3"] == 1)
    all_bear = (df["c3_dir1"] == -1) & (df["c3_dir2"] == -1) & (df["c3_dir3"] == -1)

    def _add_bg(fig, df, mask, color):
        in_zone = False; start = None
        for i, val in enumerate(mask):
            if val and not in_zone:
                in_zone = True; start = df["time"].iloc[i]
            elif not val and in_zone:
                in_zone = False
                fig.add_vrect(x0=start, x1=df["time"].iloc[i-1],
                              fillcolor=color, opacity=0.07, layer="below", line_width=0)
        if in_zone:
            fig.add_vrect(x0=start, x1=df["time"].iloc[-1],
                          fillcolor=color, opacity=0.07, layer="below", line_width=0)

    _add_bg(fig, df, all_bull, "#22c55e")
    _add_bg(fig, df, all_bear, "#ef4444")

    # Price candles (standard colour — Chart 3 focus is on signals, not candle tinting)
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"],  close=df["close"],
        name="Price",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e",  decreasing_fillcolor="#ef4444"))

    # Three ST lines — all green/red, varying line width to distinguish them
    st3_configs = [
        ("c3_st1", "c3_dir1", "ST1 (10,1.0)", 2.5),
        ("c3_st2", "c3_dir2", "ST2 (11,2.0)", 2.0),
        ("c3_st3", "c3_dir3", "ST3 (12,3.0)", 1.5),
    ]
    for st_col, dir_col, label, w in st3_configs:
        if st_col not in df.columns:
            continue
        add_st_line(fig, df, st_col, dir_col, label, width=w)

    # Fibonacci Bollinger Bands (200, 2.618)
    if df["c3_fib_upper"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["c3_fib_upper"],
            name="Fib BB Upper (2.618)", line=dict(color="#fbbf24", width=1.5, dash="dot"),
            hovertemplate="Fib BB Upper: %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["c3_fib_lower"],
            name="Fib BB Lower (2.618)", line=dict(color="#fbbf24", width=1.5, dash="dot"),
            fill="tonexty", fillcolor="rgba(251,191,36,0.05)",
            hovertemplate="Fib BB Lower: %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["c3_fib_basis"],
            name="Fib BB Basis (SMA 200)", line=dict(color="#fbbf24", width=1, dash="dash"),
            hovertemplate="Fib BB Basis: %{y:.4f}<extra></extra>"))

    # ATR Trailing Stop — only visible while in a long (NaN elsewhere = clean chart)
    if "c3_trail_stop" in df.columns and df["c3_trail_stop"].notna().any():
        # atr_trail_mult is in scope from the Streamlit sidebar slider
        try:
            mult_label = f"{atr_trail_mult:.1f}"
        except Exception:
            mult_label = "1.5"
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["c3_trail_stop"],
            name=f"ATR Trail Stop ({mult_label}×)",
            line=dict(color="#f97316", width=1.5, dash="dashdot"),
            connectgaps=False,
            hovertemplate="ATR Trail Stop: %{y:.4f}<extra></extra>"))

    # Buy / Exit markers only — no sell (long-only strategy)
    buy_df  = df[df["c3_buy"].notna()]
    exit_df = df[df["c3_exit"].notna()]

    if not buy_df.empty:
        fig.add_trace(go.Scatter(
            x=buy_df["time"], y=buy_df["low"].astype(float) * 0.990,
            mode="markers+text", name="BUY (all 3 bull + RSI>50)",
            text=["▲"] * len(buy_df),
            textfont=dict(size=20, color="#0077BB"),
            marker=dict(symbol="triangle-up", size=16, color="#0077BB",
                        line=dict(color="white", width=1.5)),
            textposition="bottom center"))

    if not exit_df.empty:
        fig.add_trace(go.Scatter(
            x=exit_df["time"], y=exit_df["high"].astype(float) * 1.015,
            mode="markers", name="EXIT long (ST turned red or upper Fib BB hit)",
            marker=dict(symbol="diamond", size=12, color="#fbbf24",
                        line=dict(color="white", width=1.5))))

    fig.update_layout(
        title=None,
        height=650,
        margin=dict(t=50, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#0e1117",
        paper_bgcolor="rgba(0,0,0,0)")
    return fig


def compute_swing_summary(df, price, is_swing_suitable):
    """
    Runs all 5 swing indicator analyses and returns a structured dict
    used both for the top summary card and reused in each chart section.
    """
    result = {}

    # ── Divergence ────────────────────────────────────────────
    result["div_rsi_bull"] = result["div_rsi_bear"]   = False
    result["div_macd_bull"] = result["div_macd_bear"] = False
    result["rsi_divs"] = []
    result["macd_divs"] = []
    if df["RSI"].notna().any() and df["MACD"].notna().any():
        try:
            rdi = detect_divergence(df["close"], df["RSI"],  window=3, lookback=60)
            mdi = detect_divergence(df["close"], df["MACD"], window=3, lookback=60)
            result["rsi_divs"]      = rdi
            result["macd_divs"]     = mdi
            result["div_rsi_bull"]  = any(t == "bullish" for _, t in rdi[-3:]) if rdi else False
            result["div_rsi_bear"]  = any(t == "bearish" for _, t in rdi[-3:]) if rdi else False
            result["div_macd_bull"] = any(t == "bullish" for _, t in mdi[-3:]) if mdi else False
            result["div_macd_bear"] = any(t == "bearish" for _, t in mdi[-3:]) if mdi else False
        except Exception:
            pass

    # ── Ichimoku ──────────────────────────────────────────────
    result["ichi_signal"] = "insufficient"
    result["ichi_tenkan"] = result["ichi_kijun"] = np.nan
    result["ichi_cloud_top"] = result["ichi_cloud_bottom"] = np.nan
    if len(df) >= 52:
        try:
            tenkan, kijun, ssa, ssb, _ = compute_ichimoku(df)
            lt = tenkan.iloc[-1]; lk = kijun.iloc[-1]
            la = ssa.iloc[-1];   lb = ssb.iloc[-1]
            result["ichi_tenkan"] = lt;  result["ichi_kijun"] = lk
            if pd.notna(la) and pd.notna(lb):
                result["ichi_cloud_top"]    = max(la, lb)
                result["ichi_cloud_bottom"] = min(la, lb)
            ct = result["ichi_cloud_top"]; cb = result["ichi_cloud_bottom"]
            if pd.notna(ct) and pd.notna(lt) and pd.notna(lk):
                if price > ct and lt > lk:
                    result["ichi_signal"] = "strong_bull"
                elif price > ct:
                    result["ichi_signal"] = "bull"
                elif price < cb and lt < lk:
                    result["ichi_signal"] = "strong_bear"
                elif price < cb:
                    result["ichi_signal"] = "bear"
                else:
                    result["ichi_signal"] = "inside_cloud"
        except Exception:
            pass

    # ── OBV ───────────────────────────────────────────────────
    result["obv_signal"] = "neutral"
    if "volume" in df.columns and df["volume"].notna().any() and len(df) >= 20:
        try:
            obv = [0]
            for i in range(1, len(df)):
                if df["close"].iloc[i] > df["close"].iloc[i-1]:
                    obv.append(obv[-1] + df["volume"].iloc[i])
                elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                    obv.append(obv[-1] - df["volume"].iloc[i])
                else:
                    obv.append(obv[-1])
            obv_s = pd.Series(obv, index=df.index)
            pt = float(df["close"].iloc[-1]) - float(df["close"].iloc[-20])
            ot = float(obv_s.iloc[-1])       - float(obv_s.iloc[-20])
            result["obv_price_trend"] = pt
            result["obv_trend"]       = ot
            if pt > 0 and ot > 0:   result["obv_signal"] = "confirming_bull"
            elif pt > 0 and ot < 0: result["obv_signal"] = "bearish_div"
            elif pt < 0 and ot > 0: result["obv_signal"] = "bullish_div"
            elif pt < 0 and ot < 0: result["obv_signal"] = "confirming_bear"
        except Exception:
            pass

    # ── Fibonacci ─────────────────────────────────────────────
    result["fib_signal"] = "neutral"
    result["fib_zone"]   = None
    try:
        fib_df     = df.tail(60)
        sh = float(fib_df["high"].max()); sl = float(fib_df["low"].min())
        fr = sh - sl
        levels = {"23.6%": 0.236, "38.2%": 0.382, "50%": 0.500,
                  "61.8%": 0.618, "78.6%": 0.786}
        fib_p  = {lbl: sh - r * fr for lbl, r in levels.items()}
        sorted_l = sorted(fib_p.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(sorted_l) - 1):
            ul, uv = sorted_l[i]; ll, lv = sorted_l[i+1]
            if lv <= price <= uv:
                result["fib_zone"] = (ll, lv, ul, uv)
                pct = (price - lv) / (uv - lv) * 100 if uv != lv else 50
                if any(k in ll for k in ["38.2", "50%", "61.8"]) and pct < 25:
                    result["fib_signal"] = "at_support"
                elif pct > 75:
                    result["fib_signal"] = "near_resistance"
                else:
                    result["fib_signal"] = "mid_zone"
                break
    except Exception:
        pass

    # ── Parabolic SAR ─────────────────────────────────────────
    result["psar_signal"]    = "insufficient"
    result["psar_bull"]      = None
    result["psar_val"]       = np.nan
    result["psar_just_flip"] = None
    if len(df) >= 10:
        try:
            pv, pb = compute_psar(df)
            cur_bull  = bool(pb.iloc[-1])
            prev_bull = bool(pb.iloc[-2]) if len(pb) > 1 else cur_bull
            result["psar_bull"]  = cur_bull
            result["psar_val"]   = float(pv.iloc[-1])
            if cur_bull and not prev_bull:
                result["psar_signal"]    = "just_flipped_bull"
                result["psar_just_flip"] = "bull"
            elif not cur_bull and prev_bull:
                result["psar_signal"]    = "just_flipped_bear"
                result["psar_just_flip"] = "bear"
            elif cur_bull:
                result["psar_signal"] = "bull"
            else:
                result["psar_signal"] = "bear"
        except Exception:
            pass

    # ── Overall recommendation ─────────────────────────────────
    bull_pts = 0; bear_pts = 0; notes = []

    # Divergence
    if result["div_rsi_bull"] or result["div_macd_bull"]:
        bull_pts += 2
        notes.append("Bullish momentum divergence on " +
                      ("RSI & MACD" if result["div_rsi_bull"] and result["div_macd_bull"]
                       else "RSI" if result["div_rsi_bull"] else "MACD"))
    elif result["div_rsi_bear"] or result["div_macd_bear"]:
        bear_pts += 2
        notes.append("Bearish momentum divergence — momentum weakening into price highs")

    # Ichimoku
    sig = result["ichi_signal"]
    if sig == "strong_bull":
        bull_pts += 3; notes.append("Ichimoku: price above cloud, Tenkan > Kijun — strong bullish structure")
    elif sig == "bull":
        bull_pts += 1; notes.append("Ichimoku: price above cloud but momentum softening")
    elif sig == "strong_bear":
        bear_pts += 3; notes.append("Ichimoku: price below cloud, Tenkan < Kijun — strong bearish structure")
    elif sig == "bear":
        bear_pts += 1; notes.append("Ichimoku: price below cloud — avoid new longs")
    elif sig == "inside_cloud":
        notes.append("Ichimoku: price inside cloud — indecision, wait for breakout")

    # OBV
    obv = result["obv_signal"]
    if obv == "confirming_bull":
        bull_pts += 2; notes.append("OBV confirming uptrend — volume flowing in with price")
    elif obv == "bullish_div":
        bull_pts += 2; notes.append("OBV bullish divergence — accumulation on price weakness")
    elif obv == "bearish_div":
        bear_pts += 2; notes.append("OBV bearish divergence — distribution into price strength")
    elif obv == "confirming_bear":
        bear_pts += 1; notes.append("OBV confirming downtrend — volume flowing out with price")

    # Fibonacci
    fib = result["fib_signal"]
    if fib == "at_support":
        ll, lv, ul, uv = result["fib_zone"]
        bull_pts += 2; notes.append(f"Price sitting on {ll} Fibonacci support (${lv:.4f}) — key entry zone")
    elif fib == "near_resistance":
        _, _, ul, uv = result["fib_zone"]
        bear_pts += 1; notes.append(f"Price approaching {ul} Fibonacci resistance (${uv:.4f}) — watch for rejection")

    # SAR
    psar = result["psar_signal"]
    if psar == "just_flipped_bull":
        bull_pts += 3; notes.append("Parabolic SAR just flipped bullish — fresh trend reversal signal")
    elif psar == "just_flipped_bear":
        bear_pts += 3; notes.append("Parabolic SAR just flipped bearish — trend reversed, exit signal")
    elif psar == "bull":
        bull_pts += 1; notes.append(f"Parabolic SAR bullish — trailing stop at ${result['psar_val']:.4f}")
    elif psar == "bear":
        bear_pts += 1; notes.append("Parabolic SAR bearish — avoid new longs")

    total = bull_pts + bear_pts
    bull_pct = bull_pts / total * 100 if total > 0 else 50

    if bull_pts >= 8:
        action = "STRONG BUY"
        action_color = "#22c55e"
        action_detail = (
            "Multiple high-conviction bullish signals aligning. "
            + ("For swing traders: consider entering now with a stop below the SAR or 1.5× ATR. "
               if is_swing_suitable else
               "For DCA holders: this is a strong accumulation zone — consider adding to position. "))
    elif bull_pts >= 5:
        action = "WATCH / ACCUMULATE"
        action_color = "#86efac"
        action_detail = (
            "Bullish bias building but not full confluence. "
            + ("For swing traders: watch for one more confirming signal (e.g. SAR flip or Fib support bounce) before entry. "
               if is_swing_suitable else
               "For DCA holders: reasonable entry zone, consider partial accumulation. "))
    elif bull_pts >= 3 and bear_pts <= 2:
        action = "HOLD / NEUTRAL"
        action_color = "#9ca3af"
        action_detail = (
            "Mixed signals — no clear directional edge. "
            "Hold existing positions. Avoid new entries until signals clarify. "
            "Watch for Ichimoku cloud breakout or SAR flip for next directional cue.")
    elif bear_pts >= 5:
        action = "CAUTION — REDUCE / WAIT"
        action_color = "#f97316"
        action_detail = (
            "Multiple bearish signals present. "
            + ("For swing traders: avoid new longs. Consider tightening stops on existing positions. "
               if is_swing_suitable else
               "For DCA holders: pause regular buys until trend stabilises. Consider waiting for RSI < 35 before next purchase. "))
    elif bear_pts >= 3:
        action = "CAUTION"
        action_color = "#fbbf24"
        action_detail = (
            "More bearish signals than bullish. "
            "Avoid new entries. Hold cash and wait for OBV or Ichimoku to turn positive.")
    else:
        action = "NEUTRAL — WAIT"
        action_color = "#9ca3af"
        action_detail = "Insufficient data or signals too mixed for a clear recommendation. Widen the date range for better indicator coverage."

    # Profit-taking check
    take_profit_note = ""
    if result["fib_signal"] == "near_resistance":
        _, _, ul, uv = result["fib_zone"]
        take_profit_note = f"⚠️ Price near {ul} Fibonacci resistance — if you're in a position, consider taking partial profits."
    if result["ichi_signal"] in ("strong_bull") and psar == "just_flipped_bear":
        take_profit_note = "⚠️ SAR just flipped bearish despite bullish Ichimoku — trend may be topping. Consider scaling out."
    if obv == "bearish_div" and bull_pts > 3:
        take_profit_note = "⚠️ OBV bearish divergence while price is elevated — distribution signal. Consider taking profits."

    result["action"]           = action
    result["action_color"]     = action_color
    result["action_detail"]    = action_detail
    result["take_profit_note"] = take_profit_note
    result["bull_pts"]         = bull_pts
    result["bear_pts"]         = bear_pts
    result["notes"]            = notes

    return result


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
        if hh and hl:   return "🟢 Uptrend - Higher Highs & Higher Lows", "green"
        elif lh and ll: return "🔴 Downtrend - Lower Highs & Lower Lows", "red"
        else:           return "⚪ Choppy - No clear structure", "gray"
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
    if score >= 7:   return "🟢 STRONG BUY"
    elif score >= 3: return "🟡 WATCH / ACCUMULATE"
    elif score >= -2:return "⚪ HOLD / NEUTRAL"
    elif score >= -5:return "🟠 CAUTION / REDUCE"
    else:            return "🔴 STRONG SELL / AVOID"


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
        signals.append(("🟢 SWING BUY - Pullback in Uptrend",
            f"Golden Cross active + price above EMA 200. RSI ({rsi:.1f}) pulled back to oversold while MACD histogram positive.",
            f"Entry: ${price:,.4f} | Stop: ${stop:,.4f} (-{((price-stop)/price)*100:.1f}%) | Target: ${target:,.4f} (+{((target-price)/price)*100:.1f}%) | R:R {rr:.1f}x",
            "green"))
    elif rsi_extreme and wr_oversold and stoch_low:
        stop   = price - atr * 1.5
        target = ema50 if pd.notna(ema50) and ema50 > price else nearest_resistance
        rr     = (target - price) / (price - stop) if (price - stop) > 0 else 0
        signals.append(("🟡 SWING WATCH - Extreme Oversold Reversal",
            f"Triple oversold confluence: RSI ({rsi:.1f}), Williams %R ({wr:.1f}), Stoch K ({stoch_k:.1f}). {'Caution - Death Cross active.' if death_cross else 'No Death Cross - more reliable.'}",
            f"Entry: ${price:,.4f} | Stop: ${stop:,.4f} (-{((price-stop)/price)*100:.1f}%) | Target: ${target:,.4f} (+{((target-price)/price)*100:.1f}%) | R:R {rr:.1f}x",
            "yellow"))
    elif trending_up and macd_bullish and not rsi_overbought and squeeze == False:
        stop   = ema50 if pd.notna(ema50) else price - atr * 2
        target = nearest_resistance
        rr     = (target - price) / (price - stop) if (price - stop) > 0 else 0
        signals.append(("🟢 SWING BUY - Momentum Breakout",
            f"TTM Squeeze released. ADX ({adx:.1f}) confirms strong trend, +DI ({adx_pos:.1f}) > -DI ({adx_neg:.1f}). MACD confirming.",
            f"Entry: ${price:,.4f} | Stop: ${stop:,.4f} (-{((price-stop)/price)*100:.1f}%) | Target: ${target:,.4f} (+{((target-price)/price)*100:.1f}%) | R:R {rr:.1f}x",
            "green"))
    elif death_cross and rsi_overbought and stoch_high:
        signals.append(("🔴 CAUTION - Overbought in Downtrend",
            f"Death Cross + RSI ({rsi:.1f}) overbought + Stoch K ({stoch_k:.1f}) above 80. High reversal risk.",
            "Watch for RSI to fall below 65 and MACD to cross bearish before reassessing.",
            "red"))
    else:
        if score >= 0:
            signals.append(("⚪ NO CLEAR SWING SETUP - Leaning Neutral/Bullish",
                f"Indicators mixed. {'Golden Cross - wait for dip to EMA 50 or RSI < 45.' if golden_cross else 'Watch for Golden Cross or oversold bounce.'}",
                "Monitor for RSI dip below 40 or TTM Squeeze firing.", "gray"))
        else:
            signals.append(("🟠 NO CLEAR SWING SETUP - Leaning Bearish",
                f"More bearish signals. {'Death Cross active.' if death_cross else 'Trend weakening.'} Avoid new longs until RSI < 30 and MACD turns positive.",
                "Wait for extreme oversold confluence before considering entry.", "orange"))

    if week52_high and week52_low and week52_high != week52_low:
        pct_from_high = ((price - week52_high) / week52_high) * 100
        pct_from_low  = ((price - week52_low)  / week52_low)  * 100
        range_pos     = ((price - week52_low) / (week52_high - week52_low)) * 100
        signals.append(("📅 52-Week Range Context",
            f"52w High: ${week52_high:,.4f} | 52w Low: ${week52_low:,.4f} | Position in range: {range_pos:.0f}%",
            f"Price is {abs(pct_from_high):.1f}% below 52w high and {abs(pct_from_low):.1f}% above 52w low. "
            f"{'Near lows - potential value zone.' if range_pos < 30 else 'Near highs - higher risk.' if range_pos > 70 else 'Mid-range.'}",
            "blue"))

    if pd.notna(atr):
        atr_pct = (atr / price) * 100
        signals.append(("📏 ATR Volatility - Position Sizing",
            f"ATR (14): ${atr:,.4f} ({atr_pct:.1f}% of price) | Stop 1.5x: ${atr*1.5:,.4f} | Stop 2x: ${atr*2:,.4f}",
            f"For 1% portfolio risk at 1.5x ATR stop: position size approx {(0.01 / ((atr * 1.5) / price)) * 100:.1f}% of capital.",
            "purple"))
    return signals


# ── LOAD DATA ──────────────────────────────────────────────────

interval = INTERVAL_MAP[timeframe]
min_bars = MIN_BARS[timeframe]

with st.spinner(f"Loading {ticker} data ({timeframe})..."):
    df   = get_stock_data(ticker, date_start, fetch_end, interval)
    info = get_stock_info(ticker)

if df.empty or len(df) < min_bars:
    friendly_combos = {
        "Hourly":  "Try a range of 7–60 days with Hourly.",
        "Daily":   "Try a range of at least 3 months with Daily.",
        "Weekly":  "Try a range of at least 6 months with Weekly.",
        "Monthly": "Try a range of at least 2 years with Monthly.",
    }
    bar_count = len(df) if not df.empty else 0
    st.error(
        f"Not enough data for **{ticker}** on **{timeframe}** timeframe "
        f"({bar_count} bars returned, need at least {min_bars}). "
        f"{friendly_combos.get(timeframe, 'Try a wider date range.')}"
    )
    st.stop()

df = compute_indicators(df)
df = df.dropna(subset=["close"]).reset_index(drop=True)

if df.empty:
    st.error(f"Not enough bars after indicator calculation for {ticker}. Try a wider date range.")
    st.stop()

# Warn if key indicators couldn't be computed (NaN across all rows)
missing = [c for c in ["RSI", "ADX", "MACD", "EMA_50"]
           if c in df.columns and df[c].isna().all()]
if missing:
    st.info(
        f"⚠️ Some indicators unavailable with the current range/timeframe "
        f"({', '.join(missing)} require more bars). Those sections will show N/A. "
        f"Try a wider date range for full coverage."
    )

df     = triple_supertrend_signals(df)
df     = compute_chart3_signals(df, atr_trail_mult=atr_trail_mult)
latest = df.iloc[-1]

if pd.isna(latest["EMA_200"]) and timeframe in ("Weekly", "Monthly"):
    st.warning("⚠️ EMA 200 not available — extend range to 4+ years for full indicator coverage on this timeframe.")

price         = info.get("price") or float(latest["close"])
score, bullish_count, bearish_count, indicators = composite_score(latest)
neutral_count = 10 - bullish_count - bearish_count
label         = signal_label(score)
ts            = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
swing_signals = swing_trade_commentary(latest, df, ticker, price, score, info)
trend_label, trend_color = detect_trend_structure(df, window=3)
swing_summary  = compute_swing_summary(df, price, is_swing_suitable=(ticker in SWING_TRADE_SUITABLE))

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

# ── SWING TRADE KEY INSIGHTS ───────────────────────────────────
ss = swing_summary
ac = ss["action_color"]
bp = ss["bull_pts"]; brp = ss["bear_pts"]
total_pts = bp + brp
bull_bar_pct = int(bp / total_pts * 100) if total_pts > 0 else 50

st.markdown(
    f"""
    <div style='border:1px solid {ac}; border-radius:8px; padding:16px 20px; margin:12px 0;
                background:rgba(0,0,0,0.25)'>
      <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px'>
        <span style='font-size:1.1rem; font-weight:700; color:{ac}'>
          📊 Swing Indicator Summary — {ss["action"]}
        </span>
        <span style='font-size:0.8rem; color:#9ca3af'>
          Based on: Divergence · Ichimoku · OBV · Fibonacci · Parabolic SAR
        </span>
      </div>

      <div style='background:#1f2937; border-radius:4px; height:8px; margin-bottom:10px; overflow:hidden'>
        <div style='background:{ac}; height:100%; width:{bull_bar_pct}%; border-radius:4px'></div>
      </div>
      <div style='display:flex; justify-content:space-between; font-size:0.75rem; color:#6b7280; margin-bottom:12px'>
        <span>🐻 Bear signals: {brp}</span>
        <span>🐂 Bull signals: {bp}</span>
      </div>

      <p style='font-size:0.9rem; color:#d1d5db; margin:0 0 10px 0'>{ss["action_detail"]}</p>

      {"<div style='border-left:3px solid #f97316; padding:6px 10px; margin-bottom:10px; background:rgba(249,115,22,0.08); border-radius:0 4px 4px 0'><span style='font-size:0.85rem; color:#f97316'>" + ss["take_profit_note"] + "</span></div>" if ss["take_profit_note"] else ""}

      <div style='border-top:1px solid #374151; padding-top:10px; margin-top:4px'>
        <span style='font-size:0.78rem; color:#9ca3af; font-weight:600; text-transform:uppercase; letter-spacing:0.05em'>Key signals</span>
        <ul style='margin:6px 0 0 0; padding-left:18px'>
          {"".join(f"<li style='font-size:0.82rem; color:#d1d5db; margin-bottom:3px'>{n}</li>" for n in ss["notes"]) if ss["notes"] else "<li style='font-size:0.82rem; color:#6b7280'>No strong signals detected — indicators may need more data</li>"}
        </ul>
      </div>
      <p style='font-size:0.72rem; color:#4b5563; margin:10px 0 0 0'>
        ⚠️ Not financial advice. Scroll to Swing Trade Indicators section below for full charts.
      </p>
    </div>
    """,
    unsafe_allow_html=True)

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
    "green": "#22c55e", "yellow": "#fbbf24", "red": "#f87171",
    "gray":  "#9ca3af", "orange": "#f97316", "blue": "#60a5fa", "purple": "#c084fc",
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

st.caption("Not financial advice. All signals are algorithmic. Confirm with volume and broader market context.")
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

# ── TRIPLE SUPERTREND STATUS ───────────────────────────────────

st.markdown("### 📊 Triple Supertrend — Current Status")
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
    status_line("All 3 Supertrends bullish — strong trend alignment, high-conviction long setup.", "strong_bullish")
elif bulls == 2:
    status_line("2 of 3 Supertrends bullish — leaning up, wait for ST3 (21,1) to confirm.", "bullish")
elif bears == 3:
    status_line("All 3 Supertrends bearish — strong downtrend, avoid new longs.", "strong_bearish")
elif bears == 2:
    status_line("2 of 3 Supertrends bearish — leaning down, caution on long entries.", "bearish")
else:
    status_line("Supertrends conflicted — mixed directions, wait for alignment.", "neutral")

# ── CHART 1 ────────────────────────────────────────────────────

st.markdown("### Chart 1 — ST Lines (any flip signals)")
fig1 = build_chart1(df)
st.plotly_chart(fig1, use_container_width=True)
st.caption(
    "Each ST line changes colour with direction: "
    "green = bullish support below price, red = bearish resistance above price. "
    "Blue ▲ = any ST flipped bullish. Orange ▼ = any ST flipped bearish.")

st.divider()

# ── CHART 2 ────────────────────────────────────────────────────

st.markdown("### Chart 2 — Nordman-style (candle tint + full-alignment arrows)")
fig2 = build_chart2(df)
st.plotly_chart(fig2, use_container_width=True)
st.caption(
    "Candle colour reflects ST alignment: bright green = 3/3 bull, light green = 2/3, light red = 1/3, bright red = 0/3. "
    "Green background = all 3 bullish. Red background = all 3 bearish. "
    "EMA 200 (yellow dotted). "
    "Blue ▲ signal: all 3 STs green + price above EMA 200 + StochRSI K < 20. "
    "Orange ▼ signal: all 3 STs red + price below EMA 200 + StochRSI K > 80.")

st.divider()

# ── CHART 3 ────────────────────────────────────────────────────

st.markdown("### Chart 3 — RSI + Fibonacci BB (long-only signals)")

# Chart 3 current status
c3_bulls = sum(1 for d in [latest.get("c3_dir1", 0), latest.get("c3_dir2", 0), latest.get("c3_dir3", 0)] if d == 1)
c3_bears = sum(1 for d in [latest.get("c3_dir1", 0), latest.get("c3_dir2", 0), latest.get("c3_dir3", 0)] if d == -1)
c3_rsi7  = latest.get("c3_rsi7", float("nan"))

def st3_badge(direction):
    return val_span("▲ Bull", "#00e676") if direction == 1 else val_span("▼ Bear", "#ff6d00")

st.markdown(
    f"<p style='font-size:0.9rem; margin-bottom:2px'>"
    f"ST1 (10,1.0): {st3_badge(latest.get('c3_dir1',0))} &nbsp;&nbsp; "
    f"ST2 (11,2.0): {st3_badge(latest.get('c3_dir2',0))} &nbsp;&nbsp; "
    f"ST3 (12,3.0): {st3_badge(latest.get('c3_dir3',0))} &nbsp;&nbsp; "
    f"RSI(7): {val_span(f'{c3_rsi7:.1f}' if pd.notna(c3_rsi7) else 'N/A', '#fbbf24')}"
    f"</p>",
    unsafe_allow_html=True)

if c3_bulls == 3 and pd.notna(c3_rsi7) and c3_rsi7 > 50:
    status_line("All 3 STs bullish + RSI(7) > 50 — BUY conditions met. Watch for first alignment bar.", "strong_bullish")
elif c3_bulls == 3:
    status_line("All 3 STs bullish but RSI(7) ≤ 50 — alignment without momentum confirmation, not a valid entry yet.", "caution")
elif c3_bears == 3:
    status_line("All 3 STs bearish — downtrend. No long entry. Hold cash and wait for STs to turn green.", "bearish")
else:
    status_line("STs conflicted — no entry signal. Wait for all 3 STs to align green + RSI(7) > 50.", "neutral")

fig3 = build_chart3(df)
st.plotly_chart(fig3, use_container_width=True)
st.caption(
    "Long-only strategy. ST params: (10,1.0), (11,2.0), (12,3.0). "
    "Blue ▲ BUY: all 3 STs turn green + RSI(7) > 50, on the first bar of alignment. "
    "Yellow ◆ EXIT: close long when any ST turns red, upper Fib BB hit, or price closes below ATR trail stop. "
    f"Orange dashed line = ATR(14) trail stop ({atr_trail_mult:.1f}× ATR below highest close since entry) — only shown while in a long. "
    "Adjust multiplier in the sidebar — lower = tighter, higher = more room.")

st.divider()

# ── RSI + Stoch ────────────────────────────────────────────────

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
            status_line(f"RSI {rsi_f:.1f} & Stoch K {k_f:.1f} — both deeply oversold, high probability bounce.", "strong_bullish")
        elif rsi_f < 40:
            status_line(f"RSI {rsi_f:.1f} — oversold, watching for Stoch K to confirm reversal.", "bullish")
        elif rsi_f > 70 and k_f > 80:
            status_line(f"RSI {rsi_f:.1f} & Stoch K {k_f:.1f} — both overbought, elevated pullback risk.", "strong_bearish")
        elif rsi_f > 70:
            status_line(f"RSI {rsi_f:.1f} — overbought, momentum may be fading.", "bearish")
        else:
            status_line(f"RSI {rsi_f:.1f} & Stoch K {k_f:.1f} — neutral range.", "neutral")
    fig_rsi = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.5],
                            subplot_titles=("RSI (14)", "Stochastic RSI"))
    fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["RSI"],
        name="RSI", line=dict(color="#F59E0B", width=2)), row=1, col=1)
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   row=1, col=1)
    fig_rsi.add_hline(y=40, line_dash="dash", line_color="green", row=1, col=1)
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="lime",  row=1, col=1)
    fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.05, row=1, col=1)
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red",   opacity=0.05, row=1, col=1)
    fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["StochRSI_k"],
        name="Stoch K", line=dict(color="#60A5FA", width=2)), row=2, col=1)
    fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["StochRSI_d"],
        name="Stoch D", line=dict(color="#F472B6", width=1, dash="dot")), row=2, col=1)
    fig_rsi.add_hline(y=80, line_dash="dash", line_color="red",  row=2, col=1)
    fig_rsi.add_hline(y=20, line_dash="dash", line_color="lime", row=2, col=1)
    fig_rsi.update_layout(height=500, margin=dict(t=30, b=10))
    st.plotly_chart(fig_rsi, use_container_width=True)
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
            status_line(f"Price below lower BB — oversold. {cross} active.", cross_sent)
        elif price > latest["BB_upper"]:
            status_line(f"Price above upper BB — overbought. {cross} active.", "bearish")
        elif bb_f < 0.2:
            status_line(f"Price near lower BB ({bb_f*100:.0f}%) — leaning oversold. {cross}.", cross_sent)
        elif bb_f > 0.8:
            status_line(f"Price near upper BB ({bb_f*100:.0f}%) — extended. {cross}.", "caution")
        else:
            status_line(f"Price mid-band ({bb_f*100:.0f}%), {cross} active.", cross_sent)
    elif pd.notna(bb_f):
        status_line("EMA 200 not available — extend date range for EMA cross signal.", "neutral")
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price"))
    fig_bb.add_trace(go.Scatter(x=df["time"], y=df["BB_upper"],
        name="BB Upper", line=dict(color="gray", dash="dot", width=1)))
    fig_bb.add_trace(go.Scatter(x=df["time"], y=df["BB_lower"],
        name="BB Lower", line=dict(color="gray", dash="dot", width=1),
        fill="tonexty", fillcolor="rgba(128,128,128,0.1)"))
    fig_bb.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"],
        name="BB Mid", line=dict(color="gray", width=1)))
    fig_bb.add_trace(go.Scatter(x=df["time"], y=df["EMA_50"],
        name="EMA 50", line=dict(color="#34D399", width=1.5)))
    fig_bb.add_trace(go.Scatter(x=df["time"], y=df["EMA_200"],
        name="EMA 200", line=dict(color="#F87171", width=1.5)))
    support_s, resistance_s = detect_support_resistance(df)
    for s in support_s:
        fig_bb.add_hline(y=s, line_dash="dot", line_color="lime", opacity=0.5,
                         annotation_text=f"S ${s:,.4f}", annotation_position="bottom left")
    for r in resistance_s:
        fig_bb.add_hline(y=r, line_dash="dot", line_color="tomato", opacity=0.5,
                         annotation_text=f"R ${r:,.4f}", annotation_position="top left")
    fig_bb.update_layout(height=500, margin=dict(t=30, b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_bb, use_container_width=True)
    st.caption("BB below lower = oversold; above upper = overbought. EMA50 (green) / EMA200 (red).")

st.divider()

# ── MACD + ADX ─────────────────────────────────────────────────

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
            status_line(f"MACD above signal, histogram positive — bullish momentum confirmed.", "bullish")
        elif mv > ms and mh < 0:
            status_line(f"MACD above signal but histogram shrinking — momentum weakening.", "caution")
        elif mv < ms and mh < 0:
            status_line(f"MACD below signal, histogram negative — bearish momentum.", "bearish")
        else:
            status_line(f"MACD crossing signal — potential momentum shift, wait for confirmation.", "caution")
    fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.45, 0.55],
                             subplot_titles=("Price", "MACD"))
    fig_macd.add_trace(go.Scatter(x=df["time"], y=df["close"],
        name="Price", line=dict(color="#F59E0B")), row=1, col=1)
    fig_macd.add_trace(go.Scatter(x=df["time"], y=df["MACD"],
        name="MACD", line=dict(color="#60A5FA", width=2)), row=2, col=1)
    fig_macd.add_trace(go.Scatter(x=df["time"], y=df["MACD_signal"],
        name="Signal", line=dict(color="#F472B6", width=1.5)), row=2, col=1)
    colors_macd = ["green" if v >= 0 else "red" for v in df["MACD_hist"].fillna(0)]
    fig_macd.add_trace(go.Bar(x=df["time"], y=df["MACD_hist"],
        name="Histogram", marker_color=colors_macd), row=2, col=1)
    fig_macd.update_layout(height=500, margin=dict(t=30, b=10))
    st.plotly_chart(fig_macd, use_container_width=True)
    st.caption("MACD above signal = bullish. Histogram turning green = momentum shift.")

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
            status_line(f"ADX {adx_f:.1f} — weak/no trend, choppy. TA signals less reliable.", "neutral")
        elif adx_f >= 25 and dp > dn:
            status_line(f"ADX {adx_f:.1f} — strong trend, bullish direction (+DI {dp:.1f} > -DI {dn:.1f}).", "bullish")
        elif adx_f >= 25 and dn > dp:
            status_line(f"ADX {adx_f:.1f} — strong trend, bearish direction (-DI {dn:.1f} > +DI {dp:.1f}).", "bearish")
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
    st.caption("ADX >25 = trending market. <20 = choppy, avoid breakout entries.")

st.divider()

# ── CCI + Williams %R ──────────────────────────────────────────

row3_l, row3_r = st.columns(2)

with row3_l:
    st.markdown(f"<h4>CCI &nbsp;{val_span('CCI: ' + cci_val, '#F59E0B')}</h4>",
                unsafe_allow_html=True)
    cci_f = latest["CCI"]
    if pd.notna(cci_f):
        if cci_f < -100:
            status_line(f"CCI {cci_f:.0f} — oversold below -100, potential swing buy zone.", "bullish")
        elif cci_f > 100:
            status_line(f"CCI {cci_f:.0f} — overbought above +100, consider tightening stops.", "bearish")
        elif cci_f < 0:
            status_line(f"CCI {cci_f:.0f} — mildly negative.", "neutral")
        else:
            status_line(f"CCI {cci_f:.0f} — mildly positive.", "neutral")
    fig_cci = go.Figure()
    fig_cci.add_trace(go.Scatter(x=df["time"], y=df["CCI"],
        name="CCI", line=dict(color="#F59E0B", width=2)))
    fig_cci.add_hline(y=100,  line_dash="dash", line_color="red")
    fig_cci.add_hline(y=-100, line_dash="dash", line_color="lime")
    fig_cci.add_hrect(y0=-300, y1=-100, fillcolor="green", opacity=0.05)
    fig_cci.add_hrect(y0=100,  y1=300,  fillcolor="red",   opacity=0.05)
    fig_cci.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_cci, use_container_width=True)
    st.caption("CCI below -100 = oversold. Above +100 = overbought.")

with row3_r:
    st.markdown(f"<h4>Williams %R &nbsp;{val_span('%R: ' + wr_val, '#60A5FA')}</h4>",
                unsafe_allow_html=True)
    wr_f = latest["WilliamsR"]
    if pd.notna(wr_f):
        if wr_f < -80:
            status_line(f"Williams %R {wr_f:.1f} — oversold, look for bounce confirmation.", "bullish")
        elif wr_f > -20:
            status_line(f"Williams %R {wr_f:.1f} — overbought, elevated reversal risk.", "bearish")
        else:
            status_line(f"Williams %R {wr_f:.1f} — neutral mid-range.", "neutral")
    fig_wr = go.Figure()
    fig_wr.add_trace(go.Scatter(x=df["time"], y=df["WilliamsR"],
        name="Williams %R", line=dict(color="#60A5FA", width=2)))
    fig_wr.add_hline(y=-20, line_dash="dash", line_color="red")
    fig_wr.add_hline(y=-80, line_dash="dash", line_color="lime")
    fig_wr.add_hrect(y0=-100, y1=-80, fillcolor="green", opacity=0.05)
    fig_wr.add_hrect(y0=-20,  y1=0,   fillcolor="red",   opacity=0.05)
    fig_wr.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_wr, use_container_width=True)
    st.caption("Williams %R below -80 = oversold. Above -20 = overbought.")

st.divider()

# ── ROC + ATR ──────────────────────────────────────────────────

row4_l, row4_r = st.columns(2)

with row4_l:
    st.markdown(f"<h4>ROC &nbsp;{val_span('ROC: ' + roc_val, '#F59E0B')}</h4>",
                unsafe_allow_html=True)
    roc_f = latest["ROC"]
    if pd.notna(roc_f):
        if roc_f > 10:
            status_line(f"ROC {roc_f:.1f}% — strong positive momentum.", "strong_bullish")
        elif roc_f > 5:
            status_line(f"ROC {roc_f:.1f}% — positive momentum building.", "bullish")
        elif roc_f < -10:
            status_line(f"ROC {roc_f:.1f}% — strong negative momentum.", "strong_bearish")
        elif roc_f < -5:
            status_line(f"ROC {roc_f:.1f}% — negative momentum, caution on new longs.", "bearish")
        else:
            status_line(f"ROC {roc_f:.1f}% — momentum flat near zero.", "neutral")
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
    st.markdown(f"<h4>ATR &nbsp;{val_span('ATR: ' + atr_val, '#C084FC')}</h4>",
                unsafe_allow_html=True)
    atr_f = latest["ATR"]
    if pd.notna(atr_f):
        atr_pct    = (atr_f / price) * 100
        recent_atr = df["ATR"].tail(20).mean()
        if atr_f > recent_atr * 1.2:
            status_line(f"ATR ${atr_f:.4f} ({atr_pct:.1f}%) — volatility expanding, widen stops.", "caution")
        elif atr_f < recent_atr * 0.8:
            status_line(f"ATR ${atr_f:.4f} ({atr_pct:.1f}%) — volatility contracting, breakout may be near.", "caution")
        else:
            status_line(f"ATR ${atr_f:.4f} ({atr_pct:.1f}%) — normal volatility, use 1.5x ATR for stops.", "neutral")
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=df["time"], y=df["ATR"],
        name="ATR", line=dict(color="#C084FC", width=2)))
    fig_atr.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_atr, use_container_width=True)
    st.caption("ATR rising = increasing volatility. Size stops at 1.5-2x ATR.")

st.divider()

# ── TTM Squeeze + Volume ───────────────────────────────────────

row5_l, row5_r = st.columns(2)

with row5_l:
    st.markdown(f"<h4>TTM Squeeze &nbsp;{val_span('Squeeze: ' + sq_val, sq_color)}</h4>",
                unsafe_allow_html=True)
    sq_f = latest["squeeze"]
    sh_f = latest["squeeze_hist"]
    if pd.notna(sq_f) and pd.notna(sh_f):
        if sq_f and sh_f > 0:
            status_line("Squeeze ON — coiling with bullish momentum building. Watch for release.", "caution")
        elif sq_f and sh_f < 0:
            status_line("Squeeze ON — coiling but bearish momentum. Release likely downside.", "caution")
        elif not sq_f and sh_f > 0:
            status_line("Squeeze FIRED — momentum released to the upside. Breakout signal.", "strong_bullish")
        elif not sq_f and sh_f < 0:
            status_line("Squeeze FIRED — momentum released to the downside. Breakdown signal.", "strong_bearish")
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
    st.caption("Black dots = coiling for breakout. Lime = released. Green bars = bullish breakout signal.")

with row5_r:
    st.markdown("<h4>Volume</h4>", unsafe_allow_html=True)
    if "volume" in df.columns and len(df) > 1:
        last_vol   = df["volume"].iloc[-1]
        avg_vol_20 = df["volume"].tail(20).mean()
        vol_ratio  = last_vol / avg_vol_20 if avg_vol_20 > 0 else 1
        up_day     = df["close"].iloc[-1] >= df["close"].iloc[-2]
        if vol_ratio > 1.5 and up_day:
            status_line(f"Volume {vol_ratio:.1f}x above average on up day — strong buying conviction.", "bullish")
        elif vol_ratio > 1.5 and not up_day:
            status_line(f"Volume {vol_ratio:.1f}x above average on down day — strong selling pressure.", "bearish")
        elif vol_ratio < 0.6:
            status_line(f"Volume {vol_ratio:.1f}x below average — low conviction.", "neutral")
        else:
            status_line(f"Volume {vol_ratio:.1f}x average — normal activity.", "neutral")
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
        st.caption("Green = up day; red = down day. Volume above average on breakouts = confirmation.")
    else:
        st.info("Volume data not available.")


st.divider()

# ══════════════════════════════════════════════════════════════════
# SWING TRADE INDICATORS SECTION
# 1. RSI / MACD Divergence
# 2. Ichimoku Cloud
# 3. On Balance Volume (OBV)
# 4. Fibonacci Retracement
# 5. Parabolic SAR
# ══════════════════════════════════════════════════════════════════

st.markdown("## 📐 Swing Trade Indicators")
st.caption("Supplementary indicators specifically suited to swing trading entries and exits.")
st.divider()

# ── Helper: detect divergence ──────────────────────────────────

def detect_divergence(price_series, indicator_series, window=5, lookback=60):
    """
    Scan the last `lookback` bars for:
      - Bullish divergence: price makes lower low, indicator makes higher low
      - Bearish divergence: price makes higher high, indicator makes lower high
    Returns list of (index, type) tuples.
    """
    results = []
    p = price_series.values
    ind = indicator_series.values
    n = len(p)
    start = max(window, n - lookback)
    for i in range(start + window, n - 1):
        # local low check
        if all(p[i] <= p[i-j] for j in range(1, window+1)) and \
           all(p[i] <= p[i+j] for j in range(1, min(window+1, n-i))):
            # find prior local low in lookback
            for j in range(i - window, max(start, i - lookback), -1):
                if all(p[j] <= p[j-k] for k in range(1, min(window+1, j+1))) and \
                   all(p[j] <= p[j+k] for k in range(1, min(window+1, i-j))):
                    if p[i] < p[j] and ind[i] > ind[j]:   # bullish div
                        results.append((i, "bullish"))
                    break
        # local high check
        if all(p[i] >= p[i-j] for j in range(1, window+1)) and \
           all(p[i] >= p[i+j] for j in range(1, min(window+1, n-i))):
            for j in range(i - window, max(start, i - lookback), -1):
                if all(p[j] >= p[j-k] for k in range(1, min(window+1, j+1))) and \
                   all(p[j] >= p[j+k] for k in range(1, min(window+1, i-j))):
                    if p[i] > p[j] and ind[i] < ind[j]:   # bearish div
                        results.append((i, "bearish"))
                    break
    return results


# ── 1. RSI / MACD DIVERGENCE ───────────────────────────────────

st.markdown("### 1. RSI & MACD Divergence")

if df["RSI"].notna().any() and df["MACD"].notna().any():
    # Reuse pre-computed values from swing_summary
    rsi_divs       = swing_summary["rsi_divs"]
    macd_divs      = swing_summary["macd_divs"]
    recent_rsi_bull  = swing_summary["div_rsi_bull"]
    recent_rsi_bear  = swing_summary["div_rsi_bear"]
    recent_macd_bull = swing_summary["div_macd_bull"]
    recent_macd_bear = swing_summary["div_macd_bear"]

    if recent_rsi_bull or recent_macd_bull:
        status_line(
            f"Bullish divergence detected ({'RSI' if recent_rsi_bull else ''}"
            f"{' & ' if recent_rsi_bull and recent_macd_bull else ''}"
            f"{'MACD' if recent_macd_bull else ''}) — "
            "price made lower low but momentum didn't follow. Potential reversal up.",
            "bullish")
    elif recent_rsi_bear or recent_macd_bear:
        status_line(
            f"Bearish divergence detected ({'RSI' if recent_rsi_bear else ''}"
            f"{' & ' if recent_rsi_bear and recent_macd_bear else ''}"
            f"{'MACD' if recent_macd_bear else ''}) — "
            "price made higher high but momentum weakening. Caution on new longs.",
            "bearish")
    else:
        status_line("No significant divergence detected in recent bars — trend and momentum are aligned.", "neutral")

    # Chart
    fig_div = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Price", "RSI (14)", "MACD"))

    fig_div.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e",  decreasing_fillcolor="#ef4444"),
        row=1, col=1)

    # Mark divergence points on price
    for idx, div_type in rsi_divs:
        color = "#0077BB" if div_type == "bullish" else "#EE7733"
        sym   = "triangle-up" if div_type == "bullish" else "triangle-down"
        y_val = float(df["low"].iloc[idx]) * 0.993 if div_type == "bullish" \
                else float(df["high"].iloc[idx]) * 1.007
        fig_div.add_trace(go.Scatter(
            x=[df["time"].iloc[idx]], y=[y_val],
            mode="markers", showlegend=False,
            marker=dict(symbol=sym, size=12, color=color,
                        line=dict(color="white", width=1))),
            row=1, col=1)

    fig_div.add_trace(go.Scatter(x=df["time"], y=df["RSI"],
        name="RSI", line=dict(color="#F59E0B", width=2)), row=2, col=1)
    fig_div.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
    fig_div.add_hline(y=30, line_dash="dash", line_color="lime",  row=2, col=1)

    # Mark RSI divergence points
    for idx, div_type in rsi_divs:
        color = "#0077BB" if div_type == "bullish" else "#EE7733"
        sym   = "triangle-up" if div_type == "bullish" else "triangle-down"
        fig_div.add_trace(go.Scatter(
            x=[df["time"].iloc[idx]], y=[df["RSI"].iloc[idx]],
            mode="markers", showlegend=False,
            marker=dict(symbol=sym, size=10, color=color,
                        line=dict(color="white", width=1))),
            row=2, col=1)

    fig_div.add_trace(go.Scatter(x=df["time"], y=df["MACD"],
        name="MACD", line=dict(color="#60A5FA", width=2)), row=3, col=1)
    fig_div.add_trace(go.Scatter(x=df["time"], y=df["MACD_signal"],
        name="Signal", line=dict(color="#F472B6", width=1.5)), row=3, col=1)
    hist_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["MACD_hist"].fillna(0)]
    fig_div.add_trace(go.Bar(x=df["time"], y=df["MACD_hist"],
        name="Hist", marker_color=hist_colors), row=3, col=1)

    # Mark MACD divergence points
    for idx, div_type in macd_divs:
        color = "#0077BB" if div_type == "bullish" else "#EE7733"
        sym   = "triangle-up" if div_type == "bullish" else "triangle-down"
        fig_div.add_trace(go.Scatter(
            x=[df["time"].iloc[idx]], y=[df["MACD"].iloc[idx]],
            mode="markers", showlegend=False,
            marker=dict(symbol=sym, size=10, color=color,
                        line=dict(color="white", width=1))),
            row=3, col=1)

    fig_div.update_layout(height=650, margin=dict(t=40, b=10),
                          xaxis_rangeslider_visible=False,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_div, use_container_width=True)
    st.caption(
        "Blue ▲ = bullish divergence (price lower low, indicator higher low) — momentum not confirming the sell-off, "
        "potential reversal up. Orange ▼ = bearish divergence (price higher high, indicator lower high) — "
        "momentum weakening into new highs, watch for rollover. Divergence is highest-conviction on RSI and MACD together.")
else:
    st.info("Not enough data to compute divergence. Try a wider date range.")

st.divider()

# ── 2. ICHIMOKU CLOUD ──────────────────────────────────────────

st.markdown("### 2. Ichimoku Cloud")

def compute_ichimoku(df):
    h, l, c = df["high"], df["low"], df["close"]
    tenkan  = (h.rolling(9).max()  + l.rolling(9).min())  / 2
    kijun   = (h.rolling(26).max() + l.rolling(26).min()) / 2
    ssa     = ((tenkan + kijun) / 2).shift(26)
    ssb     = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    chikou  = c.shift(-26)
    return tenkan, kijun, ssa, ssb, chikou

if len(df) >= 52:
    tenkan, kijun, ssa, ssb, chikou = compute_ichimoku(df)
    latest_t = tenkan.iloc[-1]
    latest_k = kijun.iloc[-1]
    latest_ssa = ssa.iloc[-1]
    latest_ssb = ssb.iloc[-1]
    price_now  = float(latest["close"])
    cloud_top    = max(latest_ssa, latest_ssb) if pd.notna(latest_ssa) and pd.notna(latest_ssb) else np.nan
    cloud_bottom = min(latest_ssa, latest_ssb) if pd.notna(latest_ssa) and pd.notna(latest_ssb) else np.nan

    # Status
    if pd.notna(cloud_top) and pd.notna(latest_t) and pd.notna(latest_k):
        if price_now > cloud_top and latest_t > latest_k:
            status_line(
                f"Price above cloud + Tenkan ({latest_t:.4f}) above Kijun ({latest_k:.4f}) — "
                "strong bullish structure. Trend is up, cloud is support.", "strong_bullish")
        elif price_now > cloud_top:
            status_line(
                f"Price above cloud but Tenkan ≤ Kijun — uptrend intact but momentum softening. "
                "Watch for Tenkan/Kijun cross.", "bullish")
        elif price_now < cloud_bottom and latest_t < latest_k:
            status_line(
                f"Price below cloud + Tenkan ({latest_t:.4f}) below Kijun ({latest_k:.4f}) — "
                "strong bearish structure. Avoid new longs.", "strong_bearish")
        elif price_now < cloud_bottom:
            status_line("Price below cloud — bearish trend. Wait for price to reclaim cloud before entering.", "bearish")
        else:
            status_line("Price inside the cloud — choppy/transitional zone. Wait for a clean break above or below.", "neutral")
    else:
        status_line("Insufficient data for full Ichimoku reading — extend range.", "neutral")

    fig_ichi = go.Figure()

    # Cloud fill (green when SSA > SSB, red when SSB > SSA)
    cloud_green_ssa = np.where(ssa >= ssb, ssa, ssb)
    cloud_green_ssb = np.where(ssa >= ssb, ssb, ssa)

    fig_ichi.add_trace(go.Scatter(
        x=df["time"], y=cloud_green_ssa,
        name="SSA", line=dict(color="rgba(34,197,94,0.3)", width=0),
        showlegend=False, hoverinfo="skip"))
    fig_ichi.add_trace(go.Scatter(
        x=df["time"], y=cloud_green_ssb,
        name="SSB", line=dict(color="rgba(239,68,68,0.3)", width=0),
        fill="tonexty", fillcolor="rgba(34,197,94,0.15)",
        showlegend=False, hoverinfo="skip"))

    fig_ichi.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e",  decreasing_fillcolor="#ef4444"))

    fig_ichi.add_trace(go.Scatter(x=df["time"], y=tenkan,
        name="Tenkan (9)", line=dict(color="#f87171", width=1.5)))
    fig_ichi.add_trace(go.Scatter(x=df["time"], y=kijun,
        name="Kijun (26)", line=dict(color="#60a5fa", width=1.5)))
    fig_ichi.add_trace(go.Scatter(x=df["time"], y=ssa,
        name="SSA (leading A)", line=dict(color="#22c55e", width=1, dash="dot")))
    fig_ichi.add_trace(go.Scatter(x=df["time"], y=ssb,
        name="SSB (leading B)", line=dict(color="#ef4444", width=1, dash="dot")))
    fig_ichi.add_trace(go.Scatter(x=df["time"], y=chikou,
        name="Chikou (lagging)", line=dict(color="#c084fc", width=1, dash="dash")))

    fig_ichi.update_layout(
        height=600, margin=dict(t=40, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        plot_bgcolor="#0e1117", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_ichi, use_container_width=True)
    st.caption(
        "Tenkan (red, 9-period) = short-term momentum line. Kijun (blue, 26-period) = medium-term baseline / dynamic S&R. "
        "Cloud (green = bullish, red = bearish) = future support/resistance zone. "
        "Price above green cloud = bullish. Price below red cloud = bearish. Price inside cloud = indecision. "
        "Tenkan crossing above Kijun = bullish signal (TK cross). Chikou (purple) above price = bullish confirmation. "
        "Best swing entry: price breaks above cloud + Tenkan > Kijun + Chikou clear of price.")
else:
    st.info("Ichimoku requires at least 52 bars. Try a wider date range.")

st.divider()

# ── 3. ON BALANCE VOLUME (OBV) ─────────────────────────────────

st.markdown("### 3. On Balance Volume (OBV)")

if "volume" in df.columns and df["volume"].notna().any():
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    obv_series = pd.Series(obv, index=df.index)
    obv_ema    = obv_series.ewm(span=20).mean()

    # Divergence: compare last 20 bars trend of price vs OBV
    price_trend = df["close"].iloc[-1] - df["close"].iloc[-20] if len(df) >= 20 else 0
    obv_trend   = obv_series.iloc[-1]  - obv_series.iloc[-20]  if len(df) >= 20 else 0

    if price_trend > 0 and obv_trend > 0:
        status_line(
            f"OBV rising with price — volume confirming the uptrend. Institutional accumulation likely.", "bullish")
    elif price_trend > 0 and obv_trend < 0:
        status_line(
            "Bearish OBV divergence — price rising but OBV falling. Distribution on strength. "
            "Smart money may be selling into rallies. Caution on new longs.", "bearish")
    elif price_trend < 0 and obv_trend < 0:
        status_line(
            "OBV falling with price — volume confirming the downtrend. Avoid new longs.", "bearish")
    elif price_trend < 0 and obv_trend > 0:
        status_line(
            "Bullish OBV divergence — price falling but OBV rising. Accumulation on weakness. "
            "Smart money may be buying the dip. Watch for price reversal.", "bullish")
    else:
        status_line("OBV and price broadly aligned — no divergence signal.", "neutral")

    fig_obv = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.5],
                            subplot_titles=("Price", "On Balance Volume"))
    fig_obv.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e",  decreasing_fillcolor="#ef4444"),
        row=1, col=1)
    fig_obv.add_trace(go.Scatter(x=df["time"], y=obv_series,
        name="OBV", line=dict(color="#60A5FA", width=2)), row=2, col=1)
    fig_obv.add_trace(go.Scatter(x=df["time"], y=obv_ema,
        name="OBV EMA (20)", line=dict(color="#F59E0B", width=1.5, dash="dot")), row=2, col=1)
    fig_obv.update_layout(height=550, margin=dict(t=40, b=10),
                          xaxis_rangeslider_visible=False,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_obv, use_container_width=True)
    st.caption(
        "OBV adds volume on up-close days, subtracts on down-close days — tracks whether volume is flowing in or out. "
        "OBV rising with price = healthy uptrend, volume confirms. "
        "OBV falling while price rises = bearish divergence, distribution — smart money selling into strength. "
        "OBV rising while price falls = bullish divergence, accumulation — smart money buying the dip. "
        "Yellow dotted line = 20-period EMA of OBV — OBV above its EMA = bullish bias.")
else:
    st.info("Volume data not available for OBV.")

st.divider()

# ── 4. FIBONACCI RETRACEMENT ───────────────────────────────────

st.markdown("### 4. Fibonacci Retracement")

lookback_fib = min(len(df), 60)
fib_df       = df.tail(lookback_fib)
swing_high   = float(fib_df["high"].max())
swing_low    = float(fib_df["low"].min())
fib_range    = swing_high - swing_low
price_now    = float(latest["close"])

FIB_LEVELS = {
    "0%   (High)": 0.0,
    "23.6%":       0.236,
    "38.2%":       0.382,
    "50%":         0.500,
    "61.8% (Golden)": 0.618,
    "78.6%":       0.786,
    "100% (Low)":  1.0,
}
fib_prices = {lbl: swing_high - ratio * fib_range for lbl, ratio in FIB_LEVELS.items()}

# Find which zone price is currently in
current_zone = None
sorted_levels = sorted(fib_prices.items(), key=lambda x: x[1], reverse=True)
for i in range(len(sorted_levels) - 1):
    upper_lbl, upper_val = sorted_levels[i]
    lower_lbl, lower_val = sorted_levels[i+1]
    if lower_val <= price_now <= upper_val:
        current_zone = (lower_lbl, lower_val, upper_lbl, upper_val)
        break

if current_zone:
    lower_lbl, lower_val, upper_lbl, upper_val = current_zone
    pct_through = (price_now - lower_val) / (upper_val - lower_val) * 100 if upper_val != lower_val else 50
    near_support = pct_through < 25
    near_resist  = pct_through > 75
    if "61.8" in lower_lbl or "50%" in lower_lbl or "38.2" in lower_lbl:
        if near_support:
            status_line(
                f"Price at ${price_now:.4f} — sitting on the {lower_lbl} Fibonacci support level (${lower_val:.4f}). "
                "Key retracement zone, high-probability swing long entry area.", "bullish")
        else:
            status_line(
                f"Price at ${price_now:.4f} — between {lower_lbl} (${lower_val:.4f}) and "
                f"{upper_lbl} (${upper_val:.4f}) Fibonacci levels. Next resistance: ${upper_val:.4f}.", "neutral")
    elif near_resist:
        status_line(
            f"Price approaching {upper_lbl} resistance at ${upper_val:.4f}. "
            "Watch for rejection here — consider tightening stops.", "caution")
    else:
        status_line(
            f"Price at ${price_now:.4f} between {lower_lbl} (${lower_val:.4f}) and "
            f"{upper_lbl} (${upper_val:.4f}). No key level immediately in play.", "neutral")
else:
    status_line(f"Price at ${price_now:.4f} — outside current Fibonacci range.", "neutral")

fig_fib = go.Figure()
fig_fib.add_trace(go.Candlestick(
    x=fib_df["time"], open=fib_df["open"], high=fib_df["high"],
    low=fib_df["low"], close=fib_df["close"], name="Price",
    increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
    increasing_fillcolor="#22c55e",  decreasing_fillcolor="#ef4444"))

FIB_COLORS = {
    "0%   (High)":      "#9ca3af",
    "23.6%":            "#60a5fa",
    "38.2%":            "#34d399",
    "50%":              "#fbbf24",
    "61.8% (Golden)":   "#f59e0b",
    "78.6%":            "#f87171",
    "100% (Low)":       "#9ca3af",
}
for lbl, price_level in fib_prices.items():
    fig_fib.add_hline(
        y=price_level,
        line_dash="dot",
        line_color=FIB_COLORS.get(lbl, "#9ca3af"),
        line_width=1.5,
        annotation_text=f"{lbl}  ${price_level:.4f}",
        annotation_position="right",
        annotation_font_size=11)

fig_fib.update_layout(
    height=550, margin=dict(t=40, b=10, r=140),
    xaxis_rangeslider_visible=False,
    plot_bgcolor="#0e1117", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_fib, use_container_width=True)
st.caption(
    f"Fibonacci levels drawn from swing low (${swing_low:.4f}) to swing high (${swing_high:.4f}) over last {lookback_fib} bars. "
    "38.2%, 50%, and 61.8% are the most-watched pullback zones for swing entries — price often finds support here during healthy uptrends. "
    "61.8% (golden ratio) is the deepest 'healthy' retracement before the move is considered failed. "
    "A bounce off any of these levels with volume confirmation and RSI oversold = high-conviction entry.")

st.divider()

# ── 5. PARABOLIC SAR ───────────────────────────────────────────

st.markdown("### 5. Parabolic SAR")

def compute_psar(df, initial_af=0.02, step_af=0.02, max_af=0.2):
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(df)
    psar  = close.copy()
    bull  = np.ones(n, dtype=bool)
    af    = np.full(n, initial_af)
    ep    = np.zeros(n)   # extreme point

    # Initialise
    bull[0] = close[1] > close[0]
    psar[0] = low[0]  if bull[0] else high[0]
    ep[0]   = high[0] if bull[0] else low[0]

    for i in range(1, n):
        prev_psar = psar[i-1]
        prev_bull = bull[i-1]
        prev_ep   = ep[i-1]
        prev_af   = af[i-1]

        # Projected SAR
        new_psar = prev_psar + prev_af * (prev_ep - prev_psar)

        if prev_bull:
            new_psar = min(new_psar, low[i-1], low[max(0, i-2)])
            if low[i] < new_psar:          # reversal
                bull[i] = False
                psar[i] = prev_ep
                ep[i]   = low[i]
                af[i]   = initial_af
            else:
                bull[i] = True
                psar[i] = new_psar
                if high[i] > prev_ep:
                    ep[i] = high[i]
                    af[i] = min(prev_af + step_af, max_af)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
        else:
            new_psar = max(new_psar, high[i-1], high[max(0, i-2)])
            if high[i] > new_psar:         # reversal
                bull[i] = True
                psar[i] = prev_ep
                ep[i]   = high[i]
                af[i]   = initial_af
            else:
                bull[i] = False
                psar[i] = new_psar
                if low[i] < prev_ep:
                    ep[i] = low[i]
                    af[i] = min(prev_af + step_af, max_af)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af

    return pd.Series(psar, index=df.index), pd.Series(bull, index=df.index)

if len(df) >= 10:
    psar_vals, psar_bull = compute_psar(df)
    current_psar_bull = bool(psar_bull.iloc[-1])
    current_psar      = float(psar_vals.iloc[-1])
    psar_distance_pct = abs(price_now - current_psar) / price_now * 100

    # Detect recent flip
    prev_bull_psar = bool(psar_bull.iloc[-2]) if len(psar_bull) > 1 else current_psar_bull
    just_flipped_bull = current_psar_bull and not prev_bull_psar
    just_flipped_bear = not current_psar_bull and prev_bull_psar

    if just_flipped_bull:
        status_line(
            f"Parabolic SAR just flipped bullish — SAR dot moved below price at ${current_psar:.4f}. "
            "Fresh uptrend signal. Strong entry confirmation.", "strong_bullish")
    elif just_flipped_bear:
        status_line(
            f"Parabolic SAR just flipped bearish — SAR dot moved above price at ${current_psar:.4f}. "
            "Trend reversed. Exit or avoid new longs.", "strong_bearish")
    elif current_psar_bull:
        status_line(
            f"SAR bullish — dot below price at ${current_psar:.4f} "
            f"({psar_distance_pct:.1f}% below). Uptrend intact. SAR is trailing stop reference.", "bullish")
    else:
        status_line(
            f"SAR bearish — dot above price at ${current_psar:.4f} "
            f"({psar_distance_pct:.1f}% above). Downtrend. Wait for SAR to flip below price.", "bearish")

    # Split into bull/bear for colour coding
    psar_bull_y = psar_vals.where(psar_bull,  other=np.nan)
    psar_bear_y = psar_vals.where(~psar_bull, other=np.nan)

    fig_psar = go.Figure()
    fig_psar.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e",  decreasing_fillcolor="#ef4444"))
    fig_psar.add_trace(go.Scatter(
        x=df["time"], y=psar_bull_y,
        mode="markers", name="SAR Bullish",
        marker=dict(symbol="circle", size=5, color="#00c853")))
    fig_psar.add_trace(go.Scatter(
        x=df["time"], y=psar_bear_y,
        mode="markers", name="SAR Bearish",
        marker=dict(symbol="circle", size=5, color="#ff1744")))

    # Add ATR from existing df for reference overlay
    if df["ATR"].notna().any():
        fig_psar.add_trace(go.Scatter(
            x=df["time"], y=df["EMA_50"],
            name="EMA 50", line=dict(color="#fbbf24", width=1.5, dash="dot")))

    fig_psar.update_layout(
        height=500, margin=dict(t=40, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        plot_bgcolor="#0e1117", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_psar, use_container_width=True)
    st.caption(
        "Parabolic SAR dots appear below price in uptrends (green = bullish) and above price in downtrends (red = bearish). "
        "When dots flip from above to below price = bullish reversal signal. Flip from below to above = exit / trend ended. "
        "SAR accelerates toward price as the trend matures — when it gets very close, reversal is near. "
        "Works best in trending markets; generates many false signals in choppy/sideways conditions. "
        "Use with ADX > 25 to filter: only trust SAR signals when ADX confirms a trend is present.")
else:
    st.info("Not enough data for Parabolic SAR.")

st.divider()

st.caption(f"Data: Yahoo Finance | Auto-refreshes every 5 min | Last run: {ts} | {timeframe} | {date_start.strftime('%d %b %Y')} to {date_end.strftime('%d %b %Y')}")
