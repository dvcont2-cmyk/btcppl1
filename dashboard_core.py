"""Shared, causal calculations. No Streamlit or network access at import time.

Conventions: OHLC uses positive prices; +1 bullish / -1 bearish / 0 unavailable.
ATR uses Wilder smoothing seeded with the first period's mean true range.
Weekly candles end Sunday. Only fully elapsed periods are used.
"""
import numpy as np
import pandas as pd
import ta

OHLC = ['open', 'high', 'low', 'close']
ST_CONFIGS = [(7, 3.0), (14, 2.0), (21, 1.0)]


def clean_ohlc(df):
    if df.empty:
        return pd.DataFrame(columns=['time'] + OHLC)
    out = df.copy()
    # Preserve the provider's calendar date (ASX local date, crypto UTC date).
    times = pd.to_datetime(out['time'])
    if times.dt.tz is not None:
        times = times.dt.tz_localize(None)
    out['time'] = times.dt.normalize()
    for col in OHLC + (['volume'] if 'volume' in out else []):
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out = out.dropna(subset=['time'] + OHLC)
    valid = np.isfinite(out[OHLC]).all(axis=1) & out[OHLC].gt(0).all(axis=1)
    valid &= out.high.ge(out[['open', 'close', 'low']].max(axis=1))
    valid &= out.low.le(out[['open', 'close', 'high']].min(axis=1))
    return out.loc[valid].sort_values('time').drop_duplicates('time', keep='last').reset_index(drop=True)


def completed_bars(df, timeframe='Daily', asof=None):
    """asof is the exclusive calendar-day boundary of available completed data.

    The first resampled period is discarded if history starts mid-period.
    Missing trading days (weekends/holidays) are not filled with fake prices.
    """
    out = clean_ohlc(df)
    cutoff = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now(tz='UTC').tz_localize(None).normalize()
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    cutoff = cutoff.normalize()
    out = out[out.time < cutoff].copy()
    if 'open_time' not in out:
        out['open_time'] = out['time']
    if out.empty or timeframe == 'Daily':
        return out.reset_index(drop=True)
    period = 'W-SUN' if timeframe == 'Weekly' else 'M'
    groups = out.time.dt.to_period(period)
    agg = {'open_time': 'first', 'time': 'last', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in out:
        agg['volume'] = 'sum'
    result = out.groupby(groups).agg(agg)
    ends = result.index.end_time.normalize()
    # Keep last trading date for execution/plotting; filter by period completion.
    result = result.loc[(ends < cutoff) & (result.index.start_time >= out.time.iloc[0])]
    return result.reset_index(drop=True)


def atr_wilder(df, period):
    if period < 1:
        raise ValueError('ATR period must be positive')
    result = pd.Series(np.nan, index=df.index, dtype=float)
    if len(df) < period:
        return result
    previous = df.close.shift()
    tr = pd.concat([df.high-df.low, (df.high-previous).abs(), (df.low-previous).abs()], axis=1).max(axis=1)
    result.iloc[period-1] = tr.iloc[:period].mean()
    for i in range(period, len(df)):
        result.iloc[i] = (result.iloc[i-1]*(period-1)+tr.iloc[i])/period
    return result


def compute_supertrend(df, period=10, multiplier=3.0):
    """Seed bearish at first valid ATR; emit no direction before that bar.

    Reference recurrence: TradingView Supertrend help article. Data-feed and
    initialization differences can still differ from a platform export.
    """
    if period < 1 or multiplier <= 0:
        raise ValueError('Period and multiplier must be positive')
    atr = atr_wilder(df, period)
    mid = (df.high + df.low)/2
    upper, lower = (mid+multiplier*atr).to_numpy(), (mid-multiplier*atr).to_numpy()
    line = np.full(len(df), np.nan)
    direction = np.zeros(len(df), dtype=int)
    if len(df) < period:
        return pd.Series(line, index=df.index), pd.Series(direction, index=df.index)
    start = period-1
    direction[start] = -1
    line[start] = upper[start]
    close = df.close.to_numpy()
    for i in range(start+1, len(df)):
        if not (upper[i] < upper[i-1] or close[i-1] > upper[i-1]):
            upper[i] = upper[i-1]
        if not (lower[i] > lower[i-1] or close[i-1] < lower[i-1]):
            lower[i] = lower[i-1]
        if direction[i-1] == -1:
            direction[i] = 1 if close[i] > upper[i] else -1
        else:
            direction[i] = -1 if close[i] < lower[i] else 1
        line[i] = lower[i] if direction[i] == 1 else upper[i]
    return pd.Series(line, index=df.index), pd.Series(direction, index=df.index)


def triple_supertrend_signals(df):
    out = df.copy()
    for j, (period, factor) in enumerate(ST_CONFIGS, 1):
        out[f'st{j}'], out[f'dir{j}'] = compute_supertrend(out, period, factor)
    dirs = out[['dir1', 'dir2', 'dir3']]
    ready = dirs.ne(0).all(axis=1)
    bull = dirs.eq(1).all(axis=1)
    prev_ready = ready.shift(fill_value=False)
    out['entry_signal'] = ready & prev_ready & bull & ~bull.shift(fill_value=False)
    out['exit_signal'] = ready & prev_ready & ~bull & bull.shift(fill_value=False)
    out['buy_signal'] = out.close.where(out.entry_signal)
    out['sell_signal'] = out.close.where(out.exit_signal)
    return out


def compute_indicators(df):
    """Insufficient history is NaN, never a fabricated neutral measurement."""
    out = df.copy()
    c, h, l = out.close, out.high, out.low
    columns = ['RSI','StochRSI_k','StochRSI_d','MACD','MACD_signal','MACD_hist',
               'BB_upper','BB_lower','BB_mid','BB_pct','EMA_50','EMA_200',
               'ADX','ADX_pos','ADX_neg','CCI','WilliamsR','ROC','ATR']
    for col in columns:
        out[col] = np.nan
    if len(out) >= 14:
        out['RSI'] = ta.momentum.RSIIndicator(c, window=14).rsi()
        sr = ta.momentum.StochRSIIndicator(c, window=14)
        out['StochRSI_k'], out['StochRSI_d'] = sr.stochrsi_k()*100, sr.stochrsi_d()*100
        out['WilliamsR'] = ta.momentum.WilliamsRIndicator(h,l,c,lbp=14).williams_r()
    if len(out) >= 26:
        macd = ta.trend.MACD(c)
        out['MACD'],out['MACD_signal'],out['MACD_hist'] = macd.macd(),macd.macd_signal(),macd.macd_diff()
    if len(out) >= 20:
        bb = ta.volatility.BollingerBands(c, window=20)
        out['BB_upper'],out['BB_lower'] = bb.bollinger_hband(),bb.bollinger_lband()
        out['BB_mid'],out['BB_pct'] = bb.bollinger_mavg(),bb.bollinger_pband()
        out['CCI'] = ta.trend.CCIIndicator(h,l,c,window=20).cci()
    for period in [50,200]:
        if len(out) >= period:
            out[f'EMA_{period}'] = ta.trend.EMAIndicator(c,window=period).ema_indicator()
    if len(out) >= 28:
        adx = ta.trend.ADXIndicator(h,l,c,window=14)
        for col, values in [('ADX',adx.adx()),('ADX_pos',adx.adx_pos()),('ADX_neg',adx.adx_neg())]:
            out[col] = values
            out.loc[out.index[:27], col] = np.nan
    if len(out) >= 13:
        out['ROC'] = ta.momentum.ROCIndicator(c,window=12).roc()
    out['ATR'] = atr_wilder(out,14)
    atr20 = atr_wilder(out,20)
    mid = c.rolling(20).mean()
    out['squeeze'] = (out.BB_upper < mid+1.5*atr20) & (out.BB_lower > mid-1.5*atr20)
    valid = atr20.notna() & out.BB_upper.notna()
    out['squeeze_valid'] = valid
    out['squeeze_fired'] = valid & valid.shift(fill_value=False) & out['squeeze'].shift(fill_value=False) & ~out['squeeze']
    delta = c - ((h.rolling(20).max()+l.rolling(20).min())/2+mid)/2
    out['squeeze_hist'] = delta.rolling(20).mean()  # Explicitly an approximation, not branded TTM.
    out['MACD_cross_up'] = (out.MACD_hist > 0) & (out.MACD_hist.shift() <= 0)
    return out


def support_resistance(df, window=3, num_levels=5, price=None):
    if df.empty:
        return [], []
    price = float(df.close.iloc[-1]) if price is None else price
    supports, resistances = [], []
    for i in range(window, len(df)-window):
        # Pivots are only available AFTER window following bars have closed.
        if df.high.iloc[i] >= df.high.iloc[i-window:i+window+1].max():
            resistances.append(float(df.high.iloc[i]))
        if df.low.iloc[i] <= df.low.iloc[i-window:i+window+1].min():
            supports.append(float(df.low.iloc[i]))
    def nearest(levels, below):
        levels = sorted(set(x for x in levels if (0 < x < price if below else x > price)), reverse=below)
        chosen = []
        for x in levels:
            if all(abs(x-y)/y > .02 for y in chosen):
                chosen.append(x)
            if len(chosen) == num_levels:
                break
        return sorted(chosen)
    return nearest(supports,True), nearest(resistances,False)


def backtest(df, initial=10000., fee_bps=10., slippage_bps=5.):
    """Long/cash. Signals at close, fills next available bar open.

    Input must already contain signals computed on warm history. Starts flat;
    no synthetic entry for an already-bullish regime. Open trades are marked at
    final close (no fictional liquidation). Uses fractional units, proportional
    fees, no leverage/tax/interest. Adjusted OHLC implies adjusted-price returns.
    """
    if initial <= 0 or not 0 <= fee_bps < 10000 or not 0 <= slippage_bps < 10000:
        raise ValueError('Invalid capital or costs')
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    fee, slip = fee_bps/10000, slippage_bps/10000
    cash, units, pending, trade = initial, 0., None, None
    records, ledger = [], []
    previous_signal_time = pd.NaT
    baseline_units = initial/(float(df.open.iloc[0])*(1+slip)*(1+fee))
    for _, row in df.iterrows():
        entry_fill = exit_fill = np.nan
        if pending == 'buy' and units == 0:
            fill = float(row.open)*(1+slip)
            units = cash/(fill*(1+fee))
            entry_fee = units*fill*fee
            trade = {'entry_signal_time':previous_signal_time,'entry_time':row.get('open_time', row.time),'entry_price':fill,'units':units,
                     'entry_fee':entry_fee,'entry_cost':cash}
            cash = 0.
            entry_fill = fill
        elif pending == 'sell' and units > 0:
            fill = float(row.open)*(1-slip)
            exit_fee = units*fill*fee
            cash = units*fill-exit_fee
            ledger.append(dict(trade,exit_signal_time=previous_signal_time,exit_time=row.get('open_time', row.time),exit_price=fill,exit_fee=exit_fee,
                               pnl=cash-trade['entry_cost'],return_pct=(cash/trade['entry_cost']-1)*100,status='Closed'))
            units, trade = 0., None
            exit_fill = fill
        pending = None
        if units == 0 and bool(row.entry_signal):
            pending = 'buy'
        elif units > 0 and bool(row.exit_signal):
            pending = 'sell'
        previous_signal_time = row.time
        records.append({'time':row.time, 'open_time':row.get('open_time', row.time),'equity':cash+units*float(row.close),
                        'buy_hold':baseline_units*float(row.close),'invested':units > 0,
                        'entry_fill':entry_fill,'exit_fill':exit_fill})
    if trade is not None:
        value = units*float(df.close.iloc[-1])
        ledger.append(dict(trade,exit_signal_time=pd.NaT,exit_time=pd.NaT,exit_price=np.nan,exit_fee=0.,
                           pnl=value-trade['entry_cost'],return_pct=(value/trade['entry_cost']-1)*100,status='Open (marked)'))
    curve = pd.DataFrame(records)
    trades = pd.DataFrame(ledger)
    equity_with_initial = np.r_[initial,curve.equity.to_numpy()]
    dd = equity_with_initial/np.maximum.accumulate(equity_with_initial)-1
    closed = [t for t in ledger if t['status']=='Closed']
    wins = sum(t['pnl']>0 for t in closed)
    metrics = {'Return %':(curve.equity.iloc[-1]/initial-1)*100,
               'Buy & hold %':(curve.buy_hold.iloc[-1]/initial-1)*100,
               'Max drawdown %':dd.min()*100,'Closed trades':len(closed),
               'Win rate %':100*wins/len(closed) if closed else np.nan,
               'Exposure %':100*curve.invested.mean()}
    return curve,trades,metrics


def render_supertrend(df, key, currency='AUD'):
    import streamlit as st
    import plotly.graph_objects as go
    st.subheader('Triple Supertrend — one strategy, one price chart')
    st.caption('ST (7,3), (14,2), (21,1). Enter when all three turn bullish; exit when alignment breaks. Completed candles only.')
    with st.expander('Backtest settings and assumptions', expanded=False):
        c1,c2,c3 = st.columns(3)
        capital = c1.number_input(f'Starting capital ({currency})',min_value=100.,value=10000.,key=key+'_capital')
        fee = c2.number_input('Fee per side (basis points)',min_value=0.,max_value=500.,value=10.,key=key+'_fee')
        slip = c3.number_input('Slippage per side (basis points)',min_value=0.,max_value=500.,value=5.,key=key+'_slip')
        st.caption('10 bps = 0.10%. Fills occur at the next available open. Fractional units; proportional fees, no minimum brokerage, taxes or cash interest. No shorting or leverage. Open trades remain marked at final close. The benchmark pays the same entry costs and remains marked too.')
        st.caption('ASX uses dividend/split-adjusted OHLC: results are an adjusted-price approximation, not a cash-dividend ledger. No parameter optimisation or independent holdout is performed. Descriptive historical results do not establish future profitability.')
        st.caption('The selected date range is the evaluation window; history before it is used only for indicators. The test starts in cash and waits for a new entry transition. A final-bar signal remains pending until a subsequent bar exists.')
    curve,trades,metrics = backtest(df,capital,fee,slip)
    fig = go.Figure(go.Candlestick(x=df.time,open=df.open,high=df.high,low=df.low,close=df.close,name='Price'))
    for j,(period,factor) in enumerate(ST_CONFIGS,1):
        for direction,color,side in [(1,'#22c55e','Bull'),(-1,'#ef4444','Bear')]:
            fig.add_trace(go.Scatter(x=df.time,y=df[f'st{j}'].where(df[f'dir{j}']==direction),
                mode='lines',connectgaps=False,line=dict(color=color,width=1+j*.4),
                name=f'ST {period},{factor:g} {side}',legendgroup=f'ST{j}'))
    if df.EMA_200.notna().any():
        fig.add_trace(go.Scatter(x=df.time,y=df.EMA_200,name='EMA 200',line=dict(color='#fbbf24',dash='dot')))
    for col,name,symbol,color in [('entry_fill','Buy fill','triangle-up','#22c55e'),('exit_fill','Sell fill','triangle-down','#ef4444')]:
        mask=curve[col].notna()
        fig.add_trace(go.Scatter(x=curve.loc[mask,'open_time'],y=curve.loc[mask,col],mode='markers',name=name,marker=dict(symbol=symbol,color=color,size=13)))
    fig.update_layout(height=600,xaxis_rangeslider_visible=False,yaxis_title=currency,legend=dict(orientation='h'))
    st.plotly_chart(fig,width="stretch")
    st.caption('Markers are executed fills, including slippage, rather than same-close signals. Band lines break at reversals. Weekly fills are dated at the next week’s first trading day; candles are labelled at the last trading day.')
    if df.entry_signal.iloc[-1]:
        st.info('Latest completed candle has an entry signal; next-open fill is not yet included.')
    elif df.exit_signal.iloc[-1]:
        st.info('Latest completed candle breaks bullish alignment; an existing long would exit at the next open.')
    cols=st.columns(3)
    for i,(name,value) in enumerate(metrics.items()):
        cols[i%3].metric(name,'N/A' if pd.isna(value) else f'{value:,.2f}')
    st.caption(f'Evaluation: {df.time.iloc[0]:%d %b %Y}–{df.time.iloc[-1]:%d %b %Y}; {len(df)} completed bars. Win rate excludes open positions. No trades means no evidence for this strategy in this window.')
    with st.expander('Equity curve and complete trade ledger'):
        st.line_chart(curve.set_index('time')[['equity','buy_hold']])
        if trades.empty:
            st.info('No trades in this evaluation window.')
        else:
            st.dataframe(trades,width="stretch")
        st.download_button('Download trade ledger',trades.to_csv(index=False),'supertrend_trades.csv','text/csv',key=key+'_trades')
        st.download_button('Download equity curve',curve.to_csv(index=False),'supertrend_equity.csv','text/csv',key=key+'_equity')
        st.download_button('Download candles, indicators and signals',df.to_csv(index=False),'supertrend_signals.csv','text/csv',key=key+'_signals')
