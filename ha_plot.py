# ha_plot.py — Pure HA visualizer with entry arrows + SL/TP lines
# Quick standalone visualization for Heikin Ashi candles + EMA + Stochastic
# Shows where entries would trigger and the suggested SL/TP levels.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Sample Data Generator
# ─────────────────────────────────────────────
def generate_sample(n=300, symbol="XAUUSD"):
    dates = pd.date_range("2024-01-01", periods=n, freq="15T")
    base = 2000 if "XAU" in symbol else 30000
    prices = np.cumsum(np.random.randn(n)) + base
    df = pd.DataFrame({
        "datetime": dates,
        "open": prices + np.random.randn(n),
        "high": prices + np.random.rand(n)*5,
        "low": prices - np.random.rand(n)*5,
        "close": prices + np.random.randn(n),
        "volume": np.random.randint(100, 1000, size=n)
    })
    return df.set_index("datetime")

# ─────────────────────────────────────────────
# Heikin Ashi + Signals + Plot
# ─────────────────────────────────────────────
def compute_ha_signals(df: pd.DataFrame,
                       ema_fast=21, ema_slow=200,
                       k_len=14, d_len=3,
                       ha_smooth=3,
                       atr_len=14,
                       rr_target=2.0,
                       ob=80, os=20):
    # HA
    ha_close_raw = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_close = ha_close_raw.ewm(span=ha_smooth).mean()
    ha_open = (df["open"] + df["close"]) / 2
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)

    # EMA
    emaF = ha_close.ewm(span=ema_fast).mean()
    emaS = ha_close.ewm(span=ema_slow).mean()

    # Stochastic
    ll = df["low"].rolling(k_len).min()
    hh = df["high"].rolling(k_len).max()
    k = (df["close"] - ll) / (hh - ll) * 100
    d = k.rolling(d_len).mean()

    # ATR (simple TR mean)
    tr = pd.concat([
        (df['high']-df['low']),
        (df['high']-df['close'].shift()).abs(),
        (df['low']-df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_len).mean()

    # Trend filters
    bull_trend = (ha_close > emaF) & (emaF > emaS)
    bear_trend = (ha_close < emaF) & (emaF < emaS)

    # Triggers (use HA bar close confirmation)
    stoch_up = (k.shift(1) < d.shift(1)) & (k > d) & (k < os)
    stoch_dn = (k.shift(1) > d.shift(1)) & (k < d) & (k > ob)

    long_sig = (bull_trend & stoch_up)
    short_sig = (bear_trend & stoch_dn)

    # Entry at HA close; SL at HA low/high ± ATR; TP by RR multiple
    entry_price_long = ha_close.where(long_sig)
    sl_long = (ha_low - atr).where(long_sig)
    tp_long = (entry_price_long + rr_target * (entry_price_long - sl_long)).where(long_sig)

    entry_price_short = ha_close.where(short_sig)
    sl_short = (ha_high + atr).where(short_sig)
    tp_short = (entry_price_short - rr_target * (sl_short - entry_price_short)).where(short_sig)

    return {
        'ha_open': ha_open, 'ha_high': ha_high, 'ha_low': ha_low, 'ha_close': ha_close,
        'emaF': emaF, 'emaS': emaS, 'k': k, 'd': d,
        'long_sig': long_sig, 'short_sig': short_sig,
        'entry_long': entry_price_long, 'sl_long': sl_long, 'tp_long': tp_long,
        'entry_short': entry_price_short, 'sl_short': sl_short, 'tp_short': tp_short
    }


def plot_ha_with_signals(df: pd.DataFrame, signals: dict, title="HA + EMA + Stoch with SL/TP"):
    ha_open = signals['ha_open']; ha_high = signals['ha_high']; ha_low = signals['ha_low']; ha_close = signals['ha_close']
    emaF = signals['emaF']; emaS = signals['emaS']
    k = signals['k']; d = signals['d']

    idx = np.arange(len(df))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,9), sharex=True, gridspec_kw={'height_ratios':[3,1]})

    # HA candles
    for i in range(len(df)):
        color = "green" if ha_close.iloc[i] >= ha_open.iloc[i] else "red"
        ax1.plot([i, i], [ha_low.iloc[i], ha_high.iloc[i]], color=color, linewidth=1)
        ax1.add_patch(plt.Rectangle((i-0.3, min(ha_open.iloc[i], ha_close.iloc[i])), 0.6, abs(ha_close.iloc[i]-ha_open.iloc[i]),
                                    facecolor=color, edgecolor=color, alpha=0.6))

    # EMAs
    ax1.plot(idx, emaF.values, label="EMA Fast", linewidth=1.2)
    ax1.plot(idx, emaS.values, label="EMA Slow", linewidth=1.2)

    # Long markers and SL/TP lines
    longs = signals['entry_long'].dropna()
    for t, price in longs.items():
        i = df.index.get_loc(t)
        ax1.annotate('▲', (i, price), textcoords='offset points', xytext=(0,-12), ha='center', fontsize=10, color='green')
        sl = signals['sl_long'].loc[t]
        tp = signals['tp_long'].loc[t]
        ax1.hlines([sl, tp], i-5, i+20, linestyles='dashed', linewidth=1)

    # Short markers and SL/TP lines
    shorts = signals['entry_short'].dropna()
    for t, price in shorts.items():
        i = df.index.get_loc(t)
        ax1.annotate('▼', (i, price), textcoords='offset points', xytext=(0,8), ha='center', fontsize=10, color='red')
        sl = signals['sl_short'].loc[t]
        tp = signals['tp_short'].loc[t]
        ax1.hlines([sl, tp], i-20, i+5, linestyles='dashed', linewidth=1)

    ax1.set_title(title)
    ax1.legend()

    # Stochastic subplot
    ax2.plot(idx, k.values, label="%K")
    ax2.plot(idx, d.values, label="%D")
    ax2.axhline(80, linestyle="--", alpha=0.5)
    ax2.axhline(20, linestyle="--", alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = generate_sample()
    sigs = compute_ha_signals(df)
    plot_ha_with_signals(df, sigs, title="HA Candles + EMA + Stoch + SL/TP")
