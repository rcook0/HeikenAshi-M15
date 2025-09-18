# Heikin Ashi Overlay Strategy (Backtrader + Sample Generator + VectorBT)

## python ha_overlay_backtrader.py --csv sample.csv

## Sample Data Generator → creates synthetic OHLCV CSVs for quick testing.
##   python ha_overlay_backtrader.py --gen --csv sample.csv --symbol XAUUSD

## VectorBT Module → run the same HA Overlay logic in vectorbt for fast backtesting and visualization.
##   python ha_overlay_backtrader.py --csv sample.csv --vectorbt

# Heikin Ashi Strategy (Backtrader + VectorBT) — Pure HA Only

# Heikin Ashi Strategy (Backtrader + VectorBT) — Pure HA Only with Shared Plot

import argparse
import datetime as dt
import math
import backtrader as bt
import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Sample OHLCV Data Generator
# ─────────────────────────────────────────────
def generate_sample_csv(path: str = "sample_data.csv", n=1000, symbol="XAUUSD"):
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
    df.to_csv(path, index=False)
    print(f"Sample data written to {path}")
    return path

# ─────────────────────────────────────────────
# Backtrader: Heikin Ashi Only Indicator
# ─────────────────────────────────────────────
class HeikinAshi(bt.Indicator):
    lines = ('ha_open', 'ha_high', 'ha_low', 'ha_close')
    params = dict(smooth_len=3)
    def __init__(self):
        ha_close_raw = (self.data.open + self.data.high + self.data.low + self.data.close) / 4.0
        self.l.ha_close = bt.ind.EMA(ha_close_raw, period=max(1, int(self.p.smooth_len)))
        self.l.ha_open = (self.data.open + self.data.close) / 2.0
        self.l.ha_high = bt.Max(self.data.high, bt.Max(self.l.ha_open, self.l.ha_close))
        self.l.ha_low = bt.Min(self.data.low, bt.Min(self.l.ha_open, self.l.ha_close))

class HAOnlyStrategy(bt.Strategy):
    params = dict(ema_fast=21, ema_slow=200, atr_len=14)
    def __init__(self):
        self.ha = HeikinAshi(self.data)
        self.ema_fast = bt.ind.EMA(self.ha.ha_close, period=self.p.ema_fast)
        self.ema_slow = bt.ind.EMA(self.ha.ha_close, period=self.p.ema_slow)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_len)
    def next(self):
        if not self.position and self.ha.ha_close[0] > self.ema_fast[0] > self.ema_slow[0]:
            self.buy()
        elif self.position and self.ha.ha_close[0] < self.ema_fast[0]:
            self.close()

class GenericCSV(bt.feeds.GenericCSVData):
    params = (
        ('dtformat', '%Y-%m-%d %H:%M:%S'),
        ('datetime', 0), ('open', 1), ('high', 2), ('low', 3), ('close', 4),
        ('volume', 5), ('openinterest', -1),
        ('timeframe', bt.TimeFrame.Minutes), ('compression', 15),
    )

# ─────────────────────────────────────────────
# VectorBT: HA Only Strategy
# ─────────────────────────────────────────────
def ha_only_vectorbt(df: pd.DataFrame, ema_fast=21, ema_slow=200, atr_len=14):
    ha_close_raw = (df['open'] + df['high'] + df['low'] + df['close'])/4
    ha_close = ha_close_raw.ewm(span=3).mean()
    ha_open = (df['open']+df['close'])/2
    ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df['low'], ha_open, ha_close], axis=1).min(axis=1)
    emaF = ha_close.ewm(span=ema_fast).mean()
    emaS = ha_close.ewm(span=ema_slow).mean()
    tr = pd.concat([(df['high']-df['low']), (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_len).mean()
    long_entries = (ha_close>emaF) & (emaF>emaS)
    long_exits = (ha_close<emaF)
    pf = vbt.Portfolio.from_signals(df['close'], entries=long_entries, exits=long_exits, size=0.1, fees=0.0002)
    return pf, ha_open, ha_high, ha_low, ha_close, emaF, emaS

# ─────────────────────────────────────────────
# Shared HA Plot
# ─────────────────────────────────────────────
def ha_plot(df, ha_open, ha_high, ha_low, ha_close, emaF, emaS, title="HA Strategy Plot"):
    fig, ax = plt.subplots(figsize=(12,6))
    for i in range(len(df)):
        color = 'green' if ha_close.iloc[i] >= ha_open.iloc[i] else 'red'
        ax.plot([i, i], [ha_low.iloc[i], ha_high.iloc[i]], color=color, linewidth=1)
        ax.add_patch(plt.Rectangle((i-0.3, min(ha_open.iloc[i], ha_close.iloc[i])),
                                   0.6,
                                   abs(ha_close.iloc[i]-ha_open.iloc[i]),
                                   facecolor=color, edgecolor=color, alpha=0.6))
    ax.plot(range(len(df)), emaF, label='EMA Fast', color='blue')
    ax.plot(range(len(df)), emaS, label='EMA Slow', color='orange')
    ax.set_title(title)
    ax.legend()
    plt.show()

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--csv', default='sample_data.csv')
    parser.add_argument('--symbol', default='XAUUSD')
    parser.add_argument('--vectorbt', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    if args.gen:
        generate_sample_csv(args.csv, n=1000, symbol=args.symbol)

    if args.vectorbt:
        df = pd.read_csv(args.csv, parse_dates=['datetime']).set_index('datetime')
        pf, ha_open, ha_high, ha_low, ha_close, emaF, emaS = ha_only_vectorbt(df)
        print(pf.stats())
        if args.plot:
            ha_plot(df, ha_open, ha_high, ha_low, ha_close, emaF, emaS, title=f"VectorBT HA {args.symbol}")
    else:
        cerebro = bt.Cerebro()
        data = GenericCSV(dataname=args.csv)
        cerebro.adddata(data)
        cerebro.addstrategy(HAOnlyStrategy)
        cerebro.run()
        if args.plot:
            df = pd.read_csv(args.csv, parse_dates=['datetime']).set_index('datetime')
            ha_close_raw = (df['open'] + df['high'] + df['low'] + df['close'])/4
            ha_close = ha_close_raw.ewm(span=3).mean()
            ha_open = (df['open']+df['close'])/2
            ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
            ha_low = pd.concat([df['low'], ha_open, ha_close], axis=1).min(axis=1)
            emaF = ha_close.ewm(span=21).mean()
            emaS = ha_close.ewm(span=200).mean()
            ha_plot(df, ha_open, ha_high, ha_low, ha_close, emaF, emaS, title=f"Backtrader HA {args.symbol}")
