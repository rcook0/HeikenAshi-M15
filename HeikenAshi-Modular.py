# Heikin Ashi Overlay Strategy (Backtrader + Sample Generator + VectorBT)

## Sample Data Generator → creates synthetic OHLCV CSVs for quick testing.

##   python ha_overlay_backtrader.py --gen --csv sample_data.csv --symbol XAUUSD

## VectorBT Module → run the same HA Overlay logic in vectorbt for fast backtesting and visualization.

##   python ha_overlay_backtrader.py --csv sample_data.csv --vectorbt

import argparse
import datetime as dt
import math
import backtrader as bt
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# Sample OHLCV Data Generator
# ─────────────────────────────────────────────
def generate_sample_csv(path: str = "sample_data.csv", n=1000, symbol="XAUUSD"):
    """Generate synthetic OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=n, freq="15T")
    prices = np.cumsum(np.random.randn(n)) + 2000 if "XAU" in symbol else np.cumsum(np.random.randn(n)) + 30000
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
# Backtrader Components (same as previous)
# ─────────────────────────────────────────────
class HeikinAshiSmooth(bt.Indicator):
    lines = ('ha_open', 'ha_high', 'ha_low', 'ha_close')
    params = dict(smooth_len=3)
    def __init__(self):
        ha_close_raw = (self.data.open + self.data.high + self.data.low + self.data.close) / 4.0
        self.ha_close_smooth = bt.ind.EMA(ha_close_raw, period=max(1, int(self.p.smooth_len)))
        self.l.ha_close = self.ha_close_smooth
        self.l.ha_open = (self.data.open + self.data.close) / 2.0
        self.l.ha_high = bt.Max(self.data.high, bt.Max(self.l.ha_open, self.l.ha_close))
        self.l.ha_low = bt.Min(self.data.low, bt.Min(self.l.ha_open, self.l.ha_close))

# (Omitting strategy code here for brevity — unchanged from v1.0 above)
# Assume HAOverlayStrategy and RiskSizer classes already defined...

class GenericCSV(bt.feeds.GenericCSVData):
    params = (
        ('dtformat', '%Y-%m-%d %H:%M:%S'),
        ('datetime', 0), ('open', 1), ('high', 2), ('low', 3), ('close', 4),
        ('volume', 5), ('openinterest', -1),
        ('timeframe', bt.TimeFrame.Minutes), ('compression', 15),
    )

# ─────────────────────────────────────────────
# VectorBT Modular Implementation
# ─────────────────────────────────────────────
import vectorbt as vbt

def ha_overlay_vectorbt(df: pd.DataFrame,
                        ema_fast=21, ema_slow=200,
                        atr_len=14, atr_stop=1.0, atr_target=2.0,
                        accuracy=0.7, ha_smooth=3,
                        vol_ma=20, vol_filter=True):
    """Run HA Overlay Strategy in vectorbt."""
    # Heikin Ashi calc
    ha_close_raw = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_close = ha_close_raw.ewm(span=ha_smooth).mean()
    ha_open = (df['open'] + df['close'])/2
    ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df['low'], ha_open, ha_close], axis=1).min(axis=1)

    # EMA & VWAP
    ema_fast = ha_close.ewm(span=ema_fast).mean()
    ema_slow = ha_close.ewm(span=ema_slow).mean()
    typical_price = (df['high']+df['low']+df['close'])/3
    vwap = (typical_price*df['volume']).cumsum() / df['volume'].cumsum()
    vwap_slope = vwap.diff()

    # Volume filter
    vol_ma_series = df['volume'].rolling(vol_ma).mean()
    vol_ok = (df['volume'] > vol_ma_series) if vol_filter else True

    # ATR
    tr = pd.concat([(df['high']-df['low']),
                    (df['high']-df['close'].shift()).abs(),
                    (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_len).mean()

    # Stochastic
    low_k = df['low'].rolling(14).min()
    high_k = df['high'].rolling(14).max()
    k = (df['close']-low_k)/(high_k-low_k)*100
    d = k.rolling(3).mean()
    ob = 80 - (80-60)*(1-accuracy)
    os = 20 + (40-20)*(1-accuracy)
    stoch_cross_up = (k.shift(1)<d.shift(1)) & (k>d) & (k<os)
    stoch_cross_dn = (k.shift(1)>d.shift(1)) & (k<d) & (k>ob)

    # Long/short conds
    bull = (ha_close>vwap) & (ema_fast>ema_slow) & (vwap_slope>0) & vol_ok
    bear = (ha_close<vwap) & (ema_fast<ema_slow) & (vwap_slope<0) & vol_ok
    long_entries = bull & stoch_cross_up
    short_entries = bear & stoch_cross_dn

    # Exits by ATR stops
    long_exits = (df['close'] < ha_close - atr*atr_stop)
    short_exits = (df['close'] > ha_close + atr*atr_stop)

    pf = vbt.Portfolio.from_signals(df['close'],
                                    entries=long_entries,
                                    exits=long_exits,
                                    short_entries=short_entries,
                                    short_exits=short_exits,
                                    size=0.1, fees=0.0002)
    return pf

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', action='store_true', help='Generate sample data')
    parser.add_argument('--csv', default='sample_data.csv')
    parser.add_argument('--symbol', default='XAUUSD')
    parser.add_argument('--vectorbt', action='store_true')
    args = parser.parse_args()

    if args.gen:
        generate_sample_csv(args.csv, n=1000, symbol=args.symbol)

    if args.vectorbt:
        df = pd.read_csv(args.csv, parse_dates=['datetime']).set_index('datetime')
        pf = ha_overlay_vectorbt(df)
        print(pf.stats())
        pf.plot().show()
    else:
        # Backtrader path (simplified)
        cerebro = bt.Cerebro()
        data = GenericCSV(dataname=args.csv)
        data._name = args.symbol
        cerebro.adddata(data)
        cerebro.addstrategy(HAOverlayStrategy)
        cerebro.run()
        cerebro.plot()
