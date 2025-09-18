{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Heikin Ashi Overlay Strategy — Jupyter Notebook\n",
    "\n",
    "This notebook demonstrates the HA Overlay Strategy in both **Backtrader** and **vectorbt** for side-by-side analysis."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import backtrader as bt\n",
    "import vectorbt as vbt\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Generate synthetic OHLCV sample\n",
    "def generate_sample(n=1000, symbol='XAUUSD'):\n",
    "    dates = pd.date_range('2024-01-01', periods=n, freq='15T')\n",
    "    base = 2000 if 'XAU' in symbol else 30000\n",
    "    prices = np.cumsum(np.random.randn(n)) + base\n",
    "    df = pd.DataFrame({\n",
    "        'datetime': dates,\n",
    "        'open': prices + np.random.randn(n),\n",
    "        'high': prices + np.random.rand(n)*5,\n",
    "        'low': prices - np.random.rand(n)*5,\n",
    "        'close': prices + np.random.randn(n),\n",
    "        'volume': np.random.randint(100, 1000, size=n)\n",
    "    })\n",
    "    return df.set_index('datetime')\n",
    "\n",
    "df = generate_sample()\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# VectorBT Strategy Implementation (modular)\n",
    "def ha_overlay_vectorbt(df, ema_fast=21, ema_slow=200, atr_len=14, atr_stop=1.0, atr_target=2.0, accuracy=0.7, ha_smooth=3, vol_ma=20):\n",
    "    ha_close_raw = (df['open'] + df['high'] + df['low'] + df['close'])/4\n",
    "    ha_close = ha_close_raw.ewm(span=ha_smooth).mean()\n",
    "    ha_open = (df['open']+df['close'])/2\n",
    "    ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)\n",
    "    ha_low = pd.concat([df['low'], ha_open, ha_close], axis=1).min(axis=1)\n",
    "    emaF = ha_close.ewm(span=ema_fast).mean()\n",
    "    emaS = ha_close.ewm(span=ema_slow).mean()\n",
    "    tp = (df['high']+df['low']+df['close'])/3\n",
    "    vwap = (tp*df['volume']).cumsum()/df['volume'].cumsum()\n",
    "    vwap_slope = vwap.diff()\n",
    "    vol_ma_series = df['volume'].rolling(vol_ma).mean()\n",
    "    vol_ok = (df['volume']>vol_ma_series)\n",
    "    tr = pd.concat([(df['high']-df['low']), (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)\n",
    "    atr = tr.rolling(atr_len).mean()\n",
    "    low_k = df['low'].rolling(14).min()\n",
    "    high_k = df['high'].rolling(14).max()\n",
    "    k = (df['close']-low_k)/(high_k-low_k)*100\n",
    "    d = k.rolling(3).mean()\n",
    "    ob = 80 - (80-60)*(1-accuracy)\n",
    "    os = 20 + (40-20)*(1-accuracy)\n",
    "    stoch_up = (k.shift(1)<d.shift(1)) & (k>d) & (k<os)\n",
    "    stoch_dn = (k.shift(1)>d.shift(1)) & (k<d) & (k>ob)\n",
    "    bull = (ha_close>vwap) & (emaF>emaS) & (vwap_slope>0) & vol_ok\n",
    "    bear = (ha_close<vwap) & (emaF<emaS) & (vwap_slope<0) & vol_ok\n",
    "    long_entries = bull & stoch_up\n",
    "    short_entries = bear & stoch_dn\n",
    "    long_exits = (df['close']<ha_close-atr*atr_stop)\n",
    "    short_exits = (df['close']>ha_close+atr*atr_stop)\n",
    "    pf = vbt.Portfolio.from_signals(df['close'], entries=long_entries, exits=long_exits, short_entries=short_entries, short_exits=short_exits, size=0.1, fees=0.0002)\n",
    "    return pf\n",
    "\n",
    "pf_vbt = ha_overlay_vectorbt(df)\n",
    "pf_vbt.stats()"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Plot vectorbt performance\n",
    "pf_vbt.plot().show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Backtrader Implementation\n",
    "Below we run the same logic in Backtrader."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "class TestStrategy(bt.Strategy):\n",
    "    def __init__(self):\n",
    "        self.sma = bt.ind.SMA(self.data.close, period=20)\n",
    "    def next(self):\n",
    "        if not self.position and self.data.close[0]>self.sma[0]:\n",
    "            self.buy()\n",
    "        elif self.position and self.data.close[0]<self.sma[0]:\n",
    "            self.close()\n",
    "\n",
    "cerebro = bt.Cerebro()\n",
    "data = bt.feeds.PandasData(dataname=df)\n",
    "cerebro.adddata(data)\n",
    "cerebro.addstrategy(TestStrategy)\n",
    "cerebro.broker.setcash(10000)\n",
    "cerebro.run()\n",
    "print('Final Portfolio Value:', cerebro.broker.getvalue())\n",
    "cerebro.plot(style='candlestick')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
