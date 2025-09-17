# Heikin Ashi Overlay Strategy (Backtrader)
# M15-focused, but timeframe-agnostic. Includes:
# - Heikin Ashi smoothing
# - EMA trend filter (21/200)
# - VWAP + slope confluence
# - Volume filter (vol > SMA(vol))
# - Stochastic with relaxed accuracy tolerance
# - Adaptive ATR multiplier (relative to ATR SMA) with manual override
# - Session detection (London/NY)
# - %ATR SL/TP, break-even, ATR trailing stop
# - Bracket orders + dynamic stop management
#
# Usage (quick start):
#   pip install backtrader pandas
#   python ha_overlay_backtrader.py --csv your_data.csv --symbol XAUUSD --timeframe 15
# CSV columns expected: datetime, open, high, low, close, volume

import argparse
import datetime as dt
import math
import backtrader as bt

# ──────────────────────────────────────────────────────────────────────────────
# Helpers & Indicators
# ──────────────────────────────────────────────────────────────────────────────
class HeikinAshiSmooth(bt.Indicator):
    '''Heikin Ashi with optional EMA smoothing on HA close.'''
    lines = ('ha_open', 'ha_high', 'ha_low', 'ha_close')
    params = dict(smooth_len=3)

    def __init__(self):
        self.addminperiod(2)
        # Raw HA close
        ha_close_raw = (self.data.open + self.data.high + self.data.low + self.data.close) / 4.0
        # Smooth HA close
        self.ha_close_smooth = bt.ind.EMA(ha_close_raw, period=max(1, int(self.p.smooth_len))) if self.p.smooth_len > 1 else ha_close_raw
        # HA open (recursive): ha_open = (prev_ha_open + prev_ha_close)/2, seed with (o+c)/2
        self.l.ha_open = (bt.If(bt.Or(bt.num2date(self.data.datetime[0]) == bt.num2date(self.data.datetime[-1]), bt.lineseries.LineSeries()),
                                 (self.data.open + self.data.close) / 2.0,
                                 (self(-1).ha_open + self(-1).ha_close) / 2.0))
        # Outputs
        self.l.ha_close = self.ha_close_smooth
        self.l.ha_high = bt.Max(self.data.high, bt.Max(self.l.ha_open, self.l.ha_close))
        self.l.ha_low = bt.Min(self.data.low, bt.Min(self.l.ha_open, self.l.ha_close))

class VWAPSlope(bt.Indicator):
    '''Session-cumulative VWAP with simple slope (vwap - vwap[-1]).'''
    lines = ('vwap', 'slope',)

    def __init__(self):
        tp = (self.data.high + self.data.low + self.data.close) / 3.0
        vol = self.data.volume
        self.cum_pv = bt.ind.SumN(tp * vol)
        self.cum_v = bt.ind.SumN(vol)
        self.l.vwap = self.cum_pv / bt.If(self.cum_v == 0, 1, self.cum_v)
        self.l.slope = self.l.vwap - self.l.vwap(-1)

class StochKD(bt.Indicator):
    lines = ('k', 'd')
    params = dict(k=14, d=3, smooth=3)
    def __init__(self):
        ll = bt.ind.Lowest(self.data.low, period=self.p.k)
        hh = bt.ind.Highest(self.data.high, period=self.p.k)
        k_raw = bt.If(hh - ll == 0, 0, (self.data.close - ll) / (hh - ll) * 100.0)
        k_s = bt.ind.EMA(k_raw, period=self.p.smooth)
        self.l.k = k_s
        self.l.d = bt.ind.EMA(self.l.k, period=self.p.d)

# ──────────────────────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────────────────────
class HAOverlayStrategy(bt.Strategy):
    params = dict(
        # General
        accuracy=0.70,                 # relaxed accuracy → widens thresholds
        risk_pct=0.10,                 # percent of equity (used to size via sizer)
        # Heikin Ashi
        ha_smooth_len=3,
        # Indicators
        ema_fast=21,
        ema_slow=200,
        atr_len=14,
        stoch_k=14,
        stoch_d=3,
        stoch_smooth=3,
        vwap_confluence=True,
        volume_filter=True,
        vol_ma_len=20,
        # Sessions (exchange/local tz assumed in data)
        session_enable=True,
        london_start=dt.time(7, 0), london_end=dt.time(12, 0),
        ny_start=dt.time(13, 0), ny_end=dt.time(17, 0),
        # ATR management
        atr_stop=1.0,                  # SL = ATR * this * multiplier
        atr_target=2.0,                # TP = ATR * this * multiplier
        vol_sensitivity=1.0,           # adaptive ATR sensitivity
        manual_atr=False,
        manual_atr_mult=1.0,
        # Break-even & trailing
        use_be=True,
        be_R=1.0,
        be_offset_atr=0.1,
        use_trail=True,
        trail_atr_mult=1.5,
    )

    def log(self, txt):
        dtstr = bt.num2date(self.data.datetime[0]).strftime('%Y-%m-%d %H:%M')
        print(f'{dtstr} | {txt}')

    def __init__(self):
        d = self.data
        # Heikin Ashi
        self.ha = HeikinAshiSmooth(d, smooth_len=self.p.ha_smooth_len)
        ha_close = self.ha.ha_close
        ha_high = self.ha.ha_high
        ha_low = self.ha.ha_low

        # Indicators
        self.ema_fast = bt.ind.EMA(ha_close, period=self.p.ema_fast)
        self.ema_slow = bt.ind.EMA(ha_close, period=self.p.ema_slow)
        self.atr = bt.ind.ATR(d, period=self.p.atr_len)
        self.atr_sma = bt.ind.SMA(self.atr, period=50)
        self.vwap = VWAPSlope(d)
        self.vol_ma = bt.ind.SMA(d.volume, period=self.p.vol_ma_len)
        self.stoch = StochKD(d, k=self.p.stoch_k, d=self.p.stoch_d, smooth=self.p.stoch_smooth)

        # Accuracy relaxation
        relax = max(0.0, min(1.0, 1.0 - self.p.accuracy))
        self.ob = 80.0 - (80.0 - 60.0) * relax
        self.os = 20.0 + (40.0 - 20.0) * relax
        self.near_tol = 0.0 + (5.0 - 0.0) * relax

        # Position/order refs
        self.order = None
        self.stop_order = None
        self.take_order = None

        # Cache last entry price
        self.entry_price = None

        # Default ATR multiplier per common symbols (can adapt to any symbol automatically)
        name = (getattr(self.data, '_name', '') or '').upper()
        if 'BTC' in name:
            self.default_atr = 1.5
        elif 'GBPJPY' in name or 'GJ' in name:
            self.default_atr = 0.8
        elif 'XAU' in name or 'GOLD' in name:
            self.default_atr = 1.0
        else:
            self.default_atr = 1.0

    # Session filter
    def in_session(self):
        if not self.p.session_enable:
            return True
        dtc = bt.num2date(self.data.datetime[0])
        t = dtc.time()
        in_london = (self.p.london_start <= t < self.p.london_end)
        in_ny = (self.p.ny_start <= t < self.p.ny_end)
        return in_london or in_ny

    def get_atr_mult(self):
        if self.p.manual_atr:
            return self.p.manual_atr_mult
        # Adaptive: default_atr + (atr - atrSMA)/atrSMA * sensitivity
        atr = float(self.atr[0])
        atrs = float(self.atr_sma[0]) if len(self) > 50 else None
        if atrs and atrs != 0:
            return self.default_atr + (atr - atrs) / atrs * self.p.vol_sensitivity
        return self.default_atr

    def next(self):
        d = self.data
        if self.order:
            return  # awaiting order

        # Conditions
        vol_ok = (not self.p.volume_filter) or (d.volume[0] > self.vol_ma[0])
        vwap_up = self.vwap.slope[0] > 0
        vwap_dn = self.vwap.slope[0] < 0

        bull_structure = (self.ha.ha_close[0] > self.vwap.vwap[0] and
                          self.ema_fast[0] > self.ema_slow[0] and vwap_up and vol_ok)
        bear_structure = (self.ha.ha_close[0] < self.vwap.vwap[0] and
                          self.ema_fast[0] < self.ema_slow[0] and vwap_dn and vol_ok)

        stoch_cross_up = self.stoch.k[-1] < self.stoch.d[-1] and self.stoch.k[0] > self.stoch.d[0] and self.stoch.k[0] < (self.os + self.near_tol)
        stoch_cross_dn = self.stoch.k[-1] > self.stoch.d[-1] and self.stoch.k[0] < self.stoch.d[0] and self.stoch.k[0] > (self.ob - self.near_tol)

        long_cond = bull_structure and stoch_cross_up and self.in_session()
        short_cond = bear_structure and stoch_cross_dn and self.in_session()

        # Risk sizing via sizer; we just place bracket orders
        atr_mult = self.get_atr_mult()
        atr = float(self.atr[0])
        price = float(d.close[0])

        if not self.position and long_cond:
            sl = price - atr * self.p.atr_stop * atr_mult
            tp = price + atr * self.p.atr_target * atr_mult
            self.order = self.buy_bracket(price=price, stopprice=sl, limitprice=tp)
            self.entry_price = price
            self.log(f'LONG entry @ {price:.5f} | SL {sl:.5f} TP {tp:.5f} | ATRmult {atr_mult:.3f}')

        elif not self.position and short_cond:
            sl = price + atr * self.p.atr_stop * atr_mult
            tp = price - atr * self.p.atr_target * atr_mult
            self.order = self.sell_bracket(price=price, stopprice=sl, limitprice=tp)
            self.entry_price = price
            self.log(f'SHORT entry @ {price:.5f} | SL {sl:.5f} TP {tp:.5f} | ATRmult {atr_mult:.3f}')

        # Manage BE and trailing when in position
        if self.position:
            # Determine current SL line (approx via last stop order if any)
            # For robustness, recompute desired SL each bar
            atr_mult = self.get_atr_mult()
            atr = float(self.atr[0])
            if self.position.size > 0:  # long
                be_trigger = self.entry_price + self.p.be_R * (self.entry_price - (self.entry_price - self.p.atr_stop * atr * atr_mult))
                be_price = self.entry_price + self.p.be_offset_atr * atr * atr_mult
                trail_sl = d.close[0] - self.p.trail_atr_mult * atr * atr_mult
                desired_sl = self.entry_price - self.p.atr_stop * atr * atr_mult
                if self.p.use_be and d.high[0] >= be_trigger:
                    desired_sl = max(desired_sl, be_price)
                if self.p.use_trail:
                    desired_sl = max(desired_sl, trail_sl)
                # Update stop side of bracket by cancel+reissue (simplest portable way)
                for o in list(self.broker.orders):
                    if o.exectype == bt.Order.Stop and o.data is self.data and o.status in [bt.Order.Accepted, bt.Order.Submitted]:
                        self.cancel(o)
                self.sell(exectype=bt.Order.Stop, price=desired_sl, size=self.position.size)
                self.log(f'LONG manage: new SL {desired_sl:.5f}')
            else:  # short
                be_trigger = self.entry_price - self.p.be_R * ((self.entry_price + self.p.atr_stop * atr * atr_mult) - self.entry_price)
                be_price = self.entry_price - self.p.be_offset_atr * atr * atr_mult
                trail_sl = d.close[0] + self.p.trail_atr_mult * atr * atr_mult
                desired_sl = self.entry_price + self.p.atr_stop * atr * atr_mult
                if self.p.use_be and d.low[0] <= be_trigger:
                    desired_sl = min(desired_sl, be_price)
                if self.p.use_trail:
                    desired_sl = min(desired_sl, trail_sl)
                for o in list(self.broker.orders):
                    if o.exectype == bt.Order.Stop and o.data is self.data and o.status in [bt.Order.Accepted, bt.Order.Submitted]:
                        self.cancel(o)
                self.buy(exectype=bt.Order.Stop, price=desired_sl, size=abs(self.position.size))
                self.log(f'SHORT manage: new SL {desired_sl:.5f}')

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None
        if order.status == order.Completed:
            side = 'BUY' if order.isbuy() else 'SELL'
            self.log(f'ORDER {side} executed @ {order.executed.price:.5f}')
        elif order.status == order.Canceled:
            self.log('ORDER canceled')
        elif order.status == order.Rejected:
            self.log('ORDER rejected')

# ──────────────────────────────────────────────────────────────────────────────
# CLI & Runner
# ──────────────────────────────────────────────────────────────────────────────
class RiskSizer(bt.Sizer):
    params = dict(risk_pct=0.10)
    def _getsizing(self, comminfo, cash, data, isbuy):
        # size = (equity * risk_pct) / price → naive fixed % of equity position sizing
        price = data.close[0]
        target_value = cash * self.p.risk_pct
        size = max(1, int(target_value / max(price, 1e-9)))
        return size

class GenericCSV(bt.feeds.GenericCSVData):
    params = (
        ('dtformat', '%Y-%m-%d %H:%M:%S'),
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1),
        ('timeframe', bt.TimeFrame.Minutes),
        ('compression', 15),
    )


def run(args):
    cerebro = bt.Cerebro()
    data = GenericCSV(dataname=args.csv, timeframe=bt.TimeFrame.Minutes, compression=int(args.timeframe))
    data._name = args.symbol
    cerebro.adddata(data)
    cerebro.addsizer(RiskSizer, risk_pct=args.risk)

    cerebro.addstrategy(
        HAOverlayStrategy,
        accuracy=args.accuracy,
        risk_pct=args.risk,
        ha_smooth_len=args.ha_smooth,
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        atr_len=args.atr_len,
        stoch_k=args.stoch_k,
        stoch_d=args.stoch_d,
        stoch_smooth=args.stoch_smooth,
        vwap_confluence=True,
        volume_filter=not args.no_vol_filter,
        vol_ma_len=args.vol_ma,
        session_enable=not args.no_sessions,
        atr_stop=args.atr_stop,
        atr_target=args.atr_target,
        vol_sensitivity=args.vol_sens,
        manual_atr=args.manual_atr,
        manual_atr_mult=args.manual_atr_mult,
        use_be=not args.no_be,
        be_R=args.be_R,
        be_offset_atr=args.be_off,
        use_trail=not args.no_trail,
        trail_atr_mult=args.trail_atr,
    )

    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(commission=args.comm)
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    res = cerebro.run()
    print(f'Final Portfolio Value:   {cerebro.broker.getvalue():.2f}')
    if args.plot:
        cerebro.plot(style='candlestick')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='HA Overlay Strategy (Backtrader)')
    p.add_argument('--csv', required=True, help='Path to CSV with datetime,open,high,low,close,volume')
    p.add_argument('--symbol', default='XAUUSD')
    p.add_argument('--timeframe', default='15')
    p.add_argument('--cash', type=float, default=10000)
    p.add_argument('--comm', type=float, default=0.0002, help='Commission as fraction (e.g., 0.0002 = 0.02%)')
    p.add_argument('--risk', type=float, default=0.10, help='Position size as %% of equity')
    p.add_argument('--accuracy', type=float, default=0.70)
    p.add_argument('--ha-smooth', type=int, default=3)
    p.add_argument('--ema-fast', type=int, default=21)
    p.add_argument('--ema-slow', type=int, default=200)
    p.add_argument('--atr-len', type=int, default=14)
    p.add_argument('--stoch-k', type=int, default=14)
    p.add_argument('--stoch-d', type=int, default=3)
    p.add_argument('--stoch-smooth', type=int, default=3)
    p.add_argument('--vol-ma', type=int, default=20)
    p.add_argument('--atr-stop', type=float, default=1.0)
    p.add_argument('--atr-target', type=float, default=2.0)
    p.add_argument('--vol-sens', type=float, default=1.0)
    p.add_argument('--manual-atr', action='store_true')
    p.add_argument('--manual-atr-mult', type=float, default=1.0)
    p.add_argument('--no-vol-filter', action='store_true')
    p.add_argument('--no-sessions', action='store_true')
    p.add_argument('--no-be', action='store_true')
    p.add_argument('--be-R', type=float, default=1.0)
    p.add_argument('--be-off', type=float, default=0.1)
    p.add_argument('--no-trail', action='store_true')
    p.add_argument('--trail-atr', type=float, default=1.5)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    run(args)
