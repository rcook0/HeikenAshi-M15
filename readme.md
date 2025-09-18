# M15 Heikin Ashi Strategy — Enhanced v1.6

## Overview
This is a fully self-configuring M15 Heikin Ashi strategy designed for TradingView. It combines HA smoothing, EMA trend, VWAP, volume filtering, Stochastic oscillator, and dynamic ATR-based trade management to generate high-probability signals with optional manual override.

The strategy includes automated session detection, symbol-based default volatility factors, and adaptive ATR multipliers.

## Features

- **Heikin Ashi:** Smooth HA candles with adjustable sensitivity.
- **EMA Trend Filter:** Fast (21) and slow (200) EMA trend confirmation.
- **VWAP Confluence:** Trades align with VWAP slope for trend confirmation.
- **Volume Filter:** Optional filter to only take trades when volume is above its moving average.
- **Stochastic Oscillator:** Configurable K/D periods with adjustable accuracy tolerance.
- **Adaptive ATR:** SL, TP, break-even, and trailing stops scale dynamically with recent volatility.
- **Symbol-Specific Defaults:** Default ATR multipliers for XAU, BTC, and GBPJPY.
- **Manual ATR Override:** Option to manually set ATR multiplier.
- **Session Detection:** Auto-detects London and New York sessions, avoids signals outside active sessions.
- **Break-Even & Trailing Stops:** Optional BE and ATR-based trailing stop.
- **Alerts:** Customizable alerts for Long and Short entries.
- **Visual Confluence Zones:** Background highlights for bullish and bearish zones.
- **Plug-and-Play:** Automatically adapts to other symbols with self-configuring ATR multipliers.

## Setup Guide

1. **Add the Script to TradingView:**
   - Copy the Pine Script code into a new TradingView Pine Editor script.
   - Save and add it to your M15 chart.

2. **Configure Inputs:**
   - `Symbol filter`: Choose specific symbols or leave as "Any".
   - `Signal accuracy`: Adjust signal tolerance between 0.3–1.0.
   - `Position size % of equity`: Set risk percentage per trade.
   - `ATR length`: Period for ATR calculation.
   - `Volatility sensitivity`: Adjust how strongly ATR adapts to recent volatility.
   - `Manual ATR multiplier`: Enable manual override if you want fixed ATR multipliers.
   - `Enable volume filter`: Toggle volume-based trade filtering.
   - `Enable session detection`: Toggle London/NY session filtering.

3. **Optional Inputs:**
   - `HA smoothing length`: Adjust HA candle smoothing.
   - `EMA fast/slow`: Modify EMA periods.
   - `Stoch K/D length and smoothing`: Customize stochastic oscillator.
   - `ATR % for Stop/Target`: Adjust risk/reward levels.
   - `Break-even`, `Trailing stop`: Enable/disable trade management features.

4. **Activate Alerts:**
   - Use the provided `alertcondition` entries for Long and Short signals.
   - Configure TradingView alerts to receive notifications.

5. **Backtesting:**
   - Use TradingView's strategy tester to evaluate performance.
   - Adjust inputs as needed for each symbol based on volatility and session behavior.

## Notes

- Ensure you test the strategy on demo or paper trading before using live capital.
- ATR multipliers and session detection are fully self-configuring but can be manually overridden.
- The strategy is designed to be flexible and adapts automatically to new symbols if no manual input is provided.

