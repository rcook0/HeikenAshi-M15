# ha_plot.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Sample Data Generator
# ─────────────────────────────────────────────
def generate_sample(n=200, symbol="XAUUSD"):
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
# HA Calculation + Plot
# ─────────────────────────────────────────────
def ha_plot(df, ema_fast=21, ema_slow=200, ha_smooth=3, title="Heikin Ashi Plot"):
    ha_close_raw = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_close = ha_close_raw.ewm(span=ha_smooth).mean()
    ha_open = (df["open"] + df["close"]) / 2
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)

    emaF = ha_close.ewm(span=ema_fast).mean()
    emaS = ha_close.ewm(span=ema_slow).mean()

    fig, ax = plt.subplots(figsize=(12,6))
    for i in range(len(df)):
        color = "green" if ha_close.iloc[i] >= ha_open.iloc[i] else "red"
        ax.plot([i, i], [ha_low.iloc[i], ha_high.iloc[i]], color=color, linewidth=1)
        ax.add_patch(plt.Rectangle(
            (i-0.3, min(ha_open.iloc[i], ha_close.iloc[i])),
            0.6,
            abs(ha_close.iloc[i]-ha_open.iloc[i]),
            facecolor=color, edgecolor=color, alpha=0.6
        ))

    ax.plot(range(len(df)), emaF, label="EMA Fast", color="blue")
    ax.plot(range(len(df)), emaS, label="EMA Slow", color="orange")
    ax.set_title(title)
    ax.legend()
    plt.show()

# ─────────────────────────────────────────────
# Demo Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_sample()
    ha_plot(df, title="HA Candles + EMA (Pure View)")
