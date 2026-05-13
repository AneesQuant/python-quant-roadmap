# ============================================================
#  Day 22 — MACD Strategy (Moving Average Convergence Divergence)
#  Author Name: Anees
#  Concepts: EMA, MACD Line, Signal Line, Histogram, Buy/Sell Signals
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ─────────────────────────────────────────────
# STEP 1: Download Stock Data
# ─────────────────────────────────────────────
ticker = "AAPL"
df = yf.download(ticker, start="2023-01-01", end="2024-12-31", auto_adjust=True)
df = df[["Close"]].copy()
df.columns = ["Close"]
print(f"✅ Data loaded: {len(df)} rows for {ticker}\n")

# ─────────────────────────────────────────────
# STEP 2: Calculate MACD Indicators
# ─────────────────────────────────────────────
# EMA = Exponential Moving Average (gives more weight to recent prices)
# MACD Line   = Fast EMA (12) - Slow EMA (26)
# Signal Line = 9-period EMA of MACD Line
# Histogram   = MACD Line - Signal Line (shows momentum strength)

EMA_FAST   = 12   # Short-term trend
EMA_SLOW   = 26   # Long-term trend
SIGNAL_WIN =  9   # Smoothing of MACD line

df["EMA_Fast"]   = df["Close"].ewm(span=EMA_FAST,   adjust=False).mean()
df["EMA_Slow"]   = df["Close"].ewm(span=EMA_SLOW,   adjust=False).mean()
df["MACD"]       = df["EMA_Fast"] - df["EMA_Slow"]
df["Signal"]     = df["MACD"].ewm(span=SIGNAL_WIN,  adjust=False).mean()
df["Histogram"]  = df["MACD"] - df["Signal"]

# ─────────────────────────────────────────────
# STEP 3: Generate Buy / Sell Signals
# ─────────────────────────────────────────────
# BUY  → MACD crosses ABOVE Signal line (momentum turning bullish)
# SELL → MACD crosses BELOW Signal line (momentum turning bearish)

df["Signal_Buy"]  = 0
df["Signal_Sell"] = 0

for i in range(1, len(df)):
    prev_diff = df["MACD"].iloc[i-1] - df["Signal"].iloc[i-1]
    curr_diff = df["MACD"].iloc[i]   - df["Signal"].iloc[i]

    if prev_diff < 0 and curr_diff > 0:   # crossover upward
        df.at[df.index[i], "Signal_Buy"] = 1

    if prev_diff > 0 and curr_diff < 0:   # crossover downward
        df.at[df.index[i], "Signal_Sell"] = 1

buy_signals  = df[df["Signal_Buy"]  == 1]
sell_signals = df[df["Signal_Sell"] == 1]
print(f"📈 Total BUY  signals : {len(buy_signals)}")
print(f"📉 Total SELL signals : {len(sell_signals)}\n")

# ─────────────────────────────────────────────
# STEP 4: Simple Backtest (no fees/slippage)
# ─────────────────────────────────────────────
capital    = 10_000   # Starting capital in USD
position   = 0        # Shares held
cash       = capital

for i in range(1, len(df)):
    if df["Signal_Buy"].iloc[i] == 1 and cash > 0:
        price    = df["Close"].iloc[i]
        position = cash / price          # Buy as many shares as possible
        cash     = 0

    elif df["Signal_Sell"].iloc[i] == 1 and position > 0:
        price    = df["Close"].iloc[i]
        cash     = position * price      # Sell all shares
        position = 0

# Final portfolio value
final_value  = cash + (position * df["Close"].iloc[-1])
total_return = ((final_value - capital) / capital) * 100
print(f"💰 Starting Capital : ${capital:,.2f}")
print(f"💼 Final Value      : ${final_value:,.2f}")
print(f"📊 Total Return     : {total_return:.2f}%\n")

# ─────────────────────────────────────────────
# STEP 5: Plot Everything
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle(f"{ticker} — MACD Strategy (Day 22)", fontsize=15, fontweight="bold")

# --- Top Panel: Price + Buy/Sell signals ---
ax1.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.2, label="Close Price")
ax1.scatter(buy_signals.index,  buy_signals["Close"],
            marker="^", color="green", s=80, zorder=5, label="BUY Signal")
ax1.scatter(sell_signals.index, sell_signals["Close"],
            marker="v", color="red",   s=80, zorder=5, label="SELL Signal")
ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left")
ax1.grid(alpha=0.3)

# --- Bottom Panel: MACD, Signal, Histogram ---
ax2.plot(df.index, df["MACD"],   color="blue",   linewidth=1.2, label="MACD Line")
ax2.plot(df.index, df["Signal"], color="orange", linewidth=1.2, label="Signal Line")
colors = ["green" if v >= 0 else "red" for v in df["Histogram"]]
ax2.bar(df.index, df["Histogram"], color=colors, alpha=0.5, label="Histogram")
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.set_ylabel("MACD Value")
ax2.set_xlabel("Date")
ax2.legend(loc="upper left")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/day22_macd_chart.png", dpi=150)
plt.show()
print("✅ Chart saved!")