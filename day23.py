# ============================================================
#  Day 23 — MACD + RSI Combined Strategy
#  Author Name: Anees
#  Concept: Use BOTH indicators together for stronger signals
#  Rule: Only BUY when MACD says bullish AND RSI is not overbought
#        Only SELL when MACD says bearish AND RSI is not oversold
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
# STEP 2: Calculate RSI
# ─────────────────────────────────────────────
# RSI > 70 = Overbought (avoid buying)
# RSI < 30 = Oversold   (avoid selling)

RSI_PERIOD = 14

delta  = df["Close"].diff()
gain   = delta.clip(lower=0)
loss   = -delta.clip(upper=0)

avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()

rs        = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# ─────────────────────────────────────────────
# STEP 3: Calculate MACD
# ─────────────────────────────────────────────
EMA_FAST   = 12
EMA_SLOW   = 26
SIGNAL_WIN =  9

df["EMA_Fast"]  = df["Close"].ewm(span=EMA_FAST,  adjust=False).mean()
df["EMA_Slow"]  = df["Close"].ewm(span=EMA_SLOW,  adjust=False).mean()
df["MACD"]      = df["EMA_Fast"] - df["EMA_Slow"]
df["Signal"]    = df["MACD"].ewm(span=SIGNAL_WIN, adjust=False).mean()
df["Histogram"] = df["MACD"] - df["Signal"]

df.dropna(inplace=True)

# ─────────────────────────────────────────────
# STEP 4: Combined Buy / Sell Logic
# ─────────────────────────────────────────────
# ✅ BUY  when: MACD crosses ABOVE Signal  AND  RSI < 70 (not overbought)
# ✅ SELL when: MACD crosses BELOW Signal  AND  RSI > 30 (not oversold)

RSI_OVERBOUGHT = 70
RSI_OVERSOLD   = 30

df["Buy_Signal"]  = 0
df["Sell_Signal"] = 0

for i in range(1, len(df)):
    prev_diff = df["MACD"].iloc[i-1] - df["Signal"].iloc[i-1]
    curr_diff = df["MACD"].iloc[i]   - df["Signal"].iloc[i]
    rsi_now   = df["RSI"].iloc[i]

    # MACD bullish crossover + RSI not overbought
    if prev_diff < 0 and curr_diff > 0 and rsi_now < RSI_OVERBOUGHT:
        df.at[df.index[i], "Buy_Signal"] = 1

    # MACD bearish crossover + RSI not oversold
    if prev_diff > 0 and curr_diff < 0 and rsi_now > RSI_OVERSOLD:
        df.at[df.index[i], "Sell_Signal"] = 1

buy_signals  = df[df["Buy_Signal"]  == 1]
sell_signals = df[df["Sell_Signal"] == 1]
print(f"📈 BUY  signals  : {len(buy_signals)}")
print(f"📉 SELL signals  : {len(sell_signals)}\n")

# ─────────────────────────────────────────────
# STEP 5: Simple Backtest
# ─────────────────────────────────────────────
capital  = 10_000
position = 0
cash     = capital
trades   = []

for i in range(1, len(df)):
    price = df["Close"].iloc[i]

    if df["Buy_Signal"].iloc[i] == 1 and cash > 0:
        position = cash / price
        cash     = 0
        trades.append(("BUY", df.index[i], price))

    elif df["Sell_Signal"].iloc[i] == 1 and position > 0:
        cash     = position * price
        position = 0
        trades.append(("SELL", df.index[i], price))

final_value  = cash + (position * df["Close"].iloc[-1])
total_return = ((final_value - capital) / capital) * 100

print(f"💰 Starting Capital : ${capital:,.2f}")
print(f"💼 Final Value      : ${final_value:,.2f}")
print(f"📊 Total Return     : {total_return:.2f}%")
print(f"🔄 Total Trades     : {len(trades)}\n")

# ─────────────────────────────────────────────
# STEP 6: Plot — 3 Panels (Price, MACD, RSI)
# ─────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
fig.suptitle(f"{ticker} — MACD + RSI Combined Strategy (Day 23)",
             fontsize=14, fontweight="bold")

# Panel 1: Price + Signals
ax1.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.2, label="Close Price")
ax1.scatter(buy_signals.index,  buy_signals["Close"],
            marker="^", color="green", s=90, zorder=5, label="BUY Signal")
ax1.scatter(sell_signals.index, sell_signals["Close"],
            marker="v", color="red",   s=90, zorder=5, label="SELL Signal")
ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(alpha=0.3)

# Panel 2: MACD
ax2.plot(df.index, df["MACD"],   color="blue",   linewidth=1.1, label="MACD")
ax2.plot(df.index, df["Signal"], color="orange", linewidth=1.1, label="Signal")
colors = ["green" if v >= 0 else "red" for v in df["Histogram"]]
ax2.bar(df.index, df["Histogram"], color=colors, alpha=0.4, label="Histogram")
ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")
ax2.set_ylabel("MACD")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.3)

# Panel 3: RSI
ax3.plot(df.index, df["RSI"], color="purple", linewidth=1.1, label="RSI")
ax3.axhline(RSI_OVERBOUGHT, color="red",   linewidth=0.8, linestyle="--", label="Overbought (70)")
ax3.axhline(RSI_OVERSOLD,   color="green", linewidth=0.8, linestyle="--", label="Oversold (30)")
ax3.fill_between(df.index, RSI_OVERBOUGHT, 100, alpha=0.08, color="red")
ax3.fill_between(df.index, 0, RSI_OVERSOLD,    alpha=0.08, color="green")
ax3.set_ylim(0, 100)
ax3.set_ylabel("RSI")
ax3.set_xlabel("Date")
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/day23_macd_rsi_chart.png", dpi=150)
plt.show()
print("✅ Chart saved!")