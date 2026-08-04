<div align="center">

# Bybit RSI Reversal Bot

**Mean-reversion algorithmic trading for Bybit perpetuals**

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Exchange](https://img.shields.io/badge/Exchange-Bybit-f7a600?style=flat-square)](https://bybit.com)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078d4?style=flat-square&logo=windows&logoColor=white)](https://github.com/ProperProperProper/bybit-rsi-reversal-bot/releases)
[![Strategy](https://img.shields.io/badge/Strategy-Mean%20Reversion-8b5cf6?style=flat-square)](https://properproperproper.github.io/bybit-rsi-reversal-bot/)
[![Symbols](https://img.shields.io/badge/Symbols-BTC%20%7C%20ETH%20%7C%20SOL%20%7C%20BNB%20%7C%20XRP-orange?style=flat-square)](https://github.com/ProperProperProper/bybit-rsi-reversal-bot)

[Website](https://properproperproper.github.io/bybit-rsi-reversal-bot/) · [Wiki](https://github.com/ProperProperProper/bybit-rsi-reversal-bot/wiki) · [Report a Bug](https://github.com/ProperProperProper/bybit-rsi-reversal-bot/issues)

</div>

---

```
┌─ BTC LONG ──────┐ ┌─ ETH FLAT ──────┐ ┌─ SOL LONG ──────┐  ┌─ Wallet ────────┐
│ RSI   38.4      │ │ RSI   61.2      │ │ RSI   29.1      │  │ Avail   124.80  │
│ KDE   0.74D     │ │ KDE   —         │ │ KDE   0.81D     │  │ Wallet  148.30  │
│ CI    43.1      │ │ CI    55.3      │ │ CI    41.7      │  │ DD        1.9%  │
│ px    62840.0   │ │ px    1870.5    │ │ px    163.40    │  │ Ses PnL%  +2.1% │
└─────────────────┘ └─────────────────┘ └─────────────────┘  └─────────────────┘

Recent Trades
Time               Sym   Side   Entry     Exit      PnL%    Why   Bars
2026-07-31 17:34   XRP   LONG   1.0628    1.0698   +6.1%    tp    8
2026-07-31 22:03   ETH   SHORT  1870.66   1865.80  +0.8%    sl    3
```

---

## What Is This?

A fully automated trading system for **Bybit linear perpetuals** that combines three independent confirmation layers to identify high-probability mean-reversion entries. Runs a **walk-forward backtester** every cycle to find optimal parameters per symbol, then executes live trades with a **real-time terminal UI**.

No black box — every decision is a deterministic indicator calculation. No machine learning.

---

## How It Works

An entry fires only when **all three layers agree simultaneously**:

```
RSI crosses oversold/overbought
        │
        ▼
KDE score confirms it's statistically extreme vs last 100 bars
        │
        ▼
Price is inside an MTF Supply or Demand zone
        │
        ▼
Choppiness Index confirms market is trending, not ranging
        │
        ▼
SL distance ≤ 1.5% from entry  →  ENTER
```

**Exits** are set on the exchange immediately after fill (`set_trading_stop`). Take-profit is placed at **70% of the next confirmed swing high/low** distance from entry — capturing most of the move while closing before the full swing target. Stop-loss is set below the demand zone (longs) or above the supply zone (shorts).

---

## Walk-Forward Backtester

The core defence against overfitting:

```
|◄──────────────── 48h window ────────────────►|
|◄── IS: 24h — search 100,000 combos ──►|◄── OOS: 24h ──►|
                                                ▲
                               params only accepted if profitable here too
```

Every 30 minutes the backtester runs fresh, searching **100,000 random parameter combinations** across a 10-dimensional space. A param set is only written to the live trader if it produces **positive return on both the in-sample and out-of-sample windows**. Selection is by Sharpe ratio — not raw return, which overfits.

---

## Features

| | |
|---|---|
| **Walk-forward OOS validation** | Prevents curve-fitting — params tested on unseen data before use |
| **100,000 combos per symbol** | Thorough random search across 10 parameters per BT cycle |
| **5 symbols simultaneously** | BTC, ETH, SOL, BNB, XRP — each with its own optimised params |
| **Exchange-managed TP/SL** | Stops sit on Bybit's server — survive bot restarts and disconnects |
| **Swing TP at 70%** | TP placed at 70% of confirmed swing high/low distance — full position close |
| **Isolated margin** | Each symbol risks only its wallet slice — one bad trade can't wipe the account |
| **Stale DB cleanup** | Orphaned open rows auto-patched on every start and WS reconnect |
| **WebSocket health guard** | Auto-reconnects if stream goes stale; cleanup runs again after |
| **Rich TUI** | Live indicators, session PnL%, per-trade PnL%, sparklines |
| **Crash-safe logging** | Rotating file log + crash dump on unhandled thread exceptions |

---

## Parameters Searched

| Parameter | Range | Type | Description |
|-----------|-------|------|-------------|
| `rsi_p` | 5 – 21 | int | RSI period |
| `rsi_os` | 20 – 40 | int | Oversold threshold — long entry crossover |
| `rsi_ob` | 60 – 80 | int | Overbought threshold — short entry crossover |
| `kde_thr` | 0.30 – 0.70 | float | Min KDE score — how extreme must the reading be |
| `zone_sl_buf` | 0.3 – 2.0 | float | ATR multiplier for zone SL distance |
| `atr_p` | 8 – 20 | int | ATR period |
| `chop_len` | 8 – 20 | int | Choppiness Index period |
| `chop_thr` | 38 – 62 | float | CI threshold — below = trending |
| `swing_len` | 2 – 8 | int | Bars each side to confirm a swing high/low |
| `partial_lvl` | 40 – 60 | int | RSI midline (backtester only) |
| `max_zone_sl_pct` | 1.5% – 3.0% | float | Max allowed zone SL distance as % of entry |

**Fixed constants:** `KDE_LB=100` · `HTF_FACTOR=4` · `SD_ZONE_MULT=1.5` · `LEVERAGE=11x` · `TAKER_FEE=0.055%`

---

## Quick Start

### Prerequisites
- Python 3.11+
- Bybit account with API key (MAINNET)
- `pip install pybit numpy pandas rich scipy`

### Run

**1. Add your API keys**
```
data/.keys_live
```
```json
{ "api_key": "YOUR_KEY", "api_secret": "YOUR_SECRET" }
```

**2. Run the backtester** (generates optimised params)
```
python bybit_rsi_reversal_bt.py
```

**3. Run the live trader**
```
python bybit_rsi_reversal_live_trader.py
```

Or use the pre-built Windows executables directly — no Python required.

The BT and trader run concurrently. The trader picks up fresh params automatically after each BT cycle.

> **Note:** `data/` is gitignored. It contains your API keys, trade database, and generated param files — never committed.

---

## Architecture

```
bybit_rsi_reversal_bt.py           bybit_rsi_reversal_live_trader.py
        │                               │
  fetch OHLCV (REST)            RSIReversalMultiBot (main thread)
  search 100k combos            ├─ WebSocket (public klines)
  walk-forward split            ├─ WebSocket (private pos/wallet)
  write results JSON ──────────►├─ SymbolBot × 5 (per-symbol threads)
                                │   ├─ compute_signal() on bar close
                                │   ├─ place market entry
                                │   └─ set TP (70% swing) / SL (zone) on exchange
                                └─ Rich TUI (live refresh)
```

---

## File Structure

```
bybit-rsi-reversal-bot/
├── bybit_rsi_reversal_bt.py              # Backtester source
├── bybit_rsi_reversal_bt.exe             # Pre-built Windows executable
├── bybit_rsi_reversal_live_trader.py     # Live trader source
├── bybit_rsi_reversal_live_trader.exe    # Pre-built Windows executable
├── docs/                                 # GitHub Pages website
│   ├── index.html
│   ├── robots.txt
│   └── sitemap.xml
├── LICENSE
└── README.md

data/                             # Runtime only — gitignored
├── .keys_live                    # API credentials
├── bybit_rsi_reversal_trades_live.db   # Live trade log
└── bybit_rsi_reversal_best_params_*.json  # Per-symbol best params
```

---

## Contributing

Issues and PRs are welcome. If you've found a bug, improved the strategy, or tested on different symbols/timeframes — open an issue or submit a pull request.

See the [Wiki](https://github.com/ProperProperProper/bybit-rsi-reversal-bot/wiki) for full strategy and architecture documentation.

---

## Disclaimer

This software is provided for educational and research purposes only. Algorithmic trading involves substantial risk of loss. Past backtester performance does not guarantee future results. The authors are not responsible for any financial losses incurred through use of this software. **Trade at your own risk.**

---

<div align="center">

**[Website](https://properproperproper.github.io/bybit-rsi-reversal-bot/) · [Wiki](https://github.com/ProperProperProper/bybit-rsi-reversal-bot/wiki) · [MIT License](LICENSE)**

If this project was useful, consider giving it a ⭐

</div>
