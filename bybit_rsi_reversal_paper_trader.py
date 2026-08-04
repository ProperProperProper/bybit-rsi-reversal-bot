#!/usr/bin/env python3
"""
RSI Kernel + MTF S/D — PAPER TRADER with AI AGENT
Simulates trades using live Bybit kline feeds. No real orders placed.
AI agent (Claude via CLI) analyses every signal and trade, manages subscription limits.
"""
import io, os, sys, time, json, math, bisect, shutil, subprocess, threading, logging, sqlite3, signal
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP, WebSocket
from rich.live    import Live
from rich.table   import Table
from rich.text    import Text
from rich.panel   import Panel
from rich.columns import Columns
from rich.console import Console
from rich         import box

# -- Paths ---------------------------------------------------------------------
if getattr(sys, "frozen", False):
    _DIR = os.path.dirname(sys.executable)
else:
    _DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

KEYS_PATH    = os.path.join(DATA_DIR, ".keys_live")
DB_PATH      = os.path.join(DATA_DIR, "bybit_rsi_reversal_trades_paper.db")
AI_LOG_PATH  = os.path.join(DATA_DIR, "bybit_rsi_reversal_ai_log.jsonl")

# -- Fixed constants (must match BT exactly) -----------------------------------
CATEGORY         = "linear"
LEVERAGE         = 11
TAKER_FEE        = 0.00055
MARGIN_HEADROOM  = 0.98
KDE_LB           = 100
HTF_FACTOR       = 4
SD_ZONE_MULT     = 0.5
MAX_ZONE_SL_PCT_DEFAULT = 0.020
TP_FACTOR        = 0.70
SEED_BARS        = 500
WS_STALE_S       = 120
BALANCE_REFRESH_S = 300
BALANCE_FALLBACK  = 600.0
MIN_BAL_PER_SYM   = 20.0
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

MIN_LIVE_CHOP_THR  = 45.0
MIN_LIVE_SHARPE    = 0.5   # 2-day BT produces sharpe 1–8; 15 was unreachable
MIN_LIVE_RETURN_PT = 0.0   # ret>0 already enforced; per-trade threshold not meaningful on 3-4 trades

# -- AI agent constants --------------------------------------------------------
AI_ENABLED          = True          # set False to disable all AI calls
AI_MAX_PER_HOUR     = 15            # max Claude calls per rolling hour
AI_MAX_PER_DAY      = 80            # hard daily cap (resets midnight UTC)
AI_MIN_INTERVAL_S   = 90            # minimum seconds between any two calls
AI_CALL_TIMEOUT_S   = 120           # subprocess timeout per call
AI_SESSION_SUMMARY_H = 4            # session summary every N hours
AI_MODEL_FAST       = "claude-haiku-4-5-20251001"   # signal checks
AI_MODEL_ANALYSIS   = "claude-sonnet-4-6"           # trade analysis & summaries
AI_SNAP_BARS            = 25            # OHLCV bars to include in prompts
AI_GATE_TIMEOUT_S       = 45            # tight timeout for synchronous entry/position gates
AI_POSITION_CHECK_BARS  = 3             # check open position every N bars (45 min for 15m)

# Priority codes (lower = higher priority)
_AI_PRI_TRADE   = 1
_AI_PRI_SIGNAL  = 2
_AI_PRI_SUMMARY = 3

# -- Select best (symbol, interval) from BT results ---------------------------

def _best_for_symbol(sym):
    best_sharpe = -999.0
    best_entry  = None
    for iv in ["3", "15", "30"]:
        path = os.path.join(DATA_DIR, f"bybit_rsi_reversal_results_{sym}_{iv}m.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                d = json.load(f)
            ret      = float(d.get("total_return_pct", -999))
            sharpe   = float(d.get("sharpe", -999))
            oos_r    = float(d.get("oos_return_pct", ret))
            wr       = float(d.get("win_rate", 0))
            n_trades = int(d.get("trades", 0))
            params   = d.get("params", {})
            chop_thr = float(params.get("chop_thr", 0))
            ret_pt   = ret / max(n_trades, 1)
            mtime    = os.path.getmtime(path)
            if ret <= 0 or sharpe <= 0 or oos_r <= 0 or wr < 0.60:
                continue
            if chop_thr < MIN_LIVE_CHOP_THR or sharpe < MIN_LIVE_SHARPE or ret_pt < MIN_LIVE_RETURN_PT:
                continue
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_entry = {
                    "interval":    iv,
                    "params":      d["params"],
                    "sharpe":      sharpe,
                    "return":      ret,
                    "win_rate":    wr,
                    "result_path": path,
                    "result_mtime": mtime,
                }
        except Exception:
            pass
    return best_entry


def _select_symbols():
    selections = {}
    for sym in ALL_SYMBOLS:
        best = _best_for_symbol(sym)
        if best:
            selections[sym] = best
    return selections

SELECTIONS = _select_symbols()
SYMBOLS    = list(SELECTIONS.keys())
N_SYMBOLS  = len(SYMBOLS)

# -- Logging -------------------------------------------------------------------
_log = logging.getLogger("bybit_rsi_reversal_paper_trader")
_log.setLevel(logging.DEBUG)
_log.propagate = False
_log_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
_fh = RotatingFileHandler(
    os.path.join(DATA_DIR, "bybit_rsi_reversal_paper_trader.log"),
    maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
_fh.setFormatter(_log_fmt); _fh.setLevel(logging.DEBUG); _log.addHandler(_fh)
_sh = logging.StreamHandler(sys.stderr)
_sh.setFormatter(_log_fmt); _sh.setLevel(logging.WARNING); _log.addHandler(_sh)


def _thread_excepthook(args):
    import traceback as _tb
    msg = "".join(_tb.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    _log.critical(f"Unhandled thread exception in "
                  f"'{(args.thread or threading.current_thread()).name}':\n{msg}")
    _write_crash(msg)

threading.excepthook = _thread_excepthook


def _write_crash(text):
    try:
        with open(os.path.join(DATA_DIR, "crash_bybit_rsi_reversal_paper_trader.log"),
                  "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n{datetime.now(timezone.utc).isoformat()}\n{text}\n")
    except Exception:
        pass


def _make_console():
    if sys.platform == "win32":
        try:
            buf = getattr(sys.stdout, "buffer", None)
            if buf is not None:
                return Console(file=io.TextIOWrapper(buf, encoding="utf-8",
                                                     write_through=True))
        except Exception:
            pass
    return Console()


# -- Helpers -------------------------------------------------------------------

def _decimals(step):
    s = f"{float(step):.10f}".rstrip("0")
    return max(0, len(s) - s.index(".") - 1) if "." in s else 0


def load_keys():
    if os.path.exists(KEYS_PATH):
        try:
            with open(KEYS_PATH) as f:
                d = json.load(f)
            return d.get("api_key", ""), d.get("api_secret", "")
        except Exception:
            pass
    return "", ""


def make_session():
    k, s = load_keys()
    if not k or not s:
        _log.critical(f"No API keys — place them in {KEYS_PATH}"); sys.exit(1)
    try:
        sess = HTTP(api_key=k, api_secret=s, demo=False)
        r = sess.get_wallet_balance(accountType="UNIFIED")
        if r.get("retCode", -1) != 0:
            _log.critical(f"Wallet check failed: {r.get('retMsg','')}"); sys.exit(1)
        _log.info(f"Session OK MAINNET read-only (key=...{k[-4:]})")
        return sess
    except Exception as e:
        import traceback
        _log.critical(f"Cannot connect: {e}\n{traceback.format_exc()}"); sys.exit(1)


_NO_RETRY         = {110007, 110006, 110012, 110013, 110017, 110025}
_RATE_LIMIT_CODES = {10006, 10018}
_API_SEM          = threading.Semaphore(5)


def _api(fn, *args, _log_fn=None, **kwargs):
    for attempt in range(3):
        with _API_SEM:
            try:
                r = fn(*args, **kwargs)
            except Exception as e:
                if attempt == 2:
                    if _log_fn: _log_fn(f"API error {fn.__name__}: {e}", "ERROR")
                    raise
                time.sleep(0.5 * (attempt + 1)); continue
        if isinstance(r, dict):
            rc = r.get("retCode", 0)
            if rc in _RATE_LIMIT_CODES:
                time.sleep(1 + attempt); continue
            if rc in _NO_RETRY:
                return r
        return r
    raise RuntimeError(f"API {fn.__name__} exhausted after 3 attempts")


def _sign(v):
    try:
        return "green" if float(v) > 0 else ("red" if float(v) < 0 else "dim")
    except Exception:
        return "dim"


def _spark(values, width=22):
    vals = list(values)
    if len(vals) < 2: return "─" * width
    lo, hi = min(vals), max(vals)
    if hi == lo: return "─" * width
    CHARS = " ▁▂▃▄▅▆▇█"
    step = max(1, len(vals) // width)
    out = ""
    for i in range(0, len(vals), step):
        chunk = vals[i:i + step]
        idx = max(0, min(len(CHARS) - 1,
                         int((sum(chunk) / len(chunk) - lo) / (hi - lo) * (len(CHARS) - 1))))
        out += CHARS[idx]
    return out[:width]


# -- Indicators (exact copies from BT, must never diverge) --------------------

def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain  = np.maximum(delta, 0.0)
    loss  = np.maximum(-delta, 0.0)
    avg_g = pd.Series(gain).ewm(alpha=1.0 / period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(alpha=1.0 / period, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rs  = np.where(avg_l == 0, 100.0, avg_g / avg_l)
        rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _kde_scores(rsi_arr: np.ndarray, lb: int) -> tuple:
    from numpy.lib.stride_tricks import sliding_window_view
    n           = len(rsi_arr)
    score_long  = np.full(n, np.nan)
    score_short = np.full(n, np.nan)
    if n < lb:
        return score_long, score_short
    windows              = sliding_window_view(rsi_arr, lb)
    cur_col              = windows[:, -1:]
    score_long[lb - 1:]  = np.mean(windows > cur_col, axis=1)
    score_short[lb - 1:] = np.mean(windows < cur_col, axis=1)
    return score_long, score_short


def _atr_wilder(high, low, close, period):
    tr    = np.empty(len(close))
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
             np.maximum(np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:]  - close[:-1])))
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values


def _chop_index(high, low, close, period):
    tr    = np.empty(len(close))
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
             np.maximum(np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:]  - close[:-1])))
    sum_tr = pd.Series(tr).rolling(period, min_periods=period).sum().values
    hh     = pd.Series(high).rolling(period, min_periods=period).max().values
    ll     = pd.Series(low).rolling(period,  min_periods=period).min().values
    rng    = hh - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.where(rng == 0, 100.0,
             100.0 * np.log10(np.maximum(sum_tr, 1e-10) / np.maximum(rng, 1e-10))
             / math.log10(period))
    ci[:period - 1] = np.nan
    return ci


def _compute_sd_zones(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray) -> tuple:
    N_BASE  = 2
    MIN_RUN = 2
    n     = len(close)
    n_htf = n // HTF_FACTOR
    if n_htf < N_BASE + MIN_RUN + 1:
        return (np.zeros(n, bool), np.zeros(n, bool),
                np.full(n, np.nan), np.full(n, np.nan))

    htf_open  = np.array([open_[i * HTF_FACTOR]                             for i in range(n_htf)])
    htf_high  = np.array([high[i * HTF_FACTOR:(i + 1) * HTF_FACTOR].max()  for i in range(n_htf)])
    htf_low   = np.array([low[i * HTF_FACTOR:(i + 1) * HTF_FACTOR].min()   for i in range(n_htf)])
    htf_close = np.array([close[(i + 1) * HTF_FACTOR - 1]                  for i in range(n_htf)])

    htf_rng  = htf_high - htf_low
    htf_body = np.abs(htf_close - htf_open)
    is_imp   = np.where(htf_rng > 0, htf_body / htf_rng, 0.0) > SD_ZONE_MULT
    is_bull  = htf_close > htf_open
    is_bear  = htf_close < htf_open

    zones_raw = []
    i = 0
    while i < n_htf:
        if is_imp[i]:
            j = i
            if is_bull[i]:
                while j < n_htf and is_imp[j] and is_bull[j]:
                    j += 1
                if (j - i) >= MIN_RUN and i >= N_BASE:
                    zones_raw.append(('demand',
                                      htf_high[i - N_BASE:i].max(),
                                      htf_low[i - N_BASE:i].min(), j))
            elif is_bear[i]:
                while j < n_htf and is_imp[j] and is_bear[j]:
                    j += 1
                if (j - i) >= MIN_RUN and i >= N_BASE:
                    zones_raw.append(('supply',
                                      htf_high[i - N_BASE:i].max(),
                                      htf_low[i - N_BASE:i].min(), j))
            i = j
        else:
            i += 1

    in_demand  = np.zeros(n, bool)
    in_supply  = np.zeros(n, bool)
    demand_bot = np.full(n, np.nan)
    supply_top = np.full(n, np.nan)

    for zone_type, z_top, z_bot, htf_active in zones_raw:
        base_start = htf_active * HTF_FACTOR
        if base_start >= n:
            continue
        if zone_type == 'demand':
            for b in range(base_start, n):
                if close[b] < z_bot:
                    break
                if z_bot <= close[b] <= z_top:
                    in_demand[b]  = True
                    demand_bot[b] = z_bot
        else:
            for b in range(base_start, n):
                if close[b] > z_top:
                    break
                if z_bot <= close[b] <= z_top:
                    in_supply[b]  = True
                    supply_top[b] = z_top

    return in_demand, in_supply, demand_bot, supply_top


def _build_swing_levels(high: np.ndarray, low: np.ndarray, swing_len: int) -> tuple:
    n = len(high); sh = []; sl = []
    for i in range(swing_len, n - swing_len):
        if (high[i] >= high[i - swing_len:i].max() and
                high[i] >= high[i + 1:i + swing_len + 1].max()):
            sh.append((i + swing_len, high[i]))
        if (low[i] <= low[i - swing_len:i].min() and
                low[i] <= low[i + 1:i + swing_len + 1].min()):
            sl.append((i + swing_len, low[i]))
    return sh, sl


def compute_signal(candles_snap: list, params: dict):
    n = len(candles_snap)
    if n < 20:
        return None

    close = np.array([c["c"] for c in candles_snap], dtype=float)
    high  = np.array([c["h"] for c in candles_snap], dtype=float)
    low   = np.array([c["l"] for c in candles_snap], dtype=float)
    open_ = np.array([c["o"] for c in candles_snap], dtype=float)

    rsi_p           = int(params["rsi_p"])
    rsi_os          = int(params["rsi_os"])
    rsi_ob          = int(params["rsi_ob"])
    kde_thr         = float(params["kde_thr"])
    atr_p           = int(params.get("atr_p", 14))
    chop_len        = int(params["chop_len"])
    chop_thr        = float(params["chop_thr"])
    swing_len       = int(params["swing_len"])
    max_zone_sl_pct = float(params.get("max_zone_sl_pct", MAX_ZONE_SL_PCT_DEFAULT))

    if n < max(rsi_p, chop_len, atr_p, KDE_LB) + 10:
        return None

    rsi_arr                     = _rsi(close, rsi_p)
    kde_long_arr, kde_short_arr = _kde_scores(rsi_arr, KDE_LB)
    ci_arr                      = _chop_index(high, low, close, chop_len)
    atr_arr                     = _atr_wilder(high, low, close, atr_p)
    in_demand, in_supply, demand_bot, supply_top = _compute_sd_zones(open_, high, low, close)

    rsi_cur       = rsi_arr[-1];  rsi_prev = rsi_arr[-2]
    kde_long_cur  = float(kde_long_arr[-1])
    kde_short_cur = float(kde_short_arr[-1])
    ci_cur        = ci_arr[-1]
    atr_cur       = atr_arr[-1]

    if not all(np.isfinite(v) for v in [rsi_cur, rsi_prev, ci_cur, atr_cur,
                                         kde_long_cur, kde_short_cur]):
        return None

    ci_ok       = bool(ci_cur > chop_thr)
    cross_os_up = (rsi_prev < rsi_os) and (rsi_cur >= rsi_os)
    cross_ob_dn = (rsi_prev > rsi_ob) and (rsi_cur <= rsi_ob)

    in_d  = bool(in_demand[-1])
    in_s  = bool(in_supply[-1])
    d_bot = float(demand_bot[-1])
    s_top = float(supply_top[-1])
    cl    = float(close[-1])

    sl_long_ok  = (in_d and np.isfinite(d_bot) and d_bot < cl
                   and abs(cl - d_bot) / cl <= max_zone_sl_pct)
    sl_short_ok = (in_s and np.isfinite(s_top) and s_top > cl
                   and abs(s_top - cl) / cl <= max_zone_sl_pct)

    sh_events, sl_events = _build_swing_levels(high, low, swing_len)
    sh_prices = sorted(p for (_, p) in sh_events)
    sl_prices = sorted(p for (_, p) in sl_events)
    has_long_tp  = bisect.bisect_right(sh_prices, cl) < len(sh_prices)
    has_short_tp = bisect.bisect_left(sl_prices, cl) - 1 >= 0

    buy_sig  = (cross_os_up and kde_long_cur  >= kde_thr
                and ci_ok and in_d and sl_long_ok and has_long_tp)
    sell_sig = (cross_ob_dn and kde_short_cur >= kde_thr
                and ci_ok and in_s and sl_short_ok and has_short_tp)

    return {
        "buy_sig": buy_sig, "sell_sig": sell_sig,
        "rsi": rsi_cur, "rsi_prev": rsi_prev,
        "ci": ci_cur, "atr": atr_cur,
        "kde_long":  kde_long_cur,
        "kde_short": kde_short_cur,
        "in_demand": in_d, "in_supply": in_s,
        "demand_bot": d_bot if in_d else None,
        "supply_top": s_top if in_s else None,
        "sh_prices": sh_prices, "sl_prices": sl_prices,
        "close": cl,
        "params_snapshot": {
            "rsi_p": rsi_p, "rsi_os": rsi_os, "rsi_ob": rsi_ob,
            "kde_thr": kde_thr, "chop_thr": chop_thr,
            "swing_len": swing_len, "max_zone_sl_pct": max_zone_sl_pct,
        },
    }


# -- DB ------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT,
            symbol       TEXT,
            interval     TEXT,
            side         TEXT,
            entry        REAL,
            partial_px   REAL,
            exit_price   REAL,
            qty          REAL,
            net_pnl      REAL,
            gross_pnl    REAL,
            entry_fee    REAL,
            exit_fee     REAL,
            mfe_pct      REAL,
            mae_pct      REAL,
            close_reason TEXT,
            bars_held    INTEGER,
            status       TEXT NOT NULL DEFAULT 'closed'
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_analysis (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT,
            event_type   TEXT,
            symbol       TEXT,
            model        TEXT,
            prompt_chars INTEGER,
            response     TEXT,
            duration_s   REAL,
            throttled    INTEGER DEFAULT 0
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bot_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            level     TEXT,
            message   TEXT
        )""")
    conn.commit()
    return conn


# =============================================================================
# AIAgent — Claude-backed trade analysis via CLI subprocess
# =============================================================================

class AIAgent:
    """
    Calls `claude -p <prompt>` as a subprocess for each analysis event.
    Manages subscription session limits: max per hour, max per day, min interval.
    Priority queue: trade analysis (1) > signal check (2) > session summary (3).
    All responses logged to SQLite ai_analysis table + JSONL file.
    """

    def __init__(self, multi):
        self._multi       = multi
        self._enabled     = AI_ENABLED and bool(shutil.which("claude"))
        self._lock        = threading.Lock()
        # (priority, enqueue_time, event_dict)
        self._queue: list = []
        self._call_log: deque = deque(maxlen=AI_MAX_PER_HOUR * 25)  # timestamps of recent calls
        self._daily_count = 0
        self._daily_reset_date = datetime.now(timezone.utc).date()
        self._last_call_ts = 0.0
        self._last_insight = ""   # last AI response snippet for TUI display
        self._calls_this_hour = 0
        self._calls_today     = 0
        self._throttle_count  = 0

        if not self._enabled:
            reason = "AI_ENABLED=False" if not AI_ENABLED else "claude CLI not found in PATH"
            _log.warning(f"AIAgent disabled — {reason}")

        threading.Thread(target=self._worker, daemon=True, name="ai-agent").start()

    # -- Scheduling helpers ----------------------------------------------------

    def _reset_daily_if_needed(self):
        today = datetime.now(timezone.utc).date()
        if today != self._daily_reset_date:
            self._daily_reset_date = today
            self._daily_count      = 0

    def _hour_count(self) -> int:
        cutoff = time.time() - 3600
        return sum(1 for t in self._call_log if t > cutoff)

    def _can_call(self) -> bool:
        if not self._enabled:
            return False
        now = time.time()
        if now - self._last_call_ts < AI_MIN_INTERVAL_S:
            return False
        self._reset_daily_if_needed()
        if self._daily_count >= AI_MAX_PER_DAY:
            return False
        if self._hour_count() >= AI_MAX_PER_HOUR:
            return False
        return True

    def _budget_str(self) -> str:
        h = self._hour_count()
        return f"AI calls: {h}/{AI_MAX_PER_HOUR} this hr | {self._daily_count}/{AI_MAX_PER_DAY} today"

    # -- Synchronous helpers ---------------------------------------------------

    def _call_claude_sync(self, model: str, prompt: str, timeout: int = AI_GATE_TIMEOUT_S) -> str:
        result = subprocess.run(
            ["claude", "-p", "--model", model, prompt],
            capture_output=True, text=True, timeout=timeout, encoding="utf-8",
        )
        return result.stdout.strip() or result.stderr.strip() or ""

    def _record_call(self, symbol: str, etype: str, model: str,
                     prompt: str, response: str, duration: float):
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self._last_call_ts    = time.time()
            self._call_log.append(self._last_call_ts)
            self._daily_count    += 1
            self._last_insight    = f"[{etype}/{symbol}] {response[:120]}"
            self._calls_this_hour = self._hour_count()
            self._calls_today     = self._daily_count
        entry = {
            "ts": ts_str, "type": etype, "symbol": symbol, "model": model,
            "prompt_chars": len(prompt), "response": response,
            "duration_s": round(duration, 2),
        }
        try:
            with open(AI_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            _log.warning(f"AI log write failed: {e}")
        try:
            self._multi.db.execute(
                "INSERT INTO ai_analysis "
                "(timestamp, event_type, symbol, model, prompt_chars, response, duration_s) "
                "VALUES (?,?,?,?,?,?,?)",
                (ts_str, etype, symbol, model, len(prompt), response, round(duration, 2)))
            self._multi.db.commit()
        except Exception as e:
            _log.warning(f"AI DB write failed: {e}")

    def gate_entry(self, symbol: str, interval: str, side: str, sig: dict,
                   params: dict, entry_px: float, sl: float, tp: float,
                   notional: float, qty: float) -> dict:
        """
        Synchronous entry gate — blocks until Claude responds.
        Returns {"proceed": bool, "reason": str}.
        Fail-safe: if AI unavailable, throttled, or times out → always PROCEED.
        """
        if not self._enabled or not self._can_call():
            return {"proceed": True, "reason": "AI throttled — proceeding"}

        p         = params
        cl        = sig["close"]
        rsi_prev  = sig.get("rsi_prev", float("nan"))
        kde_score = sig["kde_long"] if side == "long" else sig["kde_short"]
        sl_pct    = abs(cl - sl) / cl if cl else 0

        prompt = (
            f"You are gating a live paper trade entry for an RSI Kernel + MTF S/D strategy.\n"
            f"Verify BT compliance and flag any reason to SKIP this trade.\n\n"
            f"STRATEGY RULES (BT exact):\n"
            f"- Long: rsi_prev < rsi_os AND rsi_cur >= rsi_os (crossover up)\n"
            f"- Short: rsi_prev > rsi_ob AND rsi_cur <= rsi_ob (crossover down)\n"
            f"- KDE score >= kde_thr (fraction of last 100 RSI bars above/below current)\n"
            f"- CI > chop_thr (choppy market required)\n"
            f"- Price inside MTF S/D zone; SL at zone distal; dist <= max_zone_sl_pct\n"
            f"- TP = 70% of distance to next confirmed swing\n\n"
            f"SIGNAL: {symbol} {interval}m {side.upper()}\n"
            f"  rsi_prev={rsi_prev:.2f}  rsi_cur={sig['rsi']:.2f}  "
            f"rsi_os={p.get('rsi_os')}  rsi_ob={p.get('rsi_ob')}\n"
            f"  kde_score={kde_score:.4f}  kde_thr={p.get('kde_thr'):.2f}\n"
            f"  ci={sig['ci']:.2f}  chop_thr={p.get('chop_thr'):.1f}\n"
            f"  sl_pct={sl_pct:.4f}  max_sl_pct={p.get('max_zone_sl_pct', MAX_ZONE_SL_PCT_DEFAULT):.4f}\n"
            f"  entry={entry_px:.5f}  sl={sl:.5f}  tp={tp:.5f}\n"
            f"  notional={notional:.2f} USDT  qty={qty:.4f}  lev=11x isolated\n\n"
            f"FIRST LINE MUST BE EXACTLY: PROCEED or SKIP\n"
            f"Then one sentence explaining."
        )
        t0 = time.time()
        try:
            response = self._call_claude_sync(AI_MODEL_FAST, prompt)
            duration = time.time() - t0
            self._record_call(symbol, "entry_gate", AI_MODEL_FAST, prompt, response, duration)
            first_line = response.split("\n")[0].upper() if response else ""
            proceed    = "SKIP" not in first_line  # fail-safe: proceed unless explicit SKIP
            _log.info(f"AI gate {symbol} {side}: {'PROCEED' if proceed else 'SKIP'} — {response[:60]}")
            return {"proceed": proceed, "reason": response[:200]}
        except subprocess.TimeoutExpired:
            self._multi.log(f"AI entry gate {symbol} timed out — proceeding", "WARN")
            return {"proceed": True, "reason": "AI timeout — proceeding"}
        except Exception as e:
            self._multi.log(f"AI entry gate {symbol} error: {e} — proceeding", "WARN")
            return {"proceed": True, "reason": f"AI error ({e}) — proceeding"}

    def gate_position(self, symbol: str, interval: str, pos: dict,
                      h: float, l: float, c: float,
                      bars_held: int, mfe_pct: float, mae_pct: float) -> dict:
        """
        Synchronous position review — called every AI_POSITION_CHECK_BARS while in trade.
        Returns {"exit_now": bool, "reason": str}.
        Fail-safe: if AI unavailable, throttled, or times out → always HOLD.
        """
        if not self._enabled or not self._can_call():
            return {"exit_now": False, "reason": "AI throttled — holding"}

        side      = pos["side"]
        entry_px  = pos["entry_px"]
        sl        = pos["stop_level"]
        tp        = pos["tp_level"]
        p         = pos.get("params_snapshot", {})
        dist_sl   = abs(c - sl) / entry_px if entry_px else 0
        dist_tp   = abs(c - tp) / entry_px if entry_px else 0
        tp_rng    = abs(tp - entry_px)
        progress  = (((c - entry_px) / tp_rng) if side == "long" else
                     ((entry_px - c) / tp_rng)) if tp_rng > 0 else 0
        unrealised = ((c - entry_px) if side == "long" else (entry_px - c)) * pos["qty"]

        prompt = (
            f"You are reviewing an open paper trade for RSI Kernel + MTF S/D strategy.\n"
            f"Decide: EXIT NOW (before TP/SL) or HOLD (let BT rules play out).\n\n"
            f"POSITION: {symbol} {interval}m {side.upper()}\n"
            f"  Entry={entry_px:.5f}  SL={sl:.5f}  TP={tp:.5f}\n"
            f"  Bar: H={h:.5f} L={l:.5f} C={c:.5f}\n"
            f"  TP progress={progress*100:.1f}%  dist_SL={dist_sl:.3%}  dist_TP={dist_tp:.3%}\n"
            f"  Unrealised={unrealised:+.4f} USDT  MFE={mfe_pct:+.2f}%  MAE={mae_pct:+.2f}%\n"
            f"  Bars held={bars_held}  Params: rsi_os={p.get('rsi_os','?')} "
            f"kde_thr={p.get('kde_thr','?')} chop_thr={p.get('chop_thr','?')}\n\n"
            f"Default is HOLD unless setup is clearly invalidated.\n"
            f"FIRST LINE MUST BE EXACTLY: HOLD or EXIT NOW\n"
            f"Then one sentence explaining."
        )
        t0 = time.time()
        try:
            response = self._call_claude_sync(AI_MODEL_FAST, prompt)
            duration = time.time() - t0
            self._record_call(symbol, "position_gate", AI_MODEL_FAST, prompt, response, duration)
            first_line = response.split("\n")[0].upper() if response else ""
            exit_now   = "EXIT" in first_line  # HOLD is default; EXIT only if explicit
            _log.info(f"AI pos gate {symbol} {side}: {'EXIT' if exit_now else 'HOLD'} — {response[:60]}")
            return {"exit_now": exit_now, "reason": response[:200]}
        except subprocess.TimeoutExpired:
            self._multi.log(f"AI position gate {symbol} timed out — holding", "WARN")
            return {"exit_now": False, "reason": "AI timeout — holding"}
        except Exception as e:
            self._multi.log(f"AI position gate {symbol} error: {e} — holding", "WARN")
            return {"exit_now": False, "reason": f"AI error ({e}) — holding"}

    # -- Enqueue ---------------------------------------------------------------

    def enqueue(self, priority: int, event: dict):
        if not self._enabled:
            return
        with self._lock:
            # Drop duplicate signal checks for same symbol if already queued
            sym = event.get("symbol", "")
            etype = event.get("type", "")
            if etype == "signal_check":
                self._queue = [e for e in self._queue
                               if not (e[2].get("type") == "signal_check"
                                       and e[2].get("symbol") == sym)]
            self._queue.append((priority, time.time(), event))
            self._queue.sort(key=lambda x: (x[0], x[1]))

    def enqueue_signal_check(self, symbol: str, interval: str, side: str,
                              sig: dict, params: dict, entry_px: float,
                              sl: float, tp: float, notional: float, qty: float):
        p  = params
        cl = sig["close"]
        rsi_prev = sig.get("rsi_prev", float("nan"))
        cross_ok = ("✓" if ((side == "long" and rsi_prev < p["rsi_os"] and sig["rsi"] >= p["rsi_os"])
                             or (side == "short" and rsi_prev > p["rsi_ob"] and sig["rsi"] <= p["rsi_ob"]))
                    else "✗ MISMATCH")
        kde_score = sig["kde_long"] if side == "long" else sig["kde_short"]
        kde_ok    = "✓" if kde_score >= p["kde_thr"] else "✗ MISMATCH"
        ci_ok     = "✓" if sig["ci"] > p["chop_thr"] else "✗ MISMATCH"
        sl_pct    = abs(cl - sl) / cl if cl else 0
        zone_ok   = "✓" if sl_pct <= p.get("max_zone_sl_pct", MAX_ZONE_SL_PCT_DEFAULT) else "✗ MISMATCH"
        tp_exists = "✓" if tp > 0 else "✗ MISSING"

        prompt = (
            f"TASK: Verify this RSI Kernel + MTF S/D paper trade entry matches BT logic exactly.\n\n"
            f"STRATEGY (BT rules):\n"
            f"- Long: RSI crosses UP through oversold (rsi_prev < rsi_os, rsi_cur >= rsi_os)\n"
            f"- Short: RSI crosses DOWN through overbought (rsi_prev > rsi_ob, rsi_cur <= rsi_ob)\n"
            f"- KDE: score = fraction of last 100 RSI bars where RSI was above(long)/below(short) current. score >= kde_thr to enter.\n"
            f"- CI > chop_thr (choppy market filter, counterintuitive: higher CI = choppier = ENTER)\n"
            f"- Price inside MTF S/D zone. SL at zone distal line <= max_zone_sl_pct from entry.\n"
            f"- TP = entry + 0.70 × (next confirmed swing high - entry) for long; mirrored for short.\n"
            f"- Isolated margin 11x. Liq price = entry × (1 - 1/11) for long.\n\n"
            f"SIGNAL: {symbol} {interval}m {side.upper()}\n"
            f"  rsi_prev={rsi_prev:.2f}  rsi_cur={sig['rsi']:.2f}  rsi_os={p['rsi_os']}  rsi_ob={p['rsi_ob']}\n"
            f"  RSI crossover: {cross_ok}\n"
            f"  kde_score={kde_score:.4f}  kde_thr={p['kde_thr']:.2f}  → {kde_ok}\n"
            f"  ci={sig['ci']:.2f}  chop_thr={p['chop_thr']:.1f}  → {ci_ok}\n"
            f"  in_zone=True  sl_pct={sl_pct:.4f}  max_sl_pct={p.get('max_zone_sl_pct', MAX_ZONE_SL_PCT_DEFAULT):.4f}  → {zone_ok}\n"
            f"  swing_tp_exists: {tp_exists}\n"
            f"  entry={entry_px:.5f}  sl={sl:.5f}  tp={tp:.5f}\n"
            f"  notional={notional:.2f} USDT  qty={qty:.4f}  lev=11x\n\n"
            f"RESPOND IN 3 BULLET POINTS:\n"
            f"• BT match: YES/NO — flag any discrepancy\n"
            f"• Risk flags: anything close to threshold or unusual?\n"
            f"• One-line note for the log"
        )
        self.enqueue(_AI_PRI_SIGNAL, {
            "type": "signal_check", "symbol": symbol, "interval": interval,
            "side": side, "prompt": prompt, "model": AI_MODEL_FAST,
        })

    def enqueue_trade_analysis(self, symbol: str, interval: str, pos: dict,
                               exit_px: float, reason: str,
                               mfe_pct: float, mae_pct: float, bars_held: int,
                               net_pnl: float, gross_pnl: float,
                               entry_snap: list):
        side      = pos["side"]
        entry_px  = pos["entry_px"]
        sl        = pos["stop_level"]
        tp        = pos["tp_level"]
        p         = pos.get("params_snapshot", {})
        outcome   = "WINNING" if net_pnl >= 0 else "LOSING"
        net_pct   = net_pnl / max(pos["isolated_mrgn"], 1e-9) * 100
        sl_pct    = abs(entry_px - sl) / entry_px

        reason_map = {
            "tp":     "price hit TP level intrabar (high >= tp for long)",
            "sl":     "price hit SL level intrabar (low <= sl for long)",
            "sl_gap": "bar OPENED below SL (gap — filled at open price)",
            "liq":    "bar opened at or below liquidation price (isolated margin wiped)",
        }
        reason_explain = reason_map.get(reason, reason)

        snap_closes = " ".join(f"{c['c']:.2f}" for c in entry_snap[-AI_SNAP_BARS:])
        snap_ohlcv  = "\n".join(
            f"  {datetime.fromtimestamp(c['t']//1000, tz=timezone.utc).strftime('%H:%M')} "
            f"O={c['o']:.4f} H={c['h']:.4f} L={c['l']:.4f} C={c['c']:.4f}"
            for c in entry_snap[-min(AI_SNAP_BARS, len(entry_snap)):]
        )

        prompt = (
            f"TASK: Analyse this paper trade result. Identify the primary driver of the outcome.\n\n"
            f"STRATEGY: RSI crossover + KDE percentile + Choppiness Index + MTF S/D zones. 11x isolated margin.\n"
            f"TP=70% swing distance. SL=zone distal. Entry/exit simulated at confirmed bar close/high/low (BT-exact).\n\n"
            f"TRADE: {symbol} {interval}m {side.upper()} — {outcome}\n"
            f"  Entry: {entry_px:.5f}  Exit: {exit_px:.5f}  Reason: {reason} ({reason_explain})\n"
            f"  SL: {sl:.5f}  TP: {tp:.5f}  SL-dist: {sl_pct:.3%}\n"
            f"  Net P&L: {net_pnl:+.4f} USDT ({net_pct:+.2f}% of margin)  Bars held: {bars_held}\n"
            f"  MFE: {mfe_pct:+.2f}%  MAE: {mae_pct:+.2f}%\n\n"
            f"ENTRY CONDITIONS (at signal bar close):\n"
            f"  RSI: {pos.get('entry_rsi', '?')}  KDE: {pos.get('entry_kde', '?'):.4f}  "
            f"CI: {pos.get('entry_ci', '?'):.2f}\n"
            f"  Params: rsi_os={p.get('rsi_os','?')} rsi_ob={p.get('rsi_ob','?')} "
            f"kde_thr={p.get('kde_thr','?')} chop_thr={p.get('chop_thr','?')}\n\n"
            f"LAST {min(AI_SNAP_BARS, len(entry_snap))} OHLCV BARS (oldest→newest):\n{snap_ohlcv}\n\n"
            f"RESPOND IN 4 BULLET POINTS:\n"
            f"• Why {outcome.lower()}: primary driver\n"
            f"• MAE/MFE ({mae_pct:+.2f}%/{mfe_pct:+.2f}%): what this ratio reveals about entry timing\n"
            f"• BT alignment: does this outcome fit BT expectations? Any live-vs-BT divergence?\n"
            f"• Watch next: one concrete thing to look for in similar future setups"
        )
        self.enqueue(_AI_PRI_TRADE, {
            "type": "trade_analysis", "symbol": symbol, "interval": interval,
            "side": side, "outcome": outcome,
            "prompt": prompt, "model": AI_MODEL_ANALYSIS,
        })

    def enqueue_session_summary(self, multi):
        try:
            rows = multi.db.execute(
                "SELECT symbol, side, entry, exit_price, net_pnl, mfe_pct, mae_pct, "
                "close_reason, bars_held, timestamp "
                "FROM trades WHERE status='closed' ORDER BY id DESC LIMIT 20"
            ).fetchall()
        except Exception:
            rows = []

        trade_lines = "\n".join(
            f"  {r[9][-8:]} {r[0].replace('USDT','')} {r[1].upper()} "
            f"entry={r[2]:.4f} exit={r[3]:.4f} pnl={r[4]:+.4f} "
            f"mfe={r[5]:.1f}% mae={r[6]:.1f}% reason={r[7]} bars={r[8]}"
            for r in rows
        ) or "  (no trades yet)"

        with multi._lock:
            peq   = multi._paper_equity
            seq   = multi._start_equity
            cum_p = multi._cum_profit_usdt
            cum_l = multi._cum_loss_usdt
            max_p = multi._max_profit_usdt
            max_l = multi._max_loss_usdt

        ses_pnl = peq - seq
        ses_pct = ses_pnl / max(seq, 1) * 100
        hours   = (time.time() - multi._start_ts) / 3600

        try:
            exit_counts = dict(multi.db.execute(
                "SELECT close_reason, COUNT(*) FROM trades WHERE status='closed' GROUP BY close_reason"
            ).fetchall())
        except Exception:
            exit_counts = {}

        exit_str = "  ".join(f"{k}:{v}" for k, v in exit_counts.items()) or "none"

        n_t = len(rows)
        n_w = sum(1 for r in rows if r[4] > 0)
        wr  = n_w / n_t * 100 if n_t else 0

        prompt = (
            f"TASK: Session performance review for RSI Kernel + MTF S/D paper trader.\n\n"
            f"SESSION: {', '.join(SYMBOLS)}  Duration: {hours:.1f}h\n"
            f"Paper equity: {seq:.2f} → {peq:.2f} USDT  P&L: {ses_pnl:+.4f} ({ses_pct:+.2f}%)\n"
            f"Trades: {n_t} total  Wins: {n_w}  WR: {wr:.1f}%\n"
            f"Cum profit: +{cum_p:.4f}  Max profit: +{max_p:.4f}\n"
            f"Cum loss: {cum_l:.4f}  Max loss: {max_l:.4f}\n"
            f"Exit reasons: {exit_str}\n\n"
            f"LAST {min(20, n_t)} TRADES:\n{trade_lines}\n\n"
            f"RESPOND IN 5 BULLET POINTS:\n"
            f"• Session quality: is paper performance in line with BT expectations?\n"
            f"• Pattern: any symbol/direction/time showing consistent behaviour?\n"
            f"• BT drift: any sign live signals differ from what BT would have taken?\n"
            f"• Best/worst: what drove the extremes this session?\n"
            f"• Action: one specific thing to monitor or adjust going forward"
        )
        self.enqueue(_AI_PRI_SUMMARY, {
            "type": "session_summary", "symbol": "ALL",
            "prompt": prompt, "model": AI_MODEL_ANALYSIS,
        })

    # -- Worker ----------------------------------------------------------------

    def _worker(self):
        while True:
            time.sleep(15)
            with self._lock:
                if not self._queue or not self._can_call():
                    continue
                _, _, event = self._queue.pop(0)

            self._process(event)

    def _process(self, event: dict):
        prompt  = event["prompt"]
        model   = event.get("model", AI_MODEL_ANALYSIS)
        etype   = event.get("type", "unknown")
        symbol  = event.get("symbol", "—")
        t0      = time.time()

        try:
            response = self._call_claude_sync(model, prompt, timeout=AI_CALL_TIMEOUT_S)
            if not response:
                response = "(empty response)"
            duration = time.time() - t0
            self._record_call(symbol, etype, model, prompt, response, duration)
            _log.info(f"AI [{etype}][{symbol}] {duration:.1f}s: {response[:80]}")
            self._multi.log(f"AI {etype.upper()} [{symbol}]: {response[:100]}", "INFO")

        except subprocess.TimeoutExpired:
            self._multi.log(f"AI [{etype}] timed out after {AI_CALL_TIMEOUT_S}s", "WARN")
        except FileNotFoundError:
            self._multi.log("AI: claude CLI not found — disabling agent", "ERROR")
            self._enabled = False
        except Exception as e:
            self._multi.log(f"AI [{etype}] error: {e}", "WARN")

    @property
    def last_insight(self) -> str:
        with self._lock:
            return self._last_insight

    @property
    def budget_str(self) -> str:
        with self._lock:
            return self._budget_str()


# =============================================================================
# PaperSymbolBot — per-symbol paper trading logic
# =============================================================================

class PaperSymbolBot:

    def __init__(self, symbol, selection, multi):
        self.symbol   = symbol
        self.interval = selection["interval"]
        self._multi   = multi

        self._params   = dict(selection["params"])
        self.lot_step  = 1.0
        self.tick      = 0.0001
        self._price_dp = 4
        self._qty_dp   = 0

        self._candles       = deque(maxlen=SEED_BARS)
        self._last_price    = 0.0
        self._last_kline_ts = 0.0
        self._lock          = threading.Lock()
        self._bar_count     = 0
        self._indicators    = {}
        self._pnl_history   = deque(maxlen=50)
        self._spark_cache   = "─" * 22

        self.paper_position: dict | None = None
        self._bars_held = 0
        self._mfe_pct   = 0.0
        self._mae_pct   = 0.0

        self._last_evaluated_bar_t = 0

    def fetch_instrument_info(self):
        try:
            r = _api(self._multi.session.get_instruments_info,
                     category=CATEGORY, symbol=self.symbol, _log_fn=self._multi.log)
            items = r["result"]["list"]
            if items:
                info = items[0]
                self.lot_step  = float(info["lotSizeFilter"]["qtyStep"])
                self.tick      = float(info["priceFilter"]["tickSize"])
                self._price_dp = _decimals(self.tick)
                self._qty_dp   = _decimals(self.lot_step)
        except Exception as e:
            self._multi.log(f"{self.symbol}: fetch_instrument_info failed: {e}", "WARN")

    def _round_tick(self, price: float) -> float:
        t = self.tick or 0.0001
        return round(round(price / t) * t, self._price_dp)

    def _seed_candles(self):
        try:
            r = _api(self._multi.session.get_kline,
                     category=CATEGORY, symbol=self.symbol,
                     interval=self.interval, limit=SEED_BARS,
                     _log_fn=self._multi.log)
            rows = r["result"]["list"]
            for row in reversed(rows):
                ts_ms = int(row[0])
                self._candles.append({
                    "t": ts_ms,
                    "o": float(row[1]),
                    "h": float(row[2]),
                    "l": float(row[3]),
                    "c": float(row[4]),
                })
            self._bar_count = len(self._candles)
            self._multi.log(f"{self.symbol}: seeded {len(self._candles)} candles ({self.interval}m)", "INFO")
        except Exception as e:
            self._multi.log(f"{self.symbol}: seed_candles failed: {e}", "WARN")

    # -- Paper entry & exit ----------------------------------------------------

    def _paper_enter(self, sig: dict, close_px: float):
        side = "long" if sig["buy_sig"] else "short"
        sh_prices = sig["sh_prices"]
        sl_prices = sig["sl_prices"]

        if side == "long":
            sl_level = sig.get("demand_bot")
            if sl_level is None or not np.isfinite(sl_level) or sl_level >= close_px:
                return
            i_above = bisect.bisect_right(sh_prices, close_px)
            if i_above >= len(sh_prices):
                return
            tp_level  = close_px + TP_FACTOR * (sh_prices[i_above] - close_px)
            liq_price = close_px * (1.0 - 1.0 / LEVERAGE)
        else:
            sl_level = sig.get("supply_top")
            if sl_level is None or not np.isfinite(sl_level) or sl_level <= close_px:
                return
            i_below = bisect.bisect_left(sl_prices, close_px) - 1
            if i_below < 0:
                return
            tp_level  = close_px - TP_FACTOR * (close_px - sl_prices[i_below])
            liq_price = close_px * (1.0 + 1.0 / LEVERAGE)

        alloc    = self._multi.get_paper_equity() / max(N_SYMBOLS, 1)
        notional = alloc * LEVERAGE * MARGIN_HEADROOM
        qty      = notional / close_px
        entry_fee = notional * TAKER_FEE

        self._multi.adjust_paper_equity(-entry_fee)

        entry_snap = list(self._candles)[-AI_SNAP_BARS:]

        self.paper_position = {
            "side":           side,
            "entry_px":       close_px,
            "stop_level":     sl_level,
            "tp_level":       tp_level,
            "liq_price":      liq_price,
            "qty":            qty,
            "notional":       notional,
            "isolated_mrgn":  notional / LEVERAGE,
            "entry_fee":      entry_fee,
            "entry_ts":       datetime.now(timezone.utc),
            "entry_snap":     entry_snap,
            "entry_rsi":      sig["rsi"],
            "entry_ci":       sig["ci"],
            "entry_kde":      sig["kde_long"] if side == "long" else sig["kde_short"],
            "params_snapshot": sig.get("params_snapshot", {}),
        }
        self._bars_held = 0
        self._mfe_pct   = 0.0
        self._mae_pct   = 0.0

        self._multi.log(
            f"PAPER {self.symbol} {side.upper()} @ {close_px:.{self._price_dp}f}  "
            f"SL={sl_level:.{self._price_dp}f}  TP={tp_level:.{self._price_dp}f}  "
            f"RSI={sig['rsi']:.1f}  KDE={self.paper_position['entry_kde']:.3f}  "
            f"CI={sig['ci']:.1f}", "INFO")

    def _paper_exit(self, exit_px: float, reason: str):
        pos = self.paper_position
        if pos is None:
            return

        side = pos["side"]
        ef   = exit_px * pos["qty"] * TAKER_FEE
        if side == "long":
            gross_pnl = (exit_px - pos["entry_px"]) * pos["qty"]
        else:
            gross_pnl = (pos["entry_px"] - exit_px) * pos["qty"]
        net_pnl = gross_pnl - ef - pos["entry_fee"]

        self._multi.adjust_paper_equity(gross_pnl - ef)

        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            self._multi.db.execute(
                """INSERT INTO trades
                   (timestamp, symbol, interval, side, entry, exit_price, qty,
                    net_pnl, gross_pnl, entry_fee, exit_fee,
                    mfe_pct, mae_pct, close_reason, bars_held, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'closed')""",
                (ts_str, self.symbol, self.interval, side,
                 pos["entry_px"], exit_px, pos["qty"],
                 round(net_pnl, 6), round(gross_pnl - ef, 6),
                 round(pos["entry_fee"], 6), round(ef, 6),
                 round(self._mfe_pct * 100, 4), round(self._mae_pct * 100, 4),
                 reason, self._bars_held))
            self._multi.db.commit()
        except Exception as e:
            self._multi.log(f"{self.symbol}: DB write failed: {e}", "WARN")

        sign = "+" if net_pnl >= 0 else ""
        self._multi.log(
            f"PAPER {self.symbol} {side.upper()} CLOSE [{reason}] "
            f"@ {exit_px:.{self._price_dp}f}  net={sign}{net_pnl:.4f}  "
            f"MFE={self._mfe_pct*100:.2f}%  MAE={self._mae_pct*100:.2f}%  bars={self._bars_held}", "INFO")

        self._pnl_history.append(net_pnl)
        self._spark_cache = _spark(self._pnl_history)

        pnl_pct = net_pnl / max(self._multi.get_paper_equity(), 1.0) * 100
        with self._multi._lock:
            self._multi._recent_trades.appendleft((
                ts_str[-8:], self.symbol, f"{self.interval}m",
                side.upper(), round(pos["entry_px"], self._price_dp),
                round(exit_px, self._price_dp),
                round(pnl_pct, 3),
                round(self._mfe_pct * 100, 2),
                round(self._mae_pct * 100, 2),
                reason, self._bars_held,
            ))
            self._multi._refresh_pnl_stats()

        # Enqueue AI trade analysis (non-blocking)
        self._multi.ai.enqueue_trade_analysis(
            self.symbol, self.interval, pos,
            exit_px, reason,
            self._mfe_pct * 100, self._mae_pct * 100,
            self._bars_held, net_pnl, gross_pnl - ef,
            pos.get("entry_snap", []))

        self.paper_position = None
        self._bars_held     = 0
        self._mfe_pct       = 0.0
        self._mae_pct       = 0.0

    # -- Bar evaluation --------------------------------------------------------

    def _evaluate_closed_bar(self, candle: dict):
        with self._lock:
            bar_t = candle["t"]
            if bar_t <= self._last_evaluated_bar_t:
                return
            self._last_evaluated_bar_t = bar_t

            self._candles.append(candle)
            self._bar_count += 1
            self._last_price    = candle["c"]
            self._last_kline_ts = time.time()

            o = float(candle["o"])
            h = float(candle["h"])
            l = float(candle["l"])
            c = float(candle["c"])

            pos = self.paper_position

            if pos is not None:
                self._bars_held += 1

                if pos["side"] == "long":
                    mfe = (h - pos["entry_px"]) / pos["entry_px"]
                    mae = (l - pos["entry_px"]) / pos["entry_px"]
                else:
                    mfe = (pos["entry_px"] - l) / pos["entry_px"]
                    mae = (pos["entry_px"] - h) / pos["entry_px"]
                self._mfe_pct = max(self._mfe_pct, mfe)
                self._mae_pct = min(self._mae_pct, mae)

                exit_px = None
                reason  = None

                if pos["side"] == "long":
                    if o < pos["stop_level"]:
                        fill    = max(o, pos["liq_price"])
                        exit_px = fill
                        reason  = "liq" if fill <= pos["liq_price"] else "sl_gap"
                    else:
                        sl_hit = l <= pos["stop_level"]
                        tp_hit = h >= pos["tp_level"]
                        if sl_hit or tp_hit:
                            exit_px = pos["stop_level"] if sl_hit else pos["tp_level"]
                            reason  = "sl" if sl_hit else "tp"
                else:
                    if o > pos["stop_level"]:
                        fill    = min(o, pos["liq_price"])
                        exit_px = fill
                        reason  = "liq" if fill >= pos["liq_price"] else "sl_gap"
                    else:
                        sl_hit = h >= pos["stop_level"]
                        tp_hit = l <= pos["tp_level"]
                        if sl_hit or tp_hit:
                            exit_px = pos["stop_level"] if sl_hit else pos["tp_level"]
                            reason  = "sl" if sl_hit else "tp"

                if exit_px is not None:
                    self._paper_exit(exit_px, reason)
                    return

                # AI position gate: every N bars check if we should exit early
                if (self._bars_held > 0
                        and self._bars_held % AI_POSITION_CHECK_BARS == 0):
                    verdict = self._multi.ai.gate_position(
                        self.symbol, self.interval, pos, h, l, c,
                        self._bars_held,
                        self._mfe_pct * 100, self._mae_pct * 100)
                    if verdict["exit_now"]:
                        self._multi.log(
                            f"AI EXIT {self.symbol} {pos['side'].upper()} "
                            f"@ {c:.{self._price_dp}f}: {verdict['reason'][:80]}", "WARN")
                        self._paper_exit(c, "ai_exit")
                        return

            if len(self._candles) < max(int(self._params.get("rsi_p", 14)), KDE_LB) + 10:
                return

            snap = list(self._candles)
            sig  = compute_signal(snap, self._params)
            if sig is None:
                return

            self._indicators = sig

            if self.paper_position is None:
                if sig.get("buy_sig") or sig.get("sell_sig"):
                    # Pre-compute key levels for the AI gate prompt
                    _side = "long" if sig.get("buy_sig") else "short"
                    _sh   = sig["sh_prices"]
                    _sl   = sig["sl_prices"]
                    if _side == "long":
                        _sl_lvl = sig.get("demand_bot") or 0.0
                        _ia     = bisect.bisect_right(_sh, c)
                        _tp_lvl = c + TP_FACTOR * (_sh[_ia] - c) if _ia < len(_sh) else 0.0
                    else:
                        _sl_lvl = sig.get("supply_top") or 0.0
                        _ib     = bisect.bisect_left(_sl, c) - 1
                        _tp_lvl = c - TP_FACTOR * (c - _sl[_ib]) if _ib >= 0 else 0.0
                    _alloc    = self._multi.get_paper_equity() / max(N_SYMBOLS, 1)
                    _notional = _alloc * LEVERAGE * MARGIN_HEADROOM
                    _qty      = _notional / c if c else 0.0

                    verdict = self._multi.ai.gate_entry(
                        self.symbol, self.interval, _side, sig, self._params,
                        c, _sl_lvl, _tp_lvl, _notional, _qty)
                    if verdict["proceed"]:
                        self._paper_enter(sig, c)
                    else:
                        self._multi.log(
                            f"AI BLOCKED {self.symbol} {_side.upper()} "
                            f"@ {c:.{self._price_dp}f}: {verdict['reason'][:80]}", "WARN")

    def handle_kline(self, msg):
        try:
            data = msg.get("data", [])
            for bar in data:
                self._last_kline_ts = time.time()
                try:
                    self._last_price = float(bar["close"])
                except Exception:
                    pass
                if not bar.get("confirm", False):
                    continue
                candle = {
                    "t": int(bar["start"]),
                    "o": float(bar["open"]),
                    "h": float(bar["high"]),
                    "l": float(bar["low"]),
                    "c": float(bar["close"]),
                }
                self._evaluate_closed_bar(candle)
        except Exception as e:
            self._multi.log(f"{self.symbol}: handle_kline error: {e}", "WARN")

    @property
    def last_price(self) -> float:
        return self._last_price

    @property
    def ws_age(self) -> float:
        return time.time() - self._last_kline_ts if self._last_kline_ts > 0 else 999.0


# =============================================================================
# PaperMultiBot — orchestrator
# =============================================================================

class PaperMultiBot:

    def __init__(self):
        self.running        = True
        self.session        = None
        self.db             = None
        self.ai: AIAgent    = None   # type: ignore  — set in _run
        self._lock          = threading.Lock()
        self._log_buf       = deque(maxlen=120)
        self._live_wallet   = 0.0
        self._paper_equity  = 0.0
        self._start_equity  = 0.0
        self._peak_equity   = 0.0
        self._start_ts      = 0.0
        self._bots: dict    = {}
        self._ws_public     = None
        self._ws_start_ts   = 0.0
        self._recent_trades = deque(maxlen=12)
        self._cum_profit_usdt = 0.0
        self._max_profit_usdt = 0.0
        self._cum_loss_usdt   = 0.0
        self._max_loss_usdt   = 0.0
        self._next_summary_ts = 0.0

    def log(self, msg: str, level: str = "INFO"):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = (ts, level, msg)
        with self._lock:
            self._log_buf.appendleft(entry)
        getattr(_log, level.lower(), _log.info)(msg)
        try:
            if self.db:
                self.db.execute(
                    "INSERT INTO bot_log (timestamp, level, message) VALUES (?,?,?)",
                    (ts, level, msg))
                self.db.commit()
        except Exception:
            pass

    def get_log(self, n: int = 10):
        with self._lock:
            return list(self._log_buf)[:n]

    def get_paper_equity(self) -> float:
        with self._lock:
            return self._paper_equity

    def adjust_paper_equity(self, delta: float):
        with self._lock:
            self._paper_equity += delta
            if self._paper_equity > self._peak_equity:
                self._peak_equity = self._paper_equity

    def _refresh_pnl_stats(self):
        try:
            row = self.db.execute(
                "SELECT SUM(net_pnl), MIN(net_pnl) FROM trades WHERE net_pnl < 0 AND status='closed'"
            ).fetchone()
            if row and row[0] is not None:
                self._cum_loss_usdt = float(row[0])
                self._max_loss_usdt = float(row[1])
            row = self.db.execute(
                "SELECT SUM(net_pnl), MAX(net_pnl) FROM trades WHERE net_pnl > 0 AND status='closed'"
            ).fetchone()
            if row and row[0] is not None:
                self._cum_profit_usdt = float(row[0])
                self._max_profit_usdt = float(row[1])
        except Exception as e:
            self.log(f"pnl stats refresh: {e}", "WARN")

    def _seed_balance(self):
        try:
            r = self.session.get_wallet_balance(accountType="UNIFIED")
            coins = r["result"]["list"][0]["coin"]
            usdt  = next((c for c in coins if c["coin"] == "USDT"), None)
            bal   = float(usdt["equity"]) if usdt else 0.0
            if bal < 1.0:
                bal = BALANCE_FALLBACK
                self.log(f"Balance too small, using fallback {BALANCE_FALLBACK:.2f}", "WARN")
            self._live_wallet  = bal
            self._paper_equity = bal
            self._start_equity = bal
            self._peak_equity  = bal
            self.log(f"Paper equity seeded from live equity: {bal:.2f} USDT", "INFO")
        except Exception as e:
            self.log(f"seed_balance failed: {e} — using {BALANCE_FALLBACK:.2f}", "WARN")
            self._live_wallet  = BALANCE_FALLBACK
            self._paper_equity = BALANCE_FALLBACK
            self._start_equity = BALANCE_FALLBACK
            self._peak_equity  = BALANCE_FALLBACK

    def _balance_refresh_loop(self):
        while self.running:
            time.sleep(BALANCE_REFRESH_S)
            if not self.running:
                break
            try:
                r = self.session.get_wallet_balance(accountType="UNIFIED")
                coins = r["result"]["list"][0]["coin"]
                usdt  = next((c for c in coins if c["coin"] == "USDT"), None)
                bal   = float(usdt["equity"]) if usdt else 0.0
                if bal >= 1.0:
                    with self._lock:
                        self._live_wallet = bal
            except Exception as e:
                self.log(f"balance refresh: {e}", "WARN")

    def _start_ws(self):
        try:
            if self._ws_public:
                try:
                    self._ws_public.exit()
                except Exception:
                    pass
        except Exception:
            pass

        from collections import defaultdict
        by_iv: dict = defaultdict(list)
        for sym, bot in self._bots.items():
            by_iv[bot.interval].append(sym)

        self._ws_public   = WebSocket(testnet=False, channel_type="linear")
        self._ws_start_ts = time.time()

        for iv, syms in by_iv.items():
            for sym in syms:
                def _cb(msg, _sym=sym):
                    if "kline" in msg.get("topic", "") and _sym in self._bots:
                        self._bots[_sym].handle_kline(msg)
                self._ws_public.kline_stream(interval=int(iv), symbol=sym, callback=_cb)

        self.log(f"WS started — {sum(len(v) for v in by_iv.values())} streams", "INFO")

    def _check_ws_health(self):
        stale = [sym for sym, bot in self._bots.items() if bot.ws_age > WS_STALE_S]
        if stale:
            self.log(f"WS stale for {stale} — restarting", "WARN")
            time.sleep(2)
            self._start_ws()

    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            self.log("KeyboardInterrupt — stopping", "INFO")
        except Exception as e:
            import traceback
            msg = traceback.format_exc()
            self.log(f"Fatal: {e}\n{msg}", "ERROR")
            _write_crash(msg)
        finally:
            self.running = False

    def _run(self):
        self._start_ts = time.time()
        self.session   = make_session()
        self.db        = get_db()
        self.ai        = AIAgent(self)

        self._seed_balance()
        self._refresh_pnl_stats()

        for sym in SYMBOLS:
            bot = PaperSymbolBot(sym, SELECTIONS[sym], self)
            bot.fetch_instrument_info()
            bot._seed_candles()
            self._bots[sym] = bot

        self._start_ws()
        self._next_summary_ts = time.time() + AI_SESSION_SUMMARY_H * 3600

        self.log(f"Paper trader + AI agent started — {N_SYMBOLS} symbols  "
                 f"equity={self._paper_equity:.2f} USDT  "
                 f"AI={'ON' if self.ai._enabled else 'OFF'}", "INFO")

        threading.Thread(target=self._balance_refresh_loop,
                         daemon=True, name="balance-refresh").start()

        last_health = time.time()
        while self.running:
            time.sleep(10)
            now = time.time()
            if now - last_health > 60:
                self._check_ws_health()
                last_health = now
            if now >= self._next_summary_ts:
                self.ai.enqueue_session_summary(self)
                self._next_summary_ts = now + AI_SESSION_SUMMARY_H * 3600

    def stop(self):
        self.running = False
        try:
            if self._ws_public:
                self._ws_public.exit()
        except Exception:
            pass
        self.log("Paper trader stopped", "INFO")


# =============================================================================
# TUI
# =============================================================================

def build_ui(multi: PaperMultiBot):
    now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    uptime_s = int(time.time() - multi._start_ts) if multi._start_ts else 0
    h, m, s  = uptime_s // 3600, (uptime_s % 3600) // 60, uptime_s % 60

    with multi._lock:
        paper_eq = multi._paper_equity
        start_eq = multi._start_equity
        peak_eq  = multi._peak_equity
        live_wall= multi._live_wallet
        cum_p    = multi._cum_profit_usdt
        max_p    = multi._max_profit_usdt
        cum_l    = multi._cum_loss_usdt
        max_l    = multi._max_loss_usdt

    ses_pnl = paper_eq - start_eq
    ses_pct = ses_pnl / max(start_eq, 1.0) * 100
    dd_pct  = (paper_eq - peak_eq) / max(peak_eq, 1.0) * 100
    alloc   = paper_eq / max(N_SYMBOLS, 1)

    ai_enabled = multi.ai is not None and multi.ai._enabled
    ai_budget  = multi.ai.budget_str if multi.ai else "AI disabled"

    # -- Header ----------------------------------------------------------------
    hdr = Table.grid(padding=(0, 2)); hdr.add_column(); hdr.add_column()
    hdr.add_row(
        Text(f"RSI REVERSAL ⚡ PAPER TRADER  "
             f"{' / '.join(s.replace('USDT', '') for s in SYMBOLS)}",
             style="bold yellow"),
        Text(now_str, style="dim"))
    hdr.add_row(Text(
        f"Lev: {LEVERAGE}x  Fee: {TAKER_FEE*100:.3f}%  "
        f"Paper: {paper_eq:.2f}  Live wallet: {live_wall:.2f}  "
        f"Alloc/sym: {alloc:.2f}  Up: {h:02d}:{m:02d}:{s:02d}  "
        f"AI: {'ON' if ai_enabled else 'OFF'}  {ai_budget}",
        style="cyan" if ai_enabled else "dim"))

    panels = []

    # -- Per-symbol panels -----------------------------------------------------
    for sym, bot in multi._bots.items():
        pos   = bot.paper_position
        ind   = bot._indicators
        rsi   = ind.get("rsi", float("nan"))
        ci    = ind.get("ci", float("nan"))
        kde_l = ind.get("kde_long", float("nan"))
        kde_s = ind.get("kde_short", float("nan"))
        price = bot.last_price

        g = Table.grid(padding=(0, 1)); g.add_column(min_width=22); g.add_column()

        if pos is None:
            g.add_row(Text("FLAT", style="dim"), Text(f"{price:.{bot._price_dp}f}", style="white"))
            g.add_row(Text("TP / SL", style="dim"), Text("— / —", style="dim"))
        else:
            side  = pos["side"]
            epx   = pos["entry_px"]
            tp    = pos["tp_level"]
            sl    = pos["stop_level"]
            qty   = pos["qty"]
            upnl  = (price - epx) * qty if side == "long" else (epx - price) * qty
            upnl_pct = upnl / max(pos["isolated_mrgn"], 1e-9) * 100
            col   = "green" if upnl >= 0 else "red"
            dir_c = "green" if side == "long" else "red"
            g.add_row(Text(f"{side.upper()} @ {epx:.{bot._price_dp}f}", style=dir_c),
                      Text(f"UPnL {upnl:+.4f} ({upnl_pct:+.1f}%)", style=col))
            g.add_row(Text("TP", style="dim"),
                      Text(f"{tp:.{bot._price_dp}f}  +{abs(tp-price)/price*100:.2f}%", style="green"))
            g.add_row(Text("SL", style="dim"),
                      Text(f"{sl:.{bot._price_dp}f}  -{abs(sl-price)/price*100:.2f}%", style="red"))
            g.add_row(Text("Bars/MFE/MAE", style="dim"),
                      Text(f"{bot._bars_held} / {bot._mfe_pct*100:.2f}% / {bot._mae_pct*100:.2f}%",
                           style="dim"))
            e_rsi = pos.get("entry_rsi", "?")
            e_kde = pos.get("entry_kde", float("nan"))
            g.add_row(Text("Entry RSI/KDE", style="dim"),
                      Text(f"{e_rsi:.1f} / {e_kde:.3f}" if isinstance(e_rsi, float)
                           else f"{e_rsi} / {e_kde:.3f}", style="dim"))

        ci_col = "cyan" if np.isfinite(ci) and ci > 50 else "dim"
        kde_v  = kde_l if (pos and pos["side"] == "long") else kde_s
        g.add_row(Text(f"RSI {rsi:.1f}  KDE {kde_v*100:.0f}%", style="dim"),
                  Text(f"CI {ci:.1f}", style=ci_col))
        g.add_row(Text(bot._spark_cache, style="dim"), Text(f"{bot.interval}m", style="dim"))
        ws_col = "green" if bot.ws_age < WS_STALE_S else "red"
        g.add_row(Text(f"WS {bot.ws_age:.0f}s", style=ws_col), Text(""))

        panels.append(Panel(g, title=f"[bold]{sym.replace('USDT','')}", border_style="blue",
                             padding=(0, 1)))

    sym_cols = Columns(panels, equal=True, expand=True)

    # -- Wallet / stats panel --------------------------------------------------
    wt = Table.grid(padding=(0, 2)); wt.add_column(min_width=22); wt.add_column()
    wt.add_row(Text("Live wallet",  style="dim"), Text(f"{live_wall:.2f} USDT", style="cyan"))
    wt.add_row(Text("Paper equity", style="dim"), Text(f"{paper_eq:.2f} USDT",
                                                        style=_sign(paper_eq - start_eq)))
    wt.add_row(Text("Peak equity",  style="dim"), Text(f"{peak_eq:.2f} USDT", style="green"))
    wt.add_row(Text("Drawdown",     style="dim"), Text(f"{dd_pct:.2f}%",
                                                        style="red" if dd_pct < -1 else "dim"))
    wt.add_row(Text("Session P&L",  style="dim"), Text(f"{ses_pnl:+.4f} ({ses_pct:+.2f}%)",
                                                        style=_sign(ses_pnl)))
    wt.add_row(Text(""), Text(""))

    n_t = n_w = 0; wr = 0.0
    try:
        if multi.db:
            row = multi.db.execute(
                "SELECT COUNT(*), SUM(CASE WHEN net_pnl>0 THEN 1 ELSE 0 END) "
                "FROM trades WHERE status='closed'"
            ).fetchone()
            n_t = int(row[0] or 0); n_w = int(row[1] or 0)
            wr  = n_w / n_t * 100 if n_t else 0.0
    except Exception:
        pass

    wt.add_row(Text("Trades / WR",  style="dim"), Text(f"{n_t} / {wr:.1f}%"))
    wt.add_row(Text("Cum profit",   style="dim"), Text(f"+{cum_p:.4f} USDT", style="green"))
    wt.add_row(Text("Max profit",   style="dim"), Text(f"+{max_p:.4f} USDT", style="green"))
    wt.add_row(Text("Cum loss",     style="dim"), Text(f"{cum_l:.4f} USDT",  style="red"))
    wt.add_row(Text("Max loss",     style="dim"), Text(f"{max_l:.4f} USDT",  style="red"))
    wt.add_row(Text(""), Text(""))
    wt.add_row(Text("Alloc/sym",    style="dim"), Text(f"{alloc:.2f} USDT"))
    wt.add_row(Text("Leverage",     style="dim"), Text(f"{LEVERAGE}x"))

    wallet_panel = Panel(wt, title="[bold]Wallet", border_style="yellow", padding=(0, 1))

    # -- Recent trades ---------------------------------------------------------
    rt = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold",
               expand=True, padding=(0, 1))
    rt.add_column("Time",   style="dim",    min_width=8)
    rt.add_column("Symbol", min_width=6)
    rt.add_column("TF",     style="dim",    min_width=4)
    rt.add_column("Side",   min_width=5)
    rt.add_column("Entry",  justify="right", min_width=8)
    rt.add_column("Exit",   justify="right", min_width=8)
    rt.add_column("P&L%",   justify="right", min_width=6)
    rt.add_column("MFE%",   justify="right", min_width=5)
    rt.add_column("MAE%",   justify="right", min_width=5)
    rt.add_column("Reason", min_width=6)
    rt.add_column("Bars",   justify="right", min_width=4)

    with multi._lock:
        trades_snap = list(multi._recent_trades)

    for tr in trades_snap:
        t, sym, tf, side, entry, exit_, pnl_pct, mfe, mae, reason, bars = tr
        c = "green" if pnl_pct >= 0 else "red"
        rt.add_row(t, sym.replace("USDT", ""), tf,
                   Text(side, style="green" if side == "LONG" else "red"),
                   str(entry), str(exit_),
                   Text(f"{pnl_pct:+.3f}%", style=c),
                   f"{mfe:.2f}%", f"{mae:.2f}%", reason, str(bars))

    trades_panel = Panel(rt, title="[bold]Recent Paper Trades",
                         border_style="magenta", padding=(0, 1))

    # -- Log + AI insight panel ------------------------------------------------
    lt = Table.grid(padding=(0, 1)); lt.add_column(style="dim", min_width=12); lt.add_column()

    # Latest AI insight at top if available
    if multi.ai and multi.ai.last_insight:
        insight = multi.ai.last_insight
        lt.add_row(Text("AI INSIGHT", style="bold yellow"),
                   Text(insight[:120], style="yellow"))
        lt.add_row(Text(""), Text(""))

    for ts, level, msg in multi.get_log(8):
        col = "red" if level == "ERROR" else ("yellow" if level == "WARN" else "dim")
        lt.add_row(f"{ts} [{level}]", Text(msg[:110], style=col))

    log_panel = Panel(lt, title="[bold]Log + AI", border_style="dim", padding=(0, 1))

    root = Table.grid(expand=True)
    root.add_row(Panel(hdr, border_style="cyan", padding=(0, 1)))
    root.add_row(sym_cols)
    root.add_row(Columns([wallet_panel, trades_panel], expand=True))
    root.add_row(log_panel)
    return root


# =============================================================================
# Entry point
# =============================================================================

def _setup_exit_on_close():
    def _handler(sig, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)


def _excepthook(exc_type, exc_val, exc_tb):
    import traceback as _tb
    msg = "".join(_tb.format_exception(exc_type, exc_val, exc_tb))
    _log.critical(f"Unhandled exception:\n{msg}")
    _write_crash(msg)


def main():
    if not SYMBOLS:
        print("No qualifying BT results found. Run the backtester first.", flush=True)
        sys.exit(1)

    claude_ok = bool(shutil.which("claude"))
    print(f"RSI KERNEL SD — PAPER TRADER + AI AGENT")
    print(f"Symbols: {', '.join(SYMBOLS)}  |  AI: {'claude CLI found' if claude_ok else 'claude CLI NOT FOUND — AI disabled'}")
    print("No real orders placed. Live klines + read-only balance.")

    sys.excepthook = _excepthook
    _setup_exit_on_close()

    multi = PaperMultiBot()
    t = threading.Thread(target=multi.run, daemon=True, name="paper-main")
    t.start()

    time.sleep(4)

    console = _make_console()
    try:
        with Live(build_ui(multi), console=console, refresh_per_second=2,
                  screen=False) as live:
            while multi.running:
                try:
                    live.update(build_ui(multi))
                    time.sleep(0.5)
                except Exception:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        multi.stop()
        t.join(timeout=5)
        print("\nPaper trader stopped.")


if __name__ == "__main__":
    main()
