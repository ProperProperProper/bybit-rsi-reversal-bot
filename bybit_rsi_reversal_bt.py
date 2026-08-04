#!/usr/bin/env python3
"""
RSI Kernel + MTF Supply/Demand Backtester  —  multi-symbol, multi-interval
Signal:  RSI crossover from OB/OS  +  Nadaraya-Watson RQ kernel regression on RSI
          +  MTF Supply/Demand zone (Flux Charts style, re-test required)  +  Choppiness Index filter
TP:      70% of distance to next confirmed swing HIGH (long) / swing LOW (short).
SL:      Demand zone bottom (long) / Supply zone top (short) — placed at zone distal line exactly.
Exit:    Full-position TP/SL; CI regime flip closes and enters opposite direction.
Continuous 30-min loop, DB winners first, 5-retry loser removal.
"""
import os, sys, io, time, json, math, logging, random, multiprocessing, sqlite3, bisect
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from rich.live    import Live
from rich.table   import Table
from rich.text    import Text
from rich.panel   import Panel
from rich.console import Console
from rich         import box

# ── Paths ────────────────────────────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    _DIR = os.path.dirname(sys.executable)
else:
    _DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_DIR, "data")
_exe_name   = os.path.basename(sys.executable if getattr(sys, "frozen", False) else sys.argv[0])
DEMO        = "--live" not in sys.argv and "_live" not in _exe_name
KEYS_PATH   = os.path.join(DATA_DIR, ".keys" if DEMO else ".keys_live")
CONFIG_PATH = os.path.join(DATA_DIR, "bybit_rsi_reversal_config.json")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────────
_DEFAULT_CONFIG = {
    "symbols":        ["BTCUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
    "intervals":      ["3", "15", "30"],
    "n_random":       5000,
    "min_trades":     3,
    "min_avg_hold":   2.0,
    "initial_equity": 1000.0,
    "leverage":       11,
    "sd_zone_mult":   0.5,
    "htf_factor":     4,
}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH) as _f:
            _cfg = {**_DEFAULT_CONFIG, **json.load(_f)}
    except Exception:
        _cfg = _DEFAULT_CONFIG.copy()
else:
    _cfg = _DEFAULT_CONFIG.copy()
    with open(CONFIG_PATH, "w") as _f:
        json.dump(_DEFAULT_CONFIG, _f, indent=2)

SYMBOLS   = [s.upper() for s in _cfg["symbols"]]
INTERVALS = [str(i) for i in _cfg.get("intervals", ["3", "15", "30"])]

# ── Fixed constants ───────────────────────────────────────────────────────────────
FETCH_LIMIT     = 1000
BACKTEST_HOURS  = 168     # IS window: 7 days to ensure enough S/D zone entries
WF_OOS_DAYS     = 1       # 1 day OOS for proper walk-forward (IS = BACKTEST_HOURS - 24h)
TAKER_FEE       = 0.00055
MARGIN_HEADROOM = 0.98
LEVERAGE        = int(_cfg.get("leverage", 11))
INITIAL_EQUITY  = float(_cfg["initial_equity"])
CATEGORY        = "linear"
SD_ZONE_MULT    = float(_cfg.get("sd_zone_mult", 0.5))  # body/range impulse threshold (0.5 = Flux Charts)
HTF_FACTOR      = int(_cfg.get("htf_factor", 4))
KDE_LB          = 100   # fixed lookback for KDE scoring
TP_FACTOR       = 0.70  # fraction of swing distance for take-profit (Flux Charts default)
TP_CONS_SCALE   = 0.70  # conservative validation: retest at 70% of optimised TP
MAX_ZONE_SL_PCT_DEFAULT = 0.020  # default when not in param set (backwards compat)

N_RANDOM      = int(_cfg["n_random"])
N_WORKERS     = max(1, int(multiprocessing.cpu_count() * 0.8))
BATCH_SIZE    = 50
MIN_TRADES    = int(_cfg.get("min_trades", 10))
MIN_AVG_HOLD  = float(_cfg.get("min_avg_hold", 2.0))
MAX_DD_PCT         = 0.08
MIN_LIVE_RET           = 10.0   # IS total-return % required to publish params to live trader
MIN_LIVE_SHARPE        = 15.0   # Sharpe floor: rejects near-zero equity curves regardless of trade count
MIN_LIVE_RETURN_PT     = 2.0    # minimum avg return per trade (% of equity) — rejects low-EV param sets
MIN_LIVE_CHOP_THR      = 45.0   # reject params whose optimised chop_thr is below this — ensures the CI
                                 # filter actually requires a ranging market (CI > 45) before entering
N_TOP_RETEST  = 50
MAX_RETRIES   = 5
LOOP_INTERVAL = 30 * 60


# ── Interval helpers ──────────────────────────────────────────────────────────────
def _bars_per_day(interval: str) -> int:
    return 24 * 60 // int(interval)

def _backtest_bars(interval: str) -> int:
    return BACKTEST_HOURS * (60 // int(interval))

def _oos_bars(interval: str) -> int:
    return WF_OOS_DAYS * _bars_per_day(interval)

def _bars_per_year(interval: str) -> float:
    return 365.25 * _bars_per_day(interval)

def _max_pages(interval: str) -> int:
    needed = _backtest_bars(interval) + 400
    return math.ceil(needed / FETCH_LIMIT) + 3   # +3 extra pages for richer S/D zone history


# ── Parameter search space ────────────────────────────────────────────────────────
# RSI crossover + Nadaraya-Watson RQ kernel regression on RSI (Flux Charts RSI Kernel Optimized)
# SL: zone distal line (demand bottom / supply top) — no ATR buffer
# TP: 70% of distance to next confirmed swing HIGH (long) / LOW (short)
PARAM_SPACE = {
    "rsi_p":            (5,    21),    # RSI period
    "rsi_os":           (20,   40),    # oversold threshold (long entry crossover)
    "rsi_ob":           (60,   80),    # overbought threshold (short entry crossover)
    "kde_thr":          (0.30, 0.70),  # KDE score threshold (0.50 = top-50% filter from video)
    "atr_p":            (8,    20),    # ATR period (live trader fallback SL; unused in BT scoring)
    "chop_len":         (8,    20),    # Choppiness Index period
    "chop_thr":         (38.0, 62.0),  # Choppiness threshold (<thr = trending)
    "swing_len":        (2,    8),     # bars each side for swing pivot confirmation
    "max_zone_sl_pct":  (0.015, 0.030),  # max allowed zone SL as fraction of entry (1.5%–3.0%)
}
_INT_PARAMS = {"rsi_p", "rsi_os", "rsi_ob", "atr_p", "chop_len", "swing_len"}

DEFAULT_PARAMS = {
    "rsi_p":           14,
    "rsi_os":          30,
    "rsi_ob":          70,
    "kde_thr":         0.50,
    "atr_p":           14,
    "chop_len":        14,
    "chop_thr":        50.0,
    "swing_len":       4,
    "max_zone_sl_pct": 0.020,
}

# ── Logging ───────────────────────────────────────────────────────────────────────
_log = logging.getLogger("bybit_rsi_reversal_bt")
_log.setLevel(logging.DEBUG)
_log.propagate = False
_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
_fh = RotatingFileHandler(
    os.path.join(DATA_DIR, "bybit_rsi_reversal_bt.log"),
    maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_fh.setFormatter(_log_fmt); _fh.setLevel(logging.DEBUG); _log.addHandler(_fh)
_sh = logging.StreamHandler(sys.stderr)
_sh.setFormatter(_log_fmt); _sh.setLevel(logging.WARNING); _log.addHandler(_sh)


def _write_crash(text: str):
    try:
        with open(os.path.join(DATA_DIR, "crash_bybit_rsi_reversal_bt.log"), "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n{datetime.now(timezone.utc).isoformat()}\n{text}\n")
    except Exception:
        pass


def _make_console() -> Console:
    if sys.platform == "win32":
        try:
            buf = getattr(sys.stdout, "buffer", None)
            if buf is not None:
                return Console(file=io.TextIOWrapper(buf, encoding="utf-8", write_through=True))
        except Exception:
            pass
    return Console()


_console = _make_console()


# ── DB ────────────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(DATA_DIR, "bybit_rsi_reversal_params.db")


def db_init():
    with sqlite3.connect(DB_PATH) as con:
        # Drop old schema if it has the stochastic columns
        try:
            con.execute("SELECT k_len FROM param_runs LIMIT 1")
            con.execute("DROP TABLE IF EXISTS param_runs")
            con.execute("DROP INDEX IF EXISTS idx_sym_iv_score")
        except sqlite3.OperationalError:
            pass
        con.execute("""
            CREATE TABLE IF NOT EXISTS param_runs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                run_ts           TEXT    NOT NULL,
                symbol           TEXT    NOT NULL,
                interval         TEXT    NOT NULL,
                rsi_p            INTEGER,
                rsi_os           INTEGER,
                rsi_ob           INTEGER,
                kde_thr          REAL,
                zone_sl_buf      REAL,
                atr_p            INTEGER,
                chop_len         INTEGER,
                chop_thr         REAL,
                swing_len        INTEGER,
                partial_lvl      INTEGER,
                max_zone_sl_pct  REAL,
                score            REAL,
                sharpe           REAL,
                cagr_pct         REAL,
                total_ret_pct    REAL,
                max_dd_pct       REAL,
                trades           INTEGER,
                avg_hold         REAL,
                win_rate         REAL,
                profit_factor    REAL,
                final_equity     REAL
            )
        """)
        # Migration: add max_zone_sl_pct column to existing DB if missing
        try:
            con.execute("SELECT max_zone_sl_pct FROM param_runs LIMIT 1")
        except sqlite3.OperationalError:
            con.execute(f"ALTER TABLE param_runs ADD COLUMN max_zone_sl_pct REAL DEFAULT {MAX_ZONE_SL_PCT_DEFAULT}")
        con.execute("CREATE INDEX IF NOT EXISTS idx_sym_iv_score "
                    "ON param_runs(symbol, interval, score DESC)")


def db_save_candidates(symbol: str, interval: str, candidates: list, run_ts: str):
    rows = [
        (run_ts, symbol, interval,
         int(c["rsi_p"]),    int(c["rsi_os"]),    int(c["rsi_ob"]),
         float(c["kde_thr"]),
         int(c["atr_p"]),    int(c["chop_len"]),
         float(c["chop_thr"]), int(c["swing_len"]),
         float(c["max_zone_sl_pct"]),
         float(c["score"]),  float(c["sharpe"]),
         float(c["cagr_pct"]),  float(c["total_ret_pct"]),
         float(c["max_dd_pct"]), int(c["trades"]),
         float(c["avg_hold"]),   float(c["win_rate"]),
         float(c["profit_factor"]), float(c["final_equity"]))
        for c in candidates
        if float(c.get("total_ret_pct", -1)) > 0
        and float(c.get("max_dd_pct", -100)) > -MAX_DD_PCT * 100
    ]
    if not rows:
        _log.info(f"DB: no profitable candidates for {symbol} {interval}m"); return
    with sqlite3.connect(DB_PATH) as con:
        con.executemany("""
            INSERT INTO param_runs
            (run_ts,symbol,interval,
             rsi_p,rsi_os,rsi_ob,kde_thr,
             atr_p,chop_len,chop_thr,swing_len,max_zone_sl_pct,
             score,sharpe,cagr_pct,total_ret_pct,max_dd_pct,
             trades,avg_hold,win_rate,profit_factor,final_equity)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
    _log.info(f"DB: +{len(rows)} profitable rows for {symbol} {interval}m")


def db_load_all_seen(symbol: str, interval: str) -> set:
    """All param-tuples ever recorded for this symbol/interval — used to deduplicate random sampling
    across every BT run so N_RANDOM always represents truly novel combinations."""
    _keys = sorted(PARAM_SPACE)
    col_list = ", ".join(_keys)
    try:
        with sqlite3.connect(DB_PATH) as con:
            rows = con.execute(
                f"SELECT {col_list} FROM param_runs WHERE symbol=? AND interval=?",
                (symbol, interval)
            ).fetchall()
    except sqlite3.Error:
        return set()
    seen = set()
    for row in rows:
        t = tuple(
            int(v) if _keys[i] in _INT_PARAMS
            else round(float(v), 2) if v is not None
            else round(MAX_ZONE_SL_PCT_DEFAULT, 2)
            for i, v in enumerate(row)
        )
        seen.add(t)
    return seen


def db_load_top_params(symbol: str, interval: str, n: int = N_TOP_RETEST) -> list:
    try:
        with sqlite3.connect(DB_PATH) as con:
            rows = con.execute("""
                SELECT rsi_p,rsi_os,rsi_ob,kde_thr,
                       atr_p,chop_len,chop_thr,swing_len,max_zone_sl_pct
                FROM   param_runs
                WHERE  symbol=? AND interval=? AND total_ret_pct>0 AND max_dd_pct>?
                GROUP  BY rsi_p,rsi_os,rsi_ob,kde_thr,
                          atr_p,chop_len,chop_thr,swing_len,max_zone_sl_pct
                ORDER  BY MAX(score) DESC
                LIMIT  ?
            """, (symbol, interval, -MAX_DD_PCT * 100, n)).fetchall()
    except sqlite3.Error:
        return []
    return [{"rsi_p": int(r[0]),    "rsi_os": int(r[1]),    "rsi_ob": int(r[2]),
             "kde_thr": float(r[3]),
             "atr_p": int(r[4]),    "chop_len": int(r[5]),
             "chop_thr": float(r[6]), "swing_len": int(r[7]),
             "max_zone_sl_pct": float(r[8]) if r[8] is not None else MAX_ZONE_SL_PCT_DEFAULT}
            for r in rows]


# ── Session ───────────────────────────────────────────────────────────────────────
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
    global INITIAL_EQUITY
    k, s = load_keys()
    if not k or not s:
        _log.critical(f"No API keys — place them in {KEYS_PATH}"); sys.exit(1)
    try:
        sess = HTTP(api_key=k, api_secret=s, demo=DEMO)
        r = sess.get_wallet_balance(accountType="UNIFIED")
        if r.get("retCode", -1) != 0:
            _log.critical(f"Session ping failed: {r.get('retMsg','')}"); sys.exit(1)
        if not DEMO:
            try:
                coins = r["result"]["list"][0]["coin"]
                usdt  = next((c for c in coins if c["coin"] == "USDT"), None)
                # equity = walletBalance + unrealisedPnl; fall back to walletBalance if equity is empty
                raw   = (usdt.get("equity") or usdt.get("walletBalance") or "0") if usdt else "0"
                bal   = float(raw)
                if bal >= 1.0:
                    INITIAL_EQUITY = bal
                    _log.info(f"LIVE mode: INITIAL_EQUITY set to {bal:.2f} USDT from API")
                else:
                    _log.warning(
                        f"LIVE mode: API returned balance {bal:.2f} — keeping config "
                        f"INITIAL_EQUITY={INITIAL_EQUITY:.2f}. Check account or retry.")
            except Exception as be:
                _log.warning(f"LIVE mode: could not read balance ({be}) — keeping config INITIAL_EQUITY={INITIAL_EQUITY:.2f}")
        _log.info(f"Session OK ({'LIVE' if not DEMO else 'demo'} key=...{k[-4:]})")
        return sess
    except Exception as e:
        _log.critical(f"Cannot connect: {e}"); sys.exit(1)


_NO_RETRY         = {110007, 110006, 110012, 110013, 110017, 110025}
_RATE_LIMIT_CODES = {10006, 10018}


def _api(fn, *args, **kwargs):
    for attempt in range(3):
        try:
            r = fn(*args, **kwargs)
        except Exception as e:
            _log.debug(f"API {fn.__name__} attempt {attempt+1}: {e}")
            if attempt == 2: raise
            time.sleep(0.5 * (attempt + 1)); continue
        if isinstance(r, dict):
            rc = r.get("retCode", 0)
            if rc in _RATE_LIMIT_CODES: time.sleep(1 + attempt); continue
            if rc in _NO_RETRY:         return r
            if rc != 0 and attempt < 2: time.sleep(0.5 * (attempt + 1)); continue
        return r
    return r


# ── Data fetch ────────────────────────────────────────────────────────────────────
def fetch_ohlcv(session, symbol: str, interval: str) -> pd.DataFrame:
    max_pages   = _max_pages(interval)
    pub_session = None
    all_bars: dict = {}
    end_ms = None
    for page in range(max_pages):
        kwargs = dict(category=CATEGORY, symbol=symbol,
                      interval=interval, limit=FETCH_LIMIT)
        if end_ms is not None:
            kwargs["end"] = end_ms
        r = _api(session.get_kline, **kwargs)
        raw = r.get("result", {}).get("list", [])
        if not raw:
            if page == 0:
                if pub_session is None:
                    pub_session = HTTP()
                r = pub_session.get_kline(**kwargs)
                raw = r.get("result", {}).get("list", [])
            if not raw:
                break
        for b in raw:
            all_bars[int(b[0])] = b
        if len(raw) < FETCH_LIMIT:
            break
        end_ms = min(int(b[0]) for b in raw) - 1
    if not all_bars:
        raise RuntimeError(f"No kline data for {symbol} {interval}m")
    bars = sorted(all_bars.values(), key=lambda x: int(x[0]))
    idx  = pd.to_datetime([datetime.fromtimestamp(int(b[0]) / 1000, tz=timezone.utc)
                           for b in bars])
    return pd.DataFrame({
        "open":   [float(b[1]) for b in bars],
        "high":   [float(b[2]) for b in bars],
        "low":    [float(b[3]) for b in bars],
        "close":  [float(b[4]) for b in bars],
        "volume": [float(b[5]) for b in bars],
    }, index=idx)


# ── Indicators ────────────────────────────────────────────────────────────────────
def _atr_wilder(high, low, close, period):
    tr     = np.empty(len(close))
    tr[0]  = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
             np.maximum(np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:]  - close[:-1])))
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values


def _chop_index(high, low, close, period):
    tr     = np.empty(len(close))
    tr[0]  = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
             np.maximum(np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:]  - close[:-1])))
    sum_tr = pd.Series(tr).rolling(period, min_periods=period).sum().values
    hh     = pd.Series(high).rolling(period, min_periods=period).max().values
    ll     = pd.Series(low).rolling(period,  min_periods=period).min().values
    range_ = hh - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.where(range_ == 0, 100.0,
             100.0 * np.log10(np.maximum(sum_tr, 1e-10) / np.maximum(range_, 1e-10))
             / math.log10(period))
    ci[np.isnan(hh)] = np.nan
    return ci


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
    """
    Flux Charts KDE score — percentile rank of current RSI within the lb-bar lookback window.

    score_long[i]  = fraction of lb-bar window with RSI > rsi_arr[i]  (0–1)
                     Higher = current RSI is more extreme/oversold vs recent history.
    score_short[i] = fraction of lb-bar window with RSI < rsi_arr[i]  (0–1)
                     Higher = current RSI is more extreme/overbought vs recent history.

    Long entry:  score_long  >= kde_thr  (e.g. 0.50 = top-50% most oversold reading)
    Short entry: score_short >= kde_thr  (e.g. 0.50 = top-50% most overbought reading)
    Both sides use the same threshold — matches the "KDE above 50%" filter from the video.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    n           = len(rsi_arr)
    score_long  = np.full(n, np.nan)
    score_short = np.full(n, np.nan)
    if n < lb:
        return score_long, score_short
    windows              = sliding_window_view(rsi_arr, lb)   # (n-lb+1, lb)
    cur_col              = windows[:, -1:]                     # current bar is last in each window
    score_long[lb - 1:]  = np.mean(windows > cur_col, axis=1)
    score_short[lb - 1:] = np.mean(windows < cur_col, axis=1)
    return score_long, score_short


def _compute_sd_zones(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, htf_factor: int,
                      sd_zone_mult: float) -> tuple:
    """
    Flux Charts Momentum-style base-candle Supply/Demand zones on HTF aggregation.

    Drop-Base-Rally  →  Demand: N_BASE candles before MIN_RUN consecutive bullish impulses.
    Rally-Base-Drop  →  Supply: N_BASE candles before MIN_RUN consecutive bearish impulses.

    sd_zone_mult : body/range impulse threshold (0.5 = Flux Charts standard).
    Zone top/bot : max(high) / min(low) of the N_BASE candles before the impulse run (full wicks).
    Invalidated  : close below demand_bot (demand) / close above supply_top (supply).
    """
    N_BASE  = 2   # base candles before impulse run used for zone boundaries (Flux Charts: 2)
    MIN_RUN = 2   # min consecutive HTF impulse candles to form a zone

    n     = len(close)
    n_htf = n // htf_factor
    if n_htf < N_BASE + MIN_RUN + 1:
        return (np.zeros(n, bool), np.zeros(n, bool),
                np.full(n, np.nan), np.full(n, np.nan))

    htf_open  = np.array([open_[i * htf_factor]                             for i in range(n_htf)])
    htf_high  = np.array([high[i * htf_factor:(i + 1) * htf_factor].max()  for i in range(n_htf)])
    htf_low   = np.array([low[i * htf_factor:(i + 1) * htf_factor].min()   for i in range(n_htf)])
    htf_close = np.array([close[(i + 1) * htf_factor - 1]                  for i in range(n_htf)])

    htf_rng  = htf_high - htf_low
    htf_body = np.abs(htf_close - htf_open)
    # Impulse candle: body > sd_zone_mult fraction of full high-low range
    is_imp   = np.where(htf_rng > 0, htf_body / htf_rng, 0.0) > sd_zone_mult
    is_bull  = htf_close > htf_open
    is_bear  = htf_close < htf_open

    # Find impulse runs; zones are formed from N_BASE candles before each qualifying run
    zones_raw = []  # (zone_type, z_top, z_bot, htf_active_bar)
    i = 0
    while i < n_htf:
        if is_imp[i]:
            j = i
            if is_bull[i]:
                while j < n_htf and is_imp[j] and is_bull[j]:
                    j += 1
                if (j - i) >= MIN_RUN and i >= N_BASE:
                    zones_raw.append((
                        'demand',
                        htf_high[i - N_BASE:i].max(),
                        htf_low[i - N_BASE:i].min(),
                        j,  # zone active from this HTF bar onwards
                    ))
            elif is_bear[i]:
                while j < n_htf and is_imp[j] and is_bear[j]:
                    j += 1
                if (j - i) >= MIN_RUN and i >= N_BASE:
                    zones_raw.append((
                        'supply',
                        htf_high[i - N_BASE:i].max(),
                        htf_low[i - N_BASE:i].min(),
                        j,
                    ))
            i = j  # skip to first bar after this run
        else:
            i += 1

    in_demand  = np.zeros(n, bool)
    in_supply  = np.zeros(n, bool)
    demand_bot = np.full(n, np.nan)
    supply_top = np.full(n, np.nan)

    for zone_type, z_top, z_bot, htf_active in zones_raw:
        base_start = htf_active * htf_factor
        if base_start >= n:
            continue
        if zone_type == 'demand':
            for b in range(base_start, n):
                if close[b] < z_bot:
                    break  # zone invalidated — price closed below demand floor
                if z_bot <= close[b] <= z_top:
                    in_demand[b]  = True
                    demand_bot[b] = z_bot
        else:
            for b in range(base_start, n):
                if close[b] > z_top:
                    break  # zone invalidated — price closed above supply ceiling
                if z_bot <= close[b] <= z_top:
                    in_supply[b]  = True
                    supply_top[b] = z_top

    return in_demand, in_supply, demand_bot, supply_top


def _build_swing_levels(high: np.ndarray, low: np.ndarray, swing_len: int) -> tuple:
    """
    Confirmed swing highs/lows available at bar i+swing_len (no lookahead).
    Returns (sh_events, sl_events) — lists of (bar_confirmed, price).
    """
    n  = len(high)
    sh = []
    sl = []
    for i in range(swing_len, n - swing_len):
        if high[i] >= high[i - swing_len:i].max() and high[i] >= high[i + 1:i + swing_len + 1].max():
            sh.append((i + swing_len, high[i]))
        if low[i] <= low[i - swing_len:i].min() and low[i] <= low[i + 1:i + swing_len + 1].min():
            sl.append((i + swing_len, low[i]))
    return sh, sl


# ── Fast combo backtest ───────────────────────────────────────────────────────────
def _bt_combo_se(params: dict,
                 open_: np.ndarray, high: np.ndarray,
                 low: np.ndarray,   close: np.ndarray,
                 in_demand: np.ndarray, in_supply: np.ndarray,
                 demand_bot: np.ndarray, supply_top: np.ndarray,
                 rsi_cache: dict, kde_cache: dict,
                 chop_cache: dict, atr_cache: dict, swing_cache: dict,
                 backtest_bars: int, bars_per_year: float,
                 end_offset: int = 0,
                 initial_equity: float = 0.0) -> "dict | None":

    rsi_p            = int(params["rsi_p"])
    rsi_os           = int(params["rsi_os"])
    rsi_ob           = int(params["rsi_ob"])
    kde_thr          = float(params["kde_thr"])
    atr_p            = int(params["atr_p"])
    chop_len         = int(params["chop_len"])
    chop_thr         = float(params["chop_thr"])
    swing_len        = int(params["swing_len"])
    max_zone_sl_pct  = float(params.get("max_zone_sl_pct", MAX_ZONE_SL_PCT_DEFAULT))

    if rsi_os >= rsi_ob:
        return None

    rsi_arr                  = rsi_cache[rsi_p]
    kde_long, kde_short      = kde_cache[rsi_p]
    ci_arr                   = chop_cache[chop_len]
    atr_arr                  = atr_cache[atr_p]
    sh_events, sl_events     = swing_cache[swing_len]

    n        = len(close)
    rsi_prev = np.roll(rsi_arr, 1)

    # RSI crossover: bullish = crosses up through oversold; bearish = crosses down through overbought
    cross_os_up = (rsi_prev < rsi_os) & (rsi_arr >= rsi_os)
    cross_ob_dn = (rsi_prev > rsi_ob) & (rsi_arr <= rsi_ob)
    cross_os_up[0] = cross_ob_dn[0] = False

    ci_ok    = ci_arr > chop_thr
    buy_sig  = cross_os_up & (kde_long  >= kde_thr) & ci_ok & in_demand
    sell_sig = cross_ob_dn & (kde_short >= kde_thr) & ci_ok & in_supply

    sh_ptr = 0; sh_prices: list = []
    sl_ptr = 0; sl_prices: list = []

    n_end        = n - end_offset
    start_i      = max(0, n_end - backtest_bars)
    _ie          = initial_equity if initial_equity > 0.0 else INITIAL_EQUITY
    equity       = _ie
    in_long = in_short = False
    entry_price = stop_level = tp_level = pos_qty = entry_fee_pd = 0.0
    liq_price = isolated_mrgn = 0.0
    entry_i      = 0
    trades = 0; wins = 0
    gross_w = 0.0; gross_l = 0.0; total_hold = 0
    nb       = n_end - start_i
    eq_curve = np.empty(nb)

    for idx in range(start_i, n_end):
        ii = idx - start_i

        while sh_ptr < len(sh_events) and sh_events[sh_ptr][0] <= idx:
            bisect.insort(sh_prices, sh_events[sh_ptr][1]); sh_ptr += 1
        while sl_ptr < len(sl_events) and sl_events[sl_ptr][0] <= idx:
            bisect.insort(sl_prices, sl_events[sl_ptr][1]); sl_ptr += 1

        if np.isnan(atr_arr[idx]) or np.isnan(ci_arr[idx]):
            eq_curve[ii] = equity; continue
        cl = close[idx]

        if not (in_long or in_short):
            if buy_sig[idx]:
                if np.isnan(demand_bot[idx]):
                    eq_curve[ii] = equity; continue
                sl_level = demand_bot[idx]
                if sl_level >= cl:
                    eq_curve[ii] = equity; continue
                if abs(cl - sl_level) / cl > max_zone_sl_pct:
                    eq_curve[ii] = equity; continue
                i_above = bisect.bisect_right(sh_prices, cl)
                if i_above >= len(sh_prices):
                    eq_curve[ii] = equity; continue
                _tp_swing    = sh_prices[i_above]
                _tp_cand     = cl + TP_FACTOR * (_tp_swing - cl)
                notional     = equity * LEVERAGE * MARGIN_HEADROOM
                pos_qty      = notional / cl
                entry_fee_pd = notional * TAKER_FEE; equity -= entry_fee_pd
                entry_price  = cl; stop_level = sl_level; tp_level = _tp_cand
                isolated_mrgn = notional / LEVERAGE     # max loss in isolated mode
                liq_price     = cl * (1.0 - 1.0 / LEVERAGE)   # approx long liq price
                in_long = True; entry_i = idx
            elif sell_sig[idx]:
                if np.isnan(supply_top[idx]):
                    eq_curve[ii] = equity; continue
                sl_level = supply_top[idx]
                if sl_level <= cl:
                    eq_curve[ii] = equity; continue
                if abs(sl_level - cl) / cl > max_zone_sl_pct:
                    eq_curve[ii] = equity; continue
                i_below = bisect.bisect_left(sl_prices, cl) - 1
                if i_below < 0:
                    eq_curve[ii] = equity; continue
                _tp_swing    = sl_prices[i_below]
                _tp_cand     = cl - TP_FACTOR * (cl - _tp_swing)
                notional     = equity * LEVERAGE * MARGIN_HEADROOM
                pos_qty      = notional / cl
                entry_fee_pd = notional * TAKER_FEE; equity -= entry_fee_pd
                entry_price  = cl; stop_level = sl_level; tp_level = _tp_cand
                isolated_mrgn = notional / LEVERAGE
                liq_price     = cl * (1.0 + 1.0 / LEVERAGE)   # approx short liq price
                in_short = True; entry_i = idx
        elif in_long:
            bar_open = open_[idx]
            if bar_open < stop_level:
                # Gap below SL: stop market fills at open; cap at liq_price floor (isolated margin)
                fill = max(bar_open, liq_price)
                ef = fill * pos_qty * TAKER_FEE
                pnl = (fill - entry_price) * pos_qty - ef
                pnl_allin = pnl - entry_fee_pd; equity += pnl
                trades += 1; total_hold += (idx - entry_i)
                if pnl_allin > 0: wins += 1; gross_w += pnl_allin
                else:             gross_l += abs(pnl_allin)
                in_long = False; entry_fee_pd = 0.0
            else:
                sl_hit = low[idx] <= stop_level
                tp_hit = high[idx] >= tp_level
                if sl_hit or tp_hit:
                    exit_px = stop_level if sl_hit else tp_level
                    ef = exit_px * pos_qty * TAKER_FEE
                    pnl = (exit_px - entry_price) * pos_qty - ef
                    pnl_allin = pnl - entry_fee_pd; equity += pnl
                    trades += 1; total_hold += (idx - entry_i)
                    if pnl_allin > 0: wins += 1; gross_w += pnl_allin
                    else:             gross_l += abs(pnl_allin)
                    in_long = False; entry_fee_pd = 0.0
        elif in_short:
            bar_open = open_[idx]
            if bar_open > stop_level:
                # Gap above SL: fills at open; cap at liq_price ceiling (isolated margin)
                fill = min(bar_open, liq_price)
                ef = fill * pos_qty * TAKER_FEE
                pnl = (entry_price - fill) * pos_qty - ef
                pnl_allin = pnl - entry_fee_pd; equity += pnl
                trades += 1; total_hold += (idx - entry_i)
                if pnl_allin > 0: wins += 1; gross_w += pnl_allin
                else:             gross_l += abs(pnl_allin)
                in_short = False; entry_fee_pd = 0.0
            else:
                sl_hit = high[idx] >= stop_level
                tp_hit = low[idx] <= tp_level
                if sl_hit or tp_hit:
                    exit_px = stop_level if sl_hit else tp_level
                    ef = exit_px * pos_qty * TAKER_FEE
                    pnl = (entry_price - exit_px) * pos_qty - ef
                    pnl_allin = pnl - entry_fee_pd; equity += pnl
                    trades += 1; total_hold += (idx - entry_i)
                    if pnl_allin > 0: wins += 1; gross_w += pnl_allin
                    else:             gross_l += abs(pnl_allin)
                    in_short = False; entry_fee_pd = 0.0

        # MTM: cap unrealised loss at isolated margin (can't exceed locked collateral)
        if in_long:
            eq_curve[ii] = equity + max((cl - entry_price) * pos_qty, -isolated_mrgn)
        elif in_short:
            eq_curve[ii] = equity + max((entry_price - cl) * pos_qty, -isolated_mrgn)
        else:
            eq_curve[ii] = equity

    if in_long or in_short:
        cl = close[n_end - 1]; ef = cl * pos_qty * TAKER_FEE
        pnl = ((cl - entry_price) if in_long else (entry_price - cl)) * pos_qty - ef
        pnl_allin = pnl - entry_fee_pd; equity += pnl
        trades += 1; total_hold += (n_end - 1 - entry_i)
        if pnl_allin > 0: wins += 1; gross_w += pnl_allin
        else:             gross_l += abs(pnl_allin)

    if trades < MIN_TRADES or (trades > 0 and total_hold / trades < MIN_AVG_HOLD):
        return None

    avg_hold = total_hold / trades
    peak     = np.maximum.accumulate(eq_curve)
    dd_arr   = (eq_curve - peak) / np.where(peak == 0, 1e-9, peak)
    max_dd   = float(dd_arr.min())
    if max_dd < -MAX_DD_PCT:
        return None

    diffs = np.diff(eq_curve); prev = eq_curve[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = diffs / np.where(prev == 0, 1e-9, prev)
    ret    = ret[np.isfinite(ret)]
    std    = float(ret.std()) if len(ret) > 1 else 0.0
    sharpe = float(ret.mean() / std * math.sqrt(bars_per_year)) if std > 0 else 0.0

    years   = max(nb / bars_per_year, 1e-9)
    cagr    = max(equity / _ie, 1e-9) ** (1.0 / years) - 1
    tot_ret = (equity - _ie) / _ie
    pf      = gross_w / gross_l if gross_l > 0 else float("inf")
    wr      = wins / trades if trades > 0 else 0.0
    score   = sharpe * math.sqrt(trades / MIN_TRADES)

    return {
        "score":         round(score, 3),
        "sharpe":        round(sharpe, 3),
        "cagr_pct":      round(cagr * 100, 2),
        "total_ret_pct": round(tot_ret * 100, 2),
        "max_dd_pct":    round(max_dd * 100, 2),
        "trades":        int(trades),
        "avg_hold":      round(avg_hold, 1),
        "win_rate":      round(wr, 4),
        "profit_factor": round(min(pf, 99.9), 3),
        "final_equity":  round(equity, 2),
    }


def _combo_worker_se(args):
    (combos, open_, high, low, close,
     in_demand, in_supply, demand_bot, supply_top,
     rsi_cache, kde_cache, chop_cache, atr_cache, swing_cache,
     backtest_bars, bars_per_year, end_offset, initial_equity) = args
    out = []
    for combo in combos:
        m = _bt_combo_se(combo, open_, high, low, close,
                         in_demand, in_supply, demand_bot, supply_top,
                         rsi_cache, kde_cache, chop_cache, atr_cache, swing_cache,
                         backtest_bars, bars_per_year, end_offset, initial_equity)
        if m is not None:
            out.append({**combo, **m})
    return out


# ── Detailed backtest ─────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, symbol: str, interval: str, params: dict,
                 in_demand: np.ndarray, in_supply: np.ndarray,
                 demand_bot: np.ndarray, supply_top: np.ndarray,
                 bt_bars_override: int = 0, end_offset: int = 0,
                 tp_scale: float = 1.0) -> dict:
    open_  = df["open"].values; high = df["high"].values
    low    = df["low"].values;  close = df["close"].values

    bt_bars = bt_bars_override if bt_bars_override > 0 else _backtest_bars(interval)
    bpy = _bars_per_year(interval)

    rsi_p            = int(params["rsi_p"])
    rsi_os           = int(params["rsi_os"])
    rsi_ob           = int(params["rsi_ob"])
    kde_thr          = float(params["kde_thr"])
    atr_p            = int(params.get("atr_p", 14))
    chop_len         = int(params["chop_len"])
    chop_thr         = float(params["chop_thr"])
    swing_len        = int(params["swing_len"])
    max_zone_sl_pct  = float(params.get("max_zone_sl_pct", MAX_ZONE_SL_PCT_DEFAULT))

    rsi_arr                = _rsi(close, rsi_p)
    kde_long, kde_short    = _kde_scores(rsi_arr, KDE_LB)
    ci_arr                 = _chop_index(high, low, close, chop_len)
    atr_arr                = _atr_wilder(high, low, close, atr_p)

    n        = len(close)
    rsi_prev = np.roll(rsi_arr, 1)

    cross_os_up = (rsi_prev < rsi_os) & (rsi_arr >= rsi_os)
    cross_ob_dn = (rsi_prev > rsi_ob) & (rsi_arr <= rsi_ob)
    cross_os_up[0] = cross_ob_dn[0] = False

    ci_ok    = ci_arr > chop_thr
    buy_sig  = cross_os_up & (kde_long  >= kde_thr) & ci_ok & in_demand
    sell_sig = cross_ob_dn & (kde_short >= kde_thr) & ci_ok & in_supply

    sh_events, sl_events = _build_swing_levels(high, low, swing_len)
    sh_ptr = 0; sh_prices: list = []
    sl_ptr = 0; sl_prices: list = []

    n_end   = n - end_offset
    start_i = max(0, n_end - bt_bars); ts_index = df.index
    equity  = INITIAL_EQUITY
    in_long = in_short = False
    entry_price = stop_level = tp_level = pos_qty = entry_fee_pd = 0.0
    liq_price = isolated_mrgn = 0.0
    entry_ts = None; trades = []; equity_curve = []; ts_curve = []

    for idx in range(start_i, n_end):
        while sh_ptr < len(sh_events) and sh_events[sh_ptr][0] <= idx:
            bisect.insort(sh_prices, sh_events[sh_ptr][1]); sh_ptr += 1
        while sl_ptr < len(sl_events) and sl_events[sl_ptr][0] <= idx:
            bisect.insort(sl_prices, sl_events[sl_ptr][1]); sl_ptr += 1

        if np.isnan(atr_arr[idx]) or np.isnan(ci_arr[idx]):
            equity_curve.append(equity); ts_curve.append(ts_index[idx]); continue
        cl = close[idx]; ts = ts_index[idx]

        if not (in_long or in_short):
            if buy_sig[idx]:
                if np.isnan(demand_bot[idx]):
                    equity_curve.append(equity); ts_curve.append(ts); continue
                sl_level = demand_bot[idx]
                if sl_level >= cl:
                    equity_curve.append(equity); ts_curve.append(ts); continue
                if abs(cl - sl_level) / cl > max_zone_sl_pct:
                    equity_curve.append(equity); ts_curve.append(ts); continue
                i_above = bisect.bisect_right(sh_prices, cl)
                if i_above >= len(sh_prices):
                    equity_curve.append(equity); ts_curve.append(ts); continue
                _tp_swing    = sh_prices[i_above]
                _tp_cand     = cl + TP_FACTOR * tp_scale * (_tp_swing - cl)
                notional     = equity * LEVERAGE * MARGIN_HEADROOM
                pos_qty      = notional / cl; entry_fee_pd = notional * TAKER_FEE; equity -= entry_fee_pd
                entry_price  = cl; stop_level = sl_level; tp_level = _tp_cand
                isolated_mrgn = notional / LEVERAGE
                liq_price     = cl * (1.0 - 1.0 / LEVERAGE)
                in_long = True; entry_ts = ts
            elif sell_sig[idx]:
                if np.isnan(supply_top[idx]):
                    equity_curve.append(equity); ts_curve.append(ts); continue
                sl_level = supply_top[idx]
                if sl_level <= cl:
                    equity_curve.append(equity); ts_curve.append(ts); continue
                if abs(sl_level - cl) / cl > max_zone_sl_pct:
                    equity_curve.append(equity); ts_curve.append(ts); continue
                i_below = bisect.bisect_left(sl_prices, cl) - 1
                if i_below < 0:
                    equity_curve.append(equity); ts_curve.append(ts); continue
                _tp_swing    = sl_prices[i_below]
                _tp_cand     = cl - TP_FACTOR * tp_scale * (cl - _tp_swing)
                notional     = equity * LEVERAGE * MARGIN_HEADROOM
                pos_qty      = notional / cl; entry_fee_pd = notional * TAKER_FEE; equity -= entry_fee_pd
                entry_price  = cl; stop_level = sl_level; tp_level = _tp_cand
                isolated_mrgn = notional / LEVERAGE
                liq_price     = cl * (1.0 + 1.0 / LEVERAGE)
                in_short = True; entry_ts = ts
        elif in_long:
            bar_open = open_[idx]
            if bar_open < stop_level:
                # Gap below SL: stop market fills at open; isolated margin caps loss at liq_price
                fill = max(bar_open, liq_price)
                reason = "liq" if fill <= liq_price else "sl_gap"
                ef = fill*pos_qty*TAKER_FEE; pnl = (fill-entry_price)*pos_qty - ef
                pnl_ai = pnl - entry_fee_pd; equity += pnl
                hold_h = round((ts - entry_ts).total_seconds() / 3600, 1)
                trades.append({"entry_ts":str(entry_ts),"exit_ts":str(ts),"side":"long",
                               "entry_px":float(entry_price),"exit_px":float(fill),
                               "tp_px":float(tp_level),"sl_px":float(stop_level),
                               "entry_fee":round(float(entry_fee_pd),4),
                               "exit_fee":round(float(ef),4),
                               "net_pnl":round(float(pnl_ai),4),"hold_hrs":hold_h,
                               "win":bool(pnl_ai>0),"exit_reason":reason})
                in_long = False; entry_fee_pd = 0.0
            else:
                sl_hit = low[idx] <= stop_level
                tp_hit = high[idx] >= tp_level
                if sl_hit or tp_hit:
                    exit_px = stop_level if sl_hit else tp_level
                    reason  = "sl" if sl_hit else "tp"
                    ef = exit_px*pos_qty*TAKER_FEE; pnl = (exit_px-entry_price)*pos_qty - ef
                    pnl_ai = pnl - entry_fee_pd; equity += pnl
                    hold_h = round((ts - entry_ts).total_seconds() / 3600, 1)
                    trades.append({"entry_ts":str(entry_ts),"exit_ts":str(ts),"side":"long",
                                   "entry_px":float(entry_price),"exit_px":float(exit_px),
                                   "tp_px":float(tp_level),"sl_px":float(stop_level),
                                   "entry_fee":round(float(entry_fee_pd),4),
                                   "exit_fee":round(float(ef),4),
                                   "net_pnl":round(float(pnl_ai),4),"hold_hrs":hold_h,
                                   "win":bool(pnl_ai>0),"exit_reason":reason})
                    in_long = False; entry_fee_pd = 0.0
        elif in_short:
            bar_open = open_[idx]
            if bar_open > stop_level:
                # Gap above SL: fills at open; isolated margin caps loss at liq_price
                fill = min(bar_open, liq_price)
                reason = "liq" if fill >= liq_price else "sl_gap"
                ef = fill*pos_qty*TAKER_FEE; pnl = (entry_price-fill)*pos_qty - ef
                pnl_ai = pnl - entry_fee_pd; equity += pnl
                hold_h = round((ts - entry_ts).total_seconds() / 3600, 1)
                trades.append({"entry_ts":str(entry_ts),"exit_ts":str(ts),"side":"short",
                               "entry_px":float(entry_price),"exit_px":float(fill),
                               "tp_px":float(tp_level),"sl_px":float(stop_level),
                               "entry_fee":round(float(entry_fee_pd),4),
                               "exit_fee":round(float(ef),4),
                               "net_pnl":round(float(pnl_ai),4),"hold_hrs":hold_h,
                               "win":bool(pnl_ai>0),"exit_reason":reason})
                in_short = False; entry_fee_pd = 0.0
            else:
                sl_hit = high[idx] >= stop_level
                tp_hit = low[idx] <= tp_level
                if sl_hit or tp_hit:
                    exit_px = stop_level if sl_hit else tp_level
                    reason  = "sl" if sl_hit else "tp"
                    ef = exit_px*pos_qty*TAKER_FEE; pnl = (entry_price-exit_px)*pos_qty - ef
                    pnl_ai = pnl - entry_fee_pd; equity += pnl
                    hold_h = round((ts - entry_ts).total_seconds() / 3600, 1)
                    trades.append({"entry_ts":str(entry_ts),"exit_ts":str(ts),"side":"short",
                                   "entry_px":float(entry_price),"exit_px":float(exit_px),
                                   "tp_px":float(tp_level),"sl_px":float(stop_level),
                                   "entry_fee":round(float(entry_fee_pd),4),
                                   "exit_fee":round(float(ef),4),
                                   "net_pnl":round(float(pnl_ai),4),"hold_hrs":hold_h,
                                   "win":bool(pnl_ai>0),"exit_reason":reason})
                    in_short = False; entry_fee_pd = 0.0

        # MTM: cap unrealised loss at isolated margin (can't exceed locked collateral)
        if in_long:    mtm = equity + max((cl-entry_price)*pos_qty, -isolated_mrgn)
        elif in_short: mtm = equity + max((entry_price-cl)*pos_qty, -isolated_mrgn)
        else:          mtm = equity
        equity_curve.append(mtm); ts_curve.append(ts)

    if in_long or in_short:
        cl = close[n_end - 1]; ts = ts_index[n_end - 1]; ef = cl*pos_qty*TAKER_FEE
        pnl = ((cl-entry_price) if in_long else (entry_price-cl))*pos_qty - ef
        equity += pnl; pnl_ai = pnl - entry_fee_pd
        hold_h = round((ts-entry_ts).total_seconds()/3600, 1)
        trades.append({"entry_ts":str(entry_ts),"exit_ts":str(ts),
                       "side":"long" if in_long else "short",
                       "entry_px":float(entry_price),"exit_px":float(cl),
                       "tp_px":float(tp_level),"sl_px":float(stop_level),
                       "entry_fee":round(float(entry_fee_pd),4),"exit_fee":round(float(ef),4),
                       "net_pnl":round(float(pnl_ai),4),"hold_hrs":hold_h,
                       "win":bool(pnl_ai>0),"exit_reason":"open"})

    eq_arr    = np.array(equity_curve)
    peak      = np.maximum.accumulate(eq_arr)
    dd_arr    = (eq_arr - peak) / np.where(peak == 0, 1e-9, peak)
    max_dd    = float(dd_arr.min()) if len(dd_arr) else 0.0
    daily_ret = np.diff(eq_arr) / eq_arr[:-1] if len(eq_arr) > 1 else np.array([0.0])
    sharpe    = (float(daily_ret.mean() / daily_ret.std() * math.sqrt(bpy))
                 if daily_ret.std() > 0 else 0.0)
    years     = max(len(eq_arr) / bpy, 1e-9)
    tot_ret   = (equity - INITIAL_EQUITY) / INITIAL_EQUITY
    cagr      = max(equity / INITIAL_EQUITY, 1e-9) ** (1.0 / years) - 1
    n_t  = len(trades); n_w = sum(1 for t in trades if t["win"]); wr = n_w/n_t if n_t else 0.0
    profits_l = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
    losses_l  = [t["net_pnl"] for t in trades if t["net_pnl"] < 0]
    wp        = sum(profits_l) if profits_l else 0.0
    lp        = abs(sum(losses_l)) if losses_l else 0.0
    max_profit = max(profits_l) if profits_l else 0.0   # best single trade
    max_loss   = min(losses_l)  if losses_l else 0.0   # worst single trade
    pf         = wp / lp if lp > 0 else float("inf")

    return {
        "symbol": symbol, "interval": interval, "params": params,
        "initial_equity":        INITIAL_EQUITY,
        "final_equity":          round(equity, 4),
        "total_return_pct":      round(tot_ret * 100, 2),
        "cagr_pct":              round(cagr * 100, 2),
        "max_drawdown_pct":      round(max_dd * 100, 2),
        "sharpe":                round(sharpe, 3),
        "trades":                n_t, "wins": n_w, "losses": n_t - n_w,
        "win_rate":              round(wr, 4),
        "profit_factor":         round(min(pf, 99.9), 3),
        "cumulative_profit_usd":  round(wp, 2),
        "cumulative_profit_pct":  round(wp / INITIAL_EQUITY * 100, 2),
        "max_profit_usd":         round(max_profit, 2),
        "max_profit_pct":         round(max_profit / INITIAL_EQUITY * 100, 2),
        "cumulative_losses_usd":  round(lp, 2),
        "cumulative_losses_pct":  round(-lp / INITIAL_EQUITY * 100, 2),
        "max_loss_usd":           round(max_loss, 2),
        "max_loss_pct":           round(max_loss / INITIAL_EQUITY * 100, 2),
        "avg_hold_hrs":          round(sum(t["hold_hrs"] for t in trades)/n_t if n_t else 0, 1),
        "start_ts":              str(ts_curve[0])  if ts_curve else "—",
        "end_ts":                str(ts_curve[-1]) if ts_curve else "—",
        "trade_log":             trades,
    }


# ── TUI ───────────────────────────────────────────────────────────────────────────
def build_tui(phase, progress, status, current_label, all_results, err="",
              opt_state=None, loop_count=0, next_run_in=0,
              symbols=None, intervals=None):
    if symbols   is None: symbols   = SYMBOLS
    if intervals is None: intervals = INTERVALS
    now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    loop_tag = f"  Loop #{loop_count}" if loop_count > 0 else ""
    iv_str   = "/".join(f"{i}m" for i in intervals)

    hdr = Table.grid(padding=(0, 2)); hdr.add_column(); hdr.add_column()
    hdr.add_row(
        Text(f"RSI Kernel+SD BT{loop_tag}  "
             f"{' / '.join(s.replace('USDT','') for s in symbols)}  [{iv_str}]"
             f"{'  DEMO' if DEMO else '  LIVE'}",
             style="bold cyan" if DEMO else "bold red"),
        Text(now_str, style="dim"))
    eq_src = "API" if not DEMO else "cfg"
    hdr.add_row(Text(
        f"Equity: {INITIAL_EQUITY:.2f} USDT [{eq_src}]  "
        f"Lev: {LEVERAGE}x  Fee: {TAKER_FEE*100:.3f}%  BT: {BACKTEST_HOURS}h  MaxDD: <{MAX_DD_PCT*100:.0f}%  "
        f"RSI crossover + KDE(lb={KDE_LB})  Flux S/D {HTF_FACTOR}x (imp>{SD_ZONE_MULT:.0%})  "
        f"SL=zone-based  TP=next swing  DB top-{N_TOP_RETEST} + {N_RANDOM:,} random",
        style="bold green" if not DEMO else "white"))

    bar_w  = 44; pct = min(max(progress, 0.0), 1.0)
    filled = int(pct * bar_w)
    bar    = "█" * filled + "░" * (bar_w - filled)

    if phase == "waiting":
        b_sty = "bold yellow"
        m, s  = divmod(next_run_in, 60)
        status_d = f"Next run in {m:02d}:{s:02d}  —  {status}"
    elif phase == "retrying": b_sty = "bold magenta"; status_d = status
    else: b_sty = "bold green" if pct >= 1.0 else "bold cyan"; status_d = status

    prog_t = Text()
    prog_t.append(f"[{bar}]", style=b_sty)
    prog_t.append(f"  {pct*100:.1f}%  ", style=b_sty)
    prog_t.append(status_d)
    if err: prog_t.append(f"  ! {err}", style="bold red")
    pg = Table.grid(padding=(0, 1)); pg.add_column(ratio=1); pg.add_row(prog_t)
    grid = Table.grid()
    grid.add_row(hdr)
    grid.add_row(Panel(pg, border_style="dim", height=3))

    if opt_state and opt_state.get("top_combos") and phase not in ("waiting",):
        top = opt_state["top_combos"][:8]
        lb  = Table(box=box.SIMPLE, expand=True, show_header=True, header_style="dim",
                    title=f"Top Combos — {current_label}  (Score=Sharpe×√trades)",
                    title_style="bold")
        for col, w, j in [
            ("RSIp",5,"right"),("OS",4,"right"),("OB",4,"right"),
            ("KDE",5,"right"),("ATRp",5,"right"),
            ("CLn",4,"right"),("CThr",5,"right"),("SwLn",5,"right"),
            ("Score",7,"right"),("Sharpe",7,"right"),("CAGR",8,"right"),
            ("DD",6,"right"),("Hld",6,"right"),("Trd",4,"right"),
        ]:
            lb.add_column(col, justify=j, width=w)
        for r in top:
            csty = "green" if r["cagr_pct"] > 0 else "red"
            lb.add_row(
                str(r["rsi_p"]),
                str(r["rsi_os"]), str(r["rsi_ob"]),
                f"{r['kde_thr']:.2f}",
                str(r["atr_p"]),
                str(r["chop_len"]), f"{r['chop_thr']:.0f}",
                str(r["swing_len"]),
                f"{r['score']:+.3f}", f"{r['sharpe']:+.3f}",
                Text(f"{r['cagr_pct']:+.1f}%", style=csty),
                f"{r['max_dd_pct']:.1f}%",
                f"{r['avg_hold']:.1f}h", str(r["trades"]),
            )
        grid.add_row(lb)

    if all_results:
        sum_t = Table(box=box.SIMPLE, expand=True, show_header=True,
                      header_style="dim", title="Results Summary", title_style="bold")
        for col, w, j in [
            ("Symbol",7,"left"),("TF",4,"center"),("Return",9,"right"),("CAGR",8,"right"),
            ("Sharpe",7,"right"),("MaxDD",7,"right"),("WR",6,"right"),
            ("Trades",7,"right"),("PF",6,"right"),("Eq $",9,"right"),
            ("CumP$",9,"right"),("CumP%",7,"right"),
            ("MaxP$",9,"right"),("MaxP%",7,"right"),
            ("CumL$",9,"right"),("CumL%",7,"right"),
            ("MaxL$",9,"right"),("MaxL%",7,"right"),
        ]:
            sum_t.add_column(col, justify=j, width=w)
        for sym in symbols:
            for iv in intervals:
                key = (sym, iv)
                if key not in all_results: continue
                r    = all_results[key]
                rsty = "green" if r["total_return_pct"] > 0 else "red"
                cp   = r.get("cumulative_profit_usd", 0.0)
                cpp  = r.get("cumulative_profit_pct", 0.0)
                mp   = r.get("max_profit_usd", 0.0)
                mpp  = r.get("max_profit_pct", 0.0)
                cl   = r.get("cumulative_losses_usd", 0.0)
                clp  = r.get("cumulative_losses_pct", 0.0)
                ml   = r.get("max_loss_usd", 0.0)
                mlp  = r.get("max_loss_pct", 0.0)
                sum_t.add_row(
                    Text(sym.replace("USDT",""), style="bold"), Text(f"{iv}m", style="dim"),
                    Text(f"{r['total_return_pct']:+.1f}%", style=rsty),
                    Text(f"{r['cagr_pct']:+.1f}%", style=rsty),
                    f"{r['sharpe']:+.3f}", f"{r['max_drawdown_pct']:.1f}%",
                    f"{r['win_rate']*100:.0f}%", str(r["trades"]),
                    f"{r['profit_factor']:.2f}",
                    f"${r['final_equity']:.2f}",
                    Text(f"{cp:.2f}",   style="green" if cp > 0 else "dim"),
                    Text(f"{cpp:.1f}%", style="green" if cpp > 0 else "dim"),
                    Text(f"{mp:.2f}",   style="green" if mp > 0 else "dim"),
                    Text(f"{mpp:.1f}%", style="green" if mpp > 0 else "dim"),
                    Text(f"-{cl:.2f}", style="red" if cl > 0 else "dim"),
                    Text(f"{clp:.1f}%", style="red" if clp < 0 else "dim"),
                    Text(f"{ml:.2f}",   style="red" if ml < 0 else "dim"),
                    Text(f"{mlp:.1f}%", style="red" if mlp < 0 else "dim"),
                )
        grid.add_row(sum_t)

    if phase in ("done", "waiting") and all_results:
        for sym in symbols:
            for iv in intervals:
                key = (sym, iv)
                if key not in all_results: continue
                r = all_results[key]; p = r.get("params", {})
                params_txt = (
                    f"{iv}m  RSI{int(p.get('rsi_p',14))} "
                    f"OS{int(p.get('rsi_os',30))}/OB{int(p.get('rsi_ob',70))}  "
                    f"kDE≥{p.get('kde_thr',0.5)*100:.0f}%  "
                    f"ATR{int(p.get('atr_p',14))}  "
                    f"CI{int(p.get('chop_len',14))}<{p.get('chop_thr',50.0):.0f}  "
                    f"Sw{int(p.get('swing_len',4))}"
                )
                tlog_t = Table(box=box.SIMPLE, expand=True, show_header=True, header_style="dim",
                               title=f"{sym.replace('USDT','')}  {params_txt}", title_style="bold")
                for col, w in [("Entry",20),("Exit",20),("Side",6),
                               ("EntPx",11),("ExtPx",11),("TP",11),("SL",11),
                               ("Net PnL",10),("Hold",7),("Why",5)]:
                    tlog_t.add_column(col,
                        justify="right" if col in ("Net PnL","Hold","EntPx","ExtPx","TP","SL") else "left",
                        width=w)
                for t in r.get("trade_log", []):
                    pnl_sty  = "green" if t["net_pnl"] >= 0 else "red"
                    rsty     = "bold red" if t["exit_reason"] == "sl" else (
                               "bold green" if t["exit_reason"] == "tp" else "dim")
                    tlog_t.add_row(
                        t["entry_ts"][:19], t["exit_ts"][:19], t["side"],
                        f"{t['entry_px']:.4f}",
                        f"{t['exit_px']:.4f}",
                        f"{t.get('tp_px',0):.4f}", f"{t.get('sl_px',0):.4f}",
                        Text(f"{t['net_pnl']:+.2f}", style=pnl_sty),
                        f"{t['hold_hrs']:.1f}h", Text(t["exit_reason"], style=rsty),
                    )
                grid.add_row(tlog_t)

    return grid


# ── Single full pass ──────────────────────────────────────────────────────────────
def _run_pass(session, loop_count: int, live) -> tuple:
    import traceback
    global INITIAL_EQUITY
    # Refresh live account equity at the start of every loop so sizing reflects current balance
    if not DEMO:
        try:
            r = session.get_wallet_balance(accountType="UNIFIED")
            coins = r["result"]["list"][0]["coin"]
            usdt  = next((c for c in coins if c["coin"] == "USDT"), None)
            raw   = (usdt.get("equity") or usdt.get("walletBalance") or "0") if usdt else "0"
            bal   = float(raw)
            if bal >= 1.0:
                INITIAL_EQUITY = bal
                _log.info(f"Loop #{loop_count+1}: INITIAL_EQUITY refreshed to {bal:.2f} USDT")
            else:
                _log.warning(
                    f"Loop #{loop_count+1}: API balance {bal:.2f} — keeping INITIAL_EQUITY={INITIAL_EQUITY:.2f}")
        except Exception as _be:
            _log.warning(f"Balance refresh failed ({_be}) — keeping previous INITIAL_EQUITY={INITIAL_EQUITY:.2f}")

    run_ts    = datetime.now(timezone.utc).isoformat()
    symbols   = list(SYMBOLS)
    intervals = list(INTERVALS)

    _FETCH_W  = float(len(symbols) * len(intervals))
    _OPT_W    = float(len(symbols) * len(intervals))
    _BT_W     = 0.2  * len(symbols) * len(intervals)
    _TOTAL_W  = _FETCH_W + _OPT_W + _BT_W
    _done_w   = 0.0

    current_label = f"{symbols[0]} {intervals[0]}m"
    all_results: dict = {}
    err    = ""
    opt_state = {"done": 0, "total": N_RANDOM, "candidates": 0,
                 "best_sharpe": -99.0, "top_combos": []}

    phase_ref    = ["starting"]
    status_ref   = [f"Loop #{loop_count} — Starting…"]
    progress_ref = [0.0]

    def _refresh():
        try:
            live.update(build_tui(phase_ref[0], progress_ref[0], status_ref[0],
                                  current_label, all_results, err, opt_state,
                                  loop_count=loop_count,
                                  symbols=symbols, intervals=intervals))
        except Exception:
            _log.error("build_tui error:\n" + traceback.format_exc())

    ohlcv_data: dict = {}
    fetch_idx = 0
    for sym in list(symbols):
        for iv in intervals:
            fetch_idx += 1
            phase_ref[0]    = "fetching"
            status_ref[0]   = (f"Loop #{loop_count} — Fetching {sym} {iv}m "
                               f"({fetch_idx}/{len(symbols)*len(intervals)})…")
            progress_ref[0] = _done_w / _TOTAL_W; _refresh()
            try:
                df = fetch_ohlcv(session, sym, iv)
                ohlcv_data[(sym, iv)] = df
                _log.info(f"  {sym} {iv}m: {len(df)} bars")
            except Exception as e:
                _log.warning(f"{sym} {iv}m skipped: {e}")
            _done_w += 1.0

    symbols = [s for s in symbols if any((s, iv) in ohlcv_data for iv in intervals)]
    if not symbols:
        _log.critical("No data. Aborting."); return {}, {}

    n_pairs  = len(symbols) * len(intervals)
    _TOTAL_W = _FETCH_W + float(n_pairs) + 0.2 * n_pairs
    pair_idx = 0

    for sym in symbols:
        for iv in intervals:
            pair_idx += 1; key = (sym, iv)
            current_label = f"{sym} {iv}m"
            if key not in ohlcv_data:
                _done_w += 1.2; continue

            df    = ohlcv_data[key]
            high  = df["high"].values; low = df["low"].values
            close = df["close"].values; open_ = df["open"].values
            bt_bars  = _backtest_bars(iv); bpy = _bars_per_year(iv)
            oos_bars = _oos_bars(iv)
            use_wf   = oos_bars < bt_bars   # walk-forward only makes sense when window > OOS
            is_bars  = (bt_bars - oos_bars) if use_wf else bt_bars
            # Strict WF: IS indicators computed on IS-only data; no OOS bars contaminate IS search
            n_is = len(close) - oos_bars if use_wf else len(close)

            phase_ref[0]  = "optimising"
            status_ref[0] = f"Loop #{loop_count} — {current_label}: computing indicators…"
            _refresh()

            # IS-only arrays for combo optimisation (end_offset=0 since data is pre-sliced)
            in_demand, in_supply, demand_bot, supply_top = _compute_sd_zones(
                open_[:n_is], high[:n_is], low[:n_is], close[:n_is], HTF_FACTOR, SD_ZONE_MULT)
            _log.info(f"  {sym} {iv}m: demand_bars={in_demand.sum()}  supply_bars={in_supply.sum()}")

            # Pre-compute indicator caches — IS-only data, shared across all combos
            rsi_cache   = {rp: _rsi(close[:n_is], rp) for rp in range(5, 22)}
            kde_cache   = {rp: _kde_scores(rsi_cache[rp], KDE_LB) for rp in range(5, 22)}
            chop_cache  = {cl_: _chop_index(high[:n_is], low[:n_is], close[:n_is], cl_) for cl_ in range(8, 21)}
            atr_cache   = {ap: _atr_wilder(high[:n_is], low[:n_is], close[:n_is], ap) for ap in range(8, 21)}
            swing_cache = {sl_: _build_swing_levels(high[:n_is], low[:n_is], sl_) for sl_ in range(2, 9)}

            db_combos = db_load_top_params(sym, iv, N_TOP_RETEST)
            seen = db_load_all_seen(sym, iv)   # every combo ever recorded — not just top N
            for c in db_combos:                # safety: ensure retest combos are in seen
                seen.add(tuple(c[k] for k in sorted(PARAM_SPACE)))
            rng  = random.Random()
            rand_combos: list = []
            while len(rand_combos) < N_RANDOM:
                combo = {k: (rng.randint(*v) if k in _INT_PARAMS else round(rng.uniform(*v), 2))
                         for k, v in PARAM_SPACE.items()}
                _ck = tuple(combo[k] for k in sorted(PARAM_SPACE))
                if _ck not in seen:
                    seen.add(_ck)
                    rand_combos.append(combo)
            all_combos = db_combos + rand_combos
            n_total    = len(all_combos)
            batches    = [all_combos[i:i+BATCH_SIZE] for i in range(0, n_total, BATCH_SIZE)]

            status_ref[0] = (f"Loop #{loop_count} — {current_label} ({pair_idx}/{n_pairs})  "
                             f"DB:{len(db_combos)} retest + {N_RANDOM:,} novel random…")
            _refresh()

            shared    = (open_[:n_is], high[:n_is], low[:n_is], close[:n_is],
                         in_demand, in_supply, demand_bot, supply_top,
                         rsi_cache, kde_cache, chop_cache, atr_cache, swing_cache,
                         is_bars, bpy, 0, INITIAL_EQUITY)  # end_offset=0, initial_equity passed to workers
            all_cands = []; done_n = 0
            opt_state = {"done": 0, "total": n_total, "candidates": 0,
                         "best_sharpe": -99.0, "top_combos": []}

            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                futs = {executor.submit(_combo_worker_se, (batch, *shared)): len(batch)
                        for batch in batches}
                for fut in as_completed(futs):
                    bsz = futs[fut]
                    try:   all_cands.extend(fut.result())
                    except Exception as exc: _log.error(f"worker ({sym} {iv}m): {exc}")
                    done_n += bsz
                    all_cands.sort(key=lambda r: r["score"], reverse=True)
                    opt_state["done"]        = done_n; opt_state["total"] = n_total
                    opt_state["candidates"]  = len(all_cands)
                    opt_state["best_sharpe"] = all_cands[0]["score"] if all_cands else -99.0
                    opt_state["top_combos"]  = all_cands[:8]
                    progress_ref[0] = (_done_w + done_n/n_total) / _TOTAL_W
                    status_ref[0]   = (f"Loop #{loop_count} — {current_label} "
                                       f"({pair_idx}/{n_pairs})  {done_n:,}/{n_total:,}  |  "
                                       f"Cands: {len(all_cands):,}  |  "
                                       f"Best: {opt_state['best_sharpe']:+.3f}")
                    _refresh()

            db_save_candidates(sym, iv, all_cands, run_ts)
            profitable = [r for r in all_cands
                          if r["total_ret_pct"] > 0 and r["score"] > 0
                          and r["max_dd_pct"] > -MAX_DD_PCT * 100]
            best_row   = profitable[0] if profitable else (all_cands[0] if all_cands else None)

            best_params = DEFAULT_PARAMS.copy()
            if best_row:
                best_params = {k: (int(best_row[k]) if k in _INT_PARAMS else float(best_row[k]))
                               for k in DEFAULT_PARAMS}
            best_params["leverage"] = LEVERAGE

            fname = f"bybit_rsi_reversal_best_params_{sym}_{iv}m.json"
            with open(os.path.join(DATA_DIR, fname), "w") as f:
                json.dump({"symbol": sym, "interval": iv, "params": best_params,
                           "top_combos": [{k: (int(v) if isinstance(v, (np.integer,)) else
                                               float(v) if isinstance(v, (np.floating,)) else v)
                                           for k, v in r.items()}
                                          for r in all_cands[:20]]}, f, indent=2)

            phase_ref[0]  = "backtesting"
            status_ref[0] = f"Backtesting {current_label} with best params…"; _refresh()
            _done_w += 1.0

            # IS detailed BT: IS-only df slice + IS zones (no end_offset needed)
            results = run_backtest(df.iloc[:n_is], sym, iv, best_params,
                                   in_demand, in_supply, demand_bot, supply_top,
                                   bt_bars_override=is_bars)
            if use_wf:
                # OOS: recompute zones on full data so OOS zone formation is correct
                in_demand_oos, in_supply_oos, demand_bot_oos, supply_top_oos = _compute_sd_zones(
                    open_, high, low, close, HTF_FACTOR, SD_ZONE_MULT)
                oos_res = run_backtest(df, sym, iv, best_params,
                                       in_demand_oos, in_supply_oos, demand_bot_oos, supply_top_oos,
                                       bt_bars_override=oos_bars)
                results["oos_return_pct"] = oos_res.get("total_return_pct", -999.0)
            else:
                results["oos_return_pct"] = results.get("total_return_pct", -999.0)
            # Conservative stress test: same IS data, TP scaled to TP_CONS_SCALE of optimised
            cons_res = run_backtest(df.iloc[:n_is], sym, iv, best_params,
                                    in_demand, in_supply, demand_bot, supply_top,
                                    bt_bars_override=is_bars, tp_scale=TP_CONS_SCALE)
            results["conservative_return_pct"] = cons_res.get("total_return_pct", -999.0)
            tl = results.pop("trade_log")
            with open(os.path.join(DATA_DIR, f"bybit_rsi_reversal_results_{sym}_{iv}m.json"), "w") as f:
                json.dump({**results, "trade_log": tl}, f, indent=2)
            results["trade_log"] = tl
            all_results[key] = results
            _done_w += 0.2

            _log.info(f"{sym} {iv}m: return={results['total_return_pct']:+.2f}%  "
                       f"Sharpe={results['sharpe']:.3f}  DD={results['max_drawdown_pct']:.1f}%  "
                       f"trades={results['trades']}")

    loser_pairs = [(s, iv) for s in symbols for iv in intervals
                   if (all_results.get((s, iv), {}).get("total_return_pct", -1) <= 0
                       or all_results.get((s, iv), {}).get("oos_return_pct", -1) <= 0
                       or all_results.get((s, iv), {}).get("conservative_return_pct", -1) <= 0)
                   and (s, iv) in ohlcv_data]

    for retry_round in range(1, MAX_RETRIES + 1):
        if not loser_pairs: break
        still_losing = []
        for sym, iv in loser_pairs:
            key = (sym, iv); current_label = f"{sym} {iv}m"
            df  = ohlcv_data[key]; high = df["high"].values; low = df["low"].values
            close = df["close"].values; open_ = df["open"].values
            bt_bars  = _backtest_bars(iv); bpy = _bars_per_year(iv)
            oos_bars = _oos_bars(iv)
            use_wf   = oos_bars < bt_bars
            is_bars  = (bt_bars - oos_bars) if use_wf else bt_bars
            n_is_r   = len(close) - oos_bars if use_wf else len(close)

            # Strict WF: IS-only arrays for retry combo search
            in_demand_r, in_supply_r, demand_bot_r, supply_top_r = _compute_sd_zones(
                open_[:n_is_r], high[:n_is_r], low[:n_is_r], close[:n_is_r], HTF_FACTOR, SD_ZONE_MULT)

            rsi_cache_r   = {rp: _rsi(close[:n_is_r], rp) for rp in range(5, 22)}
            kde_cache_r   = {rp: _kde_scores(rsi_cache_r[rp], KDE_LB) for rp in range(5, 22)}
            chop_cache_r  = {cl_: _chop_index(high[:n_is_r], low[:n_is_r], close[:n_is_r], cl_) for cl_ in range(8, 21)}
            atr_cache_r   = {ap: _atr_wilder(high[:n_is_r], low[:n_is_r], close[:n_is_r], ap) for ap in range(8, 21)}
            swing_cache_r = {sl_: _build_swing_levels(high[:n_is_r], low[:n_is_r], sl_) for sl_ in range(2, 9)}

            rng  = random.Random()
            seen = db_load_all_seen(sym, iv)   # carry full history — retry never repeats prior work
            retry_combos: list = []
            while len(retry_combos) < N_RANDOM:
                combo = {k: (rng.randint(*v) if k in _INT_PARAMS else round(rng.uniform(*v), 2))
                         for k, v in PARAM_SPACE.items()}
                _ck = tuple(combo[k] for k in sorted(PARAM_SPACE))
                if _ck not in seen:
                    seen.add(_ck); retry_combos.append(combo)
            shared_r = (open_[:n_is_r], high[:n_is_r], low[:n_is_r], close[:n_is_r],
                        in_demand_r, in_supply_r, demand_bot_r, supply_top_r,
                        rsi_cache_r, kde_cache_r, chop_cache_r, atr_cache_r, swing_cache_r,
                        is_bars, bpy, 0, INITIAL_EQUITY)  # end_offset=0, initial_equity passed to workers
            batches = [retry_combos[i:i+BATCH_SIZE] for i in range(0, N_RANDOM, BATCH_SIZE)]

            phase_ref[0]  = "retrying"
            status_ref[0] = (f"Retry {retry_round}/{MAX_RETRIES} — {current_label}  "
                             f"{N_RANDOM:,} fresh random…")
            opt_state = {"done":0,"total":N_RANDOM,"candidates":0,"best_sharpe":-99.0,"top_combos":[]}
            _refresh()

            new_cands = []; done_n = 0
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                futs = {executor.submit(_combo_worker_se, (batch, *shared_r)): len(batch)
                        for batch in batches}
                for fut in as_completed(futs):
                    bsz = futs[fut]
                    try:   new_cands.extend(fut.result())
                    except Exception as exc: _log.error(f"retry ({sym} {iv}m): {exc}")
                    done_n += bsz
                    new_cands.sort(key=lambda r: r["score"], reverse=True)
                    opt_state["done"]=done_n; opt_state["total"]=N_RANDOM
                    opt_state["candidates"]=len(new_cands)
                    opt_state["best_sharpe"]=new_cands[0]["score"] if new_cands else -99.0
                    opt_state["top_combos"]=new_cands[:8]
                    status_ref[0] = (f"Retry {retry_round}/{MAX_RETRIES} — {current_label}  "
                                     f"{done_n:,}/{N_RANDOM:,}  Cands:{len(new_cands):,}")
                    _refresh()

            profitable = [r for r in new_cands
                          if r["total_ret_pct"] > 0 and r["score"] > 0
                          and r["max_dd_pct"] > -MAX_DD_PCT * 100]
            if profitable:
                db_save_candidates(sym, iv, new_cands, run_ts)
                best_params = {k: (int(profitable[0][k]) if k in _INT_PARAMS
                                   else float(profitable[0][k])) for k in DEFAULT_PARAMS}
                best_params["leverage"] = LEVERAGE
                with open(os.path.join(DATA_DIR, f"bybit_rsi_reversal_best_params_{sym}_{iv}m.json"), "w") as f:
                    json.dump({"symbol": sym, "interval": iv, "params": best_params,
                               "top_combos": [{k: (int(v) if isinstance(v,(np.integer,)) else
                                                   float(v) if isinstance(v,(np.floating,)) else v)
                                               for k,v in r.items()}
                                              for r in new_cands[:20]]}, f, indent=2)
                phase_ref[0]  = "backtesting"
                status_ref[0] = f"Re-BT {current_label} (retry {retry_round})…"; _refresh()
                results = run_backtest(df.iloc[:n_is_r], sym, iv, best_params,
                                       in_demand_r, in_supply_r, demand_bot_r, supply_top_r,
                                       bt_bars_override=is_bars)
                if use_wf:
                    in_demand_oos_r, in_supply_oos_r, demand_bot_oos_r, supply_top_oos_r = \
                        _compute_sd_zones(open_, high, low, close, HTF_FACTOR, SD_ZONE_MULT)
                    oos_res = run_backtest(df, sym, iv, best_params,
                                           in_demand_oos_r, in_supply_oos_r,
                                           demand_bot_oos_r, supply_top_oos_r,
                                           bt_bars_override=oos_bars)
                    results["oos_return_pct"] = oos_res.get("total_return_pct", -999.0)
                else:
                    results["oos_return_pct"] = results.get("total_return_pct", -999.0)
                cons_res_r = run_backtest(df.iloc[:n_is_r], sym, iv, best_params,
                                          in_demand_r, in_supply_r, demand_bot_r, supply_top_r,
                                          bt_bars_override=is_bars, tp_scale=TP_CONS_SCALE)
                results["conservative_return_pct"] = cons_res_r.get("total_return_pct", -999.0)
                all_results[key] = results
                tl = results.pop("trade_log")
                with open(os.path.join(DATA_DIR, f"bybit_rsi_reversal_results_{sym}_{iv}m.json"), "w") as f:
                    json.dump({**results, "trade_log": tl}, f, indent=2)
                results["trade_log"] = tl
                _log.info(f"{sym} {iv}m RECOVERED retry {retry_round} (OOS={results['oos_return_pct']:+.2f}%)")
            else:
                still_losing.append((sym, iv))
        loser_pairs = still_losing

    # All symbols always remain in scope — never eliminate tokens from backtesting
    loser_syms_all = [s for s in symbols
                      if any((s, iv) in ohlcv_data for iv in intervals)
                      and all((s, iv) not in ohlcv_data or (s, iv) in loser_pairs
                              for iv in intervals)]
    for sym in loser_syms_all:
        _log.warning(f"{sym}: all intervals failed this run — excluded from live params this cycle")

    live_params: dict = {}; excluded = []
    for sym in list(SYMBOLS):
        for iv in intervals:
            r = all_results.get((sym, iv))
            if not r: continue
            oos_r      = r.get("oos_return_pct", -999.0)
            cons_r     = r.get("conservative_return_pct", -999.0)
            wr         = float(r.get("win_rate", 0))
            n_trades   = int(r.get("trades", 0))
            chop_thr   = float(r.get("params", {}).get("chop_thr", 0))
            ret_per_t  = r["total_return_pct"] / max(n_trades, 1)
            ret_ok     = r["total_return_pct"] >= MIN_LIVE_RET or wr == 1.0
            quality    = r["sharpe"] >= MIN_LIVE_SHARPE and oos_r > 0 and wr >= 0.60
            trade_qual = ret_per_t >= MIN_LIVE_RETURN_PT
            chop_ok    = chop_thr >= MIN_LIVE_CHOP_THR
            cons_ok    = cons_r > 0   # must be profitable even at 70% of optimised TP
            if ret_ok and quality and trade_qual and chop_ok and cons_ok:
                live_params.setdefault(sym, {})[iv] = r["params"]
            else:
                reasons = []
                if not ret_ok:     reasons.append(f"ret={r['total_return_pct']:+.1f}%<{MIN_LIVE_RET:.0f}%,WR≠100%")
                if not quality:    reasons.append(f"sharpe={r['sharpe']:+.2f}/OOS={oos_r:+.1f}%/WR={wr*100:.0f}%")
                if not trade_qual: reasons.append(f"ret/trade={ret_per_t:+.1f}%<{MIN_LIVE_RETURN_PT:.0f}%")
                if not chop_ok:    reasons.append(f"chop_thr={chop_thr:.1f}<{MIN_LIVE_CHOP_THR:.0f}")
                if not cons_ok:    reasons.append(f"cons_ret={cons_r:+.1f}%≤0 (TP×{TP_CONS_SCALE})")
                excluded.append(f"{sym} {iv}m ({'; '.join(reasons)})")

    with open(os.path.join(DATA_DIR, "bybit_rsi_reversal_live_params.json"), "w") as f:
        json.dump({"generated": run_ts, "loop": loop_count, "leverage": LEVERAGE,
                   "taker_fee": TAKER_FEE, "htf_factor": HTF_FACTOR,
                   "sd_zone_mult": SD_ZONE_MULT, "kde_lb": KDE_LB,
                   "excluded": excluded, "symbols": live_params}, f, indent=2)
    _log.info(f"Live params saved  ({sum(len(v) for v in live_params.values())} tradeable pairs)")

    phase_ref[0]    = "done"; progress_ref[0] = 1.0
    n_live          = sum(len(v) for v in live_params.values())
    status_ref[0]   = (f"Loop #{loop_count} complete — {n_live} tradeable pairs"
                       + (f"  |  excluded: {len(excluded)}" if excluded else "")
                       + "  |  → data/bybit_rsi_reversal_live_params.json")
    opt_state = {"done":0,"total":0,"candidates":0,"best_sharpe":-99.0,"top_combos":[]}
    _refresh(); time.sleep(1)
    return all_results, live_params


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    import traceback
    multiprocessing.freeze_support()

    def _excepthook(et, ev, tb):
        t = "".join(traceback.format_exception(et, ev, tb))
        _log.critical(f"Uncaught:\n{t}"); _write_crash(t)
    sys.excepthook = _excepthook

    _log.info("=" * 60)
    _log.info("RSI Kernel+SD BT — starting")
    _log.info(f"  Symbols: {SYMBOLS}  Intervals: {INTERVALS}m  Lev: {LEVERAGE}x")
    _log.info(f"  Signal: RSI crossover + KDE(lb={KDE_LB}) | Flux S/D {HTF_FACTOR}x imp>{SD_ZONE_MULT:.0%}")
    _log.info(f"  SL: zone distal line (demand_bot / supply_top) — no ATR buffer")
    _log.info(f"  TP: 70% of distance to next confirmed swing level (swing_len searched)")
    _log.info(f"  MaxDD filter: <{MAX_DD_PCT*100:.0f}%")
    _log.info("=" * 60)

    db_init(); session = make_session(); loop_count = 0
    last_results: dict = {}; last_live: dict = {}

    try:
        _init_label = f"{SYMBOLS[0]} {INTERVALS[0]}m" if SYMBOLS and INTERVALS else "—"
        with Live(build_tui("starting", 0.0, "Initialising…",
                            _init_label, {},
                            "", None, loop_count=0),
                  console=_console, refresh_per_second=4, screen=False) as live:
            while True:
                loop_count += 1
                try:
                    last_results, last_live = _run_pass(session, loop_count, live)
                except Exception:
                    tb = traceback.format_exc()
                    _log.critical(f"_run_pass crashed (loop {loop_count}):\n{tb}")
                    _write_crash(tb)

                t_next = time.time() + LOOP_INTERVAL
                while time.time() < t_next:
                    remaining = int(t_next - time.time())
                    try:
                        live.update(build_tui("waiting", 1.0,
                                              f"Loop #{loop_count} complete",
                                              "", last_results, "",
                                              {"done":0,"total":0,"candidates":0,
                                               "best_sharpe":-99.0,"top_combos":[]},
                                              loop_count=loop_count,
                                              next_run_in=remaining,
                                              symbols=list(SYMBOLS),
                                              intervals=list(INTERVALS)))
                    except Exception:
                        pass
                    time.sleep(5)
    except KeyboardInterrupt:
        _log.info("Interrupted by user")
    except Exception:
        tb = traceback.format_exc()
        _log.critical(f"Fatal:\n{tb}"); _write_crash(tb)


if __name__ == "__main__":
    main()


