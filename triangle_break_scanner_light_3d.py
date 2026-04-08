import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

TIMEFRAME = "3d"
TOP_N = 300
KLINES_LIMIT = 260
REQUEST_SLEEP = 0.03
ONLY_USDT = True

EXCLUDE_SYMBOLS = {
    "BTCUSDT", "WBTCUSDT", "PAXGUSDT", "ETHUSDT", "WETHUSDT"
}

lb1 = 30
lb2 = 90
lb3 = 100
volMulti = 1.0
volLen = 20
coolBars = 100
retestBars = 10
minTouches = 2
maxNarrow = 70.0
minNarrow = 0.0
barThresh = 300

minTriH_old = 0.5
touchTol_old = 0.03
minTriH_new = 0.8
touchTol_new = 0.05

useSMAf_old = False
smaFlen_old = 30
useSMAf_new = True
smaFlen_new = 30

BASE_URL = "https://api.binance.com"


def get_json(url: str, params: Optional[dict] = None):
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_exchange_info():
    return get_json(f"{BASE_URL}/api/v3/exchangeInfo")


def fetch_tickers_24h():
    return get_json(f"{BASE_URL}/api/v3/ticker/24hr")


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    raw = get_json(
        f"{BASE_URL}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]]


def get_top_symbols(top_n: int) -> List[str]:
    info = fetch_exchange_info()
    tickers = fetch_tickers_24h()

    tradable = {}
    for s in info["symbols"]:
        if s.get("status") != "TRADING":
            continue
        if s.get("isSpotTradingAllowed") is not True:
            continue
        symbol = s["symbol"]
        quote = s.get("quoteAsset", "")
        base = s.get("baseAsset", "")
        if ONLY_USDT and quote != "USDT":
            continue
        if symbol in EXCLUDE_SYMBOLS:
            continue
        if any(x in base for x in ["UP", "DOWN", "BULL", "BEAR"]):
            continue
        if base in ["BTC", "WBTC", "ETH", "WETH", "PAXG"]:
            continue
        tradable[symbol] = True

    rows = []
    for t in tickers:
        symbol = t["symbol"]
        if symbol not in tradable:
            continue
        try:
            qv = float(t.get("quoteVolume", 0.0))
            last_price = float(t.get("lastPrice", 0.0))
        except Exception:
            continue
        if qv > 0 and last_price > 0:
            rows.append((symbol, qv))

    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:top_n]]


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) < n:
        return out
    c = np.cumsum(np.insert(a, 0, 0.0))
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return sma(tr, n)


def highest_last(arr: np.ndarray, length: int, end_idx: int) -> float:
    if end_idx < 0:
        return np.nan
    start = max(0, end_idx - length + 1)
    return float(np.max(arr[start:end_idx + 1]))


def lowest_last(arr: np.ndarray, length: int, end_idx: int) -> float:
    if end_idx < 0:
        return np.nan
    start = max(0, end_idx - length + 1)
    return float(np.min(arr[start:end_idx + 1]))


def check_tri_at_idx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    idx: int,
    lb: int,
    is_new_coin: bool,
    above_avg: bool,
) -> Tuple[bool, Optional[float]]:
    if idx < lb * 3:
        return False, None

    t = max(lb // 3, 3)

    h1 = highest_last(high, t, idx)
    h2 = highest_last(high, t, idx - t)
    h3 = highest_last(high, t, idx - 2 * t)

    l1 = lowest_last(low, t, idx)
    l2 = lowest_last(low, t, idx - t)
    l3 = lowest_last(low, t, idx - 2 * t)

    atr_value = atr[idx]
    if np.isnan(atr_value):
        return False, None

    dH = (h1 < h2) or (h1 < h3)
    dH2 = h1 < h3
    aL = (l1 > l2) or (l1 > l3)

    flat_mult = 1.5 if is_new_coin else (1.5 if lb <= 60 else 4.0)
    fL = abs(l1 - l3) < atr_value * flat_mult

    r_now = h1 - l1
    r_old = h3 - l3
    n_pct = ((r_old - r_now) / r_old * 100.0) if r_old != 0 else 0.0

    if is_new_coin:
        conv = minNarrow <= n_pct <= maxNarrow
    else:
        conv = (minNarrow <= n_pct <= maxNarrow) if lb <= 60 else True

    sym = dH and aL and conv
    desc = dH2 and fL and conv
    ok = sym or desc

    top = max(h1, h2, h3)
    bot = min(l1, l2, l3)
    tri_h = top - bot

    min_h = minTriH_new if is_new_coin else minTriH_old
    big = tri_h > atr_value * min_h

    sup = min(l1, l2, l3)
    tol_base = touchTol_new if is_new_coin else touchTol_old
    tol = sup * tol_base if is_new_coin else sup * tol_base * max(lb / 50.0, 1.0)

    tch = 0
    start = max(0, idx - lb + 1)
    for j in range(start, idx + 1):
        if abs(low[j] - sup) <= tol:
            tch += 1

    valid = ok and big and (tch >= minTouches) and above_avg
    if idx < 1:
        return False, None

    prev_h1 = highest_last(high, t, idx - 1)
    prev_h2 = highest_last(high, t, idx - 1 - t)

    brk1 = valid and close[idx] > h1 and close[idx - 1] <= prev_h1
    brk2 = valid and close[idx] > h2 and close[idx - 1] <= prev_h2 and not brk1

    return (brk1 or brk2), h1


def analyze_symbol(df: pd.DataFrame):
    n = len(df)
    if n < 120:
        return None

    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    atr = atr_np(high, low, close, 14)
    vol_ma = sma(volume, volLen)
    sma_old = sma(close, smaFlen_old)
    sma_new = sma(close, smaFlen_new)

    cnt = 999
    pending = False
    pending_bar = 0
    pending_low = np.nan
    pending_hh = np.nan

    retest_pending = False
    retest_start = 0
    retest_level = np.nan
    retest_dipped = False

    last_signal = None
    last_breakout = None
    last_signal_idx = None

    for idx in range(n):
        is_new_coin = idx < barThresh

        if is_new_coin:
            above_avg = close[idx] > sma_new[idx] if (useSMAf_new and not np.isnan(sma_new[idx])) else True
        else:
            above_avg = close[idx] > sma_old[idx] if (useSMAf_old and not np.isnan(sma_old[idx])) else True

        short_break, short_hh = check_tri_at_idx(high, low, close, atr, idx, lb1, is_new_coin, above_avg)
        mid_break, mid_hh = check_tri_at_idx(high, low, close, atr, idx, lb2, is_new_coin, above_avg)
        long_break, long_hh = check_tri_at_idx(high, low, close, atr, idx, lb3, is_new_coin, above_avg)

        vol_ok = ((volume[idx] > vol_ma[idx] * volMulti) if not np.isnan(vol_ma[idx]) else False) or (volMulti <= 1.0)

        s_break = short_break and vol_ok
        m_break = mid_break and vol_ok and not s_break
        l_break = long_break and vol_ok and not s_break and not m_break

        cnt += 1

        mid_direct = m_break and cnt >= coolBars
        long_direct = l_break and cnt >= coolBars and not mid_direct

        if mid_direct or long_direct:
            cnt = 0

        do_retest = True if is_new_coin else False

        if s_break and not pending:
            pending = True
            pending_bar = idx
            pending_low = low[idx]
            pending_hh = short_hh if short_hh is not None else np.nan

        normal_confirm = False
        if pending and (not do_retest) and (idx - pending_bar) >= 2:
            stayed_up = low[idx - 1] >= pending_low and low[idx] >= pending_low
            any_green = (close[idx - 1] > open_[idx - 1]) or (close[idx] > open_[idx])
            if stayed_up and any_green:
                normal_confirm = True
            pending = False

        if pending and do_retest and (idx - pending_bar) >= 2:
            retest_pending = True
            retest_start = idx
            retest_level = pending_hh
            retest_dipped = False
            pending = False

        retest_confirm = False
        if retest_pending and not np.isnan(retest_level):
            if low[idx] <= retest_level * 1.02:
                retest_dipped = True
            if retest_dipped and close[idx] > retest_level and close[idx] > open_[idx]:
                retest_confirm = True
                retest_pending = False
            if (idx - retest_start) >= retestBars:
                if (not retest_dipped) and close[idx] > retest_level:
                    retest_confirm = True
                retest_pending = False
            if close[idx] < retest_level * 0.90:
                retest_pending = False

        short_confirmed = (normal_confirm or retest_confirm) and cnt >= coolBars
        if short_confirmed:
            cnt = 0

        final_break = short_confirmed or mid_direct or long_direct
        if final_break:
            if short_confirmed:
                last_signal = "AL 30"
                last_breakout = short_hh
                last_signal_idx = idx
            elif mid_direct:
                last_signal = "AL 90"
                last_breakout = mid_hh
                last_signal_idx = idx
            elif long_direct:
                last_signal = "AL 100"
                last_breakout = long_hh
                last_signal_idx = idx

    if last_signal is None or last_signal_idx is None:
        return None

    if last_signal_idx not in [n - 1, n - 2]:
        return None

    return {
        "symbol": "",
        "signal_type": last_signal,
        "close": float(close[-1]),
        "breakout_level": float(last_breakout) if last_breakout is not None and not np.isnan(last_breakout) else np.nan,
        "bars": n,
        "signal_bar": int(last_signal_idx),
    }


def scan_market():
    symbols = get_top_symbols(TOP_N)
    print(f"Scanning {len(symbols)} altcoin symbols on {TIMEFRAME} ...")

    rows = []
    for i, symbol in enumerate(symbols, 1):
        try:
            df = fetch_klines(symbol, TIMEFRAME, KLINES_LIMIT)
            result = analyze_symbol(df)
            if result:
                result["symbol"] = symbol
                rows.append(result)
                print(f"[{i}/{len(symbols)}] {symbol} -> {result['signal_type']} close={result['close']:.6f}")
        except Exception as e:
            print(f"[{i}/{len(symbols)}] {symbol} -> ERROR: {e}")
        time.sleep(REQUEST_SLEEP)

    if not rows:
        print("\nNo active signal found in the last 2 closed 3D bars.")
        return

    out = pd.DataFrame(rows)[["symbol", "signal_type", "close", "breakout_level", "bars", "signal_bar"]]
    rank = {"AL 30": 1, "AL 90": 2, "AL 100": 3}
    out["rank"] = out["signal_type"].map(rank)
    out = out.sort_values(["rank", "symbol"]).drop(columns=["rank"])

    print("\n=== SIGNALS (LAST 2 BARS ONLY) ===")
    print(out.to_string(index=False))

    out.to_csv("triangle_break_scan_results_light.csv", index=False)
    print("\nSaved: triangle_break_scan_results_light.csv")


if __name__ == "__main__":
    scan_market()
