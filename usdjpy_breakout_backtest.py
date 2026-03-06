from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np


# =========================
# Indicators
# =========================
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).rolling(n, min_periods=n).mean()


# =========================
# Load CSV
# =========================
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map = {
        # English
        "Time": "time",
        "Date": "time",
        "Datetime": "time",
        "Local time": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        # Japanese
        "チャート日時": "time",
        "始値": "open",
        "高値": "high",
        "安値": "low",
        "終値": "close",
        "出来高": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"必要列が不足しています: {missing}")

    # Dukascopy系の "03.10.2025 00:00:00.000 GMT+0900" 形式に対応
    if df["time"].astype(str).str.contains("GMT", na=False).any():
        df["time"] = df["time"].astype(str).str.replace(" GMT+0900", "", regex=False)
        df["time"] = pd.to_datetime(
            df["time"],
            format="%d.%m.%Y %H:%M:%S.%f",
            errors="coerce",
        )
        df["time"] = df["time"].dt.tz_localize("Asia/Tokyo")
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("time").dropna(subset=required).reset_index(drop=True)
    return df


# =========================
# Helper
# =========================
def allowed_session(ts: pd.Timestamp) -> bool:
    hour = ts.hour

    # 東京 8-11
    if 8 <= hour <= 11:
        return True

    # 欧米時間 15-23
    if 15 <= hour <= 23:
        return True

    return False


def is_tokyo_session(ts: pd.Timestamp) -> bool:
    return 8 <= ts.hour <= 11


def get_h1_bias(df_h1: Optional[pd.DataFrame], t: pd.Timestamp) -> str:
    if df_h1 is None or len(df_h1) < 60:
        return "NEUTRAL"

    d = df_h1[df_h1["time"] <= t]
    if len(d) < 60:
        return "NEUTRAL"

    last = d.iloc[-1]

    if pd.isna(last["ema20"]) or pd.isna(last["ema50"]):
        return "NEUTRAL"

    if last["close"] > last["ema20"] > last["ema50"]:
        return "UP"
    if last["close"] < last["ema20"] < last["ema50"]:
        return "DOWN"
    return "NEUTRAL"


def count_touches(window: pd.DataFrame, level: float, atr_now: float, side: str, tol_ratio: float = 0.15) -> int:
    tol = max(atr_now * tol_ratio, 0.03)

    if side == "high":
        hits = window["high"] >= (level - tol)
    else:
        hits = window["low"] <= (level + tol)

    count = 0
    prev_hit = False
    for v in hits.to_numpy():
        if bool(v) and not prev_hit:
            count += 1
        prev_hit = bool(v)
    return count


def detect_structure(window: pd.DataFrame) -> tuple[bool, bool]:
    tail = window.tail(7)
    if len(tail) < 7:
        return False, False

    lows = tail["low"].to_numpy()
    highs = tail["high"].to_numpy()

    higher_lows = lows[-1] > lows[-3] > lows[-5]
    lower_highs = highs[-1] < highs[-3] < highs[-5]
    return bool(higher_lows), bool(lower_highs)


def candle_quality(row: pd.Series) -> tuple[float, float, float]:
    body = abs(float(row["close"]) - float(row["open"]))
    full = max(float(row["high"]) - float(row["low"]), 1e-9)
    upper_wick = float(row["high"]) - max(float(row["open"]), float(row["close"]))
    lower_wick = min(float(row["open"]), float(row["close"])) - float(row["low"])
    return body / full, upper_wick / full, lower_wick / full


# =========================
# Signal / Trade
# =========================
@dataclass
class Signal:
    time: pd.Timestamp
    direction: str
    score: int
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    range_high: float
    range_low: float
    range_width_pips: float
    touches_high: int
    touches_low: int


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry: float
    exit: float
    sl: float
    tp: float
    result_pips: float
    reason: str
    score: int
    tp_mode: str
    range_width_pips: float
    touches_high: int
    touches_low: int


# =========================
# Signal Logic
# =========================
def analyze_at(
    df_m15: pd.DataFrame,
    df_h1: Optional[pd.DataFrame],
    i: int,
    lookback: int,
    min_touches: int = 3,
    min_range_pips: float = 15.0,
    max_range_pips: float = 70.0,
) -> Optional[Signal]:
    if i < max(lookback + 2, 60):
        return None

    sub = df_m15.iloc[: i + 1]
    last = sub.iloc[-1]

    atr_now = float(last["atr14"])
    if np.isnan(atr_now):
        return None

    # 現在足を除外した直前レンジ
    window = sub.iloc[-(lookback + 1):-1].copy()
    if len(window) < lookback:
        return None

    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    range_width = range_high - range_low
    range_width_pips = range_width / 0.01

    # レンジ幅制限 15〜70pips
    if not (min_range_pips <= range_width_pips <= max_range_pips):
        return None

    current_close = float(last["close"])
    body_ratio, upper_ratio, lower_ratio = candle_quality(last)

    touches_high = count_touches(window, range_high, atr_now, "high")
    touches_low = count_touches(window, range_low, atr_now, "low")
    higher_lows, lower_highs = detect_structure(window)
    bias_h1 = get_h1_bias(df_h1, pd.Timestamp(last["time"]))

    broke_up_close = current_close > range_high
    broke_dn_close = current_close < range_low

    # Long
    if broke_up_close:
        if touches_high < min_touches:
            return None
        if not higher_lows:
            return None
        if upper_ratio > 0.25:
            return None
        if body_ratio < 0.45:
            return None
        if bias_h1 != "UP":
            return None

        score = 0
        score += 25  # breakout
        score += min(touches_high * 5, 30)
        score += 15  # structure
        score += 10  # candle quality
        score += 10 if current_close > float(last["ema20"]) else 0

        entry = current_close

        # SL: レンジ下限 - 5pips、ただし最大22pips
        sl_raw = range_low - 0.05
        sl = max(sl_raw, entry - 0.22)

        risk = entry - sl
        if risk <= 0:
            return None

        # TP: 20 / 30 / 50 pips
        tp1 = entry + 0.20
        tp2 = entry + 0.30
        tp3 = entry + 0.50

        return Signal(
            time=pd.Timestamp(last["time"]),
            direction="UP",
            score=int(score),
            entry=round(entry, 3),
            sl=round(sl, 3),
            tp1=round(tp1, 3),
            tp2=round(tp2, 3),
            tp3=round(tp3, 3),
            range_high=round(range_high, 3),
            range_low=round(range_low, 3),
            range_width_pips=round(range_width_pips, 1),
            touches_high=touches_high,
            touches_low=touches_low,
        )

    # Short
    if broke_dn_close:
        if touches_low < min_touches:
            return None
        if not lower_highs:
            return None
        if lower_ratio > 0.25:
            return None
        if body_ratio < 0.45:
            return None
        if bias_h1 != "DOWN":
            return None

        score = 0
        score += 25  # breakout
        score += min(touches_low * 5, 30)
        score += 15  # structure
        score += 10  # candle quality
        score += 10 if current_close < float(last["ema20"]) else 0

        entry = current_close

        # SL: レンジ上限 + 5pips、ただし最大22pips
        sl_raw = range_high + 0.05
        sl = min(sl_raw, entry + 0.22)

        risk = sl - entry
        if risk <= 0:
            return None

        # TP: 20 / 30 / 50 pips
        tp1 = entry - 0.20
        tp2 = entry - 0.30
        tp3 = entry - 0.50

        return Signal(
            time=pd.Timestamp(last["time"]),
            direction="DOWN",
            score=int(score),
            entry=round(entry, 3),
            sl=round(sl, 3),
            tp1=round(tp1, 3),
            tp2=round(tp2, 3),
            tp3=round(tp3, 3),
            range_high=round(range_high, 3),
            range_low=round(range_low, 3),
            range_width_pips=round(range_width_pips, 1),
            touches_high=touches_high,
            touches_low=touches_low,
        )

    return None


# =========================
# Backtest Helpers
# =========================
def choose_tp_mode(score: int) -> Optional[str]:
    if score >= 85:
        return "TP3"   # 50pips
    if score >= 70:
        return "TP2"   # 30pips
    if score >= 60:
        return "TP1"   # 20pips
    return None


def initial_followthrough_ok(df_m15: pd.DataFrame, start_i: int, sig: Signal) -> bool:
    if start_i + 1 >= len(df_m15):
        return False

    nxt = df_m15.iloc[start_i + 1]
    next_close = float(nxt["close"])

    # 東京時間は3pips以上の追随を要求
    if is_tokyo_session(sig.time):
        if sig.direction == "UP":
            return next_close >= sig.entry + 0.03
        return next_close <= sig.entry - 0.03

    # それ以外は従来どおり、レンジ外維持を確認
    if sig.direction == "UP":
        return next_close >= sig.range_high
    return next_close <= sig.range_low


def simulate_trade(df_m15: pd.DataFrame, start_i: int, sig: Signal, tp_mode: str) -> Optional[Trade]:
    entry = float(sig.entry)
    sl = float(sig.sl)

    if tp_mode == "TP3":
        tp = float(sig.tp3)
    elif tp_mode == "TP2":
        tp = float(sig.tp2)
    else:
        tp = float(sig.tp1)

    for j in range(start_i + 1, len(df_m15)):
        r = df_m15.iloc[j]
        t = pd.Timestamp(r["time"])
        hi = float(r["high"])
        lo = float(r["low"])

        # 同一バーでTP/SL両方に触れた場合は保守的にSL優先
        if sig.direction == "UP":
            hit_sl = lo <= sl
            hit_tp = hi >= tp

            if hit_sl:
                exit_price = sl
                pips = (exit_price - entry) / 0.01
                return Trade(
                    sig.time, t, "UP", entry, exit_price, sl, tp, round(pips, 1),
                    "SL", sig.score, tp_mode, sig.range_width_pips, sig.touches_high, sig.touches_low
                )
            if hit_tp:
                exit_price = tp
                pips = (exit_price - entry) / 0.01
                return Trade(
                    sig.time, t, "UP", entry, exit_price, sl, tp, round(pips, 1),
                    "TP", sig.score, tp_mode, sig.range_width_pips, sig.touches_high, sig.touches_low
                )

        else:
            hit_sl = hi >= sl
            hit_tp = lo <= tp

            if hit_sl:
                exit_price = sl
                pips = (entry - exit_price) / 0.01
                return Trade(
                    sig.time, t, "DOWN", entry, exit_price, sl, tp, round(pips, 1),
                    "SL", sig.score, tp_mode, sig.range_width_pips, sig.touches_high, sig.touches_low
                )
            if hit_tp:
                exit_price = tp
                pips = (entry - exit_price) / 0.01
                return Trade(
                    sig.time, t, "DOWN", entry, exit_price, sl, tp, round(pips, 1),
                    "TP", sig.score, tp_mode, sig.range_width_pips, sig.touches_high, sig.touches_low
                )

    return None


# =========================
# Backtest
# =========================
def backtest(
    df_m15: pd.DataFrame,
    df_h1: Optional[pd.DataFrame],
    lookback: int = 20,
    cooldown_bars: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    trades: List[Trade] = []

    i = 0
    next_allowed_i = 0

    while i < len(df_m15):
        if i < next_allowed_i:
            i += 1
            continue

        current_time = pd.Timestamp(df_m15.iloc[i]["time"])
        if not allowed_session(current_time):
            i += 1
            continue

        sig = analyze_at(df_m15, df_h1, i, lookback=lookback)
        if sig is None:
            i += 1
            continue

        tp_mode = choose_tp_mode(sig.score)
        if tp_mode is None:
            i += 1
            continue

        if not initial_followthrough_ok(df_m15, i, sig):
            i += 1
            continue

        tr = simulate_trade(df_m15, i, sig, tp_mode)
        if tr is not None:
            trades.append(tr)
            exit_idx_list = df_m15.index[df_m15["time"] == tr.exit_time].tolist()
            exit_i = exit_idx_list[0] if exit_idx_list else i + 1
            next_allowed_i = exit_i + cooldown_bars
            i = exit_i + 1
        else:
            i += 1

    df_tr = pd.DataFrame([t.__dict__ for t in trades])

    if df_tr.empty:
        summary = {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_pips": 0.0,
            "max_drawdown_pips": 0.0,
            "max_consecutive_losses": 0,
        }
        return df_tr, summary

    wins = df_tr[df_tr["result_pips"] > 0]["result_pips"].sum()
    losses = -df_tr[df_tr["result_pips"] < 0]["result_pips"].sum()
    pf = (wins / losses) if losses > 0 else float("inf")

    win_rate = (df_tr["result_pips"] > 0).mean() * 100.0
    avg_pips = df_tr["result_pips"].mean()

    equity = df_tr["result_pips"].cumsum()
    peak = equity.cummax()
    dd = peak - equity
    max_dd = dd.max()

    loss_flags = (df_tr["result_pips"] < 0).to_numpy()
    max_losses = 0
    cur = 0
    for f in loss_flags:
        if f:
            cur += 1
            max_losses = max(max_losses, cur)
        else:
            cur = 0

    summary = {
        "trades": int(len(df_tr)),
        "win_rate": round(float(win_rate), 1),
        "profit_factor": round(float(pf), 2) if pf != float("inf") else "INF",
        "avg_pips": round(float(avg_pips), 2),
        "max_drawdown_pips": round(float(max_dd), 1),
        "max_consecutive_losses": int(max_losses),
    }
    return df_tr, summary


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m15", required=True, help="15分足CSV")
    ap.add_argument("--h1", default=None, help="1時間足CSV")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--cooldown", type=int, default=8)
    ap.add_argument("--out", default="trades_v15.csv")
    args = ap.parse_args()

    df_m15 = load_csv(args.m15)
    df_h1 = load_csv(args.h1) if args.h1 else None

    df_m15["atr14"] = atr(df_m15, 14)
    df_m15["ema20"] = ema(df_m15["close"], 20)

    if df_h1 is not None:
        df_h1["ema20"] = ema(df_h1["close"], 20)
        df_h1["ema50"] = ema(df_h1["close"], 50)

    df_tr, summary = backtest(
        df_m15=df_m15,
        df_h1=df_h1,
        lookback=args.lookback,
        cooldown_bars=args.cooldown,
    )

    print("=== Backtest Summary (v1.5 session+followthrough+TP20/30/50) ===")
    for k, v in summary.items():
        print(f"{k:22s}: {v}")

    df_tr.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nTrades saved: {args.out}")

    if not df_tr.empty:
        print("\nLast 10 trades:")
        cols = [
            "entry_time",
            "exit_time",
            "direction",
            "entry",
            "exit",
            "result_pips",
            "score",
            "tp_mode",
            "range_width_pips",
            "touches_high",
            "touches_low",
        ]
        print(df_tr.tail(10)[cols].to_string(index=False))
    else:
        print("\nNo trades found.")


if __name__ == "__main__":
    main()