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


def count_touches(
    window: pd.DataFrame,
    level: float,
    atr_now: float,
    side: str,
    tol_ratio: float = 0.15
) -> int:
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
    h1_bias: str
    higher_lows: bool
    lower_highs: bool
    body_ratio: float
    upper_ratio: float
    lower_ratio: float


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
    session: str
    entry_hour: int
    weekday: str
    h1_bias: str
    higher_lows: bool
    lower_highs: bool
    body_ratio: float
    upper_ratio: float
    lower_ratio: float


# =========================
# Signal Logic
# =========================
def analyze_at(
    df_m15: pd.DataFrame,
    df_h1: Optional[pd.DataFrame],
    i: int,
    lookback: int,
    min_touches_break_side: int = 2,
    min_range_pips: float = 15.0,
    max_range_pips: float = 70.0,
    require_structure: bool = False,
    require_h1_bias: bool = False,
    use_candle_filter: bool = False,
) -> Optional[Signal]:
    if i < max(lookback + 2, 20):
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
        if touches_high < min_touches_break_side:
            return None

        if require_structure and not higher_lows:
            return None

        if require_h1_bias and bias_h1 != "UP":
            return None

        if use_candle_filter:
            if upper_ratio > 0.35:
                return None
            if body_ratio < 0.30:
                return None

        score = 0
        score += 30  # breakout close
        score += min(touches_high * 8, 24)  # 触った回数
        score += 12 if higher_lows else 0
        score += 8 if bias_h1 == "UP" else 0
        score += 5 if current_close > float(last["ema20"]) else 0
        score += 5 if body_ratio >= 0.50 else 0

        entry = current_close

        # SL: レンジ下限 - 5pips、ただし最大22pips
        sl_raw = range_low - 0.05
        sl = max(sl_raw, entry - 0.22)

        risk = entry - sl
        if risk <= 0:
            return None

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
            h1_bias=bias_h1,
            higher_lows=higher_lows,
            lower_highs=lower_highs,
            body_ratio=round(body_ratio, 4),
            upper_ratio=round(upper_ratio, 4),
            lower_ratio=round(lower_ratio, 4),
        )

    # Short
    if broke_dn_close:
        if touches_low < min_touches_break_side:
            return None

        if require_structure and not lower_highs:
            return None

        if require_h1_bias and bias_h1 != "DOWN":
            return None

        if use_candle_filter:
            if lower_ratio > 0.35:
                return None
            if body_ratio < 0.30:
                return None

        score = 0
        score += 30  # breakout close
        score += min(touches_low * 8, 24)
        score += 12 if lower_highs else 0
        score += 8 if bias_h1 == "DOWN" else 0
        score += 5 if current_close < float(last["ema20"]) else 0
        score += 5 if body_ratio >= 0.50 else 0

        entry = current_close

        # SL: レンジ上限 + 5pips、ただし最大22pips
        sl_raw = range_high + 0.05
        sl = min(sl_raw, entry + 0.22)

        risk = sl - entry
        if risk <= 0:
            return None

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
            h1_bias=bias_h1,
            higher_lows=higher_lows,
            lower_highs=lower_highs,
            body_ratio=round(body_ratio, 4),
            upper_ratio=round(upper_ratio, 4),
            lower_ratio=round(lower_ratio, 4),
        )

    return None


# =========================
# Backtest Helpers
# =========================
def choose_tp_mode(score: int) -> str:
    # v1.5a: 厳しすぎてTPモード未選択にならないよう閾値を緩和
    if score >= 65:
        return "TP3"   # 50pips
    if score >= 50:
        return "TP2"   # 30pips
    return "TP1"       # 20pips


def initial_followthrough_ok(
    df_m15: pd.DataFrame,
    start_i: int,
    sig: Signal,
    tokyo_follow_pips: float = 3.0,
    west_hold_break: bool = True,
) -> bool:
    if start_i + 1 >= len(df_m15):
        return False

    nxt = df_m15.iloc[start_i + 1]
    next_close = float(nxt["close"])

    # 東京時間は3pips以上の追随を要求
    if is_tokyo_session(sig.time):
        follow = tokyo_follow_pips * 0.01
        if sig.direction == "UP":
            return next_close >= sig.entry + follow
        return next_close <= sig.entry - follow

    # 欧米は「次足終値がブレイク水準の外に残る」だけ確認
    if west_hold_break:
        if sig.direction == "UP":
            return next_close >= sig.range_high
        return next_close <= sig.range_low

    return True


def session_name_from_time(ts: pd.Timestamp) -> str:
    h = ts.hour
    if 8 <= h <= 11:
        return "Tokyo"
    if 15 <= h <= 23:
        return "West"
    return "Other"


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
                    entry_time=sig.time,
                    exit_time=t,
                    direction="UP",
                    entry=entry,
                    exit=exit_price,
                    sl=sl,
                    tp=tp,
                    result_pips=round(pips, 1),
                    reason="SL",
                    score=sig.score,
                    tp_mode=tp_mode,
                    range_width_pips=sig.range_width_pips,
                    touches_high=sig.touches_high,
                    touches_low=sig.touches_low,
                    session=session_name_from_time(sig.time),
                    entry_hour=int(sig.time.hour),
                    weekday=sig.time.day_name(),
                    h1_bias=sig.h1_bias,
                    higher_lows=sig.higher_lows,
                    lower_highs=sig.lower_highs,
                    body_ratio=sig.body_ratio,
                    upper_ratio=sig.upper_ratio,
                    lower_ratio=sig.lower_ratio,
                )

            if hit_tp:
                exit_price = tp
                pips = (exit_price - entry) / 0.01
                return Trade(
                    entry_time=sig.time,
                    exit_time=t,
                    direction="UP",
                    entry=entry,
                    exit=exit_price,
                    sl=sl,
                    tp=tp,
                    result_pips=round(pips, 1),
                    reason="TP",
                    score=sig.score,
                    tp_mode=tp_mode,
                    range_width_pips=sig.range_width_pips,
                    touches_high=sig.touches_high,
                    touches_low=sig.touches_low,
                    session=session_name_from_time(sig.time),
                    entry_hour=int(sig.time.hour),
                    weekday=sig.time.day_name(),
                    h1_bias=sig.h1_bias,
                    higher_lows=sig.higher_lows,
                    lower_highs=sig.lower_highs,
                    body_ratio=sig.body_ratio,
                    upper_ratio=sig.upper_ratio,
                    lower_ratio=sig.lower_ratio,
                )

        else:
            hit_sl = hi >= sl
            hit_tp = lo <= tp

            if hit_sl:
                exit_price = sl
                pips = (entry - exit_price) / 0.01
                return Trade(
                    entry_time=sig.time,
                    exit_time=t,
                    direction="DOWN",
                    entry=entry,
                    exit=exit_price,
                    sl=sl,
                    tp=tp,
                    result_pips=round(pips, 1),
                    reason="SL",
                    score=sig.score,
                    tp_mode=tp_mode,
                    range_width_pips=sig.range_width_pips,
                    touches_high=sig.touches_high,
                    touches_low=sig.touches_low,
                    session=session_name_from_time(sig.time),
                    entry_hour=int(sig.time.hour),
                    weekday=sig.time.day_name(),
                    h1_bias=sig.h1_bias,
                    higher_lows=sig.higher_lows,
                    lower_highs=sig.lower_highs,
                    body_ratio=sig.body_ratio,
                    upper_ratio=sig.upper_ratio,
                    lower_ratio=sig.lower_ratio,
                )

            if hit_tp:
                exit_price = tp
                pips = (entry - exit_price) / 0.01
                return Trade(
                    entry_time=sig.time,
                    exit_time=t,
                    direction="DOWN",
                    entry=entry,
                    exit=exit_price,
                    sl=sl,
                    tp=tp,
                    result_pips=round(pips, 1),
                    reason="TP",
                    score=sig.score,
                    tp_mode=tp_mode,
                    range_width_pips=sig.range_width_pips,
                    touches_high=sig.touches_high,
                    touches_low=sig.touches_low,
                    session=session_name_from_time(sig.time),
                    entry_hour=int(sig.time.hour),
                    weekday=sig.time.day_name(),
                    h1_bias=sig.h1_bias,
                    higher_lows=sig.higher_lows,
                    lower_highs=sig.lower_highs,
                    body_ratio=sig.body_ratio,
                    upper_ratio=sig.upper_ratio,
                    lower_ratio=sig.lower_ratio,
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

        sig = analyze_at(
            df_m15=df_m15,
            df_h1=df_h1,
            i=i,
            lookback=lookback,
            min_touches_break_side=2,
            min_range_pips=15.0,
            max_range_pips=44.0,        # v1.7: 15-44 pips
            require_structure=False,
            require_h1_bias=False,
            use_candle_filter=False,
        )
        if sig is None:
            i += 1
            continue

        # v1.7: touch 2-3 のみ採用
        max_touch = max(sig.touches_high, sig.touches_low)
        if max_touch not in (2, 3):
            i += 1
            continue

        tp_mode = choose_tp_mode(sig.score)

        if not initial_followthrough_ok(
            df_m15=df_m15,
            start_i=i,
            sig=sig,
            tokyo_follow_pips=3.0,
            west_hold_break=True,
        ):
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
# Analysis
# =========================
def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for key, g in df.groupby(group_col, dropna=False, observed=False):
        trades = len(g)
        wins = g[g["result_pips"] > 0]["result_pips"].sum()
        losses = -g[g["result_pips"] < 0]["result_pips"].sum()
        pf = (wins / losses) if losses > 0 else float("inf")
        win_rate = (g["result_pips"] > 0).mean() * 100.0
        avg_pips = g["result_pips"].mean()

        rows.append({
            group_col: key,
            "trades": trades,
            "win_rate": round(float(win_rate), 1),
            "profit_factor": round(float(pf), 2) if pf != float("inf") else "INF",
            "avg_pips": round(float(avg_pips), 2),
            "total_pips": round(float(g["result_pips"].sum()), 1),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["trades", "win_rate"], ascending=[False, False]).reset_index(drop=True)


def build_analysis_tables(df_tr: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if df_tr.empty:
        return {}

    df = df_tr.copy()

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])

    # 念のため再生成
    df["entry_hour"] = df["entry_time"].dt.hour
    df["weekday"] = df["entry_time"].dt.day_name()
    df["session"] = df["entry_time"].apply(session_name_from_time)

    # touch代表値
    df["max_touch"] = df[["touches_high", "touches_low"]].max(axis=1)
    df["min_touch"] = df[["touches_high", "touches_low"]].min(axis=1)

    # touch帯
    def touch_bucket(x: float) -> str:
        if pd.isna(x):
            return "unknown"
        if x <= 1:
            return "1"
        if x == 2:
            return "2"
        if x == 3:
            return "3"
        return "4+"

    df["touch_bucket"] = df["max_touch"].apply(touch_bucket)

    # range帯
    df["range_bucket"] = pd.cut(
        df["range_width_pips"],
        bins=[15, 25, 35, 45, 55, 71],
        labels=["15-24", "25-34", "35-44", "45-54", "55-70"],
        right=False,
        include_lowest=True
    )

    # 掛け合わせ
    df["session_touch"] = df["session"].astype(str) + "_" + df["touch_bucket"].astype(str)
    df["session_range"] = df["session"].astype(str) + "_" + df["range_bucket"].astype(str)
    df["touch_range"] = df["touch_bucket"].astype(str) + "_" + df["range_bucket"].astype(str)

    tables = {
        "by_session": summarize_group(df, "session"),
        "by_weekday": summarize_group(df, "weekday"),
        "by_direction": summarize_group(df, "direction"),
        "by_tp_mode": summarize_group(df, "tp_mode"),
        "by_touch_bucket": summarize_group(df, "touch_bucket"),
        "by_range_bucket": summarize_group(df, "range_bucket"),
        "by_entry_hour": summarize_group(df, "entry_hour"),
        "by_h1_bias": summarize_group(df, "h1_bias"),
        "by_session_touch": summarize_group(df, "session_touch"),
        "by_session_range": summarize_group(df, "session_range"),
        "by_touch_range": summarize_group(df, "touch_range"),
    }
    return tables


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m15", required=True, help="15分足CSV")
    ap.add_argument("--h1", default=None, help="1時間足CSV")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--cooldown", type=int, default=8)
    ap.add_argument("--out", default="trades_v15a_analysis.csv")
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

    print("=== Backtest Summary (v1.5a + analysis) ===")
    for k, v in summary.items():
        print(f"{k:22s}: {v}")

    df_tr.to_csv(args.out, index=False, encoding="utf-8-sig")

    tables = build_analysis_tables(df_tr)

    if tables:
        for name, table in tables.items():
            out_name = f"analysis_{name}.csv"
            table.to_csv(out_name, index=False, encoding="utf-8-sig")
            print(f"Saved: {out_name}")

        print("\n=== Analysis: by_session ===")
        if not tables["by_session"].empty:
            print(tables["by_session"].to_string(index=False))

        print("\n=== Analysis: by_touch_bucket ===")
        if not tables["by_touch_bucket"].empty:
            print(tables["by_touch_bucket"].to_string(index=False))

        print("\n=== Analysis: by_range_bucket ===")
        if not tables["by_range_bucket"].empty:
            print(tables["by_range_bucket"].to_string(index=False))

        print("\n=== Analysis: by_session_touch ===")
        if not tables["by_session_touch"].empty:
            print(tables["by_session_touch"].to_string(index=False))

        print("\n=== Analysis: by_session_range ===")
        if not tables["by_session_range"].empty:
            print(tables["by_session_range"].to_string(index=False))

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
            "session",
            "h1_bias",
        ]
        print(df_tr.tail(10)[cols].to_string(index=False))
    else:
        print("\nNo trades found.")


if __name__ == "__main__":
    main()