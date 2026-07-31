from __future__ import annotations
 
import argparse
import html as html_lib
import json
import math
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
 
import numpy as np
import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
try:
    import joblib
except Exception:
    joblib = None

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
 
 
DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

BLOCK_PAGE_MARKERS: tuple[str, ...] = (
    "访问过于频繁",
    "请求过于频繁",
    "too frequent",
    "request blocked",
    "forbidden",
)
 
 
@dataclass(frozen=True)
class MarketRow:
    minute_raw: str
    score_raw: str
    odds_over_raw: str
    line_raw: str
    odds_under_raw: str
    change_time_raw: str
 
 
@dataclass(frozen=True)
class MatchRaw:
    match_id: str
    date: str
    final_total_corners: int
    home_team_name: str
    away_team_name: str
    company_id: int
    market_rows: list[MarketRow]
    fetched_at: str
 
 
@dataclass(frozen=True)
class Snapshot:
    match_id: str
    date: str
    minute: int
    corners_total: int
    line: float
    odds_over: float
    odds_under: float
 
 
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)
 
 
def dump_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
 
 
def load_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out
 
 
def iter_dates(start_yyyymmdd: str, end_yyyymmdd: str) -> list[str]:
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    cur = start
    out: list[str] = []
    while cur <= end:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out
 
 
def _safe_float(x: str) -> Optional[float]:
    s = (x or "").strip()
    if not s:
        return None
    s = s.replace("\u00a0", " ")
    s = s.replace("↑", "").replace("↓", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def parse_odds_value(odds_raw: str) -> Optional[float]:
    v = _safe_float(odds_raw)
    if v is None:
        return None
    # Site odds are often in HK format (e.g. 0.83), convert to decimal-like odds.
    if 0 < v < 1.2:
        return 1.0 + v
    if v <= 1.0:
        return None
    return v
 
 
def parse_minute(minute_raw: str) -> Optional[int]:
    s = (minute_raw or "").strip()
    if not s:
        return None
    m = re.fullmatch(r"(\d+)(?:\+(\d+))?", s)
    if not m:
        return None
    base = int(m.group(1))
    extra = int(m.group(2)) if m.group(2) else 0
    return base + extra


def parse_change_time_minute(change_time_raw: str, kickoff_dt: Optional[datetime]) -> Optional[int]:
    if kickoff_dt is None:
        return None
    s = (change_time_raw or "").strip()
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})", s)
    if not m:
        return None

    mon = int(m.group(1))
    day = int(m.group(2))
    hh = int(m.group(3))
    mm = int(m.group(4))

    year_candidates = [kickoff_dt.year, kickoff_dt.year - 1, kickoff_dt.year + 1]
    best_dt: Optional[datetime] = None
    best_abs_delta = float("inf")

    for yy in year_candidates:
        try:
            cand = datetime(yy, mon, day, hh, mm)
        except Exception:
            continue
        abs_delta = abs((cand - kickoff_dt).total_seconds())
        if abs_delta < best_abs_delta:
            best_abs_delta = abs_delta
            best_dt = cand

    if best_dt is None:
        return None

    delta_min = int(round((best_dt - kickoff_dt).total_seconds() / 60.0))
    if delta_min < 0 or delta_min > 130:
        return None
    return delta_min
 
 
def parse_corner_score_total(score_raw: str) -> Optional[int]:
    s = (score_raw or "").strip()
    if not s or s in {"比数", "比分"}:
        return None
    first = s.split("|")[0].strip()
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", first)
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2))
 
 
def parse_asian_line(line_raw: str) -> Optional[float]:
    s = (line_raw or "").strip()
    if not s:
        return None
    s = s.replace("\u00a0", " ")
    s = s.replace("↑", "").replace("↓", "").strip()
    if "/" in s:
        parts = [p.strip() for p in s.split("/") if p.strip()]
        if len(parts) == 2:
            a = _safe_float(parts[0])
            b = _safe_float(parts[1])
            if a is None or b is None:
                return None
            return (a + b) / 2.0
        return None
    return _safe_float(s)
 
 
def implied_over_prob(odds_over: float, odds_under: float) -> Optional[float]:
    if odds_over <= 1.0 or odds_under <= 1.0:
        return None
    p_over_raw = 1.0 / odds_over
    p_under_raw = 1.0 / odds_under
    denom = p_over_raw + p_under_raw
    if denom <= 0:
        return None
    return p_over_raw / denom


def _extract_matchcount(html: str) -> Optional[int]:
    m = re.search(r"matchcount\s*=\s*(\d+)", html, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        v = int(m.group(1))
    except Exception:
        return None
    return v if v > 0 else None


def _extract_match_ids_from_html(html: str) -> set[str]:
    match_ids: set[str] = set()

    soup = BeautifulSoup(html, "html.parser")
    for tr in soup.find_all("tr"):
        sid = str(tr.get("sid", "")).strip()
        if sid.isdigit() and 6 <= len(sid) <= 10:
            match_ids.add(sid)

    for a in soup.find_all("a", href=True):
        href = str(a.get("href", ""))
        for sid in re.findall(r"\bsid=([0-9]{6,10})\b", href, flags=re.IGNORECASE):
            match_ids.add(sid)

    patterns = [
        r"\b(?:sid|sId)\s*=\s*[\"']?([0-9]{6,10})\b",
        r"\b(?:analysis|AsianOdds|EuropeOdds)\(([0-9]{6,10})\)",
        r"\b(?:matchid|match_id|matchId)\s*[:=]\s*[\"']?([0-9]{6,10})\b",
    ]
    for pat in patterns:
        for sid in re.findall(pat, html, flags=re.IGNORECASE):
            match_ids.add(str(sid))

    return match_ids


def _is_plausible_corner_line(v: Optional[float]) -> bool:
    if v is None:
        return False
    # Total-corners in-play line is normally at least around 2.0.
    return 2.0 <= float(v) <= 24.0


def _normalize_corner_row_cells(cells: list[str]) -> Optional[tuple[str, str, str, str, str, str]]:
    if len(cells) < 6:
        return None

    minute_raw = str(cells[0]).strip()
    score_raw = str(cells[1]).strip()
    c2 = str(cells[2]).strip()
    c3 = str(cells[3]).strip()
    c4 = str(cells[4]).strip()
    # 支持7列格式（最后一列是状态）
    change_time_raw = str(cells[5]).strip()
    if len(cells) >= 7 and not change_time_raw and str(cells[6]).strip():
        # 如果第6列是空的，尝试使用第7列作为变化时间
        change_time_raw = str(cells[6]).strip()

    # Candidate A: [minute, score, line, over, under, time]
    a_line = parse_asian_line(c2)
    a_over = parse_odds_value(c3)
    a_under = parse_odds_value(c4)
    a_ok = _is_plausible_corner_line(a_line) and (a_over is not None) and (a_under is not None)

    # Candidate B: [minute, score, over, line, under, time]
    b_line = parse_asian_line(c3)
    b_over = parse_odds_value(c2)
    b_under = parse_odds_value(c4)
    b_ok = _is_plausible_corner_line(b_line) and (b_over is not None) and (b_under is not None)

    if a_ok and not b_ok:
        return minute_raw, score_raw, c3, c2, c4, change_time_raw
    if b_ok and not a_ok:
        return minute_raw, score_raw, c2, c3, c4, change_time_raw
    if a_ok and b_ok:
        # Prefer a better-looking odds pair (HK/decimal-like, closer to market range).
        a_penalty = abs(float(a_over) - 1.95) + abs(float(a_under) - 1.95)
        b_penalty = abs(float(b_over) - 1.95) + abs(float(b_under) - 1.95)
        if b_penalty < a_penalty:
            return minute_raw, score_raw, c2, c3, c4, change_time_raw
        return minute_raw, score_raw, c3, c2, c4, change_time_raw

    return None


def _is_valid_inplay_corner_row(row: MarketRow, kickoff_dt: Optional[datetime] = None) -> bool:
    # Must have plausible line + both odds.
    line_v = parse_asian_line(row.line_raw)
    ov_v = parse_odds_value(row.odds_over_raw)
    un_v = parse_odds_value(row.odds_under_raw)
    if (not _is_plausible_corner_line(line_v)) or (ov_v is None) or (un_v is None):
        return False

    # Must be in-play evidence: valid minute or valid corner score.
    minute_v = parse_minute(row.minute_raw)
    if minute_v is None:
        minute_v = parse_change_time_minute(row.change_time_raw, kickoff_dt)
    score_v = parse_corner_score_total(row.score_raw)
    return (minute_v is not None) or (score_v is not None)


def clean_inplay_corner_rows(
    rows: list[MarketRow],
    kickoff_dt: Optional[datetime] = None,
    require_inplay_evidence: bool = True,
) -> list[MarketRow]:
    out: list[MarketRow] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for r in rows:
        minute_v = parse_minute(r.minute_raw)
        if minute_v is None:
            minute_v = parse_change_time_minute(r.change_time_raw, kickoff_dt)

        row2 = r
        if minute_v is not None and not str(r.minute_raw or "").strip():
            row2 = MarketRow(
                minute_raw=str(int(minute_v)),
                score_raw=r.score_raw,
                odds_over_raw=r.odds_over_raw,
                line_raw=r.line_raw,
                odds_under_raw=r.odds_under_raw,
                change_time_raw=r.change_time_raw,
            )

        # 放宽验证要求：对于没有比赛时间和比分的记录，只要盘口和赔率有效也保留
        # 特别是对于刚刚开始的比赛
        if require_inplay_evidence and (not _is_valid_inplay_corner_row(row2, kickoff_dt=kickoff_dt)):
            # 即使没有时间和比分，只要有有效的盘口和赔率，也保留
            line_v = parse_asian_line(row2.line_raw)
            ov_v = parse_odds_value(row2.odds_over_raw)
            un_v = parse_odds_value(row2.odds_under_raw)
            if not (_is_plausible_corner_line(line_v) and ov_v is not None and un_v is not None):
                continue

        line_v = parse_asian_line(row2.line_raw)
        ov_v = parse_odds_value(row2.odds_over_raw)
        un_v = parse_odds_value(row2.odds_under_raw)
        if (not _is_plausible_corner_line(line_v)) or (ov_v is None) or (un_v is None):
            continue

        key = (
            str(parse_minute(row2.minute_raw)),
            str(parse_corner_score_total(row2.score_raw)),
            str(parse_asian_line(row2.line_raw)),
            str(parse_odds_value(row2.odds_over_raw)),
            str(parse_odds_value(row2.odds_under_raw)),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row2)
    return out


def _assess_market_rows_quality(rows: list[MarketRow]) -> tuple[bool, str, dict[str, Any]]:
    if not rows:
        return False, "no_rows", {"rows": 0}

    valid_minutes = 0
    valid_scores = 0
    minute_values: list[int] = []
    line_vals: list[float] = []
    for r in rows:
        mv = parse_minute(r.minute_raw)
        if mv is not None:
            valid_minutes += 1
            minute_values.append(int(mv))
        if parse_corner_score_total(r.score_raw) is not None:
            valid_scores += 1
        lv = parse_asian_line(r.line_raw)
        if lv is not None:
            line_vals.append(float(lv))

    unique_line_ticks = len({_line_to_tick(v) for v in line_vals}) if line_vals else 0
    stats = {
        "rows": int(len(rows)),
        "valid_minutes": int(valid_minutes),
        "valid_scores": int(valid_scores),
        "unique_line_ticks": int(unique_line_ticks),
        "max_minute": int(max(minute_values)) if minute_values else None,
    }

    # 降低对进行中比赛的要求
    # 对于刚开始的比赛，允许较少的行数和变化
    if len(rows) < 3:  # 从6降低到3
        return False, "too_few_rows", stats
    if unique_line_ticks < 1:  # 从3降低到1
        return False, "insufficient_line_variation", stats
    if valid_minutes == 0 and valid_scores == 0:
        return False, "no_minute_or_score", stats
    # 移除比赛时间过短的限制，允许刚开始的比赛
    # if minute_values and max(minute_values) < 20:
    #     return False, "too_early_timeline", stats

    return True, "ok", stats


def _minute_bucket(values: pd.Series) -> pd.Series:
    bins = [-1, 15, 30, 45, 60, 75, 90, 130]
    labels = ["00-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]
    return pd.cut(values.astype(float), bins=bins, labels=labels)


def _line_bucket(values: pd.Series) -> pd.Series:
    x = np.round(values.astype(float) * 2.0) / 2.0
    return x.map(lambda v: f"{float(v):.1f}")


def _line_to_tick(v: float) -> int:
    return int(round(float(v) * 2.0))


def _tick_to_line(t: int) -> float:
    return float(t) / 2.0


def _run_length_encoding(vals: list[int]) -> list[tuple[int, int, int]]:
    if not vals:
        return []
    out: list[tuple[int, int, int]] = []
    start = 0
    cur = vals[0]
    for i in range(1, len(vals)):
        if vals[i] != cur:
            out.append((start, i - 1, cur))
            start = i
            cur = vals[i]
    out.append((start, len(vals) - 1, cur))
    return out


def _compress_line_jitter(line_vals: list[float], max_jitter_ticks: int = 1) -> list[float]:
    if not line_vals:
        return []

    ticks = [_line_to_tick(v) for v in line_vals]
    runs = _run_length_encoding(ticks)
    if len(runs) < 3:
        return [_tick_to_line(t) for t in ticks]

    merged = ticks[:]
    changed = True
    while changed:
        changed = False
        runs = _run_length_encoding(merged)
        if len(runs) < 3:
            break
        for ridx in range(1, len(runs) - 1):
            s0, e0, v0 = runs[ridx - 1]
            s1, e1, v1 = runs[ridx]
            s2, e2, v2 = runs[ridx + 1]
            if (e1 - s1 + 1) > 2:
                continue
            if v0 != v2:
                continue
            if abs(v1 - v0) > int(max_jitter_ticks):
                continue
            for i in range(s1, e1 + 1):
                merged[i] = v0
            changed = True
            break

    return [_tick_to_line(t) for t in merged]


def _jitter_metrics(line_vals: list[float]) -> dict[str, Any]:
    if len(line_vals) <= 2:
        return {
            "jitter_cycles": 0,
            "jitter_ratio": 0.0,
            "max_single_jump": 0.0,
        }

    ticks = np.array([_line_to_tick(v) for v in line_vals], dtype=np.int32)
    diffs = np.diff(ticks)
    if len(diffs) == 0:
        return {
            "jitter_cycles": 0,
            "jitter_ratio": 0.0,
            "max_single_jump": 0.0,
        }

    max_single_jump = float(np.max(np.abs(diffs)) / 2.0)
    cycles = 0
    for i in range(2, len(ticks)):
        a = ticks[i - 2]
        b = ticks[i - 1]
        c = ticks[i]
        if a == c and b != a and abs(b - a) <= 1:
            cycles += 1

    jitter_ratio = float(cycles / max(1, len(ticks) - 2))
    return {
        "jitter_cycles": int(cycles),
        "jitter_ratio": jitter_ratio,
        "max_single_jump": max_single_jump,
    }


def _extract_line_series_with_corners(match: dict[str, Any]) -> pd.DataFrame:
    rows = match.get("market_rows") or []
    parsed: list[dict[str, Any]] = []
    for rr in rows:
        minute = parse_minute(str(rr.get("minute_raw", "")))
        if minute is None:
            continue
        line = parse_asian_line(str(rr.get("line_raw", "")))
        if line is None:
            continue
        corners = parse_corner_score_total(str(rr.get("score_raw", "")))
        parsed.append(
            {
                "minute": int(minute),
                "line": float(line),
                "corners": int(corners) if corners is not None else None,
            }
        )

    if not parsed:
        return pd.DataFrame(columns=["minute", "line", "corners"])

    df = pd.DataFrame(parsed).sort_values("minute").reset_index(drop=True)
    df = df.loc[(df["minute"] >= 0) & (df["minute"] <= 130)].reset_index(drop=True)
    return df


def _add_corner_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["corners"].astype(float)
    c_ff = c.ffill().fillna(0.0)
    dc = c_ff.diff().fillna(c_ff)
    out["corner_inc"] = np.maximum(0.0, dc)
    out["corner_event"] = (out["corner_inc"] > 0).astype(int)
    out["corners_ffill"] = c_ff
    return out


def _find_anchor_induced_over_pattern(
    series_df: pd.DataFrame,
    final_total_corners: int,
    min_anchor_minute: int = 46,
    max_anchor_minute: int = 75,
    min_anchor_corners: int = 5,
    min_raise_ticks: int = 4,
    max_anchor_back_ticks: int = 2,
) -> Optional[dict[str, Any]]:
    if series_df.empty:
        return None

    df = _add_corner_event_flags(series_df)
    lines_raw = df["line"].astype(float).tolist()
    lines_clean = _compress_line_jitter(lines_raw, max_jitter_ticks=1)
    df["line_clean"] = lines_clean

    jitter = _jitter_metrics(lines_raw)
    if jitter["jitter_ratio"] > 0.18 or jitter["max_single_jump"] > 2.5:
        return None

    candidates = df.loc[
        (df["minute"] >= int(min_anchor_minute))
        & (df["minute"] <= int(max_anchor_minute))
        & (df["corner_event"] == 1)
        & (df["corners_ffill"] >= int(min_anchor_corners))
    ]
    if candidates.empty:
        return None

    best: Optional[dict[str, Any]] = None
    for idx in candidates.index.tolist():
        anchor_line = float(df.at[idx, "line_clean"])
        post = df.loc[idx:].reset_index(drop=True)
        if len(post) < 3:
            continue

        post_lines = post["line_clean"].astype(float).tolist()
        peak_line = float(max(post_lines))
        raise_ticks = _line_to_tick(peak_line) - _line_to_tick(anchor_line)
        if raise_ticks < int(min_raise_ticks):
            continue

        close_line = float(post_lines[-1])
        close_back_ticks = _line_to_tick(close_line) - _line_to_tick(anchor_line)
        if close_back_ticks > int(max_anchor_back_ticks):
            continue

        peak_idx_local = int(np.argmax(post_lines))
        if peak_idx_local >= len(post) - 1:
            continue

        post_peak = post.iloc[peak_idx_local:]
        post_peak_lines = post_peak["line_clean"].astype(float).tolist()
        down_from_peak_ticks = _line_to_tick(peak_line) - _line_to_tick(min(post_peak_lines))
        if down_from_peak_ticks < 2:
            continue

        win_under_anchor_plus_1 = int(final_total_corners <= math.floor(anchor_line + 1.0))

        rec = {
            "anchor_idx": int(idx),
            "anchor_minute": int(df.at[idx, "minute"]),
            "anchor_corners": int(df.at[idx, "corners_ffill"]),
            "anchor_line": anchor_line,
            "peak_line": peak_line,
            "close_line": close_line,
            "raise_ticks": int(raise_ticks),
            "close_back_ticks": int(close_back_ticks),
            "down_from_peak_ticks": int(down_from_peak_ticks),
            "final_total_corners": int(final_total_corners),
            "target_under_line": float(anchor_line + 1.0),
            "under_anchor_plus_1_hit": int(win_under_anchor_plus_1),
            "jitter_ratio": float(jitter["jitter_ratio"]),
            "jitter_cycles": int(jitter["jitter_cycles"]),
        }

        if best is None:
            best = rec
        else:
            if rec["raise_ticks"] > best["raise_ticks"]:
                best = rec
            elif rec["raise_ticks"] == best["raise_ticks"] and rec["close_back_ticks"] < best["close_back_ticks"]:
                best = rec

    return best


def analyze_anchor_trend_patterns(
    raw_jsonl: str,
    out_json_path: str,
    out_matches_path: Optional[str] = None,
    min_anchor_minute: int = 46,
    max_anchor_minute: int = 75,
    min_anchor_corners: int = 5,
    min_raise_ticks: int = 4,
    max_anchor_back_ticks: int = 2,
) -> dict[str, Any]:
    matches = load_jsonl(raw_jsonl)
    total = len(matches)

    accepted: list[dict[str, Any]] = []
    skipped_for_jitter = 0
    skipped_for_missing = 0

    for m in matches:
        final_total = m.get("final_total_corners", None)
        if final_total is None:
            skipped_for_missing += 1
            continue

        try:
            final_total_i = int(final_total)
        except Exception:
            skipped_for_missing += 1
            continue

        series_df = _extract_line_series_with_corners(m)
        if series_df.empty or len(series_df) < 8:
            skipped_for_missing += 1
            continue

        raw_lines = series_df["line"].astype(float).tolist()
        jitter = _jitter_metrics(raw_lines)
        if jitter["jitter_ratio"] > 0.18 or jitter["max_single_jump"] > 2.5:
            skipped_for_jitter += 1
            continue

        rec = _find_anchor_induced_over_pattern(
            series_df=series_df,
            final_total_corners=final_total_i,
            min_anchor_minute=min_anchor_minute,
            max_anchor_minute=max_anchor_minute,
            min_anchor_corners=min_anchor_corners,
            min_raise_ticks=min_raise_ticks,
            max_anchor_back_ticks=max_anchor_back_ticks,
        )
        if not rec:
            continue

        rec.update(
            {
                "match_id": str(m.get("match_id", "")),
                "date": str(m.get("date", "")),
                "company_id": int(m.get("company_id", 0) or 0),
            }
        )
        accepted.append(rec)

    report_df = pd.DataFrame(accepted)
    summary: dict[str, Any] = {
        "matches_total": int(total),
        "matches_pattern": int(len(accepted)),
        "skipped_jitter": int(skipped_for_jitter),
        "skipped_missing": int(skipped_for_missing),
        "pattern_rate": float(len(accepted) / total) if total > 0 else 0.0,
    }

    if not report_df.empty:
        hit_rate = float(report_df["under_anchor_plus_1_hit"].mean())
        summary.update(
            {
                "under_anchor_plus_1_hit_rate": hit_rate,
                "avg_anchor_line": float(report_df["anchor_line"].mean()),
                "avg_peak_line": float(report_df["peak_line"].mean()),
                "avg_final_total": float(report_df["final_total_corners"].mean()),
                "median_raise_ticks": float(report_df["raise_ticks"].median()),
            }
        )

        grp = (
            report_df.groupby(["anchor_line"], observed=True)
            .agg(
                samples=("match_id", "size"),
                hit_rate=("under_anchor_plus_1_hit", "mean"),
                avg_final_total=("final_total_corners", "mean"),
            )
            .reset_index()
            .sort_values(["samples", "hit_rate"], ascending=[False, False])
        )
        summary["anchor_line_groups_top"] = grp.head(20).to_dict(orient="records")
    else:
        summary.update(
            {
                "under_anchor_plus_1_hit_rate": None,
                "anchor_line_groups_top": [],
            }
        )

    ensure_dir(os.path.dirname(out_json_path))
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "matches": accepted[:5000]}, f, ensure_ascii=False, indent=2)

    if out_matches_path:
        ensure_dir(os.path.dirname(out_matches_path))
        if report_df.empty:
            pd.DataFrame(
                columns=[
                    "match_id",
                    "date",
                    "anchor_minute",
                    "anchor_corners",
                    "anchor_line",
                    "peak_line",
                    "close_line",
                    "raise_ticks",
                    "close_back_ticks",
                    "down_from_peak_ticks",
                    "final_total_corners",
                    "target_under_line",
                    "under_anchor_plus_1_hit",
                    "jitter_ratio",
                    "jitter_cycles",
                ]
            ).to_csv(out_matches_path, index=False)
        else:
            report_df.to_csv(out_matches_path, index=False)

    return summary


def inspect_single_match_anchor_trend(
    match_id: str,
    date_yyyymmdd: str,
    company_id: int = 8,
    timeout_s: float = 12.0,
    min_anchor_minute: int = 46,
    max_anchor_minute: int = 75,
    min_anchor_corners: int = 5,
    min_raise_ticks: int = 4,
    max_anchor_back_ticks: int = 2,
) -> dict[str, Any]:
    client = TitanCornerClient(timeout_s=timeout_s)
    rows = client.fetch_corner_market_rows(str(match_id), company_id=int(company_id))
    final_total = client.fetch_final_total_corners(str(match_id))

    if not rows:
        return {
            "match_id": str(match_id),
            "date": str(date_yyyymmdd),
            "ok": False,
            "reason": "no market rows",
        }

    if final_total is None:
        inferred: list[int] = []
        for r in rows:
            c = parse_corner_score_total(r.score_raw)
            if c is not None:
                inferred.append(int(c))
        final_total = max(inferred) if inferred else None

    if final_total is None:
        return {
            "match_id": str(match_id),
            "date": str(date_yyyymmdd),
            "ok": False,
            "reason": "no final total corners",
        }

    match = {
        "match_id": str(match_id),
        "date": str(date_yyyymmdd),
        "final_total_corners": int(final_total),
        "company_id": int(company_id),
        "market_rows": [asdict(r) for r in rows],
    }

    df = _extract_line_series_with_corners(match)
    lines = df["line"].astype(float).tolist() if not df.empty else []
    jitter = _jitter_metrics(lines)
    pattern = _find_anchor_induced_over_pattern(
        series_df=df,
        final_total_corners=int(final_total),
        min_anchor_minute=min_anchor_minute,
        max_anchor_minute=max_anchor_minute,
        min_anchor_corners=min_anchor_corners,
        min_raise_ticks=min_raise_ticks,
        max_anchor_back_ticks=max_anchor_back_ticks,
    )

    return {
        "match_id": str(match_id),
        "date": str(date_yyyymmdd),
        "ok": True,
        "rows": int(len(rows)),
        "final_total_corners": int(final_total),
        "jitter": jitter,
        "pattern": pattern,
    }


def _make_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["minute_bucket"] = _minute_bucket(out["minute"]).astype(str)
    out["line_bucket"] = _line_bucket(out["line"]).astype(str)
    out["gap_bucket"] = np.round(out["line_gap"].astype(float) * 2.0) / 2.0
    out["reversal_bucket"] = np.where(out["reversal_flag_2"].astype(int) == 1, "rev", "no_rev")
    out["pace_bucket"] = pd.cut(
        out["rate_10m"].astype(float),
        bins=[-1.0, 0.2, 0.5, 0.9, 1.5, 5.0],
        labels=["very_low", "low", "mid", "high", "very_high"],
    ).astype(str)
    return out


def mine_high_probability_patterns(
    dataset_csv: str,
    out_json_path: str,
    min_samples: int = 80,
    min_prob: float = 0.80,
    top_k: int = 50,
) -> dict[str, Any]:
    df = pd.read_csv(dataset_csv)
    if df.empty:
        raise ValueError("dataset is empty")

    feat_df = _make_pattern_features(df)
    target = feat_df["y_over"].astype(int)

    group_defs = [
        ["minute_bucket", "line_bucket", "reversal_bucket"],
        ["minute_bucket", "line_bucket", "pace_bucket"],
        ["minute_bucket", "gap_bucket", "reversal_bucket"],
    ]

    rules: list[dict[str, Any]] = []
    for keys in group_defs:
        grouped = (
            feat_df.groupby(keys, observed=True)
            .agg(samples=("y_over", "size"), p_over=("y_over", "mean"), avg_line=("line", "mean"))
            .reset_index()
        )
        grouped = grouped.loc[grouped["samples"] >= int(min_samples)].copy()
        if grouped.empty:
            continue

        grouped["side"] = np.where(grouped["p_over"] >= 0.5, "over", "under")
        grouped["confidence"] = np.where(grouped["side"] == "over", grouped["p_over"], 1.0 - grouped["p_over"])
        grouped = grouped.loc[grouped["confidence"] >= float(min_prob)].copy()
        if grouped.empty:
            continue

        grouped = grouped.sort_values(["confidence", "samples"], ascending=[False, False]).reset_index(drop=True)
        for _, r in grouped.iterrows():
            cond: dict[str, Any] = {k: r[k] for k in keys}
            rules.append(
                {
                    "rule_type": "group_stat",
                    "group_keys": keys,
                    "condition": cond,
                    "samples": int(r["samples"]),
                    "side": str(r["side"]),
                    "confidence": float(r["confidence"]),
                    "p_over": float(r["p_over"]),
                    "avg_line": float(r["avg_line"]),
                }
            )

    X = df[FEATURE_COLS].astype(float)
    y = target
    groups = df["match_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=900, class_weight="balanced")),
        ]
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    p_test = model.predict_proba(X.iloc[test_idx])[:, 1]
    y_test = y.iloc[test_idx].to_numpy()

    grid = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    ml_rules: list[dict[str, Any]] = []
    for th in grid:
        mask_over = p_test >= th
        if int(mask_over.sum()) >= int(min_samples):
            conf = float(y_test[mask_over].mean())
            if conf >= float(min_prob):
                ml_rules.append(
                    {
                        "rule_type": "ml_threshold",
                        "side": "over",
                        "threshold": float(th),
                        "samples": int(mask_over.sum()),
                        "confidence": conf,
                    }
                )

        mask_under = p_test <= (1.0 - th)
        if int(mask_under.sum()) >= int(min_samples):
            conf_under = float((1 - y_test[mask_under]).mean())
            if conf_under >= float(min_prob):
                ml_rules.append(
                    {
                        "rule_type": "ml_threshold",
                        "side": "under",
                        "threshold": float(1.0 - th),
                        "samples": int(mask_under.sum()),
                        "confidence": conf_under,
                    }
                )

    all_rules = rules + ml_rules
    all_rules = sorted(all_rules, key=lambda x: (x.get("confidence", 0.0), x.get("samples", 0)), reverse=True)

    result: dict[str, Any] = {
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "params": {
            "min_samples": int(min_samples),
            "min_prob": float(min_prob),
            "top_k": int(top_k),
        },
        "model_eval": {
            "test_rows": int(len(test_idx)),
            "auc": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else float("nan"),
            "brier": float(brier_score_loss(y_test, p_test)),
            "logloss": float(log_loss(y_test, p_test, labels=[0, 1])),
        },
        "rules": all_rules[: int(top_k)],
    }

    ensure_dir(os.path.dirname(out_json_path))
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def iterate_pattern_learning(
    dataset_csv: str,
    out_json_path: str,
    rounds: int = 12,
    min_samples_grid: Optional[list[int]] = None,
    min_prob_grid: Optional[list[float]] = None,
) -> dict[str, Any]:
    if min_samples_grid is None:
        min_samples_grid = [60, 80, 100, 120, 160]
    if min_prob_grid is None:
        min_prob_grid = [0.75, 0.78, 0.80, 0.82, 0.85]

    df = pd.read_csv(dataset_csv)
    if df.empty:
        raise ValueError("dataset is empty")

    X = df[FEATURE_COLS].astype(float)
    y = df["y_over"].astype(int)
    groups = df["match_id"].astype(str)

    records: list[dict[str, Any]] = []
    rnd = max(1, int(rounds))
    for i in range(rnd):
        rs = 42 + i * 11
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=rs)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=900, class_weight="balanced")),
            ]
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        p_test = model.predict_proba(X.iloc[test_idx])[:, 1]
        y_test = y.iloc[test_idx].to_numpy()

        for ms in min_samples_grid:
            for mp in min_prob_grid:
                q = float(mp)
                q = min(0.99, max(0.50, q))
                th_over = float(np.quantile(p_test, q))
                th_under = float(np.quantile(1.0 - p_test, q))

                over_mask = p_test >= th_over
                under_mask = (1.0 - p_test) >= th_under

                over_n = int(over_mask.sum())
                under_n = int(under_mask.sum())

                over_conf = float(y_test[over_mask].mean()) if over_n > 0 else 0.0
                under_conf = float((1 - y_test[under_mask]).mean()) if under_n > 0 else 0.0

                picks = 0
                wins = 0.0
                if over_n >= int(ms):
                    picks += over_n
                    wins += float(y_test[over_mask].sum())
                if under_n >= int(ms):
                    picks += under_n
                    wins += float((1 - y_test[under_mask]).sum())

                win_rate = float(wins / picks) if picks > 0 else 0.0
                records.append(
                    {
                        "round": int(i + 1),
                        "random_state": int(rs),
                        "min_samples": int(ms),
                        "min_prob": float(mp),
                        "over_threshold": float(th_over),
                        "under_threshold": float(th_under),
                        "over_samples": over_n,
                        "under_samples": under_n,
                        "over_conf": over_conf,
                        "under_conf": under_conf,
                        "picked_samples": int(picks),
                        "picked_win_rate": win_rate,
                    }
                )

    rec_df = pd.DataFrame(records)
    if rec_df.empty:
        raise ValueError("no iteration records")

    stable = (
        rec_df.groupby(["min_samples", "min_prob"], observed=True)
        .agg(
            rounds=("round", "size"),
            avg_picked_samples=("picked_samples", "mean"),
            median_picked_samples=("picked_samples", "median"),
            avg_win_rate=("picked_win_rate", "mean"),
            p25_win_rate=("picked_win_rate", lambda x: float(np.percentile(x, 25))),
            p75_win_rate=("picked_win_rate", lambda x: float(np.percentile(x, 75))),
            min_win_rate=("picked_win_rate", "min"),
        )
        .reset_index()
    )

    stable = stable.sort_values(
        ["avg_win_rate", "p25_win_rate", "median_picked_samples"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best = stable.head(1).to_dict(orient="records")
    result: dict[str, Any] = {
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "rounds": int(rnd),
        "grid": {
            "min_samples": [int(x) for x in min_samples_grid],
            "min_prob": [float(x) for x in min_prob_grid],
        },
        "best_config": best[0] if best else None,
        "top_configs": stable.head(20).to_dict(orient="records"),
    }

    ensure_dir(os.path.dirname(out_json_path))
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
 
 
class TitanCornerClient:
    _GLOBAL_RATE_LOCK = threading.Lock()
    _GLOBAL_NEXT_ALLOWED_TS = 0.0

    def __init__(
        self,
        headers: Optional[dict[str, str]] = None,
        timeout_s: float = 12.0,
        max_retries: int = 3,
        backoff_s: float = 0.35,
        jitter_s: tuple[float, float] = (0.0, 0.12),
        min_request_interval_s: float = 0.8,
        throttle_jitter_s: tuple[float, float] = (0.03, 0.18),
    ):
        self.session = requests.Session()
        self.headers = headers or dict(DEFAULT_HEADERS)
        self.timeout_s = float(timeout_s)
        self.max_retries = max(1, int(max_retries))
        self.backoff_s = float(backoff_s)
        self.jitter_s = (float(jitter_s[0]), float(jitter_s[1]))
        self.min_request_interval_s = max(0.0, float(min_request_interval_s))
        self.throttle_jitter_s = (float(throttle_jitter_s[0]), float(throttle_jitter_s[1]))

    def _throttle(self) -> None:
        now = time.time()
        with TitanCornerClient._GLOBAL_RATE_LOCK:
            wait_s = max(0.0, TitanCornerClient._GLOBAL_NEXT_ALLOWED_TS - now)
            if wait_s > 0:
                time.sleep(wait_s)
                now = time.time()
            TitanCornerClient._GLOBAL_NEXT_ALLOWED_TS = now + self.min_request_interval_s + random.uniform(*self.throttle_jitter_s)

    @staticmethod
    def _is_block_page(status_code: int, text: str) -> bool:
        if status_code != 200:
            return True
        s = (text or "").strip().lower()
        if not s:
            return True
        if len(s) < 120 and ("<html" not in s):
            return True
        for marker in BLOCK_PAGE_MARKERS:
            if marker in s:
                return True
        return False

    def _warmup(self) -> None:
        warmups = [
            "https://vip.titan007.com/",
            "https://bf.titan007.com/football/",
        ]
        for u in warmups:
            try:
                self._throttle()
                self.session.get(u, headers=self.headers, timeout=self.timeout_s, verify=self._should_verify_ssl(u))
            except Exception:
                continue

    @staticmethod
    def _should_verify_ssl(url: str) -> bool:
        s = str(url or "").lower()
        # vip.titan007.com currently presents a cert chain that is not trusted in this environment.
        if "vip.titan007.com" in s:
            return False
        return True
 
    def _get(self, url: str, encoding: Optional[str] = None) -> str:
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                resp = self.session.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout_s,
                    verify=self._should_verify_ssl(url),
                )
                if encoding:
                    resp.encoding = encoding
                if self._is_block_page(resp.status_code, resp.text):
                    raise requests.HTTPError(f"blocked_or_bad_status={resp.status_code}")
                return resp.text
            except Exception:
                if attempt + 1 >= self.max_retries:
                    break
                # Stronger backoff to reduce anti-bot pressure.
                sleep_s = (self.backoff_s * (2**attempt)) + random.uniform(*self.jitter_s)
                if sleep_s < 0.8:
                    sleep_s = 0.8 + random.uniform(0.0, 0.5)
                if attempt >= 1:
                    self.session = requests.Session()
                    self._warmup()
                if sleep_s > 0:
                    time.sleep(sleep_s)
        return ""

    @staticmethod
    def _looks_like_soft_404(html: str) -> bool:
        if not html:
            return False
        low = html.lower()
        # Titan sometimes returns a tiny branded 404 template with HTTP 200.
        return ("error_404.gif" in low) or (len(html) < 1500 and "球探网首页" in html and "足球比分" in html)

    def fetch_match_ids_detail(self, date_yyyymmdd: str) -> tuple[list[str], Optional[int]]:
        urls = [
            f"https://bf.titan007.com/football/Over_{date_yyyymmdd}.htm",
            f"https://bf.titan007.com/football/Over_{date_yyyymmdd}.html",
            f"https://bf.titan007.com/football/Over_{date_yyyymmdd}.htm?_={int(time.time() * 1000)}",
        ]

        all_ids: set[str] = set()
        expected: Optional[int] = None
        fetched_pages = 0
        soft_404_pages = 0

        for url in urls:
            html = self._get(url, encoding="gb2312")
            if not html:
                continue

            if self._looks_like_soft_404(html):
                soft_404_pages += 1
                continue

            fetched_pages += 1
            cnt = _extract_matchcount(html)
            if cnt is not None:
                expected = max(expected or 0, cnt)

            all_ids.update(_extract_match_ids_from_html(html))

            if expected is not None and len(all_ids) >= expected:
                break

        if fetched_pages == 0:
            if soft_404_pages > 0:
                print(
                    f"fetch_match_ids({date_yyyymmdd}): list page returned soft-404 template; "
                    "date list unavailable on current endpoint"
                )
            else:
                print(f"fetch_match_ids({date_yyyymmdd}): Failed to get HTML")
            return [], None

        if expected is not None:
            print(
                f"fetch_match_ids({date_yyyymmdd}): expected={expected}, found={len(all_ids)}"
            )
        else:
            print(f"fetch_match_ids({date_yyyymmdd}): found={len(all_ids)}")

        return sorted(all_ids), expected
 
    def fetch_match_ids(self, date_yyyymmdd: str) -> list[str]:
        ids, _ = self.fetch_match_ids_detail(date_yyyymmdd)
        return ids
 
    def fetch_final_total_corners(self, match_id: str) -> Optional[int]:
        # 按顺序尝试不同的详情页面，哪个有角球数据就用哪个
        for suffix in ["sb.htm", "cn.htm", "detail.htm"]:
            url = f"https://live.titan007.com/detail/{match_id}{suffix}"
            html = self._get(url, encoding="utf-8")
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            for li in soup.find_all("li", class_="lists"):
                data_div = li.find("div", class_="data")
                if not data_div:
                    continue
                txt = data_div.get_text(strip=True)
                if "角球" not in txt or "半场角球" in txt:
                    continue
                spans = data_div.find_all("span")
                if len(spans) != 3:
                    continue
                try:
                    home = int(spans[0].get_text(strip=True))
                    away = int(spans[2].get_text(strip=True))
                    return home + away
                except Exception:
                    continue

        return None

    def fetch_match_kickoff_datetime(self, match_id: str) -> Optional[datetime]:
        patterns = [
            r"var\s+strTime\s*=\s*'([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2})'",
            r"<span\s+class=\"time\">\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2})\s*</span>",
        ]
        for suffix in ["sb.htm", "cn.htm"]:
            url = f"https://live.titan007.com/detail/{match_id}{suffix}"
            html = self._get(url, encoding="utf-8")
            if not html:
                continue
            for pat in patterns:
                m = re.search(pat, html, flags=re.IGNORECASE)
                if not m:
                    continue
                txt = m.group(1)
                try:
                    return datetime.strptime(txt, "%Y-%m-%d %H:%M")
                except Exception:
                    continue
        return None

    @staticmethod
    def _extract_team_names_from_html(html: str) -> tuple[str, str]:
        if not html:
            return "", ""

        m_home = re.search(r"homeTeamName\s*=\s*'([^']+)'", html, flags=re.IGNORECASE)
        m_away = re.search(r"guestTeamName\s*=\s*'([^']+)'", html, flags=re.IGNORECASE)
        if m_home and m_away:
            return html_lib.unescape(m_home.group(1).strip()), html_lib.unescape(m_away.group(1).strip())

        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.get_text(" ", strip=True) if soup.title else "")
        m_title = re.search(r"(.+?)\s+VS\s+(.+?)\(", title, flags=re.IGNORECASE)
        if m_title:
            return m_title.group(1).strip(), m_title.group(2).strip()

        return "", ""

    def fetch_match_teams(self, match_id: str) -> tuple[str, str]:
        # Prefer cn/sb detail pages because they contain explicit JS vars.
        for suffix in ["cn.htm", "sb.htm", "detail.htm"]:
            url = f"https://live.titan007.com/detail/{match_id}{suffix}"
            html = self._get(url, encoding="utf-8")
            if not html:
                continue
            home, away = self._extract_team_names_from_html(html)
            if home and away:
                return home, away
        return "", ""
 
    def fetch_corner_market_rows(self, match_id: str, company_id: int = 8, l: int = 0) -> list[MarketRow]:
        kickoff_dt = self.fetch_match_kickoff_datetime(match_id)
        l_candidates = [int(l)]
        for alt in [0, 1, 2]:
            if alt not in l_candidates:
                l_candidates.append(alt)

        for li in l_candidates:
            url = f"https://vip.titan007.com/ChangeDetail/corner.aspx?id={match_id}&companyid={company_id}&l={li}"
            html = self._get(url, encoding="gb2312")
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table")
            if not table:
                continue

            rows: list[MarketRow] = []
            trs = table.find_all("tr")
            for tr in trs[1:]:
                tds = tr.find_all("td")
                if len(tds) < 6:
                    continue
                cells = [td.get_text(strip=True) for td in tds[:6]]
                norm = _normalize_corner_row_cells(cells)
                if norm is None:
                    continue
                minute_raw, score_raw, odds_over_raw, line_raw, odds_under_raw, change_time_raw = norm
                rows.append(
                    MarketRow(
                        minute_raw=minute_raw,
                        score_raw=score_raw,
                        line_raw=line_raw,
                        odds_over_raw=odds_over_raw,
                        odds_under_raw=odds_under_raw,
                        change_time_raw=change_time_raw,
                    )
                )

            rows = clean_inplay_corner_rows(rows, kickoff_dt=kickoff_dt, require_inplay_evidence=True)

            if rows:
                return rows

        return []
 
 
_THREAD_LOCAL = threading.local()
 
 
def _get_thread_client(timeout_s: float, retries: int, backoff_s: float, jitter_s: tuple[float, float]) -> TitanCornerClient:
    client = getattr(_THREAD_LOCAL, "client", None)
    cfg = getattr(_THREAD_LOCAL, "cfg", None)
    cur_cfg = (float(timeout_s), int(retries), float(backoff_s), float(jitter_s[0]), float(jitter_s[1]))
    if client is None or cfg != cur_cfg:
        client = TitanCornerClient(timeout_s=timeout_s, max_retries=retries, backoff_s=backoff_s, jitter_s=jitter_s)
        setattr(_THREAD_LOCAL, "client", client)
        setattr(_THREAD_LOCAL, "cfg", cur_cfg)
    return client
 
 
def _load_seen_match_ids_fast(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    seen: set[str] = set()
    pat = re.compile(r"\"match_id\"\s*:\s*\"([^\"]+)\"")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                seen.add(m.group(1))
    return seen
 
 
def _fetch_one_match_raw(
    match_id: str,
    date_str: str,
    company_id: int,
    timeout_s: float,
    retries: int,
    backoff_s: float,
    jitter_s: tuple[float, float],
    request_jitter_s: tuple[float, float],
) -> Optional[dict[str, Any]]:
    client = _get_thread_client(timeout_s=timeout_s, retries=retries, backoff_s=backoff_s, jitter_s=jitter_s)
 
    if request_jitter_s[1] > 0:
        time.sleep(random.uniform(*request_jitter_s))
 
    rows = client.fetch_corner_market_rows(match_id, company_id=company_id)
    if not rows:
        return None

    ok_quality, reason, qstats = _assess_market_rows_quality(rows)
    if not ok_quality:
        return {
            "_reject": True,
            "reason": str(reason),
            "quality": qstats,
            "match_id": str(match_id),
            "date": str(date_str),
        }

    # Prefer inferring final corners from market rows because detail pages can change markup.
    inferred_totals: list[int] = []
    for r in rows:
        c = parse_corner_score_total(r.score_raw)
        if c is not None:
            inferred_totals.append(int(c))

    final_total = client.fetch_final_total_corners(match_id)
    if final_total is None and inferred_totals:
        final_total = max(inferred_totals)

    # 对于正在进行的比赛，如果没有最终角球数，使用当前最新的角球数
    # 即使没有也不返回None，允许模型进行实时预测
    if final_total is None:
        final_total = 0  # 对于未结束的比赛，使用0作为占位符，模型预测时会忽略这个值

    home_team_name, away_team_name = client.fetch_match_teams(match_id)
 
    rec = MatchRaw(
        match_id=str(match_id),
        date=str(date_str),
        final_total_corners=int(final_total),
        home_team_name=str(home_team_name),
        away_team_name=str(away_team_name),
        company_id=int(company_id),
        market_rows=list(rows),
        fetched_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
 
    return {
        "match_id": rec.match_id,
        "date": rec.date,
        "final_total_corners": rec.final_total_corners,
        "home_team_name": rec.home_team_name,
        "away_team_name": rec.away_team_name,
        "company_id": rec.company_id,
        "fetched_at": rec.fetched_at,
        "market_rows": [asdict(r) for r in rec.market_rows],
    }
 
 
def crawl_raw(
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    out_jsonl: str,
    company_id: int = 8,
    workers: int = 1,
    timeout_s: float = 12.0,
    retries: int = 3,
    backoff_s: float = 0.35,
    request_jitter_s: tuple[float, float] = (1.8, 3.8),
    per_date_max: Optional[int] = None,
    target_new_matches: Optional[int] = None,
    date_cooldown_s: float = 6.0,
    batch_wait_seconds: Optional[float] = None,
) -> dict[str, Any]:
    ensure_dir(os.path.dirname(out_jsonl))
 
    seen = _load_seen_match_ids_fast(out_jsonl)
    client_main = TitanCornerClient(timeout_s=timeout_s, max_retries=retries, backoff_s=backoff_s, jitter_s=(0.1, 0.3))
 
    submitted = 0
    written = 0
    skipped = 0
    expected_total = 0
    found_total = 0
    per_date_stats: list[dict[str, Any]] = []
    reject_reasons: dict[str, int] = {}
 
    dates = iter_dates(start_yyyymmdd, end_yyyymmdd)
 
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for date_str in dates:
            if target_new_matches is not None and written >= int(target_new_matches):
                break

            match_ids, expected_cnt = client_main.fetch_match_ids_detail(date_str)
            found_total += len(match_ids)
            if expected_cnt is not None:
                expected_total += int(expected_cnt)

            todo = [mid for mid in match_ids if mid not in seen]
            if per_date_max is not None:
                todo = todo[: int(per_date_max)]
 
            if not todo:
                skipped += len(match_ids)
                per_date_stats.append(
                    {
                        "date": date_str,
                        "expected": int(expected_cnt) if expected_cnt is not None else None,
                        "found_ids": len(match_ids),
                        "submitted": 0,
                        "saved": 0,
                    }
                )
                continue
 
            ex = ThreadPoolExecutor(max_workers=max(1, int(workers)))
            futures = [
                ex.submit(
                    _fetch_one_match_raw,
                    mid,
                    date_str,
                    company_id,
                    timeout_s,
                    retries,
                    backoff_s,
                    (0.0, 0.12),
                    request_jitter_s,
                )
                for mid in todo
            ]
            submitted += len(futures)
            print(f"crawl_raw({date_str}): Submitted {len(futures)} matches (Already seen: {len(match_ids)-len(todo)})")

            if batch_wait_seconds is None:
                # Prevent a single hung request from blocking the entire day forever.
                per_match_budget = max(3.0, float(timeout_s)) * max(1, int(retries))
                wait_s = min(1800.0, max(180.0, float(len(futures)) * per_match_budget * 0.7))
            else:
                wait_s = max(30.0, float(batch_wait_seconds))

            done, pending = wait(futures, timeout=wait_s)

            success_this_date = 0
            for fut in done:
                try:
                    rec = fut.result()
                except Exception:
                    reject_reasons["future_exception"] = int(reject_reasons.get("future_exception", 0) + 1)
                    continue
                if not rec:
                    continue
                if bool(rec.get("_reject", False)):
                    rs = str(rec.get("reason", "unknown"))
                    reject_reasons[rs] = int(reject_reasons.get(rs, 0) + 1)
                    continue
                mid = str(rec.get("match_id", ""))
                if not mid:
                    continue
                if mid in seen:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                success_this_date += 1
                seen.add(mid)

                if written % 200 == 0:
                    print(f"crawl_raw: progress written={written}")

                if target_new_matches is not None and written >= int(target_new_matches):
                    break

            timed_out_pending = int(len(pending))
            if timed_out_pending > 0:
                reject_reasons["date_timeout_pending"] = int(reject_reasons.get("date_timeout_pending", 0) + timed_out_pending)
                print(
                    f"crawl_raw({date_str}): Timeout after {int(wait_s)}s, "
                    f"forcing skip of {timed_out_pending} still-running matches"
                )
                for fut in pending:
                    fut.cancel()
            ex.shutdown(wait=False, cancel_futures=True)

            print(f"crawl_raw({date_str}): Successfully saved {success_this_date} matches.")

            per_date_stats.append(
                {
                    "date": date_str,
                    "expected": int(expected_cnt) if expected_cnt is not None else None,
                    "found_ids": len(match_ids),
                    "submitted": len(futures),
                    "saved": success_this_date,
                    "timed_out_pending": timed_out_pending,
                }
            )
            if date_cooldown_s > 0:
                time.sleep(float(date_cooldown_s) + random.uniform(0.1, 0.8))
 
    id_coverage = float(found_total / expected_total) if expected_total > 0 else None
    summary = {
        "submitted": submitted,
        "written": written,
        "seen": len(seen),
        "skipped": skipped,
        "expected_match_ids": expected_total if expected_total > 0 else None,
        "found_match_ids": found_total,
        "id_coverage": id_coverage,
        "dates": per_date_stats,
        "target_new_matches": int(target_new_matches) if target_new_matches is not None else None,
        "reject_reasons": reject_reasons,
    }
    return summary


def crawl_raw_from_match_ids(
    match_ids: list[str],
    date_str: str,
    out_jsonl: str,
    company_id: int = 8,
    workers: int = 1,
    timeout_s: float = 12.0,
    retries: int = 3,
    backoff_s: float = 0.35,
    request_jitter_s: tuple[float, float] = (1.8, 3.8),
    batch_wait_seconds: Optional[float] = None,
) -> dict[str, Any]:
    ensure_dir(os.path.dirname(out_jsonl))
    seen = _load_seen_match_ids_fast(out_jsonl)

    todo = [str(mid).strip() for mid in match_ids if str(mid).strip() and str(mid).strip() not in seen]
    if not todo:
        return {
            "submitted": 0,
            "written": 0,
            "seen": len(seen),
            "skipped": len(match_ids),
            "mode": "manual_match_ids",
            "date": str(date_str),
        }

    written = 0
    reject_reasons: dict[str, int] = {}
    with open(out_jsonl, "a", encoding="utf-8") as f:
        ex = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        futures = [
            ex.submit(
                _fetch_one_match_raw,
                mid,
                date_str,
                company_id,
                timeout_s,
                retries,
                backoff_s,
                (0.0, 0.12),
                request_jitter_s,
            )
            for mid in todo
        ]

        if batch_wait_seconds is None:
            per_match_budget = max(3.0, float(timeout_s)) * max(1, int(retries))
            wait_s = min(1800.0, max(180.0, float(len(futures)) * per_match_budget * 0.7))
        else:
            wait_s = max(30.0, float(batch_wait_seconds))

        done, pending = wait(futures, timeout=wait_s)

        for fut in done:
            try:
                rec = fut.result()
            except Exception:
                reject_reasons["future_exception"] = int(reject_reasons.get("future_exception", 0) + 1)
                continue
            if not rec:
                continue
            if bool(rec.get("_reject", False)):
                rs = str(rec.get("reason", "unknown"))
                reject_reasons[rs] = int(reject_reasons.get(rs, 0) + 1)
                continue
            mid = str(rec.get("match_id", ""))
            if not mid or mid in seen:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            seen.add(mid)
            written += 1

        timed_out_pending = int(len(pending))
        if timed_out_pending > 0:
            reject_reasons["date_timeout_pending"] = int(reject_reasons.get("date_timeout_pending", 0) + timed_out_pending)
            for fut in pending:
                fut.cancel()
        ex.shutdown(wait=False, cancel_futures=True)

    return {
        "submitted": len(todo),
        "written": written,
        "seen": len(seen),
        "skipped": int(len(match_ids) - len(todo)),
        "mode": "manual_match_ids",
        "date": str(date_str),
        "timed_out_pending": timed_out_pending,
        "reject_reasons": reject_reasons,
    }
 
 
def rows_to_snapshots(match: dict[str, Any]) -> list[Snapshot]:
    match_id = str(match["match_id"])
    date = str(match["date"])
    final_total = int(match["final_total_corners"])
 
    raw_rows = match.get("market_rows") or []
 
    parsed: list[tuple[int, Optional[int], Optional[float], Optional[float], Optional[float]]] = []
    for rr in raw_rows:
        minute = parse_minute(str(rr.get("minute_raw", "")))
        if minute is None:
            continue
        corners_total = parse_corner_score_total(str(rr.get("score_raw", "")))
        line = parse_asian_line(str(rr.get("line_raw", "")))
        odds_over = parse_odds_value(str(rr.get("odds_over_raw", "")))
        odds_under = parse_odds_value(str(rr.get("odds_under_raw", "")))
        parsed.append((minute, corners_total, line, odds_over, odds_under))
 
    if not parsed:
        return []
 
    parsed.sort(key=lambda x: x[0])
 
    last_corners: Optional[int] = 0
    last_line: Optional[float] = None
    last_ov: Optional[float] = None
    last_un: Optional[float] = None
 
    per_minute: dict[int, Snapshot] = {}
 
    for minute, corners_total, line, odds_over, odds_under in parsed:
        if corners_total is not None:
            last_corners = corners_total
        if line is not None:
            last_line = line
        if odds_over is not None:
            last_ov = odds_over
        if odds_under is not None:
            last_un = odds_under
 
        if last_corners is None or last_line is None or last_ov is None or last_un is None:
            continue
 
        m = int(minute)
        if m < 0 or m > 130:
            continue
 
        per_minute[m] = Snapshot(
            match_id=match_id,
            date=date,
            minute=m,
            corners_total=int(last_corners),
            line=float(last_line),
            odds_over=float(last_ov),
            odds_under=float(last_un),
        )
 
    if not per_minute:
        return []
 
    out = [per_minute[m] for m in sorted(per_minute.keys())]
 
    if out[-1].corners_total > final_total:
        out = [Snapshot(**{**asdict(s), "corners_total": min(s.corners_total, final_total)}) for s in out]
 
    return out
 
 
def _prefix_sum(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(len(arr) + 1, dtype=np.int32)
    np.cumsum(arr.astype(np.int32), out=out[1:])
    return out
 
 
def build_features_for_match(match: dict[str, Any], window_m: int = 20, drop_push: bool = True) -> pd.DataFrame:
    final_total = int(match["final_total_corners"])
    snaps = rows_to_snapshots(match)
    if not snaps:
        return pd.DataFrame()
 
    minutes = np.array([s.minute for s in snaps], dtype=np.int32)
    corners = np.array([s.corners_total for s in snaps], dtype=np.int32)
    line = np.array([s.line for s in snaps], dtype=np.float64)
    ov = np.array([s.odds_over for s in snaps], dtype=np.float64)
    un = np.array([s.odds_under for s in snaps], dtype=np.float64)
 
    p_mkt_over = np.array([
        implied_over_prob(float(a), float(b)) if implied_over_prob(float(a), float(b)) is not None else np.nan
        for a, b in zip(ov, un)
    ], dtype=np.float64)
 
    valid = ~np.isnan(p_mkt_over)
    if not valid.any():
        return pd.DataFrame()
 
    odds_log_ratio = np.log(ov) - np.log(un)
 
    corner_increments = np.zeros_like(corners)
    corner_increments[0] = max(0, corners[0])
    corner_increments[1:] = np.maximum(0, corners[1:] - corners[:-1])
 
    prefix_inc = _prefix_sum(corner_increments)
 
    last_corner_min = np.full(len(snaps), -1, dtype=np.int32)
    last = -1
    for i in range(len(snaps)):
        if corner_increments[i] > 0:
            last = int(minutes[i])
        last_corner_min[i] = last
 
    def corners_in_last(w: int) -> np.ndarray:
        target = minutes - w
        left_idx = np.searchsorted(minutes, target, side="left")
        right = np.arange(len(minutes)) + 1
        left = left_idx
        return prefix_inc[right] - prefix_inc[left]
 
    c_last_5 = corners_in_last(5)
    c_last_10 = corners_in_last(10)
    c_last_15 = corners_in_last(15)
 
    time_since_last = np.where(last_corner_min < 0, minutes, minutes - last_corner_min)
 
    prev_line = np.concatenate(([line[0]], line[:-1]))
    delta_line = line - prev_line
 
    prev_p = np.concatenate(([p_mkt_over[0]], p_mkt_over[:-1]))
    delta_p = p_mkt_over - prev_p
 
    div_sign = np.sign(delta_line) * np.sign(delta_p)
 
    w = int(window_m)
    window_left_idx = np.searchsorted(minutes, minutes - w, side="left")
 
    steps_up = np.zeros(len(snaps), dtype=np.float64)
    steps_down = np.zeros(len(snaps), dtype=np.float64)
    reversal_depth = np.zeros(len(snaps), dtype=np.float64)
    reversal_time = np.zeros(len(snaps), dtype=np.float64)
    reversal_flag_2 = np.zeros(len(snaps), dtype=np.int32)
 
    for i in range(len(snaps)):
        j = int(window_left_idx[i])
        if i == j:
            continue
        seg = line[j : i + 1]
        diffs = np.diff(seg)
        up = diffs[diffs > 0].sum()
        down = (-diffs[diffs < 0]).sum()
        steps_up[i] = up / 0.5
        steps_down[i] = down / 0.5
 
        l_max = float(np.max(seg))
        if l_max > float(line[i]):
            k_rel = int(np.argmax(seg))
            t_max = int(minutes[j + k_rel])
            depth = max(0.0, l_max - float(line[i]))
            reversal_depth[i] = depth
            if depth >= 1.0 and t_max < int(minutes[i]):
                reversal_flag_2[i] = 1
                reversal_time[i] = float(int(minutes[i]) - t_max)
 
    y_over = (final_total > line).astype(np.int32)
    if drop_push:
        keep = (final_total != line)
    else:
        keep = np.ones(len(snaps), dtype=bool)
 
    df = pd.DataFrame(
        {
            "match_id": [snaps[i].match_id for i in range(len(snaps))],
            "date": [snaps[i].date for i in range(len(snaps))],
            "minute": minutes,
            "corners_so_far": corners,
            "line": line,
            "line_gap": line - corners,
            "odds_over": ov,
            "odds_under": un,
            "p_mkt_over": p_mkt_over,
            "odds_log_ratio": odds_log_ratio,
            "delta_line": delta_line,
            "delta_p_mkt_over": delta_p,
            "divergence_sign": div_sign,
            "steps_up_20m": steps_up,
            "steps_down_20m": steps_down,
            "reversal_flag_2": reversal_flag_2,
            "reversal_depth": reversal_depth,
            "reversal_time": reversal_time,
            "corners_last_5m": c_last_5,
            "corners_last_10m": c_last_10,
            "corners_last_15m": c_last_15,
            "rate_10m": c_last_10 / 10.0,
            "time_since_last_corner": time_since_last,
            "final_total_corners": final_total,
            "y_over": y_over,
        }
    )
 
    df = df.loc[keep].reset_index(drop=True)
    df = df.loc[np.isfinite(df["p_mkt_over"])].reset_index(drop=True)
    return df
 
 
FEATURE_COLS: list[str] = [
    "minute",
    "corners_so_far",
    "line",
    "line_gap",
    "p_mkt_over",
    "odds_log_ratio",
    "delta_line",
    "delta_p_mkt_over",
    "divergence_sign",
    "steps_up_20m",
    "steps_down_20m",
    "reversal_flag_2",
    "reversal_depth",
    "reversal_time",
    "corners_last_5m",
    "corners_last_10m",
    "corners_last_15m",
    "rate_10m",
    "time_since_last_corner",
]
 
 
def build_dataset(raw_jsonl: str, out_csv: str, drop_push: bool = True) -> dict[str, Any]:
    matches = load_jsonl(raw_jsonl)
    frames: list[pd.DataFrame] = []
    for m in matches:
        df = build_features_for_match(m, drop_push=drop_push)
        if not df.empty:
            frames.append(df)
 
    if frames:
        dataset = pd.concat(frames, ignore_index=True)
    else:
        dataset = pd.DataFrame()
 
    ensure_dir(os.path.dirname(out_csv))
    dataset.to_csv(out_csv, index=False)
 
    summary = {
        "matches_in": len(matches),
        "rows_out": int(len(dataset)),
        "unique_matches": int(dataset["match_id"].nunique()) if not dataset.empty else 0,
    }
    return summary
 
 
def train_model(dataset_csv: str, out_model_path: str) -> dict[str, Any]:
    df = pd.read_csv(dataset_csv)
    if df.empty:
        raise ValueError("dataset is empty")
 
    X = df[FEATURE_COLS].astype(float)
    y = df["y_over"].astype(int)
    groups = df["match_id"].astype(str)
 
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
 
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
 
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=800, class_weight="balanced")),
        ]
    )
 
    model.fit(X_train, y_train)
 
    p_test = model.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= 0.5).astype(int)
 
    metrics: dict[str, Any] = {
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "test_rows": int(len(test_idx)),
        "test_matches": int(df.iloc[test_idx]["match_id"].nunique()),
        "auc": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_test, p_test)),
        "logloss": float(log_loss(y_test, p_test, labels=[0, 1])),
        "acc": float(accuracy_score(y_test, pred_test)),
    }
 
    if joblib is None:
        raise RuntimeError("joblib not available; install joblib / scikit-learn")
 
    ensure_dir(os.path.dirname(out_model_path))
    joblib.dump({"model": model, "features": FEATURE_COLS, "metrics": metrics}, out_model_path)
 
    return metrics
 
 
def _compute_edge(p_model_over: float, odds_over: float) -> Optional[float]:
    if not (odds_over and odds_over > 1.0):
        return None
    p_mkt_raw = 1.0 / odds_over
    return float(p_model_over - p_mkt_raw)
 
 
def predict_match(
    match_id: str,
    date_yyyymmdd: str,
    model_path: str,
    company_id: int = 8,
    minute: Optional[int] = None,
    timeout_s: float = 12.0,
) -> dict[str, Any]:
    if joblib is None:
        raise RuntimeError("joblib not available; install joblib / scikit-learn")
 
    bundle = joblib.load(model_path)
    model: Any = bundle["model"]
    features: list[str] = list(bundle["features"])
 
    client = TitanCornerClient(timeout_s=timeout_s)
 
    final_total = client.fetch_final_total_corners(match_id)
    rows = client.fetch_corner_market_rows(match_id, company_id=company_id)
    match = {
        "match_id": str(match_id),
        "date": str(date_yyyymmdd),
        "final_total_corners": int(final_total) if final_total is not None else 0,
        "company_id": int(company_id),
        "fetched_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "market_rows": [asdict(r) for r in rows],
    }
 
    df = build_features_for_match(match, drop_push=False)
    if df.empty:
        raise ValueError("no usable snapshots/features for this match")
 
    if minute is None:
        row = df.iloc[-1]
    else:
        df2 = df.loc[df["minute"] <= int(minute)]
        if df2.empty:
            row = df.iloc[0]
        else:
            row = df2.iloc[-1]
 
    X = pd.DataFrame([row[features].astype(float).to_dict()])
    p_over = float(model.predict_proba(X)[0, 1])
 
    odds_over = float(row["odds_over"])
    odds_under = float(row["odds_under"])
    p_mkt_over = implied_over_prob(odds_over, odds_under)
 
    return {
        "match_id": str(match_id),
        "minute": int(row["minute"]),
        "corners_so_far": int(row["corners_so_far"]),
        "line": float(row["line"]),
        "odds_over": odds_over,
        "odds_under": odds_under,
        "p_model_over": p_over,
        "p_mkt_over": float(p_mkt_over) if p_mkt_over is not None else None,
        "edge_vs_raw_over": _compute_edge(p_over, odds_over),
        "reversal_flag_2": int(row["reversal_flag_2"]),
        "reversal_depth": float(row["reversal_depth"]),
        "reversal_time": float(row["reversal_time"]),
    }


def discover_entry_points(
    dataset_csv: str,
    out_json_path: str,
    min_samples: int = 120,
    min_edge: float = 0.06,
    min_prob_gap: float = 0.08,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    df = pd.read_csv(dataset_csv)
    if df.empty:
        raise ValueError("dataset is empty")

    X = df[FEATURE_COLS].astype(float)
    y = df["y_over"].astype(int)
    groups = df["match_id"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=900, class_weight="balanced")),
        ]
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    p_test = model.predict_proba(test_df[FEATURE_COLS].astype(float))[:, 1]
    test_df["p_model_over"] = p_test
    test_df["edge_over"] = test_df["p_model_over"] - test_df["p_mkt_over"]
    test_df["edge_under"] = (1.0 - test_df["p_model_over"]) - (1.0 - test_df["p_mkt_over"])

    over_signal = (test_df["edge_over"] >= float(min_edge)) & (
        test_df["p_model_over"] >= (test_df["p_mkt_over"] + float(min_prob_gap))
    )
    under_signal = (test_df["edge_under"] >= float(min_edge)) & (
        (1.0 - test_df["p_model_over"]) >= (1.0 - test_df["p_mkt_over"] + float(min_prob_gap))
    )
    test_df["side"] = np.where(over_signal, "over", np.where(under_signal, "under", "none"))

    test_df["minute_bucket"] = _minute_bucket(test_df["minute"])
    test_df["line_bucket"] = _line_bucket(test_df["line"])
    test_df["won"] = np.where(test_df["side"] == "over", test_df["y_over"] == 1, test_df["y_over"] == 0)
    test_df["edge_used"] = np.where(test_df["side"] == "over", test_df["edge_over"], test_df["edge_under"])
    test_df["odds_used"] = np.where(test_df["side"] == "over", test_df["odds_over"], test_df["odds_under"])

    test_df["profit"] = np.where(
        test_df["side"] == "over",
        np.where(test_df["y_over"] == 1, test_df["odds_over"] - 1.0, -1.0),
        np.where(test_df["side"] == "under", np.where(test_df["y_over"] == 0, test_df["odds_under"] - 1.0, -1.0), 0.0),
    )

    all_df = df.copy()
    all_df["minute_bucket"] = _minute_bucket(all_df["minute"])
    all_df["line_bucket"] = _line_bucket(all_df["line"])
    assoc = (
        all_df.groupby(["minute_bucket", "line_bucket"], observed=True)
        .agg(
            samples=("y_over", "size"),
            p_over_empirical=("y_over", "mean"),
            p_mkt_over_avg=("p_mkt_over", "mean"),
            avg_line=("line", "mean"),
        )
        .reset_index()
    )
    assoc = assoc.loc[assoc["samples"] >= int(min_samples)].copy()
    assoc["empirical_vs_market"] = assoc["p_over_empirical"] - assoc["p_mkt_over_avg"]
    assoc = assoc.sort_values(["samples", "empirical_vs_market"], ascending=[False, False]).reset_index(drop=True)

    bets = test_df.loc[test_df["side"] != "none"].copy()
    rules = pd.DataFrame()
    if not bets.empty:
        rules = (
            bets.groupby(["minute_bucket", "line_bucket", "side"], observed=True)
            .agg(
                bets=("profit", "size"),
                hit_rate=("won", "mean"),
                roi=("profit", "mean"),
                avg_edge=("edge_used", "mean"),
                avg_odds=("odds_used", "mean"),
                exp_profit=("profit", "sum"),
            )
            .reset_index()
        )
        rules = rules.loc[rules["bets"] >= int(min_samples)].copy()
        rules = rules.sort_values(["roi", "hit_rate", "bets"], ascending=[False, False, False]).reset_index(drop=True)

    test_pred = (p_test >= 0.5).astype(int)
    strategy_roi = float(bets["profit"].mean()) if not bets.empty else 0.0
    strategy_hit_rate = float(bets["won"].mean()) if not bets.empty else 0.0

    result: dict[str, Any] = {
        "model_eval": {
            "rows": int(len(df)),
            "matches": int(df["match_id"].nunique()),
            "test_rows": int(len(test_df)),
            "test_matches": int(test_df["match_id"].nunique()),
            "auc": float(roc_auc_score(test_df["y_over"], p_test)) if len(np.unique(test_df["y_over"])) > 1 else float("nan"),
            "brier": float(brier_score_loss(test_df["y_over"], p_test)),
            "logloss": float(log_loss(test_df["y_over"], p_test, labels=[0, 1])),
            "acc": float(accuracy_score(test_df["y_over"], test_pred)),
        },
        "signal_params": {
            "min_samples": int(min_samples),
            "min_edge": float(min_edge),
            "min_prob_gap": float(min_prob_gap),
            "test_size": float(test_size),
        },
        "strategy_summary": {
            "bets": int(len(bets)),
            "bet_rate": float(len(bets) / len(test_df)) if len(test_df) > 0 else 0.0,
            "hit_rate": strategy_hit_rate,
            "roi_per_bet": strategy_roi,
            "total_profit_units": float(bets["profit"].sum()) if not bets.empty else 0.0,
        },
        "association_top": assoc.head(30).to_dict(orient="records"),
        "entry_points_top": rules.head(20).to_dict(orient="records"),
    }

    ensure_dir(os.path.dirname(out_json_path))
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
 
 
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline_e2e_v2")
    sub = p.add_subparsers(dest="cmd", required=True)
 
    p_crawl = sub.add_parser("crawl")
    p_crawl.add_argument("--start", required=True, help="YYYYMMDD")
    p_crawl.add_argument("--end", required=True, help="YYYYMMDD")
    p_crawl.add_argument("--out", default="data/raw_matches.jsonl")
    p_crawl.add_argument("--company", type=int, default=8)
    p_crawl.add_argument("--workers", type=int, default=1)
    p_crawl.add_argument("--timeout", type=float, default=12.0)
    p_crawl.add_argument("--retries", type=int, default=3)
    p_crawl.add_argument("--backoff", type=float, default=1.2)
    p_crawl.add_argument("--sleep-min", type=float, default=1.8)
    p_crawl.add_argument("--sleep-max", type=float, default=3.8)
    p_crawl.add_argument("--per-date-max", type=int, default=None)
    p_crawl.add_argument("--target-new-matches", type=int, default=None)
    p_crawl.add_argument("--date-cooldown", type=float, default=6.0)
    p_crawl.add_argument("--batch-wait-seconds", type=float, default=None, help="Max wait time for one date batch before skipping pending matches")
    p_crawl.add_argument("--match-ids", default="", help="Comma-separated match IDs to crawl directly")
    p_crawl.add_argument("--match-ids-file", default="", help="Path to text file with match IDs (comma or newline separated)")
 
    p_ds = sub.add_parser("build-dataset")
    p_ds.add_argument("--raw", default="data/raw_matches.jsonl")
    p_ds.add_argument("--out", default="data/dataset.csv")
    p_ds.add_argument("--keep-push", action="store_true")
 
    p_tr = sub.add_parser("train")
    p_tr.add_argument("--dataset", default="data/dataset.csv")
    p_tr.add_argument("--out", default="data/model_over_line.joblib")
 
    p_pr = sub.add_parser("predict")
    p_pr.add_argument("--match", required=True)
    p_pr.add_argument("--date", required=True, help="YYYYMMDD")
    p_pr.add_argument("--model", default="data/model_over_line.joblib")
    p_pr.add_argument("--company", type=int, default=8)
    p_pr.add_argument("--minute", type=int, default=None)
    p_pr.add_argument("--timeout", type=float, default=12.0)

    p_rule = sub.add_parser("discover-rules")
    p_rule.add_argument("--dataset", default="data/dataset.csv")
    p_rule.add_argument("--out", default="data/rule_report.json")
    p_rule.add_argument("--min-samples", type=int, default=120)
    p_rule.add_argument("--min-edge", type=float, default=0.06)
    p_rule.add_argument("--min-prob-gap", type=float, default=0.08)
    p_rule.add_argument("--test-size", type=float, default=0.25)
    p_rule.add_argument("--seed", type=int, default=42)

    p_anchor = sub.add_parser("analyze-anchor-trend")
    p_anchor.add_argument("--raw", default="data/raw_matches.jsonl")
    p_anchor.add_argument("--out", default="data/anchor_trend_report.json")
    p_anchor.add_argument("--out-matches", default="data/anchor_trend_matches.csv")
    p_anchor.add_argument("--min-anchor-minute", type=int, default=46)
    p_anchor.add_argument("--max-anchor-minute", type=int, default=75)
    p_anchor.add_argument("--min-anchor-corners", type=int, default=5)
    p_anchor.add_argument("--min-raise-ticks", type=int, default=4)
    p_anchor.add_argument("--max-anchor-back-ticks", type=int, default=2)

    p_mine = sub.add_parser("mine-patterns")
    p_mine.add_argument("--dataset", default="data/dataset.csv")
    p_mine.add_argument("--out", default="data/high_prob_patterns.json")
    p_mine.add_argument("--min-samples", type=int, default=80)
    p_mine.add_argument("--min-prob", type=float, default=0.80)
    p_mine.add_argument("--top-k", type=int, default=50)

    p_iter = sub.add_parser("iterate-learning")
    p_iter.add_argument("--dataset", default="data/dataset.csv")
    p_iter.add_argument("--out", default="data/iterative_learning_report.json")
    p_iter.add_argument("--rounds", type=int, default=12)
    p_iter.add_argument("--min-samples-grid", default="60,80,100,120,160")
    p_iter.add_argument("--min-prob-grid", default="0.75,0.78,0.80,0.82,0.85")

    p_inspect = sub.add_parser("inspect-anchor-trend")
    p_inspect.add_argument("--match", required=True)
    p_inspect.add_argument("--date", required=True, help="YYYYMMDD")
    p_inspect.add_argument("--company", type=int, default=8)
    p_inspect.add_argument("--timeout", type=float, default=12.0)
    p_inspect.add_argument("--min-anchor-minute", type=int, default=46)
    p_inspect.add_argument("--max-anchor-minute", type=int, default=75)
    p_inspect.add_argument("--min-anchor-corners", type=int, default=5)
    p_inspect.add_argument("--min-raise-ticks", type=int, default=4)
    p_inspect.add_argument("--max-anchor-back-ticks", type=int, default=2)
 
    return p
 
 
def main() -> None:
    args = build_arg_parser().parse_args()
 
    if args.cmd == "crawl":
        manual_ids: list[str] = []
        if str(args.match_ids).strip():
            manual_ids.extend([x.strip() for x in str(args.match_ids).split(",") if x.strip()])
        if str(args.match_ids_file).strip():
            with open(str(args.match_ids_file), "r", encoding="utf-8") as f:
                txt = f.read()
            manual_ids.extend([x.strip() for x in re.split(r"[\s,]+", txt) if x.strip()])

        if manual_ids:
            # Use explicit match IDs when date list endpoint is unavailable.
            summary = crawl_raw_from_match_ids(
                match_ids=manual_ids,
                date_str=args.start,
                out_jsonl=args.out,
                company_id=args.company,
                workers=args.workers,
                timeout_s=args.timeout,
                retries=args.retries,
                backoff_s=args.backoff,
                request_jitter_s=(args.sleep_min, args.sleep_max),
                batch_wait_seconds=args.batch_wait_seconds,
            )
        else:
            summary = crawl_raw(
                start_yyyymmdd=args.start,
                end_yyyymmdd=args.end,
                out_jsonl=args.out,
                company_id=args.company,
                workers=args.workers,
                timeout_s=args.timeout,
                retries=args.retries,
                backoff_s=args.backoff,
                request_jitter_s=(args.sleep_min, args.sleep_max),
                per_date_max=args.per_date_max,
                target_new_matches=args.target_new_matches,
                date_cooldown_s=float(args.date_cooldown),
                batch_wait_seconds=args.batch_wait_seconds,
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
 
    if args.cmd == "build-dataset":
        summary = build_dataset(raw_jsonl=args.raw, out_csv=args.out, drop_push=not args.keep_push)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
 
    if args.cmd == "train":
        metrics = train_model(dataset_csv=args.dataset, out_model_path=args.out)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
 
    if args.cmd == "predict":
        out = predict_match(
            match_id=str(args.match),
            date_yyyymmdd=str(args.date),
            model_path=str(args.model),
            company_id=int(args.company),
            minute=args.minute,
            timeout_s=float(args.timeout),
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.cmd == "discover-rules":
        out = discover_entry_points(
            dataset_csv=str(args.dataset),
            out_json_path=str(args.out),
            min_samples=int(args.min_samples),
            min_edge=float(args.min_edge),
            min_prob_gap=float(args.min_prob_gap),
            test_size=float(args.test_size),
            random_state=int(args.seed),
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.cmd == "analyze-anchor-trend":
        out = analyze_anchor_trend_patterns(
            raw_jsonl=str(args.raw),
            out_json_path=str(args.out),
            out_matches_path=str(args.out_matches),
            min_anchor_minute=int(args.min_anchor_minute),
            max_anchor_minute=int(args.max_anchor_minute),
            min_anchor_corners=int(args.min_anchor_corners),
            min_raise_ticks=int(args.min_raise_ticks),
            max_anchor_back_ticks=int(args.max_anchor_back_ticks),
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.cmd == "mine-patterns":
        out = mine_high_probability_patterns(
            dataset_csv=str(args.dataset),
            out_json_path=str(args.out),
            min_samples=int(args.min_samples),
            min_prob=float(args.min_prob),
            top_k=int(args.top_k),
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.cmd == "iterate-learning":
        min_samples_grid = [int(x.strip()) for x in str(args.min_samples_grid).split(",") if x.strip()]
        min_prob_grid = [float(x.strip()) for x in str(args.min_prob_grid).split(",") if x.strip()]
        out = iterate_pattern_learning(
            dataset_csv=str(args.dataset),
            out_json_path=str(args.out),
            rounds=int(args.rounds),
            min_samples_grid=min_samples_grid,
            min_prob_grid=min_prob_grid,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.cmd == "inspect-anchor-trend":
        out = inspect_single_match_anchor_trend(
            match_id=str(args.match),
            date_yyyymmdd=str(args.date),
            company_id=int(args.company),
            timeout_s=float(args.timeout),
            min_anchor_minute=int(args.min_anchor_minute),
            max_anchor_minute=int(args.max_anchor_minute),
            min_anchor_corners=int(args.min_anchor_corners),
            min_raise_ticks=int(args.min_raise_ticks),
            max_anchor_back_ticks=int(args.max_anchor_back_ticks),
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
 
    raise RuntimeError(f"unknown cmd: {args.cmd}")
 
 
if __name__ == "__main__":
    main()