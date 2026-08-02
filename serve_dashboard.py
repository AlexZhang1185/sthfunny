from __future__ import annotations

import argparse
import json
import os
import re
import threading
import tempfile
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import joblib

from update_live_best_strategy_from_oldindexall import (
    build_live_strategy_rows,
    crawl_live_matches_from_ids,
    extract_live_matches_from_feed,
    fetch_oldindexall_feed_text,
)

HOST = "127.0.0.1"
PORT = 8765

MODEL_PATH = "data/settlement_relation_model/settlement_relation_model.joblib"
RAW_OUTPUT_NOW = "data/raw_matches_live_runtime_with_teams.jsonl"
# 月度数据文件目录，文件格式：raw_matches_YYYYMM.jsonl
MONTHLY_DATA_DIR = "data/monthly"
# 实时爬取的历史数据文件
LIVE_HISTORY_FILE = f"{MONTHLY_DATA_DIR}/raw_matches_live_oldindexall_with_teams.jsonl"
# 7月29日临时数据，需要查看今天数据时再取消注释
# "data/raw_matches_20260729_test.jsonl",   # 今天的测试比赛（包含勒沃库森vs亨克）
# "data/raw_matches_20260729_full.jsonl",   # 今天的全部爬取数据
# "data/raw_matches_20260729_first50.jsonl", # 今天的前50场
# "data/raw_matches_20260729.jsonl",        # 增量爬取的今天数据


_MODEL_BUNDLE: dict[str, Any] | None = None
_MODEL_LOCK = threading.Lock()
_CURRENT_CACHE: dict[str, dict[str, Any]] = {}
_CURRENT_REFRESHING: set[str] = set()
_CURRENT_LOCK = threading.Lock()
CURRENT_CACHE_TTL_SECONDS = 45.0
FORCE_TIME_BUDGET_S = 800.0  # 手动强制刷新的串行抓取时间预算(秒), 到点返回已抓部分, 避免前端超时空白
LIVE_REFRESH_SNAPSHOT_DIR = "data/live_refresh_snapshots"
# 历史数据缓存
_HISTORY_CACHE: dict[str, dict[str, Any]] = {}  # 按日期缓存预测结果
_RAW_DATA_CACHE: dict[str, list[dict[str, Any]]] = {}  # 按月份缓存原始数据，key: YYYYMM
_RAW_DATA_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _load_model_bundle() -> dict[str, Any]:
    global _MODEL_BUNDLE
    with _MODEL_LOCK:
        if _MODEL_BUNDLE is None:
            _MODEL_BUNDLE = joblib.load(MODEL_PATH)
    return _MODEL_BUNDLE


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return out

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                continue
    return out


def _build_empty_payload(source: str, date: str | None = None, note: str = "No matches available") -> dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "strategy": "first_3x_08_any",
        "source": source,
        "date": date,
        "note": note,
        "feed_live_match_count": 0,
        "match_count": 0,
        "rows": [],
    }


def _attach_cache_meta(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["cached_at"] = _utc_now()
    out["_cached_epoch"] = time.time()
    return out


def _build_live_placeholders(live_matches: list[dict[str, str]], note: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in live_matches:
        rows.append(
            {
                "match_id": str(item.get("match_id", "")),
                "home_team_name": str(item.get("home_team_feed", "") or ""),
                "away_team_name": str(item.get("away_team_feed", "") or ""),
                "league_name": str(item.get("league_name", "") or ""),
                "kickoff_time": str(item.get("kickoff_time", "") or ""),
                "current_score": str(item.get("current_score", "") or ""),
                "current_corner_score": None,
                "latest_minute": None,
                "latest_line": None,
                "latest_corners_so_far": None,
                "latest_pred": None,
                "latest_p_under": None,
                "latest_p_push": None,
                "latest_p_over": None,
                "triggered": False,
                "trigger_minute": None,
                "line": None,
                "corners_so_far": None,
                "pred_side": None,
                "pred_prob": None,
                "first_entry_minute": None,
                "first_entry_line": None,
                "first_entry_side": None,
                "first_entry_prob": None,
                "first_entry_text": None,
                "line_changes": None,
                "opp_conf_max_after": None,
                "risk_flag": None,
                "risk_reasons": [],
                "risk_alert": False,
                "risk_alert_text": "none",
                "risk_events": [],
                "hedge_recommended": False,
                "hedge_minute": None,
                "hedge_line": None,
                "hedge_side": None,
                "hedge_prob": None,
                "hedge_text": None,
                "all_triggers": [],
                "final_total_proxy_from_page": None,
                "final_relation_proxy": None,
                "hit_proxy": None,
                "minute_rows_last10": [],
                "live_placeholder": True,
                "live_placeholder_note": note,
            }
        )
    return rows


def _extract_latest_corner_score(record: dict[str, Any]) -> str | None:
    market_rows = record.get("market_rows") or []
    for rr in reversed(market_rows):
        raw = str(rr.get("score_raw", "") or "").strip()
        if not raw:
            continue
        m = re.search(r"(\d+)\s*[-:\uFF1A]\s*(\d+)", raw)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
        m2 = re.search(r"(\d+)\D+(\d+)", raw)
        if m2:
            return f"{m2.group(1)}-{m2.group(2)}"

    # Fallback: if side split is unavailable, expose total corners to avoid fake "0-0".
    for rr in reversed(market_rows):
        total = rr.get("corners_so_far")
        if total is None:
            continue
        try:
            return f"总{int(total)}"
        except Exception:
            continue
    return None


# --- 高置信度触发锁存(sticky latch): 首个因果高置信trigger一旦出现即固定, 不随刷新丢弃 ---
_STICKY_LATCH: dict[str, dict[str, Any]] = {}
_STICKY_LATCH_LOCK = threading.Lock()

# 每15分钟段 历史回测正确率(因果+锁存, 高置信>=0.88, 本地全量数据)
_SEGMENT_HIST_ACCURACY = {
    "0-15": 0.943, "15-30": 0.936, "30-45": 0.941,
    "45-60": 0.975, "60-75": 1.000, "75-90": 1.000, "90+": 1.000,
}


def _sticky_signal_from_row(row: dict, is_finished: bool) -> dict | None:
    if not row.get("sticky_triggered"):
        return None
    m = row.get("sticky_minute")
    prob = row.get("sticky_prob")
    if m is None or prob is None:
        return None
    return {
        "match_id": str(row.get("match_id", "")),
        "home_team_name": row.get("home_team_name", ""),
        "away_team_name": row.get("away_team_name", ""),
        "league_name": row.get("league_name", ""),
        "kickoff_time": row.get("kickoff_time", ""),
        "current_score": row.get("current_score", ""),
        "current_corner_score": row.get("current_corner_score", None),
        "trigger_minute": int(m),
        "pred_side": row.get("sticky_side"),
        "line": row.get("sticky_line"),
        "pred_prob": row.get("sticky_prob"),
        "corners_so_far": row.get("sticky_corners_so_far"),
        "is_finished": bool(is_finished),
        "final_total_corners": row.get("final_total_proxy_from_page"),
        "sticky_hit": row.get("sticky_hit"),
    }


def _build_segment_predictions(records: list[dict], model_bundle: dict, latch: bool = False,
                               conf_thr: float = 0.88, live_meta_by_match: dict[str, dict[str, Any]] | None = None) -> list[dict]:
    """按15分钟段汇总【因果+锁存】高置信触发。

    不预知(causal): 用 sticky_* 字段(首个 max(p_under,p_over)>=conf_thr 的3连点,
      触发时刻定方向/定线), 不做主导方向投票、不用全场加权线, 不预知未来。
    锁存(sticky): latch=True(实时看板)时把首见信号写入持久store, 之后刷新只保留、
      永不因方向翻转/风险变化而丢弃。latch=False(历史按日期)时按当前数据直接分段。
    """
    SEGMENTS = [
        ("0-15", 0, 15), ("15-30", 15, 30), ("30-45", 30, 45),
        ("45-60", 45, 60), ("60-75", 60, 75), ("75-90", 75, 90),
        ("90+", 90, 999),
    ]
    all_rows = build_live_strategy_rows(records, model_bundle)
    record_map = {str(r.get("match_id", "")): r for r in records}
    live_meta_by_match = live_meta_by_match or {}

    fresh: dict[str, dict] = {}
    for row in all_rows:
        sig = _sticky_signal_from_row(row, is_finished=(not latch))
        if sig is None or sig["pred_prob"] is None or float(sig["pred_prob"]) < conf_thr:
            continue
        mid = str(sig.get("match_id", ""))
        live_meta = live_meta_by_match.get(mid, {})
        rec = record_map.get(mid, {})
        if live_meta:
            sig["league_name"] = str(live_meta.get("league_name", "") or sig.get("league_name", "") or "")
            sig["kickoff_time"] = str(live_meta.get("kickoff_time", "") or sig.get("kickoff_time", "") or "")
            sig["current_score"] = str(live_meta.get("current_score", "") or sig.get("current_score", "") or "")
        if sig.get("current_corner_score") in (None, ""):
            sig["current_corner_score"] = _extract_latest_corner_score(rec)
        fresh[mid] = sig

    if latch:
        with _STICKY_LATCH_LOCK:
            for mid, sig in fresh.items():
                if mid not in _STICKY_LATCH:
                    _STICKY_LATCH[mid] = sig          # 首见锁存, 之后不覆盖方向/线/分钟
                elif sig.get("is_finished") and not _STICKY_LATCH[mid].get("is_finished"):
                    _STICKY_LATCH[mid]["is_finished"] = True
                    _STICKY_LATCH[mid]["final_total_corners"] = sig.get("final_total_corners")
                    _STICKY_LATCH[mid]["sticky_hit"] = sig.get("sticky_hit")
            signals = list(_STICKY_LATCH.values())
    else:
        signals = list(fresh.values())

    segment_map = {seg[0]: [] for seg in SEGMENTS}
    for sig in signals:
        mnt = int(sig["trigger_minute"])
        for seg_name, lo, hi in SEGMENTS:
            if lo <= mnt < hi:
                segment_map[seg_name].append(sig)
                break

    segment_results = []
    for seg_name, lo, hi in SEGMENTS:
        seg_rows = segment_map[seg_name]
        if not seg_rows:
            continue
        seg_rows.sort(key=lambda r: (float(r.get("pred_prob") or 0)), reverse=True)
        segment_results.append({
            "segment": seg_name,
            "seg_start": lo,
            "seg_end": hi,
            "match_count": len(seg_rows),
            "hist_accuracy": _SEGMENT_HIST_ACCURACY.get(seg_name),
            "rows": seg_rows,
        })
    return segment_results


def _build_current_live_payload(date_yyyymmdd: str | None = None, time_budget_s: float | None = None) -> dict[str, Any]:
    date_str = date_yyyymmdd or datetime.now().strftime("%Y%m%d")
    try:
        # 延长超时时间到30秒，适配海外网络环境
        feed_text = fetch_oldindexall_feed_text(timeout_s=12.0)
        live_matches = extract_live_matches_from_feed(feed_text)
        live_ids = [x["match_id"] for x in live_matches]
    except Exception as e:
        return _build_empty_payload(
            source="oldIndexall/bfdata_ut.js",
            date=date_str,
            note=f"Failed to fetch oldIndexall feed: {e}",
        )

    if not live_ids:
        return _build_empty_payload(
            source="oldIndexall/bfdata_ut.js",
            date=date_str,
            note="No in-play matches currently.",
        )

    summary = crawl_live_matches_from_ids(
        match_ids=live_ids,
        date_str=date_str,
        out_jsonl=RAW_OUTPUT_NOW,
        company_id=8,
        timeout_s=8.0,
        retries=1,
        backoff_s=0.3,
        time_budget_s=time_budget_s,
    )

    records = _load_jsonl(RAW_OUTPUT_NOW)
    rows = build_live_strategy_rows(records, _load_model_bundle())
    live_meta_by_match = {str(x.get("match_id", "")): x for x in live_matches}

    # 先建立match_id到原始记录的映射，避免顺序错位
    rec_map = {str(rec.get("match_id", "")): rec for rec in records}

    # 增加终场角球数字段
    for row in rows:
        match_id = str(row.get("match_id", ""))
        rec = rec_map.get(match_id, {})
        meta = live_meta_by_match.get(match_id, {})
        row["league_name"] = str(meta.get("league_name", "") or "")
        row["kickoff_time"] = str(meta.get("kickoff_time", "") or "")
        row["current_score"] = str(meta.get("current_score", "") or "")
        row["current_corner_score"] = _extract_latest_corner_score(rec)
        row["final_total_corners"] = rec.get("final_total_corners", None)
        # 实时比赛如果终场角球数存在说明已经结束
        row["is_finished"] = row["final_total_corners"] is not None

    note = "For live matches, final relation uses current page corners as proxy, not full-time settled result."

    if not rows:
        note = "Current matches detected, but the corner detail source returned no usable market rows for these matches, so strategy results cannot be computed right now."
        rows = _build_live_placeholders(
            live_matches,
            note=note,
        )

    # 构建各时间段预测结果
    segment_predictions = _build_segment_predictions(
        records,
        _load_model_bundle(),
        latch=True,
        live_meta_by_match=live_meta_by_match,
    )

    return {
        "generated_at": _utc_now(),
        "strategy": "first_3x_08_any",
        "source": "oldIndexall/bfdata_ut.js",
        "date": date_str,
        "note": note,
        "feed_live_match_count": len(live_ids),
        "crawl_summary": summary,
        "match_count": len(rows),
        "rows": rows,
        "segment_predictions": segment_predictions,  # 新增：各时间段高置信度推荐
    }


def _refresh_current_cache(date_str: str) -> None:
    try:
        payload = _build_current_live_payload(date_str)
        with _CURRENT_LOCK:
            _CURRENT_CACHE[date_str] = _attach_cache_meta(payload)
    except Exception as e:
        fallback = _build_empty_payload(
            source="oldIndexall/bfdata_ut.js",
            date=date_str,
            note=f"Background refresh failed: {e}",
        )
        with _CURRENT_LOCK:
            if date_str not in _CURRENT_CACHE:
                _CURRENT_CACHE[date_str] = _attach_cache_meta(fallback)
    finally:
        with _CURRENT_LOCK:
            _CURRENT_REFRESHING.discard(date_str)


def _get_current_live_cached_or_start(date_str: str, allow_background_refresh: bool = True) -> dict[str, Any]:
    now = time.time()
    with _CURRENT_LOCK:
        cached = _CURRENT_CACHE.get(date_str)
        is_refreshing = date_str in _CURRENT_REFRESHING

        if cached is not None:
            age_s = now - float(cached.get("_cached_epoch", 0.0))
            if age_s <= CURRENT_CACHE_TTL_SECONDS:
                out = dict(cached)
                out.pop("_cached_epoch", None)
                return out

        if allow_background_refresh and not is_refreshing:
            _CURRENT_REFRESHING.add(date_str)
            t = threading.Thread(target=_refresh_current_cache, args=(date_str,), daemon=True)
            t.start()

        if cached is not None:
            out = dict(cached)
            out.pop("_cached_epoch", None)
            if allow_background_refresh:
                out["note"] = f"Using cached live data while refreshing in background. {out.get('note', '')}".strip()
            else:
                out["note"] = f"Using cached live data only. {out.get('note', '')}".strip()
            return out

    if not allow_background_refresh:
        return _build_empty_payload(
            source="oldIndexall/bfdata_ut.js",
            date=date_str,
            note="No cached live data yet. Click '立即刷新当前进行中' to run one refresh round.",
        )

    try:
        # 延长超时时间到20秒，适配海外网络环境
        feed_text = fetch_oldindexall_feed_text(timeout_s=4.0)
        live_matches = extract_live_matches_from_feed(feed_text)
    except Exception:
        live_matches = []

    if live_matches:
        placeholder_payload = {
            "generated_at": _utc_now(),
            "strategy": "first_3x_08_any",
            "source": "oldIndexall/bfdata_ut.js",
            "date": date_str,
            "note": "Current matches detected. Strategy details are refreshing in background.",
            "feed_live_match_count": len(live_matches),
            "match_count": len(live_matches),
            "rows": _build_live_placeholders(
                live_matches,
                note="Current matches detected. Strategy details are refreshing in background.",
            ),
        }
        with _CURRENT_LOCK:
            _CURRENT_CACHE[date_str] = _attach_cache_meta(placeholder_payload)
        return placeholder_payload

    return _build_empty_payload(
        source="oldIndexall/bfdata_ut.js",
        date=date_str,
        note="Refreshing current live data in background. Please retry in a few seconds.",
    )


def _normalize_date(date_text: str) -> str | None:
    s = str(date_text or "").strip()
    if not s:
        return None
    s = s.replace("-", "")
    if re.fullmatch(r"\d{8}", s):
        return s
    return None


def _normalize_match_ids(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for part in str(raw or "").split(","):
            mid = part.strip()
            if not re.fullmatch(r"\d{6,10}", mid):
                continue
            if mid in seen:
                continue
            seen.add(mid)
            out.append(mid)
    return out


def _build_rows_payload_from_match_ids(date_str: str, match_ids: list[str], source: str, note: str) -> dict[str, Any]:
    Path("data").mkdir(parents=True, exist_ok=True)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", prefix="live_match_query_", dir="data", delete=False) as tmp:
            tmp_path = tmp.name

        summary = crawl_live_matches_from_ids(
            match_ids=match_ids,
            date_str=date_str,
            out_jsonl=tmp_path,
            company_id=8,
            timeout_s=8.0,
            retries=1,
            backoff_s=0.3,
        )
        records = _load_jsonl(tmp_path)
        rows = build_live_strategy_rows(records, _load_model_bundle())

        # 先建立match_id到原始记录的映射，避免顺序错位
        rec_map = {str(rec.get("match_id", "")): rec for rec in records}

        # 增加终场角球数字段
        for row in rows:
            match_id = str(row.get("match_id", ""))
            rec = rec_map.get(match_id, {})
            row["final_total_corners"] = rec.get("final_total_corners", None)
            row["is_finished"] = row["final_total_corners"] is not None

        if not rows:
            note = f"Requested match ids returned no usable market rows: {', '.join(match_ids)}"

        return {
            "generated_at": _utc_now(),
            "strategy": "first_3x_08_any",
            "source": source,
            "date": date_str,
            "note": note,
            "feed_live_match_count": len(match_ids),
            "crawl_summary": summary,
            "match_count": len(rows),
            "rows": rows,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _load_month_data(month_str: str) -> list[dict[str, Any]]:
    """加载指定月份的原始数据，月份格式：YYYYMM
    先检查是否有按周拆分的文件（格式YYYYMM_Wx），如果有则加载所有周文件
    否则尝试加载整月文件
    """
    global _RAW_DATA_CACHE
    with _RAW_DATA_LOCK:
        if month_str in _RAW_DATA_CACHE:
            return _RAW_DATA_CACHE[month_str]

        data = []
        month_file = f"{MONTHLY_DATA_DIR}/raw_matches_{month_str}.jsonl"

        # 先检查是否有按周拆分的文件
        if os.path.exists(MONTHLY_DATA_DIR):
            for filename in os.listdir(MONTHLY_DATA_DIR):
                # 匹配周文件格式：raw_matches_YYYYMM_Wx.jsonl
                if filename.startswith(f"raw_matches_{month_str}_W") and filename.endswith(".jsonl"):
                    week_file = os.path.join(MONTHLY_DATA_DIR, filename)
                    week_data = _load_jsonl(week_file)
                    data.extend(week_data)
                    print(f"Loaded week file: {filename}, {len(week_data)} records")

        # 如果没有周文件，尝试加载整月文件
        if not data and os.path.exists(month_file):
            data = _load_jsonl(month_file)
            print(f"Loaded month file: raw_matches_{month_str}.jsonl, {len(data)} records")

        # 同时加载实时历史文件中的数据（可能包含近期数据）
        if os.path.exists(LIVE_HISTORY_FILE):
            live_data = _load_jsonl(LIVE_HISTORY_FILE)
            # 筛选属于该月份的数据
            live_data_month = [
                rec for rec in live_data
                if str(rec.get("date", "")).startswith(month_str)
            ]
            data.extend(live_data_month)

        # 存入缓存
        _RAW_DATA_CACHE[month_str] = data
        return data


def _build_date_payload(date_yyyymmdd: str) -> dict[str, Any]:
    target = _normalize_date(date_yyyymmdd)
    if not target:
        return _build_empty_payload(source="historical-jsonl", date=date_yyyymmdd, note="Invalid date format")

    # 先检查日期缓存
    global _HISTORY_CACHE
    if target in _HISTORY_CACHE:
        return _HISTORY_CACHE[target]

    # 提取月份（前6位：YYYYMM）
    month_str = target[:6]

    # 加载对应月份的数据
    month_data = _load_month_data(month_str)
    if not month_data:
        payload = _build_empty_payload(
            source="historical-jsonl",
            date=target,
            note=f"No records found for month {month_str}.",
        )
        _HISTORY_CACHE[target] = payload
        return payload

    # 筛选目标日期
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in month_data:
        if str(rec.get("date", "")) != target:
            continue
        mid = str(rec.get("match_id", "")).strip()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        filtered.append(rec)

    if not filtered:
        payload = _build_empty_payload(
            source="historical-jsonl",
            date=target,
            note=f"No records found for date {target}.",
        )
        _HISTORY_CACHE[target] = payload
        return payload

    rows = build_live_strategy_rows(filtered, _load_model_bundle())

    # 先建立match_id到原始记录的映射，避免顺序错位
    rec_map = {str(rec.get("match_id", "")): rec for rec in filtered}

    # 增加终场角球数字段（历史数据都是已结束的）
    for row in rows:
        match_id = str(row.get("match_id", ""))
        rec = rec_map.get(match_id, {})
        row["final_total_corners"] = rec.get("final_total_corners", None)
        row["is_finished"] = True

    # 构建各时间段预测结果
    segment_predictions = _build_segment_predictions(filtered, _load_model_bundle())

    payload = {
        "generated_at": _utc_now(),
        "strategy": "first_3x_08_any",
        "source": "historical-jsonl",
        "date": target,
        "note": "Computed from stored monthly match snapshots for the selected date.",
        "feed_live_match_count": None,
        "match_count": len(rows),
        "rows": rows,
        "segment_predictions": segment_predictions,  # 新增：各时间段高置信度推荐
    }

    # 存入缓存，下次直接返回
    _HISTORY_CACHE[target] = payload
    return payload


def _build_date_match_payload(date_yyyymmdd: str, match_ids: list[str]) -> dict[str, Any]:
    target = _normalize_date(date_yyyymmdd)
    if not target:
        return _build_empty_payload(source="historical-jsonl", date=date_yyyymmdd, note="Invalid date format")

    match_id_set = set(match_ids)
    # 提取月份
    month_str = target[:6]
    # 加载对应月份的数据
    month_data = _load_month_data(month_str)

    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in month_data:
        if str(rec.get("date", "")) != target:
            continue
        mid = str(rec.get("match_id", "")).strip()
        if not mid or mid in seen or mid not in match_id_set:
            continue
        seen.add(mid)
        filtered.append(rec)

    if not filtered:
        return _build_empty_payload(
            source="historical-jsonl",
            date=target,
            note=f"No records found for date {target} and match ids: {', '.join(match_ids)}",
        )

    rows = build_live_strategy_rows(filtered, _load_model_bundle())

    # 先建立match_id到原始记录的映射，避免顺序错位
    rec_map = {str(rec.get("match_id", "")): rec for rec in filtered}

    # 增加终场角球数字段
    for row in rows:
        match_id = str(row.get("match_id", ""))
        rec = rec_map.get(match_id, {})
        row["final_total_corners"] = rec.get("final_total_corners", None)
        row["is_finished"] = True

    # 构建各时间段预测结果
    segment_predictions = _build_segment_predictions(filtered, _load_model_bundle())

    return {
        "generated_at": _utc_now(),
        "strategy": "first_3x_08_any",
        "source": "historical-jsonl",
        "date": target,
        "note": f"Computed from stored monthly match snapshots for the selected date and match ids: {', '.join(match_ids)}.",
        "feed_live_match_count": None,
        "match_count": len(rows),
        "rows": rows,
        "segment_predictions": segment_predictions,  # 新增：各时间段高置信度推荐
    }


def _persist_live_refresh_snapshot(
    payload: dict[str, Any],
    endpoint: str,
    date_str: str,
    raw_query: str,
    match_ids: list[str],
) -> str | None:
    # Best-effort persistence for debugging trigger drift across refreshes.
    try:
        target_date = _normalize_date(date_str) or datetime.now().strftime("%Y%m%d")
        utc_now = datetime.utcnow()
        ts = utc_now.strftime("%Y%m%dT%H%M%S")
        micros = f"{utc_now.microsecond:06d}"

        day_dir = Path(LIVE_REFRESH_SNAPSHOT_DIR) / target_date
        day_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{ts}_{micros}_{int(time.time_ns() % 1000000):06d}.json"
        out_path = day_dir / file_name

        to_write = {
            "snapshot_written_at": _utc_now(),
            "endpoint": endpoint,
            "query": raw_query,
            "date": target_date,
            "requested_match_ids": list(match_ids),
            "payload": payload,
        }
        out_path.write_text(json.dumps(to_write, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out_path)
    except Exception:
        return None


class DashboardHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        # Prevent stale frontend JS/HTML from being reused after server-side refresh logic changes.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _send_json(self, payload: dict[str, Any], status_code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Client aborted the request (e.g. frontend fetch timeout). Safe to ignore.
            pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        match_ids = _normalize_match_ids(query.get("match_id", []))

        if path == "/api/live/current":
            date_text = query.get("date", [""])[0]
            date_str = _normalize_date(date_text) or datetime.now().strftime("%Y%m%d")
            if match_ids:
                payload = _build_rows_payload_from_match_ids(
                    date_str=date_str,
                    match_ids=match_ids,
                    source="manual-match-id",
                    note=f"Computed by explicitly refreshing match ids: {', '.join(match_ids)}",
                )
            else:
                force = query.get("force", ["0"])[0] in ("1", "true", "yes")
                if force:
                    payload = _build_current_live_payload(date_str, time_budget_s=FORCE_TIME_BUDGET_S)
                    with _CURRENT_LOCK:
                        _CURRENT_CACHE[date_str] = _attach_cache_meta(payload)
                    payload = dict(payload)
                    payload["note"] = f"Refresh requested (one-shot). {payload.get('note', '')}".strip()
                else:
                    # 普通读取只返回缓存，不再隐式触发后台刷新，避免“点一次后持续自动刷新”。
                    payload = _get_current_live_cached_or_start(date_str, allow_background_refresh=True)
                    # 兼容旧前端：若未带 force 且当前日期缓存为空，则补一次同步刷新。
                    # 这只会在该日期“首次无缓存”时触发，后续请求继续走纯缓存读取。
                    if (payload.get("match_count") in (None, 0)
                            and "No cached live data yet" in str(payload.get("note", ""))):
                        payload = _build_current_live_payload(date_str)
                        with _CURRENT_LOCK:
                            _CURRENT_CACHE[date_str] = _attach_cache_meta(payload)
                        payload = dict(payload)
                        payload["note"] = f"Refresh requested (compat mode, cache bootstrap). {payload.get('note', '')}".strip()

            snapshot_path = _persist_live_refresh_snapshot(
                payload=payload,
                endpoint=path,
                date_str=date_str,
                raw_query=parsed.query,
                match_ids=match_ids,
            )
            if snapshot_path:
                payload = dict(payload)
                payload["snapshot_file"] = snapshot_path

            self._send_json(payload)
            return

        if path == "/api/live/by-date":
            date_text = query.get("date", [""])[0]
            if not _normalize_date(date_text):
                self._send_json(
                    _build_empty_payload(source="historical-jsonl", date=date_text, note="Query date must be YYYYMMDD or YYYY-MM-DD"),
                    status_code=400,
                )
                return
            if match_ids:
                payload = _build_date_match_payload(date_text, match_ids)
            else:
                payload = _build_date_payload(date_text)
            self._send_json(payload)
            return

        if path == "/":
            self.path = "/live_strategy_dashboard.html"
        return super().do_GET()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serve live strategy dashboard and APIs")
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", type=int, default=PORT)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    Path("data").mkdir(parents=True, exist_ok=True)
    host = str(args.host)
    port = int(args.port)

    try:
        server = ThreadingHTTPServer((host, port), DashboardHandler)
    except OSError as e:
        if e.errno == 48:
            print(f"Port {port} is already in use on {host}. Try another port, e.g. --port {port + 1}")
        raise

    print(f"Serving dashboard at http://{host}:{port}/live_strategy_dashboard.html")
    print(f"API current: http://{host}:{port}/api/live/current")
    print(f"API by date: http://{host}:{port}/api/live/by-date?date=2026-07-28")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
