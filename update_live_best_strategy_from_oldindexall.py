from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

from pipeline_e2e_v2 import FEATURE_COLS, _fetch_one_match_raw, build_features_for_match
from train_settlement_relation_model import SETTLEMENT_FEATURE_COLS


OLDINDEXALL_FEED = "https://livestatic.titan007.com/vbsxml/bfdata_ut.js"
LIVE_STATES = {"1", "2", "3"}


def _build_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Referer": "https://live.titan007.com/oldIndexall.aspx",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }


def fetch_oldindexall_feed_text(timeout_s: float = 20.0) -> str:
    r_value = f"007{int(time.time())}000"
    url = f"{OLDINDEXALL_FEED}?r={r_value}"
    resp = requests.get(url, headers=_build_headers(), timeout=timeout_s)
    resp.raise_for_status()
    return resp.text


def _clean_html_text(s: str) -> str:
    # Team names sometimes include inline tags like "<font ...>(中)</font>".
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_live_matches_from_feed(feed_text: str) -> list[dict[str, str]]:
    pat = re.compile(r"A\[\d+\]=\"(.*?)\"\.split\('\^'\);?", flags=re.S)
    out: list[dict[str, str]] = []
    seen: set[str] = set()

    for m in pat.finditer(feed_text):
        fields = m.group(1).split("^")
        if len(fields) < 16:
            continue

        match_id = fields[0].strip()
        if not re.fullmatch(r"\d{6,10}", match_id):
            continue

        state = fields[13].strip() if len(fields) > 13 else ""
        if state not in LIVE_STATES:
            continue
        if match_id in seen:
            continue
        seen.add(match_id)

        league_name = _clean_html_text(fields[2] if len(fields) > 2 else "")
        home_team = _clean_html_text(fields[5] if len(fields) > 5 else "")
        away_team = _clean_html_text(fields[8] if len(fields) > 8 else "")

        out.append(
            {
                "match_id": match_id,
                "state": state,
                "league_name": league_name,
                "home_team_feed": home_team,
                "away_team_feed": away_team,
            }
        )

    return out


def _line_to_ticks(line: float) -> int:
    return int(round(float(line) * 2.0))


def _line_fraction(line: float) -> float:
    frac = abs(float(line) - round(float(line)))
    return float(round(frac, 2))


def _build_settlement_features_for_match(match: dict[str, Any]) -> pd.DataFrame:
    feat_df = build_features_for_match(match, drop_push=False)
    if feat_df.empty:
        return pd.DataFrame()

    feat_df = feat_df.sort_values(["minute", "line", "corners_so_far"]).reset_index(drop=True)
    final_total = int(match["final_total_corners"])

    corners = feat_df["corners_so_far"].astype(float).to_numpy()
    line = feat_df["line"].astype(float).to_numpy()
    minute = feat_df["minute"].astype(int).to_numpy()
    second_half = (minute >= 46).astype(int)
    is_late = (minute >= 75).astype(int)

    line_fraction = np.array([_line_fraction(v) for v in line], dtype=np.float32)
    line_ticks = np.array([_line_to_ticks(v) for v in line], dtype=np.int32)
    line_minus_corners = line - corners
    final_proxy = np.maximum(corners, line)

    corner_inc = np.zeros_like(corners)
    corner_inc[0] = max(0, corners[0])
    corner_inc[1:] = np.maximum(0, corners[1:] - corners[:-1])

    second_half_corners = np.zeros_like(corners)
    second_half_corners[second_half == 1] = corner_inc[second_half == 1]
    second_half_corners = np.cumsum(second_half_corners)

    def moving_sum(values: np.ndarray, window: int) -> np.ndarray:
        out = np.zeros(len(values), dtype=np.float32)
        for i in range(len(values)):
            left = max(0, i - window + 1)
            out[i] = float(np.sum(values[left : i + 1]))
        return out

    rate_10m = moving_sum(corner_inc, 10) / 10.0
    rate_5m = moving_sum(corner_inc, 5) / 5.0
    rate_2m = moving_sum(corner_inc, 2) / 2.0
    second_half_rate = np.where(second_half == 1, second_half_corners / np.maximum(1, minute - 45), 0.0)

    # Keep the same feature recipe used by settlement model training.
    out_df = feat_df.assign(
        line_rank_in_match=feat_df["line"].rank(method="average", pct=True).astype(float),
        minute_progress=feat_df["minute"].astype(float) / 100.0,
        corners_progress=np.minimum(feat_df["corners_so_far"].astype(float) / 20.0, 1.5),
        line_minus_corners=line_minus_corners,
        line_minus_final_proxy=line - final_proxy,
        is_second_half=second_half,
        is_late_game=is_late,
        is_integer_line=(line_fraction == 0.0).astype(int),
        is_half_line=(line_fraction == 0.5).astype(int),
        distance_to_int=np.minimum(line_fraction, np.abs(line_fraction - 1.0)),
        distance_to_half=np.abs(line_fraction - 0.5),
        corner_rate_10m=rate_10m,
        corner_rate_5m=rate_5m,
        corner_rate_2m=rate_2m,
        second_half_corners_so_far=second_half_corners,
        second_half_corner_rate=second_half_rate,
        second_half_3_4_flag=((second_half_corners >= 3) & (second_half_corners <= 4)).astype(int),
        second_half_0_2_flag=((second_half_corners >= 0) & (second_half_corners <= 2)).astype(int),
        second_half_5plus_flag=(second_half_corners >= 5).astype(int),
        line_ticks=line_ticks,
    )

    keep_raw = ["match_id", "minute", "line", "corners_so_far", "final_total_corners"] + FEATURE_COLS + [
        c for c in SETTLEMENT_FEATURE_COLS if c not in FEATURE_COLS
    ]
    keep = list(dict.fromkeys(keep_raw))
    out_df = out_df.assign(final_total_corners=final_total)
    return out_df[keep]


def _label_name(label: int) -> str:
    if int(label) == 0:
        return "under"
    if int(label) == 1:
        return "push"
    return "over"


def _proxy_relation(final_total: int, line: float) -> str:
    if float(final_total) > float(line):
        return "over"
    if float(final_total) < float(line):
        return "under"
    return "push"


def _find_first_3x_08_any(df: pd.DataFrame) -> tuple[bool, int | None, str | None, float | None, float | None, int | None]:
    if df.empty or len(df) < 3:
        return False, None, None, None, None, None

    sides = []
    confs = []
    for _, row in df.iterrows():
        pu = float(row["p_under"])
        po = float(row["p_over"])
        if pu >= po:
            sides.append("under")
            confs.append(pu)
        else:
            sides.append("over")
            confs.append(po)

    for i in range(len(df) - 2):
        c0, c1, c2 = confs[i], confs[i + 1], confs[i + 2]
        s0, s1, s2 = sides[i], sides[i + 1], sides[i + 2]
        if c0 >= 0.8 and c1 >= 0.8 and c2 >= 0.8 and s0 == s1 == s2:
            row = df.iloc[i]
            return (
                True,
                int(row["minute"]),
                s0,
                float(max(float(row["p_under"]), float(row["p_over"]))),
                float(row["line"]),
                int(row["corners_so_far"]),
            )

    return False, None, None, None, None, None


def _collect_all_triggers(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or len(df) < 3:
        return []

    sides: list[str] = []
    confs: list[float] = []
    for _, row in df.iterrows():
        pu = float(row["p_under"])
        po = float(row["p_over"])
        if pu >= po:
            sides.append("under")
            confs.append(pu)
        else:
            sides.append("over")
            confs.append(po)

    out: list[dict[str, Any]] = []
    seen_windows = set()  # 避免重复触发同一时间窗口
    for i in range(len(df) - 2):
        c0, c1, c2 = confs[i], confs[i + 1], confs[i + 2]
        s0, s1, s2 = sides[i], sides[i + 1], sides[i + 2]
        if c0 >= 0.8 and c1 >= 0.8 and c2 >= 0.8 and s0 == s1 == s2:
            row0 = df.iloc[i]
            row2 = df.iloc[i + 2]
            # 避免同一分钟的重复触发
            window_key = (int(row0["minute"]), s0, round(float(row0["line"]), 1))
            if window_key in seen_windows:
                continue
            seen_windows.add(window_key)

            out.append(
                {
                    "start_index": int(i),
                    "start_minute": int(row0["minute"]),
                    "end_minute": int(row2["minute"]),
                    "side": s0,
                    "line": round(float(row0["line"]), 1),
                    "corners_so_far": int(row0["corners_so_far"]),
                    "pred_prob": round(float(max(float(row0["p_under"]), float(row0["p_over"]))), 4),
                    "conf_triplet": [round(float(c0), 4), round(float(c1), 4), round(float(c2), 4)],
                    "line_triplet": [
                        round(float(df.iloc[i]["line"]), 1),
                        round(float(df.iloc[i + 1]["line"]), 1),
                        round(float(df.iloc[i + 2]["line"]), 1),
                    ],
                }
            )
    return out


def _aggregate_triggers_by_side(all_triggers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    将触发点按方向分组
    :param all_triggers: 所有触发点列表
    :return: 按over/under分组的触发点字典
    """
    groups = {"over": [], "under": []}
    for t in all_triggers:
        side = t["side"]
        if side in groups:
            groups[side].append(t)
    return groups


def _calculate_recommendation_interval(triggers: list[dict[str, Any]]) -> dict[str, Any]:
    """
    计算同方向触发点的推荐区间和加权盘口
    :param triggers: 同方向的触发点列表
    :return: 区间计算结果
    """
    if not triggers:
        return {}

    # 按时间排序（越晚的触发点权重越高）
    sorted_triggers = sorted(triggers, key=lambda x: x["start_minute"])
    n = len(sorted_triggers)

    # 提取数据
    lines = [t["line"] for t in sorted_triggers]
    probs = [t["pred_prob"] for t in sorted_triggers]
    minutes = [t["start_minute"] for t in sorted_triggers]

    # 计算区间
    min_line = min(lines)
    max_line = max(lines)
    interval_width = max_line - min_line

    # 时间加权平均盘口（越新的权重越高，权重 = 1 + 0.1 * 时间顺序）
    weights = [1.0 + 0.1 * i for i in range(n)]
    weighted_line = sum(l * w for l, w in zip(lines, weights)) / sum(weights)

    # 平均置信度
    avg_confidence = sum(probs) / n

    # 稳定性评级
    if interval_width <= 0.5:
        stability = "high"
    elif interval_width <= 1.0:
        stability = "medium"
    else:
        stability = "low"

    return {
        "trigger_count": n,
        "trigger_interval": [round(min_line, 1), round(max_line, 1)],
        "interval_width": round(interval_width, 1),
        "weighted_recommended_line": round(weighted_line, 2),
        "average_confidence": round(avg_confidence, 4),
        "stability_rating": stability,
        "trigger_minutes": minutes,
        "trigger_lines": lines,
        "trigger_probs": probs
    }


def build_live_strategy_rows(matches: list[dict[str, Any]], model_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    model = model_bundle["model"]
    model_features = list(model_bundle.get("features", SETTLEMENT_FEATURE_COLS))
    expected_features = list(getattr(model, "feature_names_in_", model_features))
    rows: list[dict[str, Any]] = []

    for m in matches:
        df = _build_settlement_features_for_match(m)
        if df.empty:
            continue

        X = df.reindex(columns=expected_features).fillna(0.0).astype(float)
        p = model.predict_proba(X)
        if p.shape[1] != 3:
            continue

        out = df[["minute", "line", "corners_so_far"]].copy()
        out["p_under"] = p[:, 0]
        out["p_push"] = p[:, 1]
        out["p_over"] = p[:, 2]
        out["pred"] = np.argmax(p, axis=1)
        out["pred_label"] = out["pred"].apply(_label_name)
        out = out.sort_values(["minute", "line", "corners_so_far"]).reset_index(drop=True)

        latest = out.iloc[-1]
        latest_minute = int(latest["minute"])
        latest_line = float(latest["line"])
        latest_corners = int(latest["corners_so_far"])
        latest_pred = str(latest["pred_label"])

        all_triggers = _collect_all_triggers(out)
        trigger_summary = {}
        first_entry_text: str | None = None
        final_relation_proxy = None
        hit_proxy = None

        if all_triggers:
            # 按方向分组触发点
            trigger_groups = _aggregate_triggers_by_side(all_triggers)
            over_count = len(trigger_groups["over"])
            under_count = len(trigger_groups["under"])

            # 确定主导方向（触发点更多的方向）
            dominant_side = None
            dominant_triggers = []
            if over_count > under_count:
                dominant_side = "over"
                dominant_triggers = trigger_groups["over"]
            elif under_count > over_count:
                dominant_side = "under"
                dominant_triggers = trigger_groups["under"]
            else:
                # 数量相等时选出现时间更早的方向
                if all_triggers:
                    dominant_side = all_triggers[0]["side"]
                    dominant_triggers = trigger_groups[dominant_side]

            # 计算推荐区间
            if dominant_triggers:
                interval_data = _calculate_recommendation_interval(dominant_triggers)
                opposite_triggers = trigger_groups["under"] if dominant_side == "over" else trigger_groups["over"]
                opposite_count = len(opposite_triggers)

                # 构建触发汇总信息
                trigger_summary = {
                    "dominant_side": dominant_side,
                    "trigger_count": interval_data["trigger_count"],
                    "trigger_interval": interval_data["trigger_interval"],
                    "interval_width": interval_data["interval_width"],
                    "weighted_recommended_line": interval_data["weighted_recommended_line"],
                    "average_confidence": interval_data["average_confidence"],
                    "stability_rating": interval_data["stability_rating"],
                    "has_opposite_triggers": opposite_count > 0,
                    "opposite_trigger_count": opposite_count,
                    "all_triggers": all_triggers  # 保留所有触发点用于前端展示
                }

                # 兼容原有字段：使用加权推荐盘口作为主trigger值
                triggered = True
                first = dominant_triggers[0]  # 第一个触发点
                trigger_idx = int(first["start_index"])
                trigger_minute = int(first["start_minute"])
                pred_side = dominant_side
                pred_prob = interval_data["average_confidence"]
                trigger_line = interval_data["weighted_recommended_line"]  # 使用加权盘口
                trigger_corners = int(first["corners_so_far"])

                # 构建显示文本
                interval_str = f"{interval_data['trigger_interval'][0]}-{interval_data['trigger_interval'][1]}"
                first_entry_text = (
                    f"{trigger_minute}' {pred_side} {interval_str} "
                    f"(avg conf: {interval_data['average_confidence']:.4f}, "
                    f"{interval_data['trigger_count']} triggers)"
                )
            else:
                triggered = False
                trigger_idx = None
                trigger_minute = None
                pred_side = None
                pred_prob = None
                trigger_line = None
                trigger_corners = None
        else:
            triggered = False
            trigger_idx = None
            trigger_minute = None
            pred_side = None
            pred_prob = None
            trigger_line = None
            trigger_corners = None

        # first_entry_text 已经在trigger处理阶段定义，避免重复覆盖

        final_proxy = int(m.get("final_total_corners", 0))
        if triggered and trigger_line is not None:
            final_relation_proxy = _proxy_relation(final_proxy, trigger_line)
            hit_proxy = bool(final_relation_proxy == pred_side) if final_relation_proxy is not None and pred_side else None
        else:
            final_relation_proxy = None
            hit_proxy = None

        line_arr = out["line"].to_numpy()
        line_changes = int(np.sum(np.abs(np.diff(line_arr)) > 1e-9)) if len(out) > 1 else 0

        opp_conf_max_after: float | None
        risk_flag: int | None
        risk_reasons: list[str]
        if triggered and trigger_idx is not None and pred_side is not None and pred_prob is not None and trigger_line is not None:
            post = out.iloc[trigger_idx + 1 :]
            if post.empty:
                opp_conf_max_after = 0.0
            elif pred_side == "under":
                opp_conf_max_after = float(post["p_over"].max())
            else:
                opp_conf_max_after = float(post["p_under"].max())

            # 基础风险评分
            risk_flag = (
                int(float(pred_prob) < 0.88)
                + int(float(opp_conf_max_after) >= 0.8)
                + int(line_changes >= 12)
                + int(float(trigger_line) >= 11.5)
            )
            risk_reasons = []
            if float(pred_prob) < 0.88:
                risk_reasons.append("low_trigger_prob")
            if float(opp_conf_max_after) >= 0.8:
                risk_reasons.append("strong_opposite_after_trigger")
            if line_changes >= 12:
                risk_reasons.append("high_line_volatility")
            if float(trigger_line) >= 11.5:
                risk_reasons.append("high_trigger_line")

            # 新增：基于多触发点的风险因素
            if trigger_summary:
                # 区间过宽风险
                if trigger_summary.get("interval_width", 0) > 1.0:
                    risk_flag += 1
                    risk_reasons.append("wide_trigger_interval")

                # 反向触发点过多风险
                if trigger_summary.get("opposite_trigger_count", 0) >= 2:
                    risk_flag += 1
                    risk_reasons.append("multiple_opposite_triggers")

                # 触发点数量过少风险（只有1个触发点）
                if trigger_summary.get("trigger_count", 0) == 1:
                    risk_flag += 1
                    risk_reasons.append("single_trigger_point")
        else:
            opp_conf_max_after = None
            risk_flag = None
            risk_reasons = []

        risk_events: list[dict[str, Any]] = []
        hedge_recommended = False
        hedge_minute: int | None = None
        hedge_line: float | None = None
        hedge_side: str | None = None
        hedge_prob: float | None = None
        hedge_text: str | None = None

        if triggered and trigger_idx is not None and pred_side is not None:
            opposite_triggers = [
                t
                for t in all_triggers
                if int(t.get("start_index", -1)) > int(trigger_idx) and str(t.get("side", "")) != str(pred_side)
            ]
            if opposite_triggers:
                t0 = opposite_triggers[0]
                hedge_recommended = True
                hedge_minute = int(t0["start_minute"])
                hedge_line = float(t0["line"])
                hedge_side = str(t0["side"])
                hedge_prob = float(t0["pred_prob"])
                hedge_text = f"{hedge_minute}' {hedge_side} {round(hedge_line, 1)} ({round(hedge_prob, 4)})"
                risk_events.append(
                    {
                        "type": "opposite_trigger",
                        "minute": hedge_minute,
                        "line": round(hedge_line, 1),
                        "side": hedge_side,
                        "prob": round(hedge_prob, 4),
                    }
                )

        risk_alert = bool((risk_flag is not None and int(risk_flag) >= 3) or len(risk_events) > 0)
        risk_alert_text: str
        if not risk_alert:
            risk_alert_text = "none"
        else:
            chunks: list[str] = []
            if risk_reasons:
                chunks.append("reasons=" + ",".join(risk_reasons))
            if hedge_text:
                chunks.append("hedge=" + hedge_text)
            risk_alert_text = " | ".join(chunks) if chunks else "alert"

        minute_rows_last10 = []
        for _, r in out.tail(10).iterrows():
            minute_rows_last10.append(
                {
                    "minute": int(r["minute"]),
                    "line": float(r["line"]),
                    "corners_so_far": int(r["corners_so_far"]),
                    "pred": str(r["pred_label"]),
                    "p_under": round(float(r["p_under"]), 4),
                    "p_push": round(float(r["p_push"]), 4),
                    "p_over": round(float(r["p_over"]), 4),
                }
            )

        rows.append(
            {
                "match_id": str(m.get("match_id", "")),
                "home_team_name": str(m.get("home_team_name", "") or ""),
                "away_team_name": str(m.get("away_team_name", "") or ""),
                "latest_minute": latest_minute,
                "latest_line": round(latest_line, 1),
                "latest_corners_so_far": latest_corners,
                "latest_pred": latest_pred,
                "latest_p_under": round(float(latest["p_under"]), 4),
                "latest_p_push": round(float(latest["p_push"]), 4),
                "latest_p_over": round(float(latest["p_over"]), 4),
                "triggered": bool(triggered),
                "trigger_minute": int(trigger_minute) if trigger_minute is not None else None,
                "line": round(float(trigger_line), 1) if trigger_line is not None else None,
                "corners_so_far": int(trigger_corners) if trigger_corners is not None else None,
                "pred_side": pred_side,
                "pred_prob": round(float(pred_prob), 4) if pred_prob is not None else None,
                "first_entry_minute": int(trigger_minute) if trigger_minute is not None else None,
                "first_entry_line": round(float(trigger_line), 1) if trigger_line is not None else None,
                "first_entry_side": pred_side,
                "first_entry_prob": round(float(pred_prob), 4) if pred_prob is not None else None,
                "first_entry_text": first_entry_text,
                "line_changes": line_changes,
                "opp_conf_max_after": round(float(opp_conf_max_after), 4) if opp_conf_max_after is not None else None,
                "risk_flag": risk_flag,
                "risk_reasons": risk_reasons,
                "risk_alert": risk_alert,
                "risk_alert_text": risk_alert_text,
                "risk_events": risk_events,
                "hedge_recommended": hedge_recommended,
                "hedge_minute": hedge_minute,
                "hedge_line": round(float(hedge_line), 1) if hedge_line is not None else None,
                "hedge_side": hedge_side,
                "hedge_prob": round(float(hedge_prob), 4) if hedge_prob is not None else None,
                "hedge_text": hedge_text,
                "all_triggers": all_triggers,
                "trigger_summary": trigger_summary,  # 新增：多触发点汇总信息
                "final_total_proxy_from_page": final_proxy,
                "final_relation_proxy": final_relation_proxy,
                "hit_proxy": hit_proxy,
                "minute_rows_last10": minute_rows_last10,
            }
        )

    rows.sort(key=lambda x: (x.get("latest_minute") or -1, x.get("match_id") or ""), reverse=True)
    return rows


def crawl_live_matches_from_ids(
    match_ids: list[str],
    date_str: str,
    out_jsonl: str,
    company_id: int,
    timeout_s: float,
    retries: int,
    backoff_s: float,
) -> dict[str, Any]:
    Path(out_jsonl).write_text("", encoding="utf-8")

    written = 0
    rejects: dict[str, int] = {}
    total = len(match_ids)

    with open(out_jsonl, "a", encoding="utf-8") as f:
        for idx, mid in enumerate(match_ids, start=1):
            rec = _fetch_one_match_raw(
                match_id=str(mid),
                date_str=str(date_str),
                company_id=int(company_id),
                timeout_s=float(timeout_s),
                retries=max(1, int(retries)),
                backoff_s=float(backoff_s),
                jitter_s=(0.0, 0.12),
                request_jitter_s=(0.6, 1.4),
            )

            if not rec:
                print(f"[{idx}/{total}] {mid} -> no data")
                continue

            if bool(rec.get("_reject", False)):
                reason = str(rec.get("reason", "unknown"))
                rejects[reason] = int(rejects.get(reason, 0) + 1)
                print(f"[{idx}/{total}] {mid} -> reject: {reason}")
                continue

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            print(f"[{idx}/{total}] {mid} -> saved ({written})")

    return {
        "mode": "manual_match_ids_sequential",
        "submitted": total,
        "written": written,
        "reject_reasons": rejects,
        "date": str(date_str),
    }


def _parse_match_ids_arg(values: list[str]) -> list[str]:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Update live best-strategy results from oldIndexall live list")
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"), help="Date used for output records")
    ap.add_argument("--company-id", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout-s", type=float, default=12.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff-s", type=float, default=0.35)
    ap.add_argument("--model", default="data/settlement_relation_model/settlement_relation_model.joblib")
    ap.add_argument("--out-live", default="data/live_best_strategy_results.json")
    ap.add_argument("--out-raw", default="data/raw_matches_live_oldindexall_with_teams.jsonl")
    ap.add_argument(
        "--match-id",
        action="append",
        default=[],
        help="Optional match id to infer explicitly. Repeat or pass comma-separated ids.",
    )
    args = ap.parse_args()

    explicit_match_ids = _parse_match_ids_arg(list(args.match_id or []))
    live_matches: list[dict[str, str]] = []
    if explicit_match_ids:
        live_ids = explicit_match_ids
    else:
        feed_text = fetch_oldindexall_feed_text(timeout_s=max(10.0, float(args.timeout_s)))
        live_matches = extract_live_matches_from_feed(feed_text)
        live_ids = [x["match_id"] for x in live_matches]

    if not live_ids:
        payload = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "strategy": "first_3x_08_any",
            "source": "oldIndexall/bfdata_ut.js",
            "note": "No in-play matches found in current feed snapshot.",
            "match_count": 0,
            "rows": [],
        }
        Path(args.out_live).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"live_ids": 0, "written": args.out_live}, ensure_ascii=False))
        return

    summary = crawl_live_matches_from_ids(
        match_ids=live_ids,
        date_str=str(args.date),
        out_jsonl=str(args.out_raw),
        company_id=int(args.company_id),
        timeout_s=float(args.timeout_s),
        retries=max(1, int(args.retries)),
        backoff_s=float(args.backoff_s),
    )

    raw_path = Path(args.out_raw)
    records: list[dict[str, Any]] = []
    if raw_path.exists() and raw_path.stat().st_size > 0:
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    records.append(json.loads(s))

    model_bundle = joblib.load(args.model)
    rows = build_live_strategy_rows(records, model_bundle)

    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "strategy": "first_3x_08_any",
        "source": "oldIndexall/bfdata_ut.js" if not explicit_match_ids else "manual-match-id",
        "note": "For live matches, final relation uses current page corners as proxy, not full-time settled result." if not explicit_match_ids else "Computed from explicitly requested match ids; final relation uses current page corners as proxy.",
        "feed_live_match_count": len(live_ids),
        "crawl_summary": summary,
        "match_count": len(rows),
        "rows": rows,
    }

    Path(args.out_live).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "feed_live_ids": len(live_ids),
                "explicit_match_ids": explicit_match_ids,
                "raw_records": len(records),
                "strategy_rows": len(rows),
                "out_live": args.out_live,
                "out_raw": args.out_raw,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
