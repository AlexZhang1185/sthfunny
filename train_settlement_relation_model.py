from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except Exception:
    joblib = None

from pipeline_e2e_v2 import FEATURE_COLS, build_features_for_match, ensure_dir, load_jsonl


SETTLEMENT_FEATURE_COLS: list[str] = FEATURE_COLS + [
    "line_rank_in_match",
    "minute_progress",
    "corners_progress",
    "line_minus_corners",
    "line_minus_final_proxy",
    "is_second_half",
    "is_late_game",
    "is_integer_line",
    "is_half_line",
    "distance_to_int",
    "distance_to_half",
    "corner_rate_10m",
    "corner_rate_5m",
    "corner_rate_2m",
    "second_half_corners_so_far",
    "second_half_corner_rate",
    "second_half_3_4_flag",
    "second_half_0_2_flag",
    "second_half_5plus_flag",
]


def load_unique_matches(paths: list[str], limit_matches: Optional[int] = None) -> list[dict[str, Any]]:
    seen: set[str] = set()
    matches: list[dict[str, Any]] = []
    for path in paths:
        for match in load_jsonl(path):
            match_id = str(match.get("match_id", "")).strip()
            if not match_id or match_id in seen:
                continue
            seen.add(match_id)
            matches.append(match)
            if limit_matches is not None and len(matches) >= int(limit_matches):
                return matches
    return matches


def _line_to_ticks(line: float) -> int:
    return int(round(float(line) * 2.0))


def _line_fraction(line: float) -> float:
    frac = abs(float(line) - round(float(line)))
    return float(round(frac, 2))


def _label_settlement(final_total: int, line: float) -> int:
    if final_total > line:
        return 2
    if final_total < line:
        return 0
    return 1


def _build_settlement_dataset(matches: list[dict[str, Any]]) -> pd.DataFrame:
    frame_parts: list[pd.DataFrame] = []

    for match in matches:
        feat_df = build_features_for_match(match, drop_push=False)
        if feat_df.empty or len(feat_df) < 4:
            continue

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

        feat_df = feat_df.assign(
            target_settlement=np.array([_label_settlement(final_total, v) for v in line], dtype=np.int32),
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
        )

        # 去重: minute/line/corners_so_far 已在 SETTLEMENT_FEATURE_COLS 中, 避免重复列(否则 X 会变 41 列)
        keep_cols = list(dict.fromkeys(
            ["match_id", "minute", "line", "corners_so_far", "final_total_corners", "target_settlement"]
            + SETTLEMENT_FEATURE_COLS
        ))
        feat_df = feat_df.assign(final_total_corners=final_total)[keep_cols]
        frame_parts.append(feat_df)

    if not frame_parts:
        return pd.DataFrame()

    return pd.concat(frame_parts, ignore_index=True)


def _compute_metrics(y_true: np.ndarray, p_pred: np.ndarray, labels: list[int]) -> dict[str, Any]:
    pred = np.argmax(p_pred, axis=1)
    out = {
        "samples": int(len(y_true)),
        "acc": float(accuracy_score(y_true, pred)),
        "logloss": float(log_loss(y_true, p_pred, labels=labels)),
    }
    try:
        out["auc_ovr"] = float(roc_auc_score(y_true, p_pred, multi_class="ovr", labels=labels))
    except Exception:
        out["auc_ovr"] = float("nan")
    return out


def train_settlement_model(dataset: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    X_train = dataset.iloc[train_idx][SETTLEMENT_FEATURE_COLS].astype(float)
    y_train = dataset.iloc[train_idx]["target_settlement"].astype(int).to_numpy()
    X_test = dataset.iloc[test_idx][SETTLEMENT_FEATURE_COLS].astype(float)
    y_test = dataset.iloc[test_idx]["target_settlement"].astype(int).to_numpy()

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=8,
        max_iter=350,
        min_samples_leaf=30,
        random_state=42,
    )
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)
    metrics = _compute_metrics(y_test, p_test, labels=[0, 1, 2])

    cm = confusion_matrix(y_test, np.argmax(p_test, axis=1), labels=[0, 1, 2])
    precision = {}
    recall = {}
    for idx, name in enumerate(["under", "push", "over"]):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - cm[idx, idx])
        fn = float(cm[idx, :].sum() - cm[idx, idx])
        precision[name] = float(tp / (tp + fp)) if tp + fp > 0 else float("nan")
        recall[name] = float(tp / (tp + fn)) if tp + fn > 0 else float("nan")

    metrics.update({"precision": precision, "recall": recall, "confusion_matrix": cm.tolist()})
    return metrics, {"model": model}


def save_artifacts(out_dir: str, report: dict[str, Any], model_bundle: dict[str, Any]) -> None:
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "settlement_relation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if joblib is not None:
        joblib.dump(
            {"model": model_bundle["model"], "features": SETTLEMENT_FEATURE_COLS, "labels": [0, 1, 2]},
            os.path.join(out_dir, "settlement_relation_model.joblib"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train settlement relation model: under / push / over")
    parser.add_argument(
        "--raw",
        nargs="+",
        default=["data/raw_matches_10k_v2.jsonl", "data/raw_matches_20260101_20260722_all.jsonl"],
    )
    parser.add_argument("--out-dir", default="data/settlement_relation_model")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-matches", type=int, default=None)
    args = parser.parse_args()

    matches = load_unique_matches(args.raw, limit_matches=args.limit_matches)
    dataset = _build_settlement_dataset(matches)
    if dataset.empty:
        raise ValueError("settlement dataset is empty")

    X = dataset[SETTLEMENT_FEATURE_COLS].astype(float)
    y = dataset["target_settlement"].astype(int).to_numpy()
    groups = dataset["match_id"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=int(args.seed))
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    metrics, model_bundle = train_settlement_model(dataset, train_idx, test_idx)
    report = {
        "raw_files": [str(x) for x in args.raw],
        "matches": int(len(matches)),
        "samples": int(len(dataset)),
        "unique_matches": int(dataset["match_id"].nunique()),
        "feature_count": int(len(SETTLEMENT_FEATURE_COLS)),
        "class_balance": {
            "under": float(np.mean(y == 0)),
            "push": float(np.mean(y == 1)),
            "over": float(np.mean(y == 2)),
        },
        "split": {
            "train_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),
            "train_matches": int(dataset.iloc[train_idx]["match_id"].nunique()),
            "test_matches": int(dataset.iloc[test_idx]["match_id"].nunique()),
        },
        "metrics": metrics,
        "feature_columns": list(SETTLEMENT_FEATURE_COLS),
    }

    save_artifacts(args.out_dir, report, model_bundle)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()