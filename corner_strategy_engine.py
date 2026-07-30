#!/usr/bin/env python3
"""Utilities for exporting and querying high-confidence corner line strategy rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_report(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_strategy_package(report: dict[str, Any], min_transition_samples: int = 500) -> dict[str, Any]:
    all_lines = report.get("all_lines_vs_closing_line") or {}
    transition_rules = all_lines.get("line_to_close_high_conf_rules") or []
    transition_rules = [
        r
        for r in transition_rules
        if int(r.get("samples", 0) or 0) >= int(min_transition_samples)
    ]

    high_precision_rules = report.get("high_precision_rules") or {}
    high_precision_actionable = {
        "precision_threshold": high_precision_rules.get("precision_threshold"),
        "actionable_filter": high_precision_rules.get("actionable_filter"),
        "train_samples": high_precision_rules.get("actionable_train_samples"),
        "test_samples": high_precision_rules.get("actionable_test_samples"),
        "under_or_push_rules": high_precision_rules.get("under_or_push_rules") or [],
        "over_rules": high_precision_rules.get("over_rules") or [],
    }

    return {
        "source_report": report.get("input_file"),
        "matches": report.get("matches"),
        "snapshots": report.get("snapshots"),
        "feature_cols": report.get("feature_cols") or [],
        "metrics": report.get("metrics") or [],
        "transition_rules": transition_rules,
        "high_precision_actionable_rules": high_precision_actionable,
    }


def save_strategy_package(
    report_path: str | Path,
    out_path: str | Path,
    min_transition_samples: int = 500,
) -> dict[str, Any]:
    report = load_report(report_path)
    package = build_strategy_package(report, min_transition_samples=min_transition_samples)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(package, f, ensure_ascii=False, indent=2)
    return package


def transition_rule_summary(package: dict[str, Any]) -> dict[str, Any]:
    transition_rules = package.get("transition_rules") or []
    high_precision = package.get("high_precision_actionable_rules") or {}
    return {
        "transition_rule_count": len(transition_rules),
        "high_precision_under_rule_count": len(high_precision.get("under_or_push_rules") or []),
        "high_precision_over_rule_count": len(high_precision.get("over_rules") or []),
    }
