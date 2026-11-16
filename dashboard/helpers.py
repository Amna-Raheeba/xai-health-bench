"""Data-shaping helpers for the Streamlit dashboard.

Everything the dashboard displays is shaped here first: overview
scores, per-agent comparison rows, per-case detail, counterfactual
rows, and category breakdowns. This module is pure Python on purpose
— no streamlit import — so the shaping logic is unit-tested in
``tests/test_dashboard_helpers.py`` and the app file stays a thin
presentation layer.

Reports are the JSON files the CLI writes (``results/*_report.json``);
when none exists the helpers return empty shapes, and the app offers
to run the bench instead.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

METRIC_LABELS = [
    ("bench_score", "Bench score"),
    ("faithfulness", "Faithfulness"),
    ("citation_precision", "Citation precision"),
    ("fact_coverage", "Fact coverage"),
    ("grounding_rate", "Grounding rate"),
    ("abstention_accuracy", "Abstention accuracy"),
    ("distractor_avoidance", "Distractor avoidance"),
]


def available_reports(results_dir: str | Path = RESULTS_DIR) -> list[Path]:
    """Report JSON files in the results dir, newest first."""
    d = Path(results_dir)
    if not d.exists():
        return []
    return sorted(d.glob("*_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_report(path: str | Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def overview(report: dict) -> dict[str, Any]:
    """Top strip: agent name + aggregate metrics."""
    agg = report.get("aggregate", {}) if report else {}
    return {
        "agent": report.get("agent"),
        "cases": report.get("cases", 0),
        "counterfactuals": agg.get("counterfactuals", 0),
        "metrics": [
            {"key": key, "label": label, "value": agg.get(key)}
            for key, label in METRIC_LABELS
        ],
    }


def comparison_rows(reports: list[dict]) -> list[dict[str, Any]]:
    """One row per agent report, for the comparison table."""
    rows = []
    for report in reports:
        agg = report.get("aggregate", {})
        rows.append({
            "agent": report.get("agent"),
            **{key: agg.get(key) for key, _ in METRIC_LABELS},
        })
    return rows


def case_rows(report: dict) -> list[dict[str, Any]]:
    """Per-case rows: id, category, abstained, scores, failed citations."""
    rows = []
    for s in report.get("case_scores", []):
        rows.append({
            "case_id": s.get("case_id"),
            "category": s.get("category"),
            "abstained": s.get("abstained"),
            "abstention_correct": s.get("abstention_correct"),
            "fact_coverage": s.get("fact_coverage"),
            "citation_precision": s.get("citation_precision"),
            "grounding_rate": s.get("grounding_rate"),
            "distractor_avoidance": s.get("distractor_avoidance"),
            "failed_citations": len(s.get("failed_citations") or []),
        })
    return rows


def counterfactual_rows(report: dict) -> list[dict[str, Any]]:
    """Per-perturbation rows: did the answer change as expected?"""
    return [
        {"case_id": c.get("case_id"), "cf_id": c.get("cf_id"),
         "cf_type": c.get("cf_type"), "passed": c.get("passed"),
         "detail": c.get("detail"),
         "abstained_on_perturbed": c.get("abstained_on_perturbed")}
        for c in report.get("counterfactual_scores", [])
    ]


def category_rows(report: dict) -> list[dict[str, Any]]:
    """Per-category aggregate rows."""
    rows = []
    for category, row in sorted((report.get("by_category") or {}).items()):
        rows.append({"category": category, **row})
    return rows


def run_demo_bench() -> dict:
    """Run the demo agent bench and return the report dict.

    This genuinely executes the scorer — the same code path as
    ``python -m xai_health_bench run --agent demo``.
    """
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from xai_health_bench.scorer import run_bench
    return run_bench("demo")
