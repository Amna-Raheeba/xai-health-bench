"""Bench runner: score one agent over the whole dataset.

Runs every case unperturbed, then re-runs every counterfactual
perturbation of the counterfactual cases. Returns a full report dict
(also written to disk by the CLI): aggregate metrics, per-case scores,
per-counterfactual scores, and per-category breakdowns.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import agents, metrics
from .dataset import DEFAULT_DATASET, load_dataset


def run_bench(agent_name: str, cases: list[dict] | None = None,
              dataset_path: str | Path = DEFAULT_DATASET) -> dict:
    """Score ``agent_name`` on ``cases`` (or the full dataset)."""
    if cases is None:
        cases = load_dataset(dataset_path)
    started = time.time()
    agent_run = lambda case: agents.run(agent_name, case)  # noqa: E731
    case_scores = []
    cf_scores = []
    swap_scores = []
    for case in cases:
        result = agent_run(case)
        case_scores.append(metrics.score_case(case, result))
        for cf in case.get("counterfactuals", []):
            cf_scores.append(metrics.score_counterfactual(case, cf, agent_run))
        for probe in case.get("swap_probes", []):
            swap_scores.append(metrics.score_swap_probe(case, probe, agent_run))
    agg = metrics.aggregate(case_scores, cf_scores, swap_scores)
    by_category: dict[str, dict] = {}
    for category in sorted({s["category"] for s in case_scores}):
        rows = [s for s in case_scores if s["category"] == category]
        answered = [s for s in rows if not s["abstained"]]
        def mean(key, rs):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        by_category[category] = {
            "cases": len(rows),
            "answered": len(answered),
            "fact_coverage": mean("fact_coverage", answered),
            "citation_precision": mean("citation_precision", answered),
            "grounding_rate": mean("grounding_rate", answered),
            "distractor_avoidance": mean("distractor_avoidance", answered),
            "abstention_accuracy": mean("abstention_correct", rows),
        }
    return {
        "agent": agent_name,
        "dataset": str(dataset_path),
        "cases": len(cases),
        "elapsed_s": round(time.time() - started, 3),
        "aggregate": agg,
        "by_category": by_category,
        "case_scores": case_scores,
        "counterfactual_scores": cf_scores,
        "swap_probe_scores": swap_scores,
    }


def save_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
