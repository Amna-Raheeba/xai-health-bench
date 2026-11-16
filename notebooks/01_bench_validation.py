#!/usr/bin/env python3
"""Experiment 01 — bench validation: does the bench discriminate?

Question: do the three shipped agents rank in the expected order
(demo > naive > confident), and does each metric separate the agent
it is meant to catch?

Method: run all three agents over the full 40-case dataset, save a
report JSON per agent, and assert the ranking. The assertions are the
point: if a future change makes the bench unable to tell a
hallucinating agent from a careful one, this notebook fails loudly.

Expected outcome (measured 2026-11-17):
- demo 1.000 > naive 0.970 > confident 0.896 on bench_score.
- The naive agent fails exactly the 8 abstain cases (it never
  abstains) and all 8 citation-trap cases (it states every distractor).
- The confident agent fails citation precision on every case (its
  invented "94%" statistic is caught by the numbers check).
- Faithfulness is 1.000 for all three agents: extractive agents are
  trivially faithful to whatever text they copy, so faithfulness
  alone cannot separate them — the bench's discrimination comes from
  citation precision, abstention accuracy, and distractor avoidance.
  This is a documented finding, not a bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xai_health_bench.report import render_comparison  # noqa: E402
from xai_health_bench.scorer import run_bench, save_report  # noqa: E402


def main() -> None:
    reports = []
    for agent in ("demo", "naive", "confident"):
        report = run_bench(agent)
        reports.append(report)
        save_report(report, ROOT / "results" / f"{agent}_report.json")
        agg = report["aggregate"]
        print(f"{agent}: bench_score={agg['bench_score']} "
              f"faithfulness={agg['faithfulness']} "
              f"citation_precision={agg['citation_precision']} "
              f"fact_coverage={agg['fact_coverage']} "
              f"abstention_accuracy={agg['abstention_accuracy']} "
              f"distractor_avoidance={agg['distractor_avoidance']}")
    (ROOT / "results" / "comparison.md").write_text(
        render_comparison(reports), encoding="utf-8")

    scores = {r["agent"]: r["aggregate"]["bench_score"] for r in reports}
    assert scores["demo"] > scores["naive"] > scores["confident"], scores

    naive = next(r for r in reports if r["agent"] == "naive")
    abstain_wrong = [s for s in naive["case_scores"]
                     if s["category"] == "abstain" and s["abstention_correct"] == 0.0]
    assert len(abstain_wrong) == 8, "naive must fail all 8 abstain cases"
    trap_wrong = [s for s in naive["case_scores"]
                  if s["category"] == "citation-trap"
                  and s.get("distractor_avoidance") == 0.0]
    assert len(trap_wrong) == 8, "naive must state all 8 distractor facts"

    confident = next(r for r in reports if r["agent"] == "confident")
    assert confident["aggregate"]["citation_precision"] < 0.8, (
        "confident agent's invented statistic must be caught by the audit")
    assert confident["aggregate"]["citation_precision"] < \
        reports[0]["aggregate"]["citation_precision"]

    print("\nVALIDATION PASSED: demo > naive > confident, "
          "and each baseline fails exactly where it should.")


if __name__ == "__main__":
    main()
