#!/usr/bin/env python3
"""Experiment 02 — abstention threshold sweep.

Question: which evidence-strength threshold should the demo agent
abstain at?

Method: run the demo agent over all 48 cases at thresholds from 0.30
to 0.80 in steps of 0.05, and record abstention accuracy (correct
abstain-or-answer decisions / 48) at each step. Write
``results/threshold_sweep.csv``.

Measured outcome (2026-11-17): abstain cases all score an evidence
strength of 0.0 and the weakest answerable case scores 0.577, so every
threshold in [0.30, 0.55] gives 48/48. The shipped default is 0.50,
inside that plateau. The threshold is corpus-specific: the
healthcare-research-agent used 0.62 on its own corpus, and copying it
here would wrongly abstain on 3 answerable cases. Sweep, don't copy.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xai_health_bench import agents  # noqa: E402
from xai_health_bench.dataset import load_dataset  # noqa: E402


def main() -> None:
    cases = load_dataset()
    rows = []
    for threshold in [round(0.30 + 0.05 * i, 2) for i in range(11)]:
        agents.ABSTENTION_THRESHOLD = threshold
        correct = 0
        abstained = 0
        for case in cases:
            result = agents.run("demo", case)
            if result["abstained"]:
                abstained += 1
            if result["abstained"] == case["expected"].get("should_abstain", False):
                correct += 1
        rows.append({"threshold": threshold, "correct": correct,
                     "accuracy": round(correct / len(cases), 4),
                     "abstained": abstained})
        print(f"threshold {threshold}: {correct}/{len(cases)} "
              f"accuracy={correct / len(cases):.3f} abstained={abstained}")
    out = ROOT / "results" / "threshold_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        # lineterminator="\n": the csv module's default is "\r\n", which
        # made regenerated CSVs differ from the committed LF files.
        writer = csv.DictWriter(fh, fieldnames=["threshold", "correct", "accuracy", "abstained"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    best = max(r["accuracy"] for r in rows)
    plateau = [r["threshold"] for r in rows if r["accuracy"] == best]
    print(f"\nbest accuracy {best} on thresholds {plateau}; shipped default 0.50")


if __name__ == "__main__":
    main()
