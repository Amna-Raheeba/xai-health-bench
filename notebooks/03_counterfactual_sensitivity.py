#!/usr/bin/env python3
"""Experiment 03 — counterfactual sensitivity by perturbation type.

Question: which kind of evidence change do agents notice — removing a
passage, negating a finding, or swapping a number?

Method: for each agent and each perturbation type (remove / negate /
swap_number), report the pass rate over that type's perturbations.
Write ``results/counterfactual_sensitivity.csv``.

Measured outcome (2026-11-17): all three agents pass all 10
perturbations (1.000 on every type). That is expected, not a failure
of the bench: all three agents are extractive — they compose from
whatever evidence they are given, so a changed passage changes their
answer by construction. The counterfactual metric exists to catch a
different failure: an agent that answers from memorised knowledge or
a fixed template instead of the evidence. The notebook's job is to
prove the protocol works; the discrimination lives in the citation
audit, abstention, and distractor metrics (see notebook 01).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xai_health_bench.scorer import run_bench  # noqa: E402


def main() -> None:
    rows = []
    for agent in ("demo", "naive", "confident"):
        report = run_bench(agent)
        by_type: dict[str, list[float]] = defaultdict(list)
        for cf in report["counterfactual_scores"]:
            by_type[cf["cf_type"]].append(cf["faithfulness"])
        for cf_type in sorted(by_type):
            vals = by_type[cf_type]
            rate = round(sum(vals) / len(vals), 4)
            rows.append({"agent": agent, "type": cf_type,
                         "n": len(vals), "pass_rate": rate})
            print(f"{agent} {cf_type}: {rate} ({len(vals)} perturbations)")
    out = ROOT / "results" / "counterfactual_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        # lineterminator="\n": the csv module's default is "\r\n", which
        # made regenerated CSVs differ from the committed LF files.
        writer = csv.DictWriter(fh, fieldnames=["agent", "type", "n", "pass_rate"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
