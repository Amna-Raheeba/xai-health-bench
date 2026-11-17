#!/usr/bin/env python3
"""Experiment 04 — citation-swap probe: does the recommendation move?

Question: when a case's supporting passage is replaced by a
contradicting guideline, does the agent follow the flipped source —
or does it copy both passages and assert both directions at once?

Method: for each agent, run all 8 swap probes (one per ``swap``
case) and report swap sensitivity — the fraction of probes where the
recommendation moved (original recommendation absent AND
contradicting guideline stated, or the agent abstained). Write
``results/swap_probe.csv``.

Measured outcome (2026-11-17): demo 1.000 (8/8 moved), naive 0.000
and confident 0.000 (both stated BOTH directions on all 8 probes).
The probe discriminates exactly where the counterfactual metric
cannot: all three agents score 1.000 on faithfulness, because
extractive agents copy whatever evidence they are given. The swap
probe is the only metric in this bench that separates a careful
agent from an extractive one on its own — the demo's passage-trust
rule drops the corroborating passage (coverage below 0.40), so it
follows the flipped guideline, while the extractive agents copy it
and the corroboration side by side.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xai_health_bench.scorer import run_bench  # noqa: E402


def main() -> None:
    rows = []
    for agent in ("demo", "naive", "confident"):
        report = run_bench(agent)
        agg = report["aggregate"]
        print(f"{agent}: swap_sensitivity={agg['swap_sensitivity']} "
              f"({agg['swap_probes']} probes)")
        for p in report["swap_probe_scores"]:
            rows.append({"agent": agent, "case_id": p["case_id"],
                         "probe_id": p["probe_id"],
                         "passed": int(p["passed"]),
                         "stated_both": int(p["stated_both"]),
                         "abstained": int(p["abstained_on_swapped"]),
                         "detail": p["detail"]})
            print(f"  {agent} {p['case_id']}: "
                  f"{'MOVED' if p['passed'] else 'STAYED'} — {p['detail']}")
    out = ROOT / "results" / "swap_probe.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        # lineterminator="\n": the csv module's default is "\r\n", which
        # made regenerated CSVs differ from the committed LF files.
        writer = csv.DictWriter(fh, fieldnames=["agent", "case_id", "probe_id",
                                                "passed", "stated_both",
                                                "abstained", "detail"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
