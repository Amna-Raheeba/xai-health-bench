#!/usr/bin/env python3
"""Export the bench dataset in ai-evaluation-platform format.

The eval platform's dataset loader expects one JSON object per line
with ``id``, ``project``, ``input``, ``expected``, ``category``,
``difficulty`` and ``tags``. The bench's evidence and counterfactuals
travel inside ``input["case"]`` so the platform's orchestrator can
pass the whole case to the ``XaiHealthBenchAdapter`` unchanged.

Output: ``datasets/xai_health_eval_platform.jsonl`` in this repo; copy
it to the eval platform repo as ``datasets/xai_health.jsonl``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xai_health_bench.dataset import load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "xai_health_eval_platform.jsonl"


def main() -> None:
    cases = load_dataset()
    with OUT.open("w", encoding="utf-8") as fh:
        for case in cases:
            exported = {
                "id": case["id"],
                "project": "xai-health",
                "input": {
                    "question": case["input"]["question"],
                    "case": case,
                },
                "expected": {
                    "required_facts": [f["text"] for f in case["expected"].get("facts", [])],
                    "must_include_citations": True,
                    "should_abstain": case["expected"].get("should_abstain", False),
                },
                "category": case["category"],
                "difficulty": case.get("difficulty", "medium"),
                "tags": case.get("tags", []),
            }
            fh.write(json.dumps(exported) + "\n")
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
