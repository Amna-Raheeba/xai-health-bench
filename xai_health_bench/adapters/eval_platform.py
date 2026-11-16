"""Plug XAI-Health Bench into the ai-evaluation-platform.

The eval platform's contract (see ``eval_platform.adapter``) is: an
``AgentAdapter`` with ``name``/``version``/``project`` and a ``run``
method returning a standard result dict with ``output``,
``execution_time_s``, ``token_usage``, ``tool_calls``, ``trace_id``
and ``metadata``.

``XaiHealthBenchAdapter`` wraps the bench's demo agent so the eval
platform can run it against the exported dataset
(``scripts/export_eval_platform_dataset.py``) with its existing
orchestrator, regression engine, and quality gates. The bench's own
metrics (faithfulness, citation precision, abstention accuracy) stay
in this repo — the eval platform reads them through the ``structured``
field, which carries the per-case score dict.

Setup (two files, no core changes):
1. Install this package in the eval platform's environment
   (``pip install -e /path/to/xai-health-bench`` or from GitHub).
2. Copy this file into the eval platform repo as
   ``eval_platform/adapters/xai_health.py`` and register it in
   ``eval_platform.adapter.default_registry``; copy
   ``datasets/xai_health_eval_platform.jsonl`` to
   ``datasets/xai_health.jsonl``.
"""
from __future__ import annotations

import time
import uuid

try:
    from xai_health_bench import agents, metrics
except ImportError as exc:  # pragma: no cover - setup error, not a test path
    raise ImportError(
        "xai-health-bench package is required: pip install -e /path/to/xai-health-bench"
    ) from exc


class XaiHealthBenchAdapter:
    """Eval-platform adapter around the bench's demo agent."""

    name = "xai-health-bench"
    version = "0.1.0"
    project = "xai-health"

    def run(self, input_data: dict) -> dict:
        """Run the demo agent on one eval-platform case.

        ``input_data`` is the case's ``input`` object. The caller must
        also pass the case's evidence: the exported dataset puts the
        full case (evidence + expected) under ``input["case"]``.
        """
        started = time.time()
        case = input_data.get("case")
        if case is None:
            raise ValueError("xai-health case needs input['case'] with evidence + expected")
        result = agents.run("demo", case)
        score = metrics.score_case(case, result)
        output = result["answer"]
        return {
            "output": output,
            "execution_time_s": round(time.time() - started, 4),
            "token_usage": {"input": 0, "output": len(output.split())},
            "tool_calls": [{"name": "bm25_retrieve"},
                           {"name": "verify_claims"},
                           {"name": "abstention_gate"}],
            "trace_id": uuid.uuid4().hex[:12],
            "metadata": {"agent_name": self.name, "agent_version": self.version,
                         "model": "lexical-demo-agent"},
            "status": result["status"],
            "citations": result["citations"],
            "structured": {"bench_case_score": score},
        }

    def get_agent_info(self) -> dict:
        return {"name": self.name, "version": self.version, "project": self.project}
