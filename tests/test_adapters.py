"""Tests for the ai-evaluation-platform adapter and dataset export."""
import json
import subprocess
import sys
from pathlib import Path

from xai_health_bench.adapters.eval_platform import XaiHealthBenchAdapter
from xai_health_bench.dataset import load_dataset

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_KEYS = ("output", "execution_time_s", "token_usage", "tool_calls",
                 "trace_id", "metadata")


def test_adapter_info():
    adapter = XaiHealthBenchAdapter()
    info = adapter.get_agent_info()
    assert info["name"] == "xai-health-bench"
    assert info["project"] == "xai-health"


def test_adapter_run_returns_standard_contract():
    case = load_dataset()[0]
    adapter = XaiHealthBenchAdapter()
    result = adapter.run({"question": case["input"]["question"], "case": case})
    for key in REQUIRED_KEYS:
        assert key in result, f"missing {key}"
    assert isinstance(result["output"], str)
    assert result["metadata"]["agent_name"] == "xai-health-bench"
    assert "bench_case_score" in result["structured"]
    assert result["structured"]["bench_case_score"]["case_id"] == case["id"]


def test_adapter_run_missing_case_raises():
    adapter = XaiHealthBenchAdapter()
    try:
        adapter.run({"question": "What?"})
    except ValueError as exc:
        assert "input['case']" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_export_script_produces_eval_platform_dataset(tmp_path):
    out = ROOT / "datasets" / "xai_health_eval_platform.jsonl"
    if out.exists():
        out.unlink()
    proc = subprocess.run(
        [sys.executable, "scripts/export_eval_platform_dataset.py"],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 48
    row = rows[0]
    assert row["project"] == "xai-health"
    assert "case" in row["input"]
    assert row["input"]["case"]["id"] == "xh-001"
    assert "required_facts" in row["expected"]
