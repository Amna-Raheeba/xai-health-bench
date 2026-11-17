"""Tests for dashboard.helpers (pure shaping logic, no streamlit)."""
from pathlib import Path

from xai_health_bench import scorer
from dashboard import helpers

ROOT = Path(__file__).resolve().parents[1]


def _demo_report():
    return scorer.run_bench("demo")


def test_overview_shapes_metrics():
    ov = helpers.overview(_demo_report())
    assert ov["agent"] == "demo"
    assert ov["cases"] == 48
    assert len(ov["metrics"]) == 8
    bench = [m for m in ov["metrics"] if m["key"] == "bench_score"][0]
    assert bench["value"] == 1.0
    swap = [m for m in ov["metrics"] if m["key"] == "swap_sensitivity"][0]
    assert swap["value"] == 1.0


def test_overview_empty_report():
    ov = helpers.overview({})
    assert ov["agent"] is None
    assert ov["cases"] == 0
    assert all(m["value"] is None for m in ov["metrics"])


def test_comparison_rows():
    demo = _demo_report()
    naive = scorer.run_bench("naive")
    rows = helpers.comparison_rows([demo, naive])
    assert [r["agent"] for r in rows] == ["demo", "naive"]
    assert rows[0]["bench_score"] > rows[1]["bench_score"]


def test_case_rows_count_and_fields():
    rows = helpers.case_rows(_demo_report())
    assert len(rows) == 48
    assert all("case_id" in r and "category" in r for r in rows)
    abstain_rows = [r for r in rows if r["category"] == "abstain"]
    assert len(abstain_rows) == 8
    assert all(r["abstained"] is True for r in abstain_rows)


def test_counterfactual_rows():
    rows = helpers.counterfactual_rows(_demo_report())
    assert len(rows) == 10
    assert all(r["passed"] is True for r in rows)


def test_swap_probe_rows():
    demo_rows = helpers.swap_probe_rows(_demo_report())
    assert len(demo_rows) == 8
    assert all(r["passed"] is True for r in demo_rows)
    assert all(r["stated_both"] is False for r in demo_rows)
    naive_rows = helpers.swap_probe_rows(scorer.run_bench("naive"))
    assert len(naive_rows) == 8
    assert all(r["passed"] is False for r in naive_rows)
    assert all(r["stated_both"] is True for r in naive_rows)


def test_swap_probe_rows_empty_report():
    assert helpers.swap_probe_rows({}) == []


def test_category_rows():
    rows = helpers.category_rows(_demo_report())
    assert [r["category"] for r in rows] == [
        "abstain", "citation-trap", "counterfactual", "faithful", "swap"]
    assert all(r["cases"] > 0 for r in rows)


def test_available_reports_empty_dir(tmp_path):
    assert helpers.available_reports(tmp_path) == []


def test_load_report_bad_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert helpers.load_report(bad) == {}


def test_run_demo_bench_returns_report():
    report = helpers.run_demo_bench()
    assert report["agent"] == "demo"
    assert report["aggregate"]["bench_score"] == 1.0
