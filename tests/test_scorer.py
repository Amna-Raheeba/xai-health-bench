"""Tests for xai_health_bench.scorer and report rendering."""
from pathlib import Path

from xai_health_bench import scorer
from xai_health_bench.report import render_comparison, render_report

ROOT = Path(__file__).resolve().parents[1]


def test_run_bench_demo_scores_perfectly_on_reference():
    report = scorer.run_bench("demo")
    agg = report["aggregate"]
    assert agg["bench_score"] == 1.0
    assert agg["cases"] == 40
    assert agg["counterfactuals"] == 10
    assert len(report["case_scores"]) == 40
    assert len(report["counterfactual_scores"]) == 10


def test_run_bench_ranking_demo_naive_confident():
    demo = scorer.run_bench("demo")["aggregate"]["bench_score"]
    naive = scorer.run_bench("naive")["aggregate"]["bench_score"]
    confident = scorer.run_bench("confident")["aggregate"]["bench_score"]
    assert demo > naive > confident


def test_run_bench_naive_fails_all_abstain_cases():
    report = scorer.run_bench("naive")
    abstain_wrong = [s for s in report["case_scores"]
                     if s["category"] == "abstain" and s["abstention_correct"] == 0.0]
    assert len(abstain_wrong) == 8


def test_run_bench_by_category_covers_all_categories():
    report = scorer.run_bench("demo")
    assert set(report["by_category"]) == {
        "faithful", "citation-trap", "abstain", "counterfactual"}


def test_save_and_load_report_roundtrip(tmp_path):
    report = scorer.run_bench("demo")
    path = scorer.save_report(report, tmp_path / "report.json")
    loaded = scorer.load_report(path)
    assert loaded["agent"] == "demo"
    assert loaded["aggregate"]["bench_score"] == 1.0


def test_render_report_contains_numbers():
    report = scorer.run_bench("demo")
    text = render_report(report)
    assert "# XAI-Health Bench — demo" in text
    assert "bench_score (composite) | 1.000" in text
    assert "Counterfactual failures" in text


def test_render_comparison_orders_agents():
    demo = scorer.run_bench("demo")
    naive = scorer.run_bench("naive")
    text = render_comparison([demo, naive])
    assert "| Metric | demo | naive |" in text
    assert "bench_score | 1.000 | 0.970" in text
