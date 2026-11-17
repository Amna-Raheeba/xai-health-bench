"""Tests for xai_health_bench.dataset."""
import json
from pathlib import Path

import pytest

from xai_health_bench.dataset import (
    DatasetError,
    apply_counterfactual,
    dataset_summary,
    load_dataset,
    validate_case,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "xai_health.jsonl"


def test_dataset_loads_and_counts():
    cases = load_dataset(DATASET)
    assert len(cases) == 48
    summary = dataset_summary(cases)
    assert summary["by_category"] == {
        "faithful": 14, "citation-trap": 8, "abstain": 8, "counterfactual": 10,
        "swap": 8}
    assert summary["facts"] == 63
    assert summary["counterfactuals"] == 10
    assert summary["swap_probes"] == 8
    assert summary["abstain_cases"] == 8


def test_dataset_ids_unique_and_prefixed():
    cases = load_dataset(DATASET)
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("xh-") for i in ids)


def test_all_evidence_titles_marked_synthetic():
    for case in load_dataset(DATASET):
        for ev in case["evidence"]:
            assert ev["title"].startswith("Synthetic summary"), case["id"]


def test_load_missing_file_raises():
    with pytest.raises(DatasetError):
        load_dataset(ROOT / "datasets" / "nope.jsonl")


def test_validate_rejects_unknown_category():
    bad = {
        "id": "xh-999", "project": "xai-health", "category": "bogus",
        "input": {"question": "Q?"}, "evidence": [],
        "expected": {"facts": [], "should_abstain": False},
    }
    with pytest.raises(DatasetError):
        validate_case(bad)


def test_validate_rejects_fact_without_key_phrases():
    bad = {
        "id": "xh-999", "project": "xai-health", "category": "faithful",
        "input": {"question": "Q?"},
        "evidence": [{"id": "S1", "title": "Synthetic summary — x", "text": "Some text here."}],
        "expected": {"facts": [{"text": "A fact.", "evidence_id": "S1", "key_phrases": []}],
                     "should_abstain": False},
    }
    with pytest.raises(DatasetError):
        validate_case(bad)


def test_validate_rejects_non_synthetic_title():
    bad = {
        "id": "xh-999", "project": "xai-health", "category": "faithful",
        "input": {"question": "Q?"},
        "evidence": [{"id": "S1", "title": "Real study (2024)", "text": "Some text here."}],
        "expected": {"facts": [], "should_abstain": False},
    }
    with pytest.raises(DatasetError):
        validate_case(bad)


def test_apply_counterfactual_remove():
    cases = {c["id"]: c for c in load_dataset(DATASET)}
    case = cases["xh-033"]
    cf = case["counterfactuals"][0]
    perturbed = apply_counterfactual(case, cf)
    assert [ev["id"] for ev in perturbed["evidence"]] == ["S2"]
    assert case["evidence"][0]["id"] == "S1"  # original untouched


def test_apply_counterfactual_swap_number():
    cases = {c["id"]: c for c in load_dataset(DATASET)}
    case = cases["xh-031"]
    cf = case["counterfactuals"][0]
    perturbed = apply_counterfactual(case, cf)
    assert "8.0%" in perturbed["evidence"][0]["text"]
    assert "7.0%" not in perturbed["evidence"][0]["text"]
    assert "7.0%" in case["evidence"][0]["text"]  # original untouched


def test_apply_counterfactual_negate():
    cases = {c["id"]: c for c in load_dataset(DATASET)}
    case = cases["xh-032"]
    cf = case["counterfactuals"][0]
    perturbed = apply_counterfactual(case, cf)
    assert "did not reduce" in perturbed["evidence"][0]["text"]
    assert "25%" not in perturbed["evidence"][0]["text"]
