"""Tests for the citation-swap probe (novel feature, 0.2.0).

A swap probe replaces a case's supporting passage with a contradicting
guideline and requires the agent's recommendation to move. It differs
from a ``negate`` counterfactual: the original recommendation survives
in a corroborating passage, so the probe tests conflict handling —
an extractive agent copies both passages and asserts both directions
at once (``stated_both``), while a careful agent follows the flipped
source or abstains.
"""
import copy

import pytest

from xai_health_bench import agents, metrics
from xai_health_bench.dataset import (
    DatasetError,
    apply_swap_probe,
    load_dataset,
    validate_case,
)
from xai_health_bench.scorer import run_bench


def swap_cases():
    return [c for c in load_dataset() if c.get("swap_probes")]


def test_dataset_has_eight_swap_probes():
    cases = swap_cases()
    assert len(cases) == 8
    assert all(c["category"] == "swap" for c in cases)
    assert all(len(c["swap_probes"]) == 1 for c in cases)


def test_swap_probe_validation_rejects_bad_probes():
    case = copy.deepcopy(swap_cases()[0])
    probe = case["swap_probes"][0]

    bad = copy.deepcopy(case)
    bad["swap_probes"][0]["target_evidence_id"] = "S9"
    with pytest.raises(DatasetError):
        validate_case(bad)

    bad = copy.deepcopy(case)
    bad["swap_probes"][0]["swapped_text"] = "   "
    with pytest.raises(DatasetError):
        validate_case(bad)

    bad = copy.deepcopy(case)
    bad["swap_probes"][0]["swapped_key_phrases"] = []
    with pytest.raises(DatasetError):
        validate_case(bad)

    bad = copy.deepcopy(case)
    bad["swap_probes"][0]["affected_fact"] = 99
    with pytest.raises(DatasetError):
        validate_case(bad)

    bad = copy.deepcopy(case)
    bad["swap_probes"][0]["expected"] = "fact_changed"
    with pytest.raises(DatasetError):
        validate_case(bad)

    bad = copy.deepcopy(case)
    bad["expected"]["should_abstain"] = True
    with pytest.raises(DatasetError):
        validate_case(bad)


def test_apply_swap_probe_replaces_only_the_target():
    case = swap_cases()[0]
    probe = case["swap_probes"][0]
    swapped = apply_swap_probe(case, probe)
    assert swapped["evidence"][0]["text"] == probe["swapped_text"]
    assert swapped["evidence"][1]["text"] == case["evidence"][1]["text"]
    assert swapped["evidence"][2]["text"] == case["evidence"][2]["text"]
    # The original case is untouched.
    assert case["evidence"][0]["text"] != probe["swapped_text"]


def test_demo_moves_on_all_swap_probes():
    for case in swap_cases():
        result = metrics.score_swap_probe(
            case, case["swap_probes"][0], lambda c: agents.run("demo", c))
        assert result["passed"], (case["id"], result)
        assert result["swap_sensitivity"] == 1.0
        assert not result["stated_both"]
        assert not result["abstained_on_swapped"]


def test_naive_and_confident_state_both_directions():
    for agent in ("naive", "confident"):
        for case in swap_cases():
            result = metrics.score_swap_probe(
                case, case["swap_probes"][0], lambda c, a=agent: agents.run(a, c))
            assert not result["passed"], (agent, case["id"], result)
            assert result["swap_sensitivity"] == 0.0
            assert result["stated_both"], (agent, case["id"], result)
            assert "BOTH" in result["detail"]


def test_abstention_counts_as_moving():
    case = swap_cases()[0]
    probe = case["swap_probes"][0]

    def abstaining_agent(_case):
        return {"answer": "I abstain: the evidence conflicts.",
                "abstained": True}

    result = metrics.score_swap_probe(case, probe, abstaining_agent)
    assert result["passed"]
    assert result["abstained_on_swapped"]


def test_mock_agents_cover_each_failure_mode():
    case = swap_cases()[0]
    probe = case["swap_probes"][0]
    target_text = next(ev["text"] for ev in case["evidence"]
                       if ev["id"] == probe["target_evidence_id"])

    def only_swapped(_case):
        return {"answer": probe["swapped_text"], "abstained": False}

    assert metrics.score_swap_probe(case, probe, only_swapped)["passed"]

    def only_original(_case):
        return {"answer": target_text, "abstained": False}

    failed = metrics.score_swap_probe(case, probe, only_original)
    assert not failed["passed"]
    assert "still stated" in failed["detail"]

    def neither(_case):
        return {"answer": "No relevant guidance was found in the evidence.",
                "abstained": False}

    failed = metrics.score_swap_probe(case, probe, neither)
    assert not failed["passed"]
    assert "not stated" in failed["detail"]


def test_aggregate_swap_sensitivity():
    demo = run_bench("demo")
    naive = run_bench("naive")
    assert demo["aggregate"]["swap_sensitivity"] == 1.0
    assert naive["aggregate"]["swap_sensitivity"] == 0.0
    assert len(demo["swap_probe_scores"]) == 8
    assert len(naive["swap_probe_scores"]) == 8
    # bench_score is unchanged by the probe: it is reported separately.
    assert demo["aggregate"]["bench_score"] == 1.0


def test_aggregate_without_probes_is_none():
    cases = [c for c in load_dataset() if not c.get("swap_probes")]
    report = run_bench("demo", cases)
    assert report["aggregate"]["swap_sensitivity"] is None
    assert report["swap_probe_scores"] == []
