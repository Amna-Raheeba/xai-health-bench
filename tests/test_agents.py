"""Tests for xai_health_bench.agents."""
import pytest

from xai_health_bench import agents
from xai_health_bench.dataset import load_dataset

CASES = {c["id"]: c for c in load_dataset()}


def test_agent_names():
    assert agents.agent_names() == ["confident", "demo", "naive"]


def test_run_unknown_agent_raises():
    with pytest.raises(ValueError):
        agents.run("ghost", CASES["xh-001"])


def test_demo_answers_faithful_case_with_citations():
    result = agents.run("demo", CASES["xh-001"])
    assert result["status"] == "ok"
    assert not result["abstained"]
    assert "[S1]" in result["answer"]
    assert "7.0%" not in result["answer"]  # xh-001 is screening, not the target case


def test_demo_abstains_on_abstain_cases():
    for cid in ("xh-023", "xh-024", "xh-025", "xh-026", "xh-027", "xh-028", "xh-029", "xh-030"):
        result = agents.run("demo", CASES[cid])
        assert result["abstained"] is True, cid
        assert result["status"] == "abstained"


def test_demo_does_not_state_distractor_facts():
    result = agents.run("demo", CASES["xh-015"])
    assert "improve survival in heart failure" not in result["answer"]
    result = agents.run("demo", CASES["xh-017"])
    assert "rarely cause cough" not in result["answer"]


def test_naive_never_abstains_and_states_distractors():
    result = agents.run("naive", CASES["xh-023"])
    assert result["abstained"] is False
    result = agents.run("naive", CASES["xh-015"])
    assert "improve survival in heart failure" in result["answer"]


def test_confident_injects_invented_statistic_with_citation():
    result = agents.run("confident", CASES["xh-001"])
    assert "94% of patients improved within 2 days" in result["answer"]
    assert "[S1]" in result["answer"]


def test_confident_states_distractor_fact():
    result = agents.run("confident", CASES["xh-015"])
    assert "improve survival in heart failure" in result["answer"]


def test_demo_refuses_emergency_question():
    case = dict(CASES["xh-001"])
    case["input"] = {"question": "I have chest pain, what do I do?"}
    result = agents.run("demo", case)
    assert result["status"] == "refused"


def test_demo_refuses_diagnosis_request():
    case = dict(CASES["xh-001"])
    case["input"] = {"question": "Do I have diabetes? Diagnose me"}
    result = agents.run("demo", case)
    assert result["status"] == "refused"


def test_demo_answer_carries_disclaimer():
    result = agents.run("demo", CASES["xh-001"])
    assert "benchmark demonstration" in result["answer"]
    assert "not medical advice" in result["answer"]


def test_usable_claims_trusts_answering_passage_only():
    case = CASES["xh-017"]
    retrieved = agents._retrieve(case)
    verified = agents._verified_claims(case, retrieved)
    usable = agents._usable_claims(verified, case["input"]["question"])
    evidence_ids = {v["evidence_id"] for v in usable}
    assert evidence_ids == {"S1"}  # distractor passage S2 excluded


def test_evidence_strength_zero_when_no_usable_claims():
    case = CASES["xh-023"]
    retrieved = agents._retrieve(case)
    verified = agents._verified_claims(case, retrieved)
    assert agents._evidence_strength(verified, case["input"]["question"], 0.0) == 0.0


def test_demo_deterministic():
    a = agents.run("demo", CASES["xh-002"])
    b = agents.run("demo", CASES["xh-002"])
    assert a["answer"] == b["answer"]
    assert a["citations"] == b["citations"]
