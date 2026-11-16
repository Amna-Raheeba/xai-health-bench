"""Tests for xai_health_bench.metrics."""
from xai_health_bench import agents, metrics
from xai_health_bench.dataset import load_dataset

CASES = {c["id"]: c for c in load_dataset()}


def test_fact_coverage_full_and_empty():
    case = CASES["xh-001"]
    result = agents.run("demo", case)
    assert metrics.fact_coverage(case, result["answer"]) == 1.0
    assert metrics.fact_coverage(case, "Nothing relevant here.") == 0.0


def test_fact_present_is_per_sentence():
    # "below" from one sentence and "30" from another must NOT combine.
    answer = "Metformin should be stopped. The eGFR is 30 in this patient."
    case = CASES["xh-037"]
    assert metrics.fact_coverage(case, answer) == 0.0
    answer2 = "Metformin should be stopped when eGFR falls below 30."
    assert metrics.fact_coverage(case, answer2) == 1.0


def test_citation_precision_catches_invented_number():
    case = CASES["xh-001"]
    answer = "In a large trial, 94% of patients improved within 2 days. [S1]"
    cp = metrics.citation_precision(case, answer)
    assert cp["citation_precision"] == 0.0
    assert "94%" in cp["audited"][0]["sentence"]
    assert "does not appear" in cp["audited"][0]["reason"]


def test_citation_precision_passes_verbatim_claim():
    case = CASES["xh-001"]
    result = agents.run("demo", case)
    cp = metrics.citation_precision(case, result["answer"])
    assert cp["citation_precision"] == 1.0


def test_citation_to_missing_passage_fails():
    case = CASES["xh-001"]
    cp = metrics.citation_precision(case, "Some claim here. [S9]")
    assert cp["citation_precision"] == 0.0
    assert "no evidence passage" in cp["audited"][0]["reason"]


def test_citation_marker_digit_not_counted_as_number():
    # Regression: "[S1]" contains "1"; the audit must not read it as a
    # number the claim states.
    case = CASES["xh-001"]
    answer = ("Adults aged 35 to 70 years who have overweight or obesity should be "
              "screened for type 2 diabetes [S1].")
    cp = metrics.citation_precision(case, answer)
    assert cp["citation_precision"] == 1.0


def test_grounding_rate_excludes_disclaimer():
    case = CASES["xh-001"]
    result = agents.run("demo", case)
    assert metrics.grounding_rate(result["answer"]) == 1.0
    assert metrics.grounding_rate("This is an uncited claim about treatment.") == 0.0


def test_distractor_avoidance():
    case = CASES["xh-015"]
    demo = agents.run("demo", case)
    naive = agents.run("naive", case)
    assert metrics.distractor_avoidance(case, demo["answer"]) == 1.0
    assert metrics.distractor_avoidance(case, naive["answer"]) == 0.0


def test_score_case_abstained_shape():
    case = CASES["xh-023"]
    result = agents.run("demo", case)
    score = metrics.score_case(case, result)
    assert score["abstained"] is True
    assert score["abstention_correct"] == 1.0
    assert score["fact_coverage"] is None


def test_score_case_wrong_abstention():
    case = CASES["xh-023"]
    result = agents.run("naive", case)
    score = metrics.score_case(case, result)
    assert score["abstention_correct"] == 0.0


def test_counterfactual_remove_passes_for_demo():
    case = CASES["xh-033"]
    cf = case["counterfactuals"][0]
    scored = metrics.score_counterfactual(case, cf, lambda c: agents.run("demo", c))
    assert scored["passed"] is True
    assert scored["faithfulness"] == 1.0


def test_counterfactual_swap_number_passes_for_demo():
    case = CASES["xh-031"]
    cf = case["counterfactuals"][0]
    scored = metrics.score_counterfactual(case, cf, lambda c: agents.run("demo", c))
    assert scored["passed"] is True


def test_counterfactual_negate_passes_for_demo():
    case = CASES["xh-032"]
    cf = case["counterfactuals"][0]
    scored = metrics.score_counterfactual(case, cf, lambda c: agents.run("demo", c))
    assert scored["passed"] is True


def test_counterfactual_fails_when_answer_does_not_change():
    case = CASES["xh-031"]

    def stubborn(_case):
        return {"answer": "Most adults with type 2 diabetes should aim for an "
                         "HbA1c below 7.0% [S1].", "abstained": False}

    cf = case["counterfactuals"][0]
    scored = metrics.score_counterfactual(case, cf, stubborn)
    assert scored["passed"] is False
    assert scored["faithfulness"] == 0.0


def test_aggregate_weights_sum_to_one():
    assert abs(sum(metrics.WEIGHTS.values()) - 1.0) < 1e-9


def test_aggregate_composite_matches_weights():
    case_scores = [{"case_id": "a", "category": "faithful", "abstained": False,
                    "abstention_correct": 1.0, "fact_coverage": 1.0,
                    "citation_precision": 1.0, "grounding_rate": 1.0,
                    "distractor_avoidance": 1.0}]
    cf_scores = [{"faithfulness": 1.0}]
    agg = metrics.aggregate(case_scores, cf_scores)
    assert agg["bench_score"] == 1.0
    assert agg["faithfulness"] == 1.0
