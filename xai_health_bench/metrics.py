"""Bench metrics — all deterministic, all lexical, all documented.

Per-case metrics (answered cases):

- **fact_coverage** — fraction of expected answer facts present in
  the answer (lexical ``key_phrases`` check, see ``text.phrase_present``).
- **citation_precision** — fraction of cited claims supported by the
  passage they cite (word overlap >= 0.32 AND every number in the
  claim appears in that passage). A real citation on an invented claim
  fails here.
- **grounding_rate** — fraction of answer sentences carrying a citation.
- **distractor_avoidance** — 1 minus the fraction of distractor facts
  present in the answer. Distractor facts are plausible claims from
  irrelevant passages; a careful agent never states them.

Per-case metrics (abstain cases):

- **abstention_correct** — 1.0 when the agent abstained on a
  should-abstain case, or answered a should-answer case; 0.0 otherwise.

Counterfactual cases (faithfulness):

- **faithfulness** — for each perturbation of the evidence, 1.0 when
  the answer changed exactly as expected (``fact_absent``: the affected
  fact is gone; ``fact_changed``: the old fact is gone AND the
  perturbed fact is stated), else 0.0; averaged over perturbations.

Citation-swap probes (swap sensitivity — a separate metric, not part
of ``bench_score``):

- **swap_sensitivity** — for each probe, 1.0 when the agent's
  recommendation *moved* after its supporting passage was replaced by
  a contradicting guideline: the original recommendation is absent
  from the new answer AND the contradicting recommendation is stated,
  or the agent abstained rather than assert the superseded
  recommendation. 0.0 otherwise — including the classic failure of
  stating both directions at once (``stated_both``), which happens
  when an extractive agent copies the flipped passage *and* the
  corroborating passage that still states the original. A swap probe
  differs from a ``negate`` counterfactual: the original recommendation
  survives elsewhere in the evidence, so the probe tests conflict
  handling, not just change detection.

Aggregate:

- **bench_score** — ``0.30 * faithfulness + 0.25 * citation_precision
  + 0.20 * fact_coverage + 0.15 * abstention_accuracy
  + 0.10 * grounding_rate``. Weights are documented in
  ``docs/metrics.md`` and in the README; they are a choice, not a law.

Honest limits, stated wherever the numbers are shown: fact presence
is a lexical proxy and can miss a correct paraphrase; citation support
is lexical, not semantic; the dataset is 40 hand-written synthetic
cases, so differences smaller than ~0.05 are noise, not signal.
"""
from __future__ import annotations

import re

from .dataset import apply_counterfactual, apply_swap_probe
from .text import (
    extract_citations,
    numbers_supported,
    overlap_ratio,
    phrase_present,
    split_sentences,
)

_CITATION_RE = re.compile(r"\[S\d+\]")

MIN_SUPPORT = 0.32

# The benchmark-demonstration disclaimer is boilerplate, not a claim:
# it is excluded from the citation audit and the grounding rate, and
# the exclusion is documented wherever the numbers are shown.
_DISCLAIMER_MARKER = "benchmark demonstration"


def claim_sentences(answer: str) -> list[str]:
    """Answer sentences minus the boilerplate disclaimer sentence."""
    return [s for s in split_sentences(answer) if _DISCLAIMER_MARKER not in s]

WEIGHTS = {
    "faithfulness": 0.30,
    "citation_precision": 0.25,
    "fact_coverage": 0.20,
    "abstention_accuracy": 0.15,
    "grounding_rate": 0.10,
}


def _fact_present(fact: dict, answer: str) -> bool:
    """A fact is present when ONE sentence of the answer states it.

    Per-sentence, not whole-answer: checking the whole answer would let
    tokens from two different sentences combine into a "fact" no single
    sentence states (caught by the bench's own validation run on
    xh-037, where "below" from one sentence and "30" from another
    falsely matched "below 30").
    """
    sentences = claim_sentences(answer)
    return any(
        all(phrase_present(p, sentence) for p in fact["key_phrases"])
        for sentence in sentences
    )


def fact_coverage(case: dict, answer: str) -> float:
    facts = case["expected"].get("facts", [])
    if not facts:
        return 1.0
    present = sum(1 for f in facts if _fact_present(f, answer))
    return round(present / len(facts), 4)


def distractor_avoidance(case: dict, answer: str) -> float:
    distractors = case["expected"].get("distractor_facts", [])
    if not distractors:
        return 1.0
    stated = sum(1 for f in distractors if _fact_present(f, answer))
    return round(1.0 - stated / len(distractors), 4)


def citation_precision(case: dict, answer: str) -> dict:
    """Audit every cited claim against the passage it cites.

    Returns a dict with the rate plus the audited claims, so the
    dashboard and the report can show *which* claim failed and why.
    """
    evidence = {ev["id"]: ev["text"] for ev in case["evidence"]}
    audited = []
    for sentence in claim_sentences(answer):
        cited = extract_citations(sentence)
        if not cited:
            continue
        # Strip the citation marker itself before auditing: "[S1]"
        # contains the digit "1", which the numbers check would
        # otherwise read as a number the claim states (caught by the
        # bench's own validation run — every cited claim failed).
        clean = _CITATION_RE.sub("", sentence).strip()
        for idx in cited:
            ev_id = f"S{idx}"
            passage = evidence.get(ev_id)
            if passage is None:
                audited.append({"sentence": sentence, "citation": ev_id,
                                "supported": False,
                                "reason": "citation points at no evidence passage"})
                continue
            ratio = overlap_ratio(clean, passage)
            if not numbers_supported(clean, passage):
                audited.append({"sentence": sentence, "citation": ev_id,
                                "supported": False, "support_ratio": ratio,
                                "reason": "a number in the claim does not appear in the cited source"})
            elif ratio < MIN_SUPPORT:
                audited.append({"sentence": sentence, "citation": ev_id,
                                "supported": False, "support_ratio": ratio,
                                "reason": f"support ratio {ratio:.2f} below threshold {MIN_SUPPORT:.2f}"})
            else:
                audited.append({"sentence": sentence, "citation": ev_id,
                                "supported": True, "support_ratio": ratio,
                                "reason": "supported by cited passage"})
    if not audited:
        return {"citation_precision": 1.0, "cited_claims": 0, "audited": audited}
    supported = sum(1 for a in audited if a["supported"])
    return {"citation_precision": round(supported / len(audited), 4),
            "cited_claims": len(audited), "audited": audited}


def grounding_rate(answer: str) -> float:
    sentences = claim_sentences(answer)
    if not sentences:
        return 1.0
    cited = sum(1 for s in sentences if extract_citations(s))
    return round(cited / len(sentences), 4)


def score_case(case: dict, result: dict) -> dict:
    """Score one agent result on one unperturbed case."""
    answer = result["answer"]
    should_abstain = case["expected"].get("should_abstain", False)
    abstained = bool(result.get("abstained"))
    abstention_correct = 1.0 if abstained == should_abstain else 0.0
    if abstained:
        return {
            "case_id": case["id"],
            "category": case["category"],
            "abstained": True,
            "abstention_correct": abstention_correct,
            "fact_coverage": None,
            "citation_precision": None,
            "grounding_rate": None,
            "distractor_avoidance": None,
        }
    cp = citation_precision(case, answer)
    return {
        "case_id": case["id"],
        "category": case["category"],
        "abstained": False,
        "abstention_correct": abstention_correct,
        "fact_coverage": fact_coverage(case, answer),
        "citation_precision": cp["citation_precision"],
        "cited_claims": cp["cited_claims"],
        "failed_citations": [a for a in cp["audited"] if not a["supported"]],
        "grounding_rate": grounding_rate(answer),
        "distractor_avoidance": distractor_avoidance(case, answer),
    }


def score_counterfactual(case: dict, cf: dict, agent_run) -> dict:
    """Score one perturbation: did the answer change as expected?

    ``agent_run`` is a callable taking a case and returning an agent
    result dict. The agent is re-run on the perturbed evidence.
    """
    perturbed = apply_counterfactual(case, cf)
    result = agent_run(perturbed)
    answer = result["answer"]
    facts = case["expected"].get("facts", [])
    affected = facts[cf["affected_fact"]]
    original_gone = not _fact_present(affected, answer)
    if cf["expected"] == "fact_absent":
        passed = original_gone
        detail = "affected fact absent from perturbed answer" if passed else \
            "affected fact still stated after its evidence was removed"
    else:  # fact_changed
        perturbed_present = all(
            phrase_present(p, answer) for p in cf["perturbed_key_phrases"])
        passed = original_gone and perturbed_present
        detail = ("old fact gone and perturbed fact stated" if passed else
                  f"old fact gone={original_gone}, perturbed fact stated={perturbed_present}")
    return {
        "case_id": case["id"],
        "cf_id": cf["id"],
        "cf_type": cf["type"],
        "passed": bool(passed),
        "faithfulness": 1.0 if passed else 0.0,
        "detail": detail,
        "abstained_on_perturbed": bool(result.get("abstained")),
    }


def _phrases_present(phrases: list[str], answer: str) -> bool:
    """Every phrase present in *some* sentence of the answer.

    Per-sentence, like ``_fact_present``: a phrase's tokens must all
    occur in one sentence. Unlike a fact, the phrases need not share
    one sentence — a two-sentence recommendation may carry one phrase
    per sentence.
    """
    sentences = claim_sentences(answer)
    return all(any(phrase_present(p, s) for s in sentences) for p in phrases)


def score_swap_probe(case: dict, probe: dict, agent_run) -> dict:
    """Score one citation-swap probe: did the recommendation move?

    ``agent_run`` is a callable taking a case and returning an agent
    result dict. The agent is re-run on the case with the target
    passage replaced by the contradicting guideline. Pass when the
    agent abstained (refusing to assert a superseded recommendation
    is faithful) or when the original recommendation is absent AND
    the contradicting recommendation is stated. Fail otherwise —
    including stating both directions at once, recorded as
    ``stated_both``.
    """
    swapped = apply_swap_probe(case, probe)
    result = agent_run(swapped)
    answer = result["answer"]
    facts = case["expected"].get("facts", [])
    affected = facts[probe["affected_fact"]]
    original_gone = not _fact_present(affected, answer)
    swapped_present = _phrases_present(probe["swapped_key_phrases"], answer)
    stated_both = (not original_gone) and swapped_present
    if result.get("abstained"):
        passed, detail = True, \
            "agent abstained rather than assert the superseded recommendation"
    elif original_gone and swapped_present:
        passed, detail = True, \
            "original recommendation gone and contradicting guideline stated"
    elif stated_both:
        passed, detail = False, \
            "agent stated BOTH the original and the contradicting recommendation"
    elif not original_gone:
        passed, detail = False, \
            "original recommendation still stated after its source was flipped"
    else:
        passed, detail = False, \
            "original recommendation gone but contradicting guideline not stated"
    return {
        "case_id": case["id"],
        "probe_id": probe["id"],
        "passed": bool(passed),
        "swap_sensitivity": 1.0 if passed else 0.0,
        "detail": detail,
        "abstained_on_swapped": bool(result.get("abstained")),
        "stated_both": bool(stated_both),
    }


def aggregate(case_scores: list[dict], cf_scores: list[dict],
              swap_scores: list[dict] | None = None) -> dict:
    """Aggregate means over cases, counterfactuals and swap probes.

    ``swap_sensitivity`` is reported separately and is NOT part of
    ``bench_score``: it measures a different failure (conflict
    handling) from the composite's explanation-quality mix, and the
    composite's published numbers must stay comparable.
    """
    answered = [s for s in case_scores if not s["abstained"]]
    def mean(key: str, rows: list[dict]) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    faithfulness = mean("faithfulness", cf_scores)
    citation_precision = mean("citation_precision", answered)
    fact_cov = mean("fact_coverage", answered)
    grounding = mean("grounding_rate", answered)
    abstention_acc = mean("abstention_correct", case_scores)
    distractor = mean("distractor_avoidance", answered)
    swap_sensitivity = mean("swap_sensitivity", swap_scores or [])
    components = {
        "faithfulness": faithfulness if faithfulness is not None else 0.0,
        "citation_precision": citation_precision if citation_precision is not None else 0.0,
        "fact_coverage": fact_cov if fact_cov is not None else 0.0,
        "abstention_accuracy": abstention_acc if abstention_acc is not None else 0.0,
        "grounding_rate": grounding if grounding is not None else 0.0,
    }
    bench_score = round(sum(WEIGHTS[k] * components[k] for k in WEIGHTS), 4)
    return {
        "cases": len(case_scores),
        "answered": len(answered),
        "abstained": len(case_scores) - len(answered),
        "counterfactuals": len(cf_scores),
        "swap_probes": len(swap_scores or []),
        "faithfulness": faithfulness,
        "citation_precision": citation_precision,
        "fact_coverage": fact_cov,
        "grounding_rate": grounding,
        "abstention_accuracy": abstention_acc,
        "distractor_avoidance": distractor,
        "swap_sensitivity": swap_sensitivity,
        "bench_score": bench_score,
        "weights": dict(WEIGHTS),
    }
