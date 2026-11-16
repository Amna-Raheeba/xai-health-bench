"""Demo agents scored by the bench.

Every agent shares one interface — ``run(agent_name, case)`` returns::

    {"answer": str, "citations": [int, ...], "abstained": bool,
     "status": "ok" | "abstained" | "refused", "trace": {...}}

Three agents ship so the bench can prove it discriminates:

- **demo** — selective answering in the healthcare-research-agent
  style: BM25 retrieval, lexical claim verification, evidence-strength
  abstention gate (threshold 0.62). Composes only verified claims.
- **naive** — extractive baseline: top-3 BM25 passages, every sentence
  composed with a citation, never abstains. Shows what the bench
  catches when verification and abstention are missing.
- **confident** — synthetic bad agent, documented as such: like naive,
  but injects one invented statistic and states the first distractor
  fact when the case has one. Exists only to validate that the bench
  ranks a hallucinating agent lowest. Never use it as a real agent.

All three are deterministic, standard-library only, and offline.
"""
from __future__ import annotations

from .text import (
    bm25_scores,
    content_tokens,
    expand_question,
    numbers_supported,
    overlap_ratio,
    question_coverage,
    split_sentences,
    stemmed_tokens,
)

# Abstention threshold, tuned by sweep on this dataset (see
# notebooks/02_abstention_threshold_sweep.py): abstain cases score an
# evidence strength of 0.0 and the weakest answerable case scores
# 0.577, so every threshold in [0.30, 0.55] gives 40/40 abstention
# accuracy; 0.50 sits inside that plateau. (The
# healthcare-research-agent used 0.62 on its own corpus — thresholds
# are corpus-specific and must be swept, not copied.)
ABSTENTION_THRESHOLD = 0.50
_MIN_SUPPORT = 0.32
_TOP_K = 3

# One sentence on purpose: the bench's citation audit and grounding
# rate exclude exactly this sentence (matched on "benchmark
# demonstration"), so it must not split into two sentences, one of
# which would be scored as an uncited claim.
DISCLAIMER = (
    "This answer is a benchmark demonstration on synthetic evidence, "
    "not medical advice, consult a clinician for decisions about care."
)

# Injected by the confident agent only. A fixed invented statistic:
# 94% and "2 days" appear in no evidence passage, so the citation
# audit must catch it. Documented in the README as a synthetic fault.
_INVENTED_STAT = "In a large trial, 94% of patients improved within 2 days."

_EMERGENCY_WORDS = ("chest pain", "can't breathe", "cannot breathe",
                    "unconscious", "overdose", "suicide")
_DIAGNOSIS_WORDS = ("do i have", "diagnose me", "am i suffering from")


def _is_refusal(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in _EMERGENCY_WORDS + _DIAGNOSIS_WORDS)


def _retrieve(case: dict, top_k: int = _TOP_K) -> list[tuple[dict, float]]:
    """Top-k evidence passages by BM25, with scores."""
    evidence = case["evidence"]
    query = expand_question(case["input"]["question"])
    scores = bm25_scores(query, [ev["text"] for ev in evidence])
    ranked = sorted(zip(evidence, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


def _verify(sentence: str, passage_text: str) -> tuple[bool, float, str]:
    """Lexical claim check: wording and numbers separately."""
    ratio = overlap_ratio(sentence, passage_text)
    if not numbers_supported(sentence, passage_text):
        return False, ratio, "a number in the claim does not appear in the cited source"
    if ratio < _MIN_SUPPORT:
        return False, ratio, f"support ratio {ratio:.2f} below threshold {_MIN_SUPPORT:.2f}"
    return True, ratio, "supported by cited passage"


def _usable_claims(verified: list[dict], question: str) -> list[dict]:
    """Verified claims that are actually about the question.

    Passage-trust selection: a passage is trusted when its best
    sentence covers at least 40% of the question's content tokens;
    every supported sentence of a trusted passage is usable, minus
    near-duplicates. Two earlier rules failed and are recorded here so
    they are not re-tried: a relative-BM25 floor dropped second facts
    (a second fact often shares only the drug name with the question),
    and greedy set-cover over question tokens dropped second facts for
    the same reason while keeping distractors that shared the drug
    name. Passage trust separates them because a distractor passage's
    best sentence covers far less of the question than the answering
    passage's best sentence — provided the dataset is curated so that
    answering passages echo the question's terms, which the dataset
    builder checks (see scripts/build_dataset.py).
    """
    supported = [v for v in verified if v["supported"]]
    if not supported:
        return []
    best_by_passage: dict[str, float] = {}
    for v in supported:
        best_by_passage[v["evidence_id"]] = max(
            best_by_passage.get(v["evidence_id"], 0.0), v["coverage"])
    trusted = {ev for ev, cov in best_by_passage.items() if cov >= 0.40}
    usable = [v for v in supported if v["evidence_id"] in trusted]
    usable.sort(key=lambda v: (-v["coverage"], -v["bm25"]))
    seen: set[str] = set()
    deduped: list[dict] = []
    for v in usable:
        key = " ".join(sorted(set(content_tokens(v["text"]))))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    return deduped


def _evidence_strength(verified: list[dict], question: str, top_score: float) -> float:
    """Strength of the evidence behind a would-be answer, in [0, 1].

    Same shape as the healthcare-research-agent's evidence-strength
    score: verifier support of usable claims, question coverage,
    retrieval saturation.
    """
    usable = _usable_claims(verified, question)
    if not usable:
        return 0.0
    mean_support = sum(v["support_ratio"] for v in usable) / len(usable)
    coverages = [v["coverage"] for v in usable]
    saturation = top_score / (top_score + 4.0) if top_score > 0 else 0.0
    strength = (
        0.35 * mean_support
        + 0.30 * max(coverages)
        + 0.20 * (sum(coverages) / len(coverages))
        + 0.15 * saturation
    )
    return round(min(max(strength, 0.0), 1.0), 4)


def _verified_claims(case: dict, retrieved: list[tuple[dict, float]]) -> list[dict]:
    """Draft claims from retrieved passages, each verified against its passage."""
    question = case["input"]["question"]
    sentences: list[tuple[str, dict]] = []
    for rank, (ev, score) in enumerate(retrieved, start=1):
        for sentence in split_sentences(ev["text"]):
            sentences.append((sentence, ev))
    bm25 = bm25_scores(expand_question(question), [s for s, _ in sentences])
    verified = []
    for (sentence, ev), score in zip(sentences, bm25):
        supported, ratio, reason = _verify(sentence, ev["text"])
        verified.append({
            "text": sentence,
            "citation_index": int(ev["id"][1:]),
            "evidence_id": ev["id"],
            "support_ratio": ratio,
            "supported": supported,
            "coverage": round(question_coverage(sentence, question), 4),
            "bm25": score,
            "reason": reason,
        })
    return verified


def _compose(verified: list[dict], question: str) -> str:
    """Answer from usable claims only, each cited."""
    usable = _usable_claims(verified, question)
    usable.sort(key=lambda v: (-v["bm25"], v["citation_index"]))
    seen: set[str] = set()
    sentences = []
    for v in usable:
        key = " ".join(sorted(set(content_tokens(v["text"]))))
        if key in seen:
            continue
        seen.add(key)
        # Citation goes BEFORE the full stop: "claim [S1]." — a citation
        # after the period would attach to the next sentence when the
        # answer is split back into sentences for auditing.
        sentences.append(f"{v['text'].rstrip('.!?')} [S{v['citation_index']}].")
    return " ".join(sentences)


def run_demo(case: dict) -> dict:
    """Selective-answering agent: retrieve, verify, abstain if weak."""
    question = case["input"]["question"]
    if _is_refusal(question):
        return {"answer": "I can't help with that here. " + DISCLAIMER,
                "citations": [], "abstained": False, "status": "refused",
                "trace": {"reason": "guardrail refusal"}}
    retrieved = _retrieve(case)
    verified = _verified_claims(case, retrieved)
    top_score = retrieved[0][1] if retrieved else 0.0
    strength = _evidence_strength(verified, question, top_score)
    if strength < ABSTENTION_THRESHOLD:
        return {
            "answer": ("I'm going to abstain on this one: the evidence does not "
                       "contain enough verified material to answer this reliably. "
                       + DISCLAIMER),
            "citations": [], "abstained": True, "status": "abstained",
            "trace": {"evidence_strength": strength,
                      "threshold": ABSTENTION_THRESHOLD,
                      "reason": "evidence strength below threshold"},
        }
    answer = _compose(verified, question)
    if not answer:
        return {"answer": "No verified claim covers this question. " + DISCLAIMER,
                "citations": [], "abstained": True, "status": "abstained",
                "trace": {"evidence_strength": strength,
                          "threshold": ABSTENTION_THRESHOLD,
                          "reason": "no verified claim covers the question"}}
    return {"answer": answer + " " + DISCLAIMER,
            "citations": sorted({v["citation_index"] for v in _usable_claims(verified, question)}),
            "abstained": False, "status": "ok",
            "trace": {"evidence_strength": strength,
                      "threshold": ABSTENTION_THRESHOLD,
                      "retrieved": [ev["id"] for ev, _ in retrieved]}}


def run_naive(case: dict) -> dict:
    """Extractive baseline: every retrieved sentence, cited, never abstains."""
    question = case["input"]["question"]
    if _is_refusal(question):
        return {"answer": "I can't help with that here. " + DISCLAIMER,
                "citations": [], "abstained": False, "status": "refused",
                "trace": {"reason": "guardrail refusal"}}
    retrieved = _retrieve(case)
    sentences = []
    citations = set()
    for ev, _ in retrieved:
        for sentence in split_sentences(ev["text"]):
            sentences.append(f"{sentence.rstrip('.!?')} [S{ev['id'][1:]}].")
            citations.add(int(ev["id"][1:]))
    return {"answer": " ".join(sentences) + " " + DISCLAIMER,
            "citations": sorted(citations), "abstained": False, "status": "ok",
            "trace": {"retrieved": [ev["id"] for ev, _ in retrieved]}}


def run_confident(case: dict) -> dict:
    """Synthetic hallucinating agent, for bench validation only.

    Like naive, plus one invented statistic and — when the case has
    one — the first distractor fact, stated as if supported. The bench
    must rank this agent lowest; if it does not, the bench is broken.
    """
    result = run_naive(case)
    if result["status"] == "refused":
        return result
    answer = result["answer"]
    distractors = case["expected"].get("distractor_facts", [])
    if distractors:
        answer = distractors[0]["text"] + " " + answer
    # The invented statistic is cited to [S1] on purpose: the classic
    # failure is a real citation attached to a claim the source never
    # made. That is exactly what the citation audit must catch.
    answer = answer + " " + _INVENTED_STAT + " [S1]"
    return {"answer": answer, "citations": sorted(set(result["citations"]) | {1}),
            "abstained": False, "status": "ok",
            "trace": {**result["trace"], "injected_statistic": True,
                      "injected_distractor": bool(distractors)}}


AGENTS = {"demo": run_demo, "naive": run_naive, "confident": run_confident}


def run(agent_name: str, case: dict) -> dict:
    """Run one named agent on one case."""
    if agent_name not in AGENTS:
        raise ValueError(f"unknown agent {agent_name!r}; known: {sorted(AGENTS)}")
    return AGENTS[agent_name](case)


def agent_names() -> list[str]:
    return sorted(AGENTS)
