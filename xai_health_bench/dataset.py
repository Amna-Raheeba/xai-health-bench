"""Dataset schema, loading, validation, and counterfactual perturbation.

Case schema (one JSON object per line in ``datasets/xai_health.jsonl``)::

    {
      "id": "xh-001",
      "project": "xai-health",
      "category": "faithful" | "citation-trap" | "abstain" | "counterfactual",
      "difficulty": "easy" | "medium" | "hard",
      "tags": ["..."],
      "input": {"question": "..."},
      "evidence": [{"id": "S1", "title": "...", "text": "..."}],
      "expected": {
        "facts": [{"text": "...", "evidence_id": "S1",
                   "key_phrases": ["...", "..."]}],
        "distractor_facts": [{"text": "...", "evidence_id": "S2",
                              "key_phrases": ["..."]}],
        "should_abstain": false
      },
      "counterfactuals": [
        {"id": "cf-1", "type": "remove" | "negate" | "swap_number",
         "target_evidence_id": "S2", "description": "...",
         "perturbed_text": "...",            # for negate / swap_number
         "affected_fact": 0,                 # index into expected.facts
         "perturbed_key_phrases": ["..."],   # for negate / swap_number
         "expected": "fact_absent" | "fact_changed"}
      ],
      "swap_probes": [
        {"id": "sp-1",
         "target_evidence_id": "S1", "description": "...",
         "swapped_text": "...",              # contradicting guideline
         "affected_fact": 0,                 # index into expected.facts
         "swapped_key_phrases": ["..."],     # the new recommendation
         "expected": "moved"}
      ]
    }

All evidence is synthetic and hand-written for benchmarking. Every
evidence title starts with "Synthetic summary" so no passage can be
mistaken for a real study.

A citation-swap probe differs from a counterfactual ``negate`` in
three ways: (1) the replacement is a contradicting *guideline* — a
directive reversal ("advises against X; use Y instead"), not a
negated finding; (2) the original recommendation survives elsewhere
in the evidence (a corroborating passage restates it), so the probe
tests whether the agent follows the flipped authoritative source
rather than copying everything and asserting both directions at once;
(3) abstaining counts as moving — refusing to assert a superseded
recommendation is faithful. See ``metrics.score_swap_probe``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

CATEGORIES = ("faithful", "citation-trap", "abstain", "counterfactual", "swap")
DIFFICULTIES = ("easy", "medium", "hard")
CF_TYPES = ("remove", "negate", "swap_number")

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "datasets" / "xai_health.jsonl"


class DatasetError(ValueError):
    pass


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[dict]:
    """Load and validate the dataset; raises DatasetError on any problem."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"dataset not found: {path}")
    cases = []
    seen_ids = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        validate_case(case, lineno)
        if case["id"] in seen_ids:
            raise DatasetError(f"{path}:{lineno}: duplicate case id {case['id']!r}")
        seen_ids.add(case["id"])
        cases.append(case)
    if not cases:
        raise DatasetError(f"{path}: no cases")
    return cases


def validate_case(case: dict, lineno: int = 0) -> None:
    """Validate one case dict; raises DatasetError with the location."""
    where = f"line {lineno}" if lineno else f"case {case.get('id', '?')!r}"
    for key in ("id", "project", "category", "input", "evidence", "expected"):
        if key not in case:
            raise DatasetError(f"{where}: missing key {key!r}")
    if case["project"] != "xai-health":
        raise DatasetError(f"{where}: project must be 'xai-health'")
    if case["category"] not in CATEGORIES:
        raise DatasetError(f"{where}: unknown category {case['category']!r}")
    if case.get("difficulty", "medium") not in DIFFICULTIES:
        raise DatasetError(f"{where}: unknown difficulty {case.get('difficulty')!r}")
    if "question" not in case["input"] or not case["input"]["question"].strip():
        raise DatasetError(f"{where}: input.question must be non-empty")
    evidence_ids = set()
    for ev in case["evidence"]:
        if "id" not in ev or "text" not in ev or not ev["text"].strip():
            raise DatasetError(f"{where}: evidence entries need id and non-empty text")
        if not ev["title"].startswith("Synthetic summary"):
            raise DatasetError(
                f"{where}: evidence {ev['id']!r} title must start with "
                "'Synthetic summary' (no passage may look like a real study)")
        evidence_ids.add(ev["id"])
    expected = case["expected"]
    if not isinstance(expected.get("facts", []), list):
        raise DatasetError(f"{where}: expected.facts must be a list")
    for i, fact in enumerate(expected.get("facts", [])):
        if not fact.get("key_phrases"):
            raise DatasetError(f"{where}: fact {i} needs key_phrases")
        if fact.get("evidence_id") not in evidence_ids:
            raise DatasetError(f"{where}: fact {i} cites unknown evidence {fact.get('evidence_id')!r}")
    for i, fact in enumerate(expected.get("distractor_facts", [])):
        if not fact.get("key_phrases"):
            raise DatasetError(f"{where}: distractor fact {i} needs key_phrases")
        if fact.get("evidence_id") not in evidence_ids:
            raise DatasetError(f"{where}: distractor fact {i} cites unknown evidence {fact.get('evidence_id')!r}")
    if not isinstance(expected.get("should_abstain", False), bool):
        raise DatasetError(f"{where}: expected.should_abstain must be a boolean")
    for cf in case.get("counterfactuals", []):
        if cf.get("type") not in CF_TYPES:
            raise DatasetError(f"{where}: unknown counterfactual type {cf.get('type')!r}")
        if cf.get("target_evidence_id") not in evidence_ids:
            raise DatasetError(f"{where}: counterfactual targets unknown evidence {cf.get('target_evidence_id')!r}")
        if cf["type"] in ("negate", "swap_number") and not cf.get("perturbed_text"):
            raise DatasetError(f"{where}: {cf['type']} counterfactual needs perturbed_text")
        if cf["type"] in ("negate", "swap_number") and not cf.get("perturbed_key_phrases"):
            raise DatasetError(f"{where}: {cf['type']} counterfactual needs perturbed_key_phrases")
        if not (0 <= cf.get("affected_fact", -1) < len(expected.get("facts", []))):
            raise DatasetError(f"{where}: counterfactual affected_fact out of range")
    for sp in case.get("swap_probes", []):
        if sp.get("expected") != "moved":
            raise DatasetError(f"{where}: swap probe expected must be 'moved'")
        if sp.get("target_evidence_id") not in evidence_ids:
            raise DatasetError(f"{where}: swap probe targets unknown evidence {sp.get('target_evidence_id')!r}")
        if not sp.get("swapped_text") or not sp["swapped_text"].strip():
            raise DatasetError(f"{where}: swap probe needs non-empty swapped_text")
        if not sp.get("swapped_key_phrases") or not all(
                isinstance(p, str) and p.strip() for p in sp["swapped_key_phrases"]):
            raise DatasetError(f"{where}: swap probe needs non-empty swapped_key_phrases")
        if not (0 <= sp.get("affected_fact", -1) < len(expected.get("facts", []))):
            raise DatasetError(f"{where}: swap probe affected_fact out of range")
        if expected.get("should_abstain"):
            raise DatasetError(f"{where}: swap probes only make sense on answerable cases")


def apply_counterfactual(case: dict, cf: dict) -> dict:
    """Return a copy of ``case`` with one counterfactual perturbation applied.

    - ``remove``: the target evidence passage is deleted entirely.
    - ``negate``: the target passage text is replaced by ``perturbed_text``
      (a negated version of its key finding).
    - ``swap_number``: the target passage text is replaced by
      ``perturbed_text`` (same finding, a different number).
    """
    perturbed = copy.deepcopy(case)
    target = cf["target_evidence_id"]
    if cf["type"] == "remove":
        perturbed["evidence"] = [ev for ev in perturbed["evidence"] if ev["id"] != target]
    else:
        for ev in perturbed["evidence"]:
            if ev["id"] == target:
                ev["text"] = cf["perturbed_text"]
    return perturbed


def apply_swap_probe(case: dict, probe: dict) -> dict:
    """Return a copy of ``case`` with one citation-swap probe applied.

    The target evidence passage is replaced wholesale by the
    contradicting guideline (``swapped_text``). Everything else —
    including the corroborating passage that still states the original
    recommendation — is left untouched, so the agent faces a genuine
    conflict between two passages, not a single changed fact.
    """
    swapped = copy.deepcopy(case)
    target = probe["target_evidence_id"]
    for ev in swapped["evidence"]:
        if ev["id"] == target:
            ev["text"] = probe["swapped_text"]
    return swapped


def dataset_summary(cases: list[dict]) -> dict:
    """Counts per category / difficulty, for the README and the dashboard."""
    by_category: dict[str, int] = {c: 0 for c in CATEGORIES}
    by_difficulty: dict[str, int] = {d: 0 for d in DIFFICULTIES}
    n_facts = 0
    n_cfs = 0
    n_sps = 0
    for case in cases:
        by_category[case["category"]] += 1
        by_difficulty[case.get("difficulty", "medium")] += 1
        n_facts += len(case["expected"].get("facts", []))
        n_cfs += len(case.get("counterfactuals", []))
        n_sps += len(case.get("swap_probes", []))
    return {
        "cases": len(cases),
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "facts": n_facts,
        "counterfactuals": n_cfs,
        "swap_probes": n_sps,
        "abstain_cases": sum(1 for c in cases if c["expected"].get("should_abstain")),
    }
