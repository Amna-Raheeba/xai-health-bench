# Metrics

All metrics are deterministic and lexical. Scores are in [0, 1];
higher is better on every metric.

## 1. Per-case metrics

### fact_coverage

Fraction of expected answer facts present in the answer. A fact is
present when **one sentence** of the answer contains all of the
fact's `key_phrases` (stemmed-token subset check). Per-sentence, not
whole-answer: checking the whole answer would let tokens from two
different sentences combine into a "fact" no single sentence states
(caught by the bench's validation run on xh-037).

### citation_precision

Every cited claim — an answer sentence carrying `[Sn]` — is audited
against the passage it cites:

- the claim's content-word overlap with the passage must be >= 0.32;
- every number in the claim must appear verbatim in the passage.

The citation marker itself is stripped before auditing: `[S1]`
contains the digit "1", which the numbers check would otherwise read
as a number the claim states (caught by the validation run — every
cited claim failed before this fix). The boilerplate disclaimer
sentence is excluded from the audit.

### grounding_rate

Fraction of answer sentences (excluding the disclaimer) that carry a
citation.

### distractor_avoidance

`1 - (distractor facts stated / distractor facts)`. Distractor facts
are plausible claims from passages irrelevant to the question; a
careful agent never states them.

### abstention_correct

1.0 when the agent abstained on a should-abstain case, or answered a
should-answer case; 0.0 otherwise.

## 2. Faithfulness (counterfactual perturbation)

For each perturbation of a counterfactual case, the agent is re-run
on the perturbed evidence:

- **remove** — the passage carrying the affected fact is deleted.
  Pass when the affected fact is absent from the new answer (or the
  agent abstains, which is also a faithful response to missing
  evidence).
- **negate** — the passage's key finding is negated and its number
  removed. Pass when the old fact is absent AND the perturbed fact is
  stated.
- **swap_number** — the passage's number is changed. Pass when the
  old fact (which includes the old number) is absent AND the
  perturbed fact (with the new number) is stated.

`faithfulness` is the mean pass rate over all perturbations.

## 3. Citation-swap probe (swap sensitivity)

For each probe, the agent is re-run on the case with one supporting
passage replaced by a **contradicting guideline** — a directive
reversal ("advises against X; use Y instead"), not a negated finding.
The original recommendation survives in a corroborating passage, so
the agent faces a genuine conflict between two passages.

- **Pass** when the agent abstained (refusing to assert a superseded
  recommendation is faithful) or when the original recommendation is
  absent from the new answer AND the contradicting recommendation is
  stated.
- **Fail** otherwise — including the classic failure of stating both
  directions at once (`stated_both`), recorded in the probe result.

`swap_sensitivity` is the mean pass rate over all probes. It is
reported **separately and is NOT part of `bench_score`**: it measures
a different failure (conflict handling) from the composite's
explanation-quality mix, and the composite's published numbers must
stay comparable.

How it differs from a `negate` counterfactual: a `negate` removes or
reverses one finding and the original is gone everywhere, so any
agent that copies evidence passes. A swap probe keeps the original
alive in a second passage — an agent that copies everything passes
the counterfactual and fails the probe. Measured 2026-11-17: demo
1.000 (8/8), naive 0.000, confident 0.000.

The probe only discriminates because the dataset is curated to a
coverage split, enforced by the builder (`scripts/build_dataset.py`):
the flipped passage and the target passage sit at or above 0.40
question coverage (so the demo's passage-trust rule follows them),
while every corroborating sentence stating the original fact sits
below 0.40 (so the demo drops it). A differently curated dataset
could move that split — the probe measures conflict handling on this
corpus, not general contradiction detection.

## 4. Composite: bench_score

```
bench_score = 0.30 * faithfulness
            + 0.25 * citation_precision
            + 0.20 * fact_coverage
            + 0.15 * abstention_accuracy
            + 0.10 * grounding_rate
```

Weights are a choice, documented here and in the README. They weight
explanation quality (faithfulness + citation precision = 0.55) above
answer completeness (fact coverage = 0.20), because this bench exists
to score explanations, not answers.

## 5. Honest limits

- **Lexical proxy.** Fact presence and citation support are token
  checks. A correct paraphrase can score 0 on fact coverage, and a
  sentence that reuses the right words with the wrong meaning can
  score 1. An LLM-based agent would score lower here than an
  extractive one for this reason — that is a limitation of the bench,
  not of the agent.
- **Small dataset.** 48 hand-written cases. Differences smaller than
  ~0.05 on any metric are noise, not signal.
- **Reference agent ceiling.** The demo agent scores 1.000 because it
  was co-designed with the dataset. Treat 1.000 as "the bench is
  solvable", and judge the bench by whether it ranks the baselines
  correctly (naive 0.975, confident 0.905).
- **Extractive agents are trivially faithful.** An agent that copies
  evidence verbatim passes every counterfactual by construction, so
  faithfulness alone cannot separate agents. The bench's
  discrimination comes from citation precision, abstention accuracy,
  and distractor avoidance together — plus swap sensitivity, the one
  metric that separates the agents on its own.
- **Swap probes are single-flip and curated.** Each probe flips
  exactly one passage with a hand-written contradicting sentence, and
  the probe discriminates only because the corroborating passage is
  curated below the 0.40 coverage line. The numbers (demo 1.000,
  naive 0.000, confident 0.000) prove the probe works on this corpus;
  they do not prove a general contradiction detector.
