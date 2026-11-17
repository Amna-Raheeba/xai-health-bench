# Datasets

## `xai_health.jsonl` — the bench (48 cases)

One JSON object per line. Schema and validation rules are in
`xai_health_bench/dataset.py`; the human-readable source of truth is
`scripts/build_dataset.py` — edit cases there and re-run it, never
edit the JSONL by hand.

| Category | Cases | What it tests |
|---|---|---|
| faithful | 14 | Normal answerable questions; the agent must state the expected facts with citations |
| citation-trap | 8 | A plausible but irrelevant passage is present; the agent must never state its facts |
| abstain | 8 | Evidence is topically distant; the correct behaviour is to abstain |
| counterfactual | 10 | Evidence is perturbed (remove / negate / swap_number); the answer must change accordingly |
| swap | 8 | Supporting passage replaced by a contradicting guideline; the recommendation must move |

63 expected facts, 10 counterfactual perturbations, 8 swap probes,
8 should-abstain cases. Difficulty: 10 easy, 24 medium, 14 hard.

## Synthetic-data rule

**Every evidence passage is synthetic and hand-written for
benchmarking.** Titles all start with "Synthetic summary". Numbers
are invented. Nothing here is a real study, real medical guidance, or
real patient data, and nothing in this repo is medical advice. The
dataset loader rejects any evidence title that does not start with
"Synthetic summary".

## Alignment rule

The demo agent trusts a passage when its best sentence covers at
least 40% of the question's content tokens, so the dataset is curated
to that rule and the builder checks it: every expected fact lives in a
passage at or above 0.40 coverage, every distractor fact lives in a
passage below 0.40, and abstain cases have no passage at or above
0.40. Swap probes add a second rule: the flipped passage and the
swapped text sit at or above 0.40, every corroborating sentence
stating the original fact sits below 0.40, the affected fact is
absent from the swapped text, and at least one corroborating sentence
exists. Run `python scripts/build_dataset.py` — it fails loudly if
any case violates the rule.

## `xai_health_eval_platform.jsonl` — eval-platform export

Same 48 cases in ai-evaluation-platform format, produced by
`scripts/export_eval_platform_dataset.py`. Copy it into the eval
platform repo as `datasets/xai_health.jsonl` (see `docs/api.md`).
