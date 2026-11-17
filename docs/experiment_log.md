# Experiment Log

## Experiment 01 — bench validation (2026-11-17)

**Question:** does the bench rank the three agents demo > naive >
confident, and does each metric catch the failure it is meant to catch?

**Result:** demo 1.000, naive 0.970, confident 0.896 on bench_score
(the original 40-case run; re-measured at 48 cases in Experiment 04
as demo 1.000, naive 0.975, confident 0.905 — the 8 added swap cases
are answerable, so they dilute the baselines' abstain/trap failures
slightly). Naive fails exactly the 8 abstain cases and all 8
citation-trap cases. Confident fails citation precision on every case
(its invented "94%" statistic is caught by the numbers check).
Faithfulness is 1.000 for all three agents — extractive agents are
trivially faithful to whatever text they copy, so faithfulness alone
cannot separate them; the bench's discrimination comes from citation
precision, abstention accuracy, and distractor avoidance.

**Bugs found by this experiment (all fixed, with regression tests):**

1. Citations were appended *after* the sentence period ("claim. [S1]"),
   so the audit attached `[S1]` to the *next* sentence. Fixed by
   placing the citation before the period ("claim [S1].").
2. The citation audit read the digit inside "[S1]" as a number the
   claim states, so every cited claim failed the numbers check. Fixed
   by stripping citation markers before auditing.
3. Fact presence was checked against the whole answer, letting tokens
   from two sentences combine into a "fact" (xh-037: "below" from one
   sentence + "30" from another falsely matched "below 30"). Fixed by
   checking per sentence.
4. The demo agent's relative-BM25 floor dropped second facts, and
   greedy set-cover over question tokens dropped second facts while
   keeping distractors. Fixed with passage-trust selection (a passage
   is trusted when its best sentence covers >= 40% of the question;
   all supported sentences of trusted passages are used).
5. The stemmer was inconsistent: "causes" stemmed to "caus" while
   "cause" stayed "cause". Fixed by stripping a trailing "e" after
   suffix removal.
6. The disclaimer was two sentences, so its second half was scored as
   an uncited claim and dragged grounding rate down. Fixed by making
   it one sentence.

**Dataset fixes from the alignment checker:** xh-007 (metformin
evidence now echoes "kidney disease" and "unsafe"), xh-012
(sunscreen evidence now echoes "stop the body from making"), xh-017
(distractor passage rewritten to share almost no vocabulary with the
question).

## Experiment 02 — abstention threshold sweep (2026-11-17)

**Question:** which evidence-strength threshold should the demo agent
abstain at?

**Result:** abstain cases all score 0.0 and the weakest answerable
case scores 0.577, so every threshold in [0.30, 0.55] gives 40/40
abstention accuracy. Shipped default 0.50, inside that plateau.
Copying the healthcare-research-agent's 0.62 would wrongly abstain on
3 answerable cases — thresholds are corpus-specific and must be
swept, not copied.

## Experiment 03 — counterfactual sensitivity (2026-11-17)

**Question:** which kind of evidence change do agents notice — remove,
negate, or swap_number?

**Result:** all three agents pass all 10 perturbations (1.000 on
every type). Expected: all three agents are extractive, so a changed
passage changes their answer by construction. The counterfactual
metric exists to catch an agent that answers from memorised knowledge
or a fixed template instead of the evidence; the protocol works, and
the discrimination lives in the other metrics.

## Experiment 04 — citation-swap probe (2026-11-17)

**Question:** when a supporting passage is replaced by a
contradicting guideline, does the agent follow the flipped source —
or copy both passages and assert both directions at once?

**Result:** demo 1.000 (8/8 probes moved), naive 0.000, confident
0.000 (both stated BOTH directions on all 8 probes). The probe is the
only metric in the bench that separates the agents on its own:
faithfulness is 1.000 for all three, because extractive agents copy
whatever evidence they are given. The demo moves because its
passage-trust rule drops the corroborating passage (coverage below
0.40) and follows the flipped guideline; the extractive agents copy
the flipped passage and the corroboration side by side.

**Dataset curation behind the result (enforced by the builder, not
assumed):** for every probe the target passage and the swapped text
sit at or above 0.40 question coverage, every corroborating sentence
stating the original fact sits below 0.40, the affected fact is
absent from the swapped text (checked lexically — three cases needed
rewording before they passed: xh-044's S1, xh-045's swapped text, and
xh-047's S1 all sat below 0.40 until their sentences were made to
echo the question more closely), and at least one corroborating
sentence exists (otherwise the probe degenerates into a `negate`
counterfactual). Two smaller gaps were fixed in the same pass: the
threshold-sweep and sensitivity notebooks wrote CSVs with CRLF line
endings (the csv module's default), which differed from the committed
LF files — both now set `lineterminator="\n"` — and the dashboard's
`use_container_width=True` was replaced with `width="stretch"` after
Streamlit deprecated it.
