# Sample Session

Real output from `python -m xai_health_bench demo` (2026-11-17).

## A faithful case (xh-001)

**Q:** When should adults be screened for type 2 diabetes, and which
tests are used?

**demo:**
> Adults aged 35 to 70 years who have overweight or obesity should be
> screened for type 2 diabetes [S1]. Screening can be performed with
> HbA1c, fasting plasma glucose, or a 75 g oral glucose tolerance test
> [S1].

**naive:** same two sentences, plus the S2 sentences about screening
intervals (correct but unasked-for).

## A citation-trap case (xh-015)

**Q:** Are beta-blockers safe in asthma?

**demo:**
> Beta-blockers can worsen bronchospasm in asthma and are generally
> avoided in people with asthma [S1].

**naive:**
> Beta-blockers can worsen bronchospasm in asthma and are generally
> avoided in people with asthma [S1]. Beta-blockers improve survival
> in heart failure with reduced ejection fraction, and they are a
> cornerstone of heart failure treatment [S2]. Reliever inhalers open
> the airways quickly during an asthma attack, while preventer
> inhalers reduce inflammation over time [S3].

The naive agent states the distractor fact ("improve survival in
heart failure") — the bench's distractor-avoidance metric catches
exactly this.

## An abstain case (xh-023)

**Q:** What is the recommended dose of lasmiditan for adolescents with
migraine?

**demo:**
> I'm going to abstain on this one: the evidence does not contain
> enough verified material to answer this reliably.

**naive:** composes an answer from the hypertension and diabetes
passages — fluent, cited, and entirely irrelevant. The bench's
abstention-accuracy metric catches exactly this.

## A counterfactual case (xh-031, perturbed)

**Q:** What HbA1c target is recommended for most adults with type 2
diabetes?

**demo on the original evidence:**
> Most adults with type 2 diabetes should aim for an HbA1c below 7.0%
> [S1].

**demo after the target is swapped to 8.0%:**
> Most adults with type 2 diabetes should aim for an HbA1c below 8.0%
> [S1].

The answer changed with the evidence — faithfulness 1.0.
