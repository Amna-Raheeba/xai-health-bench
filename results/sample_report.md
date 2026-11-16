# XAI-Health Bench — demo

Cases: 40 · counterfactual perturbations: 10 · elapsed: 0.034s

## Aggregate

| Metric | Score |
|---|---|
| bench_score (composite) | 1.000 |
| faithfulness (counterfactual) | 1.000 |
| citation_precision | 1.000 |
| fact_coverage | 1.000 |
| grounding_rate | 1.000 |
| abstention_accuracy | 1.000 |
| distractor_avoidance | 1.000 |

Weights: faithfulness=0.3, citation_precision=0.25, fact_coverage=0.2, abstention_accuracy=0.15, grounding_rate=0.1

## By category

| Category | Cases | Answered | Fact coverage | Citation precision | Abstention acc. |
|---|---|---|---|---|---|
| abstain | 8 | 0 | n/a | n/a | 1.000 |
| citation-trap | 8 | 8 | 1.000 | 1.000 | 1.000 |
| counterfactual | 10 | 10 | 1.000 | 1.000 | 1.000 |
| faithful | 14 | 14 | 1.000 | 1.000 | 1.000 |

## Counterfactual failures

None — every perturbation changed the answer as expected.

## Citation failures

None — every cited claim was supported by its source.

# XAI-Health Bench — agent comparison

| Metric | demo | naive | confident |
|---|---|---|---|
| bench_score | 1.000 | 0.970 | 0.896 |
| faithfulness | 1.000 | 1.000 | 1.000 |
| citation_precision | 1.000 | 1.000 | 0.720 |
| fact_coverage | 1.000 | 1.000 | 1.000 |
| grounding_rate | 1.000 | 1.000 | 0.963 |
| abstention_accuracy | 1.000 | 0.800 | 0.800 |
| distractor_avoidance | 1.000 | 0.800 | 0.800 |
