# XAI-Health Bench — demo

Cases: 48 · counterfactual perturbations: 10 · elapsed: 0.214s

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
| swap_sensitivity (separate probe, not in bench_score) | 1.000 |

Weights: faithfulness=0.3, citation_precision=0.25, fact_coverage=0.2, abstention_accuracy=0.15, grounding_rate=0.1

## By category

| Category | Cases | Answered | Fact coverage | Citation precision | Abstention acc. |
|---|---|---|---|---|---|
| abstain | 8 | 0 | n/a | n/a | 1.000 |
| citation-trap | 8 | 8 | 1.000 | 1.000 | 1.000 |
| counterfactual | 10 | 10 | 1.000 | 1.000 | 1.000 |
| faithful | 14 | 14 | 1.000 | 1.000 | 1.000 |
| swap | 8 | 8 | 1.000 | 1.000 | 1.000 |

## Counterfactual failures

None — every perturbation changed the answer as expected.

## Swap-probe failures

None — every recommendation moved when its source was flipped.

## Citation failures

None — every cited claim was supported by its source.
