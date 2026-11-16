"""XAI-Health Bench — explanation-faithfulness benchmark for healthcare agents.

Scores a healthcare agent's natural-language explanation on three
things accuracy benchmarks ignore:

1. **Faithfulness** — does the explanation change when the cited
   evidence is removed, negated, or re-numbered? (counterfactual
   perturbation)
2. **Citation grounding** — does every cited claim actually appear in
   the source it cites, numbers included? (citation audit)
3. **Abstention calibration** — does the agent abstain when the
   evidence is insufficient, and answer when it is not?

All evidence is synthetic and hand-written for benchmarking. Nothing
here is medical advice and nothing is real patient data.
"""

__version__ = "0.1.0"
