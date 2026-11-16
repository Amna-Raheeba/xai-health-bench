# Changelog

## 0.1.0 — 2026-11-17

- Initial release.
- 40-case synthetic dataset (14 faithful, 8 citation-trap, 8 abstain,
  10 counterfactual) with a builder script and alignment checks.
- Metrics: counterfactual faithfulness, citation precision (word
  overlap + numbers), fact coverage, grounding rate, abstention
  accuracy, distractor avoidance, composite bench_score.
- Three agents: demo (selective answering, passage-trust selection,
  threshold 0.50 swept on this dataset), naive (extractive baseline),
  confident (synthetic hallucinating agent for validation).
- Eval-platform plug-in: `AgentAdapter` + dataset export.
- Streamlit dashboard, three experiment notebooks, full docs.
- Measured: demo 1.000, naive 0.970, confident 0.896.
