# Changelog

## 0.2.0 — 2026-11-17

- New: citation-swap probe — 8 `swap` cases where a supporting
  passage is replaced by a contradicting guideline and the
  recommendation must move; scored as `swap_sensitivity`, reported
  separately (not part of `bench_score`). Measured: demo 1.000 (8/8
  moved), naive 0.000, confident 0.000 — both baselines state BOTH
  directions on every probe. New notebook `04_swap_probe.py`, new
  dashboard section, builder alignment checks enforcing the coverage
  split the probe depends on.
- Fixed: threshold-sweep and sensitivity notebooks wrote CSVs with
  CRLF line endings (csv module default); both now use
  `lineterminator="\n"` so regenerated files match the committed LF
  files. Dashboard `use_container_width=True` replaced with
  `width="stretch"` (Streamlit deprecated the former).
- Dataset grew 40 → 48 cases (63 facts); measured bench scores are
  now demo 1.000, naive 0.975, confident 0.905.

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
