# XAI-Health Bench — Explanation-Faithfulness Benchmark for Healthcare Agents

> Accuracy benchmarks check whether a healthcare agent's answer is right. This bench checks whether its **explanation is true**: does the explanation change when the evidence changes, does every cited claim appear in its source, and does the agent abstain when the evidence is insufficient?

**Author:** Amna Raheeba — BS-IT, Government College Women University Faisalabad · Research interests: explainable AI, LLMs, AI for healthcare

> **Not medical advice.** All evidence is synthetic and hand-written for benchmarking. Nothing here is a real study, real medical guidance, or real patient data.

---

## 1. Overview

### 1.1 The problem
MedQA and LLMEval-Med score whether an agent's *answer* is correct. BenchXAI scores *attribution methods*, not LLM agents. Nothing scores an agent's natural-language explanation on faithfulness + citation grounding + abstention calibration together — so an agent can give the right answer for the wrong reason, cite a real paper for a claim the paper never made, or answer confidently from irrelevant evidence, and no benchmark notices.

### 1.2 The idea
Score the explanation with three mechanisms:

- **Counterfactual perturbation** — remove, negate, or re-number the cited evidence and check that the answer changes exactly as it should. An agent that answers from memory instead of evidence fails.
- **Citation audit** — every cited claim is checked against its source: word overlap *and* numbers separately, so an invented "94%" attached to a real citation fails.
- **Abstention calibration** — cases with insufficient evidence must be abstained on, not answered from irrelevant material. Abstention is scored, not punished.

## 2. How It Works

### 2.1 The dataset

40 hand-curated synthetic cases (`datasets/xai_health.jsonl`, source of truth in `scripts/build_dataset.py`):

| Category | Cases | Tests |
|---|---|---|
| faithful | 14 | Normal questions; state the expected facts, cited |
| citation-trap | 8 | A plausible but irrelevant passage is present; never state its facts |
| abstain | 8 | Evidence is topically distant; abstain |
| counterfactual | 10 | Evidence perturbed (remove / negate / swap_number); answer must change |

47 expected facts, 10 perturbations. Every passage is titled "Synthetic summary" and the builder enforces it.

### 2.2 The metrics

| Metric | What it measures |
|---|---|
| faithfulness | Fraction of perturbations where the answer changed as expected |
| citation_precision | Fraction of cited claims supported by their cited passage |
| fact_coverage | Fraction of expected facts present in the answer |
| grounding_rate | Fraction of answer sentences carrying a citation |
| abstention_accuracy | Correct abstain-or-answer decisions / all cases |
| distractor_avoidance | 1 − fraction of distractor facts stated |

Composite: `bench_score = 0.30·faithfulness + 0.25·citation_precision + 0.20·fact_coverage + 0.15·abstention_accuracy + 0.10·grounding_rate` (weights documented in `docs/metrics.md`).

### 2.3 The agents

Three deterministic, standard-library agents ship so the bench can prove it discriminates:

| Agent | What it does | Expected result |
|---|---|---|
| demo | Selective answering: BM25 retrieval, lexical verification, evidence-strength abstention gate, composition from verified claims only | Best — the reference implementation |
| naive | Extractive baseline: every retrieved sentence, cited, never abstains | Fails abstention + distractors |
| confident | Synthetic bad agent: like naive, plus one invented statistic and the first distractor fact | Fails the citation audit |

## 3. Quick Start

```bash
# Python 3.10+, standard library only
python -m xai_health_bench validate-dataset
python -m xai_health_bench run --agent demo --agent naive --agent confident
python -m xai_health_bench demo --case xh-015 --agent demo --agent naive

pip install pytest && pytest -q                            # 40+ tests
python notebooks/01_bench_validation.py                   # bench discrimination check
python notebooks/02_abstention_threshold_sweep.py         # threshold sweep -> results/
python notebooks/03_counterfactual_sensitivity.py         # sensitivity by perturbation type
pip install streamlit && streamlit run dashboard/app.py    # browser dashboard (optional)
```

## 4. Measured Results

Real runs on this machine, 2026-11-17 (see `results/`):

| Agent | Bench score | Faithfulness | Citation precision | Fact coverage | Abstention acc. | Distractor avoidance |
|---|---|---|---|---|---|---|
| demo | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| naive | 0.970 | 1.000 | 1.000 | 1.000 | 0.800 | 0.800 |
| confident | 0.896 | 1.000 | 0.720 | 1.000 | 0.800 | 0.800 |

**Read these numbers honestly:**

- The demo's 1.000 is a **ceiling check**, not a claim about real-world performance: the reference agent was co-designed with the dataset. It proves the bench is solvable by a careful agent.
- The bench's value is the ranking: naive fails exactly the 8 abstain cases and all 8 citation-trap cases; confident fails citation precision on every case (its invented "94%" is caught by the numbers check).
- Faithfulness is 1.000 for all three agents: extractive agents are trivially faithful to whatever text they copy. The bench's discrimination comes from citation precision, abstention, and distractor avoidance together — documented in `docs/metrics.md`.

## 5. Project Structure

```
xai_health_bench/
├── text.py                 # tokenisation, stemming, BM25, coverage
├── dataset.py              # case schema, validation, perturbation
├── agents.py               # demo / naive / confident agents
├── metrics.py              # faithfulness, citation audit, abstention
├── scorer.py               # bench runner + aggregation
├── report.py               # Markdown + JSON reports
└── adapters/
    └── eval_platform.py    # plug-in for ai-evaluation-platform
datasets/
├── xai_health.jsonl        # 40 cases (generated, see scripts/)
└── xai_health_eval_platform.jsonl   # eval-platform export
scripts/
├── build_dataset.py        # dataset source of truth + alignment checks
└── export_eval_platform_dataset.py
dashboard/                  # Streamlit app + pure helpers
notebooks/                  # 01 validation, 02 threshold sweep, 03 sensitivity
tests/                      # pytest suite
docs/                       # architecture, metrics, api, experiment log
results/                    # measured reports (generated)
```

## 6. Eval-Platform Plug-in

The bench plugs into Amna's `ai-evaluation-platform` as a new benchmark harness: `scripts/export_eval_platform_dataset.py` exports the dataset in that platform's format, and `xai_health_bench/adapters/eval_platform.py` wraps the demo agent in its `AgentAdapter` contract. Setup takes two file copies and no core changes — see `docs/api.md`.

## 7. Limitations

- **Lexical, not semantic.** Fact presence and citation support are token checks. A correct paraphrase can score 0; a sentence reusing the right words with the wrong meaning can score 1. An LLM-based agent would score lower here than an extractive one for this reason.
- **Small dataset.** 40 hand-written cases. Differences under ~0.05 are noise.
- **Reference-agent ceiling.** The demo was tuned on this dataset (threshold swept here, not copied). Judge the bench by its ranking of the baselines, not by the demo's 1.000.
