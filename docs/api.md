# API

## 1. CLI

```bash
python -m xai_health_bench validate-dataset
python -m xai_health_bench run --agent demo
python -m xai_health_bench run --agent demo --agent naive --agent confident
python -m xai_health_bench run --agent demo --out results/demo_report.json
python -m xai_health_bench demo --case xh-001
python -m xai_health_bench demo --case xh-015 --agent demo --agent naive
```

`run` prints one line per agent with the aggregate metrics and, with
`--out`, writes a report JSON per agent plus `comparison.md` when more
than one agent ran.

## 2. Python API

```python
from xai_health_bench.dataset import load_dataset
from xai_health_bench.scorer import run_bench
from xai_health_bench import agents, metrics

cases = load_dataset()
report = run_bench("demo", cases)
print(report["aggregate"]["bench_score"])  # 1.0

# Score any agent that follows the interface:
#   run(case) -> {"answer": str, "abstained": bool, ...}
result = agents.run("demo", cases[0])
score = metrics.score_case(cases[0], result)
```

## 3. Scoring your own agent

Implement a function `my_agent(case) -> dict` returning at least
`{"answer": str, "abstained": bool}` and pass it to
`metrics.score_case` / `metrics.score_counterfactual`:

```python
from xai_health_bench import metrics
from xai_health_bench.dataset import load_dataset

for case in load_dataset():
    result = my_agent(case)
    print(case["id"], metrics.score_case(case, result))
```

For counterfactual scoring, pass the same function to
`metrics.score_counterfactual(case, cf, my_agent)` — it re-runs your
agent on the perturbed evidence and checks whether the answer changed
as expected.

For swap-probe scoring, pass it to
`metrics.score_swap_probe(case, probe, my_agent)` — it re-runs your
agent on the case with the target passage replaced by a contradicting
guideline and checks whether the recommendation moved (original
absent AND contradicting guideline stated, or abstained).

## 4. Eval-platform plug-in

```bash
python scripts/export_eval_platform_dataset.py
# copy datasets/xai_health_eval_platform.jsonl to the eval platform repo
#   as datasets/xai_health.jsonl
# copy xai_health_bench/adapters/eval_platform.py to the eval platform repo
#   as eval_platform/adapters/xai_health.py
# register XaiHealthBenchAdapter in eval_platform.adapter.default_registry
```

## 5. Dashboard

```bash
pip install streamlit
streamlit run dashboard/app.py
```

The dashboard reads `results/*_report.json` if present, or runs the
demo bench live from the sidebar.
