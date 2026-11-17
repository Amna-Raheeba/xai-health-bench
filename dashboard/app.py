"""XAI-Health Bench dashboard.

Runs the real bench (demo agent) or loads a saved report JSON, and
shows aggregate scores, per-agent comparison, per-case detail,
counterfactual results, and category breakdowns.

Run: ``streamlit run dashboard/app.py`` (streamlit is an optional
dashboard-only dependency: ``pip install streamlit``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import helpers  # noqa: E402


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


st.set_page_config(page_title="XAI-Health Bench", layout="wide")
st.title("XAI-Health Bench")
st.caption("Explanation-faithfulness benchmark for healthcare agents — "
           "synthetic evidence only, not medical advice.")

reports = helpers.available_reports()
options = ["Run demo bench now"] + [p.name for p in reports]
choice = st.sidebar.selectbox("Report", options)

if choice == "Run demo bench now":
    with st.spinner("Running the demo agent over 48 cases…"):
        report = helpers.run_demo_bench()
else:
    report = helpers.load_report(next(p for p in reports if p.name == choice))

ov = helpers.overview(report)
st.header(f"Agent: {ov['agent'] or '—'}")
cols = st.columns(4)
for i, metric in enumerate(ov["metrics"]):
    with cols[i % 4]:
        st.metric(metric["label"], _fmt(metric["value"]))
st.write(f"Cases: {ov['cases']} · counterfactual perturbations: "
         f"{ov['counterfactuals']}")

st.header("Per-case detail")
rows = helpers.case_rows(report)
if rows:
    st.dataframe(rows, width="stretch")
else:
    st.info("No report loaded yet.")

st.header("Counterfactual perturbations")
cf_rows = helpers.counterfactual_rows(report)
if cf_rows:
    st.dataframe(cf_rows, width="stretch")
    failed = [r for r in cf_rows if not r["passed"]]
    st.write(f"Passed: {len(cf_rows) - len(failed)} / {len(cf_rows)}")
else:
    st.info("No counterfactual results in this report.")

st.header("Citation-swap probes")
sp_rows = helpers.swap_probe_rows(report)
if sp_rows:
    st.dataframe(sp_rows, width="stretch")
    passed = [r for r in sp_rows if r["passed"]]
    st.write(f"Recommendations moved: {len(passed)} / {len(sp_rows)}")
    both = [r for r in sp_rows if r["stated_both"]]
    if both:
        st.write(f"Stated BOTH directions at once: {len(both)} — "
                 "the agent copied the flipped passage and the "
                 "corroborating passage without noticing the conflict.")
else:
    st.info("No swap-probe results in this report.")

st.header("By category")
cat_rows = helpers.category_rows(report)
if cat_rows:
    st.dataframe(cat_rows, width="stretch")

st.sidebar.markdown(
    "Metrics: faithfulness (answer changes when evidence changes), "
    "citation precision (cited claims supported by their source), "
    "fact coverage, grounding, abstention accuracy, distractor "
    "avoidance. Weights in docs/metrics.md.")
