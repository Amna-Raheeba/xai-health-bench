"""Markdown report rendering for a bench run (or a comparison)."""
from __future__ import annotations


def _pct(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def render_report(report: dict) -> str:
    agg = report["aggregate"]
    lines = [
        f"# XAI-Health Bench — {report['agent']}",
        "",
        f"Cases: {report['cases']} · counterfactual perturbations: "
        f"{agg['counterfactuals']} · elapsed: {report['elapsed_s']}s",
        "",
        "## Aggregate",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| bench_score (composite) | {_pct(agg['bench_score'])} |",
        f"| faithfulness (counterfactual) | {_pct(agg['faithfulness'])} |",
        f"| citation_precision | {_pct(agg['citation_precision'])} |",
        f"| fact_coverage | {_pct(agg['fact_coverage'])} |",
        f"| grounding_rate | {_pct(agg['grounding_rate'])} |",
        f"| abstention_accuracy | {_pct(agg['abstention_accuracy'])} |",
        f"| distractor_avoidance | {_pct(agg['distractor_avoidance'])} |",
        "",
        "Weights: " + ", ".join(f"{k}={v}" for k, v in agg["weights"].items()),
        "",
        "## By category",
        "",
        "| Category | Cases | Answered | Fact coverage | Citation precision | Abstention acc. |",
        "|---|---|---|---|---|---|",
    ]
    for category, row in sorted(report["by_category"].items()):
        lines.append(
            f"| {category} | {row['cases']} | {row['answered']} | "
            f"{_pct(row['fact_coverage'])} | {_pct(row['citation_precision'])} | "
            f"{_pct(row['abstention_accuracy'])} |")
    lines += ["", "## Counterfactual failures", ""]
    failures = [c for c in report["counterfactual_scores"] if not c["passed"]]
    if not failures:
        lines.append("None — every perturbation changed the answer as expected.")
    else:
        for c in failures:
            lines.append(f"- {c['case_id']} / {c['cf_id']} ({c['cf_type']}): {c['detail']}")
    lines += ["", "## Citation failures", ""]
    failed_cites = []
    for s in report["case_scores"]:
        for f in s.get("failed_citations") or []:
            failed_cites.append((s["case_id"], f))
    if not failed_cites:
        lines.append("None — every cited claim was supported by its source.")
    else:
        for case_id, f in failed_cites:
            lines.append(f"- {case_id} [{f['citation']}]: {f['reason']} — "
                         f"\"{f['sentence'][:120]}\"")
    lines.append("")
    return "\n".join(lines)


def render_comparison(reports: list[dict]) -> str:
    """One table comparing several agents' aggregate metrics."""
    metrics = ["bench_score", "faithfulness", "citation_precision",
               "fact_coverage", "grounding_rate", "abstention_accuracy",
               "distractor_avoidance"]
    header = "| Metric | " + " | ".join(r["agent"] for r in reports) + " |"
    lines = ["# XAI-Health Bench — agent comparison", "", header,
             "|" + "---|" * (len(reports) + 1)]
    for metric in metrics:
        row = [metric] + [_pct(r["aggregate"][metric]) for r in reports]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)
