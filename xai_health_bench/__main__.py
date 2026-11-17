"""CLI: run the bench, validate the dataset, print reports.

Usage:
    python -m xai_health_bench validate-dataset
    python -m xai_health_bench run --agent demo
    python -m xai_health_bench run --agent demo --agent naive --agent confident
    python -m xai_health_bench run --agent demo --out results/demo_report.json
    python -m xai_health_bench demo --case xh-001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import agents
from .dataset import DEFAULT_DATASET, dataset_summary, load_dataset
from .report import render_comparison, render_report
from .scorer import load_report, run_bench, save_report

ROOT = Path(__file__).resolve().parents[1]


def cmd_validate_dataset(args) -> int:
    cases = load_dataset(args.dataset)
    print(f"{args.dataset}: {len(cases)} cases, valid")
    print(json.dumps(dataset_summary(cases), indent=2))
    return 0


def cmd_run(args) -> int:
    cases = load_dataset(args.dataset)
    agent_names = args.agent or ["demo"]
    reports = []
    for agent in agent_names:
        report = run_bench(agent, cases, dataset_path=args.dataset)
        reports.append(report)
        agg = report["aggregate"]
        print(f"{agent}: bench_score={agg['bench_score']} "
              f"faithfulness={agg['faithfulness']} "
              f"citation_precision={agg['citation_precision']} "
              f"fact_coverage={agg['fact_coverage']} "
              f"abstention_accuracy={agg['abstention_accuracy']} "
              f"grounding={agg['grounding_rate']} "
              f"distractor_avoidance={agg['distractor_avoidance']} "
              f"swap_sensitivity={agg.get('swap_sensitivity')}")
        if args.out:
            out = Path(args.out)
            if len(agent_names) > 1:
                out = out.with_name(f"{out.stem}_{agent}{out.suffix}")
            save_report(report, out)
            print(f"  wrote {out}")
    if len(reports) > 1:
        print()
        print(render_comparison(reports))
        if args.out:
            comp_path = Path(args.out).with_name("comparison.md")
            comp_path.write_text(render_comparison(reports), encoding="utf-8")
            print(f"wrote {comp_path}")
    return 0


def cmd_demo(args) -> int:
    cases = {c["id"]: c for c in load_dataset(args.dataset)}
    case = cases.get(args.case)
    if case is None:
        print(f"no case {args.case!r}; known: {sorted(cases)}", file=sys.stderr)
        return 2
    print(f"Q: {case['input']['question']}\n")
    for agent in args.agent or ["demo"]:
        result = agents.run(agent, case)
        print(f"--- {agent} (status={result['status']}, "
              f"abstained={result['abstained']}) ---")
        print(result["answer"])
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xai_health_bench")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-dataset")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.set_defaults(func=cmd_validate_dataset)

    p = sub.add_parser("run")
    p.add_argument("--agent", action="append", default=None,
                   choices=agents.agent_names())
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("demo")
    p.add_argument("--case", default="xh-001")
    p.add_argument("--agent", action="append", default=None,
                   choices=agents.agent_names())
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
