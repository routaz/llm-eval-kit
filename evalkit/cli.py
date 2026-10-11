from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evalkit.config import load_config
from evalkit.context import ContextError, load_cases, load_risks
from evalkit.coverage import build_coverage, collect_test_covers
from evalkit.evaluate import evaluate
from evalkit.mutate import mutate
from evalkit.report import coverage_text, eval_markdown, mutation_text, write_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit", description="Evaluate an AI-assisted decision pipeline.")
    parser.add_argument("--root", type=Path, default=None, help="folder holding evalkit.toml (default: cwd)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate the risk register and golden cases")
    ev = sub.add_parser("eval", help="run the golden dataset through the real pipeline")
    ev.add_argument("--runs", type=int, help="runs per case (default from evalkit.toml)")
    cov = sub.add_parser("coverage", help="map risks to golden cases, tests and faults")
    cov.add_argument("--strict", action="store_true", help="fail on unknown ids and uncovered risks")
    sub.add_parser("mutate", help="seed faults and check the evaluation notices each one")
    args = parser.parse_args(argv)

    config = load_config(args.root)
    try:
        risks = load_risks(config.risks)
        cases = load_cases(config.cases, {r.id for r in risks})
    except ContextError as err:
        print("Context problems:", *err.problems, sep="\n  ")
        return 1

    if args.command == "check":
        print(f"OK: {len(risks)} risks, {len(cases)} golden cases")
        return 0

    if args.command == "eval":
        report = evaluate(config.load_adapter(), cases, args.runs or config.runs, config.thresholds)
        md, _ = write_eval(report, config.report_dir)
        print(eval_markdown(report))
        print(f"Report: {md}")
        return 0 if report.passed else 1

    if args.command == "coverage":
        report = build_coverage(risks, cases, collect_test_covers(config.tests), config.load_faults())
        print(coverage_text(report))
        problems = report.problems(args.strict)
        if problems:
            print("\nProblems:", *problems, sep="\n  ")
        return 1 if problems else 0

    if args.command == "mutate":
        run = mutate(config.load_adapter, cases, config.load_faults(), config.runs, config.thresholds)
        print(mutation_text(run))
        return 0 if run.passed else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
