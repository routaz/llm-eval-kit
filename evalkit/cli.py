from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from evalkit.adapter import StopEvaluation
from evalkit.config import load_config
from evalkit.context import ContextError, load_cases, load_risks
from evalkit.coverage import build_coverage, collect_test_covers
from evalkit.evaluate import evaluate
from evalkit.leakage import find_leaks, read_rows
from evalkit.mutate import mutate
from evalkit.report import coverage_text, eval_markdown, mutation_text, write_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit", description="Evaluate an AI-assisted decision pipeline.")
    parser.add_argument("--root", type=Path, default=None, help="folder holding evalkit.toml (default: cwd)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate the risk register and golden cases")
    ev = sub.add_parser("eval", help="run the golden dataset through the real pipeline")
    ev.add_argument("--runs", type=int, help="runs per case (default from evalkit.toml)")
    ev.add_argument("--cases", help="comma-separated case ids, to keep a paid run small")
    ev.add_argument("--adapter", help="'module:factory' to evaluate instead of the configured adapter")
    cov = sub.add_parser("coverage", help="map risks to golden cases, tests and faults")
    cov.add_argument("--strict", action="store_true", help="fail on unknown ids and uncovered risks")
    mut = sub.add_parser("mutate", help="seed faults and check the evaluation notices each one")
    mut.add_argument("--adapter", help="'module:factory' to use instead of the configured adapter")
    leak = sub.add_parser("leakcheck", help="fail if any golden case appears in training data")
    leak.add_argument("files", nargs="+", type=Path, help="training JSONL files")
    args = parser.parse_args(argv)

    config = load_config(args.root)
    try:
        risks = load_risks(config.risks)
        cases = load_cases(config.cases, {r.id for r in risks})
    except ContextError as err:
        print("Context problems:", *err.problems, sep="\n  ")
        return 1

    try:
        return _run(args, config, risks, cases)
    except StopEvaluation as stop:
        print(f"Evaluation stopped: {stop}")
        return 2


def _run(args, config, risks, cases) -> int:
    if args.command == "check":
        print(f"OK: {len(risks)} risks, {len(cases)} golden cases")
        return 0

    if args.command == "eval":
        if args.cases:
            wanted = args.cases.split(",")
            unknown = sorted(set(wanted) - {c.id for c in cases})
            if unknown:
                print(f"Unknown case ids: {', '.join(unknown)}")
                return 1
            cases = [c for c in cases if c.id in wanted]
        report = evaluate(
            config.load_adapter(args.adapter), cases, args.runs or config.runs, config.thresholds
        )
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
        make_adapter = partial(config.load_adapter, args.adapter)
        run = mutate(make_adapter, cases, config.load_faults(), config.runs, config.thresholds)
        print(mutation_text(run))
        return 0 if run.passed else 1

    if args.command == "leakcheck":
        rows = [row for path in args.files for row in read_rows(path)]
        leaks = find_leaks(cases, rows)
        if leaks:
            print(f"{len(leaks)} golden documents found in training data:", *leaks, sep="\n  ")
            return 1
        print(f"OK: no golden document in {len(rows)} training rows")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
