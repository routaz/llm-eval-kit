from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

from evalkit.adapter import StopEvaluation
from evalkit.config import load_config
from evalkit.context import ContextError, load_cases, load_risks
from evalkit.coverage import build_coverage, collect_test_covers
from evalkit.evaluate import evaluate
from evalkit.history import compare, latest_baseline, save, summarise
from evalkit.leakage import find_leaks, read_rows
from evalkit.mutate import mutate
from evalkit.oversight import answer_key, build_plan, queue_rows, read_decisions, score_decisions
from evalkit.report import (
    coverage_text,
    drift_text,
    eval_json,
    eval_markdown,
    mutation_text,
    oversight_plan_text,
    oversight_score_text,
    write_eval,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit", description="Evaluate an AI-assisted decision pipeline.")
    parser.add_argument("--root", type=Path, default=None, help="folder holding evalkit.toml (default: cwd)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="validate the risk register and golden cases")
    eval_parser = commands.add_parser("eval", help="run the golden dataset through the real pipeline")
    eval_parser.add_argument("--runs", type=int, help="runs per case (default from evalkit.toml)")
    eval_parser.add_argument("--cases", help="comma-separated case ids, to keep a paid run small")
    eval_parser.add_argument(
        "--adapter", help="'module:factory' to evaluate instead of the configured adapter"
    )
    eval_parser.add_argument("--save", action="store_true", help="keep this run as the baseline for `drift`")
    eval_parser.add_argument("--label", help="a note saved with the baseline, e.g. 'prompt v2'")
    commands.add_parser("drift", help="compare the last eval report with the saved baseline for its adapter")
    oversight_parser = commands.add_parser("oversight", help="test whether reviewers catch known AI errors")
    oversight_commands = oversight_parser.add_subparsers(dest="oversight_command", required=True)
    plan_parser = oversight_commands.add_parser(
        "plan", help="build a review queue seeded with known-wrong proposals"
    )
    plan_parser.add_argument("--out", type=Path, help="folder for queue.json and answer_key.json")
    plan_parser.add_argument("--seed", type=int, default=0, help="shuffles the queue order")
    plan_parser.add_argument("--adapter", help="'module:factory' to use instead of the configured adapter")
    score_parser = oversight_commands.add_parser("score", help="score the team's decisions on a seeded queue")
    score_parser.add_argument("--key", type=Path, help="answer_key.json from `oversight plan`")
    score_parser.add_argument("--decisions", type=Path, required=True, help="CSV with item_id,decision")
    coverage_parser = commands.add_parser("coverage", help="map risks to golden cases, tests and faults")
    coverage_parser.add_argument(
        "--strict", action="store_true", help="fail on unknown ids and uncovered risks"
    )
    mutate_parser = commands.add_parser(
        "mutate", help="seed faults and check the evaluation notices each one"
    )
    mutate_parser.add_argument("--adapter", help="'module:factory' to use instead of the configured adapter")
    leakcheck_parser = commands.add_parser(
        "leakcheck", help="fail if any golden case appears in training data"
    )
    leakcheck_parser.add_argument("files", nargs="+", type=Path, help="training JSONL files")
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
        adapter_ref = args.adapter or config.adapter
        report = evaluate(
            config.load_adapter(adapter_ref), cases, args.runs or config.runs, config.thresholds
        )
        md, _ = write_eval(report, config.report_dir, adapter_ref)
        print(eval_markdown(report))
        print(f"Report: {md}")
        if args.save:
            if not config.history:
                print("Cannot save: set [paths] history in evalkit.toml")
                return 1
            save(config.history, summarise(eval_json(report, adapter_ref), args.label))
            print(f"Saved as the baseline for {adapter_ref} in {config.history.name}")
        return 0 if report.passed else 1

    if args.command == "drift":
        current_path = config.report_dir / "eval.json"
        if not current_path.exists():
            print("No eval report yet: run `evalkit eval` first")
            return 1
        current = summarise(json.loads(current_path.read_text(encoding="utf-8")))
        baseline = latest_baseline(config.history, current["adapter"]) if config.history else None
        if baseline is None:
            print(f"No baseline for {current['adapter']}: save one with `evalkit eval --save`")
            return 1
        drift = compare(baseline, current, config.thresholds.drift_tolerance)
        print(drift_text(drift, baseline))
        return 1 if drift.worse else 0

    if args.command == "oversight":
        return _oversight(args, config, cases)

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


def _oversight(args, config, cases) -> int:
    folder = config.report_dir / "oversight"
    if args.oversight_command == "plan":
        out = args.out or folder
        make_adapter = partial(config.load_adapter, args.adapter)
        plan = build_plan(make_adapter, cases, config.load_faults(), args.seed)
        out.mkdir(parents=True, exist_ok=True)
        (out / "queue.json").write_text(json.dumps(queue_rows(plan, cases), indent=2), encoding="utf-8")
        (out / "answer_key.json").write_text(json.dumps(answer_key(plan), indent=2), encoding="utf-8")
        print(oversight_plan_text(plan))
        print(f"\nqueue.json goes to the review system; answer_key.json stays with QA. Both in {out}")
        return 0

    key = json.loads((args.key or folder / "answer_key.json").read_text(encoding="utf-8"))
    try:
        decisions = read_decisions(args.decisions, key)
    except ValueError as err:
        print(f"Decisions file problems:\n{err}")
        return 1
    score = score_decisions(key, decisions)
    print(oversight_score_text(score))
    floor = config.thresholds.min_catch_rate
    if floor is not None and score.catch_rate is not None and score.catch_rate < floor:
        print(f"\nCatch rate below the agreed {floor:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
