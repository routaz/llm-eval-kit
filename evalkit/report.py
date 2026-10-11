from __future__ import annotations

import json
from pathlib import Path

from evalkit.coverage import CoverageReport
from evalkit.evaluate import EvalReport
from evalkit.mutate import MutationRun


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def eval_markdown(report: EvalReport) -> str:
    m = report.metrics
    lines = [
        f"# Evaluation: {'PASS' if report.passed else 'FAIL'}",
        "",
        f"{len(report.results)} cases, {report.runs_per_case} runs each",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Error rate | {_pct(m['error_rate'])} |",
        f"| Unflagged error rate | {_pct(m['unflagged_error_rate'])} |",
        f"| Consistency | {_pct(m['consistency'])} |",
        f"| Fairness agreement | {_pct(m['fairness_agreement'])} |",
        "",
    ]
    if report.failures:
        lines += ["## Thresholds broken", "", *[f"- {f}" for f in report.failures], ""]
    lines += ["## Cases", "", "| Case | Covers | Result | Detail |", "| --- | --- | --- | --- |"]
    for res in report.results:
        lines.append(
            f"| {res.case.id} | {', '.join(res.case.covers)} | {_case_status(res)} | {_case_detail(res)} |"
        )
    return "\n".join(lines) + "\n"


def _case_status(res) -> str:
    if res.unflagged_error:
        return "UNFLAGGED ERROR"
    if res.wrong:
        return "flagged error"
    return "ok" if res.consistent else "inconsistent"


def _case_detail(res) -> str:
    details: list[str] = []
    for i, run in enumerate(res.runs, 1):
        if run.exception:
            details.append(f"run {i}: {run.exception}")
            continue
        if not run.decision_ok:
            details.append(f"run {i}: decision {run.proposal.decision}/{run.proposal.risk_rating}")
        details += [f"run {i}: {e}" for e in run.field_errors]
        details += [f"run {i}: missing flag {f}" for f in run.missing_flags]
    return "; ".join(dict.fromkeys(details)) or ""


def eval_json(report: EvalReport) -> dict:
    return {
        "passed": report.passed,
        "runs_per_case": report.runs_per_case,
        "metrics": report.metrics,
        "failures": report.failures,
        "cases": [
            {
                "id": res.case.id,
                "covers": list(res.case.covers),
                "status": _case_status(res),
                "runs": [
                    {
                        "decision": r.proposal.decision if r.proposal else None,
                        "risk_rating": r.proposal.risk_rating if r.proposal else None,
                        "flags": list(r.proposal.flags) if r.proposal else [],
                        "exception": r.exception,
                        "wrong": r.wrong,
                        "unflagged_error": r.unflagged_error,
                    }
                    for r in res.runs
                ],
            }
            for res in report.results
        ],
    }


def write_eval(report: EvalReport, folder: Path) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    md, js = folder / "eval.md", folder / "eval.json"
    md.write_text(eval_markdown(report), encoding="utf-8")
    js.write_text(json.dumps(eval_json(report), indent=2), encoding="utf-8")
    return md, js


def coverage_text(report: CoverageReport) -> str:
    lines = [f"{'Risk':<6}{'Status':<10}{'Cases':>6}{'Tests':>7}{'Faults':>8}  Title"]
    for row in report.rows:
        lines.append(
            f"{row.risk.id:<6}{row.status:<10}{len(row.cases):>6}{len(row.tests):>7}{len(row.faults):>8}"
            f"  {row.risk.title}"
        )
        if row.status == "manual":
            lines.append(f"{'':<16}manual: {row.risk.manual}")
    if report.untagged_tests:
        lines += ["", f"{len(report.untagged_tests)} tests carry no covers marker:"]
        lines += [f"  {t}" for t in report.untagged_tests]
    return "\n".join(lines)


def mutation_text(run: MutationRun) -> str:
    if not run.baseline.passed:
        return "Baseline evaluation failed, so no fault result counts:\n" + "\n".join(
            f"  {f}" for f in run.baseline.failures
        )
    lines = []
    for r in run.results:
        lines.append(
            f"{r.status.upper():<10}{r.fault.id} ({', '.join(r.fault.covers)}): {r.fault.description}"
        )
        if r.status == "caught":
            lines += [f"{'':<10}{f}" for f in r.failures]
        elif r.status == "survived":
            lines.append(f"{'':<10}cases {', '.join(r.case_ids)} did not notice. Strengthen them.")
        else:
            lines.append(f"{'':<10}no golden case covers {', '.join(r.fault.covers)}")
    caught = sum(r.status == "caught" for r in run.results)
    lines += ["", f"{caught}/{len(run.results)} faults caught"]
    return "\n".join(lines)
