from __future__ import annotations

import json
from pathlib import Path

from evalkit.coverage import CoverageReport
from evalkit.evaluate import EvalReport
from evalkit.history import Drift, git_commit
from evalkit.mutate import MutationRun
from evalkit.oversight import OversightPlan, OversightScore


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} s"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


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
        f"| Latency, mean / max | {_seconds(m['mean_latency_s'])} / {_seconds(m['max_latency_s'])} |",
        f"| Model cost, total (as recorded, when replaying) | {_usd(m['total_cost_usd'])} |",
        "",
    ]
    if report.failures:
        lines += ["## Thresholds broken", "", *[f"- {f}" for f in report.failures], ""]
    lines += [
        "## Cases",
        "",
        "| Case | Covers | Result | Latency | Detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for res in report.results:
        lines.append(
            f"| {res.case.id} | {', '.join(res.case.covers)} | {_case_status(res)} "
            f"| {_seconds(_mean([r.latency_s for r in res.runs]))} | {_case_detail(res)} |"
        )
    return "\n".join(lines) + "\n"


def _mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


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


def eval_json(report: EvalReport, adapter: str | None = None) -> dict:
    return {
        "passed": report.passed,
        "adapter": adapter,
        "models": sorted(
            {r.proposal.model for res in report.results for r in res.runs if r.proposal and r.proposal.model}
        ),
        "git_commit": git_commit(),
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
                        "latency_s": r.latency_s,
                        "cost_usd": r.cost_usd,
                        "model": r.proposal.model if r.proposal else None,
                    }
                    for r in res.runs
                ],
            }
            for res in report.results
        ],
    }


def write_eval(report: EvalReport, folder: Path, adapter: str | None = None) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    md, js = folder / "eval.md", folder / "eval.json"
    md.write_text(eval_markdown(report), encoding="utf-8")
    js.write_text(json.dumps(eval_json(report, adapter), indent=2), encoding="utf-8")
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


def drift_text(drift: Drift, baseline: dict) -> str:
    when = baseline["saved_at"] + (f" ({baseline['label']})" if baseline.get("label") else "")
    lines = [f"Compared with the baseline saved {when}", ""]
    if drift.models_changed:
        before, after = (", ".join(m) or "unknown" for m in drift.models_changed)
        lines += [f"The model changed: {before} -> {after}", ""]
    lines.append(f"{'Metric':<24}{'Baseline':>12}{'Now':>12}")
    for name, old, new, worse in drift.metrics:
        fmt = _usd if name == "total_cost_usd" else _seconds if name.endswith("_s") else _pct
        lines.append(f"{name:<24}{fmt(old):>12}{fmt(new):>12}{'  WORSE' if worse else ''}")
    if not drift.comparable:
        lines += ["", "The runs covered different cases, so the metrics are shown but not judged."]
    if drift.regressed:
        lines += ["", "Passed before, failing now:", *[f"  {c}" for c in drift.regressed]]
    if drift.fixed:
        lines += ["", "Failing before, passing now:", *[f"  {c}" for c in drift.fixed]]
    if drift.not_compared:
        lines += ["", f"In only one of the two runs: {', '.join(drift.not_compared)}"]
    lines += ["", "WORSE than the baseline" if drift.worse else "No drift"]
    return "\n".join(lines)


def oversight_plan_text(plan: OversightPlan) -> str:
    seeded = [i for i in plan.items if i.kind == "seeded"]
    lines = [
        f"{len(plan.items)} queue items: {len(seeded)} seeded with a known error, "
        f"{len(plan.items) - len(seeded)} correct controls"
    ]
    for item in seeded:
        lines.append(
            f"  {item.item_id}  {item.case_id}: {item.fault_id} "
            f"proposes {item.proposal.decision}, correct is {item.correct_decision}"
        )
    if plan.unused_faults:
        lines.append(
            f"No decision-changing seed from: {', '.join(plan.unused_faults)} "
            "(a single proposal's decision doesn't change, so reviewers can't show they caught it)"
        )
    return "\n".join(lines)


def oversight_score_text(score: OversightScore) -> str:
    lines = [
        f"{score.reviews} reviews, team level",
        "",
        f"{'Catch rate':<22}{_pct(score.catch_rate):>8}   known errors not accepted as proposed",
        f"{'Wrong-approval rate':<22}{_pct(score.wrong_approval_rate):>8}   wrong approvals approved anyway",
        f"{'False-alarm rate':<22}{_pct(score.false_alarm_rate):>8}   correct proposals overridden",
    ]
    if score.by_fault:
        lines += ["", "By kind of error (feeds reviewer training):"]
        lines += [f"  {fault:<44}{caught}/{seen} caught" for fault, (caught, seen) in score.by_fault.items()]
    if score.unreviewed:
        lines += ["", f"Not reviewed yet: {', '.join(score.unreviewed)}"]
    return "\n".join(lines)
