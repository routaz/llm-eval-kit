from __future__ import annotations

from dataclasses import dataclass, field

from evalkit.adapter import PipelineAdapter
from evalkit.config import Thresholds
from evalkit.context import GoldenCase
from evalkit.metrics import RunResult, is_consistent, majority_decision, rate, score


@dataclass
class CaseResult:
    case: GoldenCase
    runs: list[RunResult]

    @property
    def wrong(self) -> bool:
        return any(r.wrong for r in self.runs)

    @property
    def unflagged_error(self) -> bool:
        return any(r.unflagged_error for r in self.runs)

    @property
    def consistent(self) -> bool:
        return is_consistent(self.runs)


@dataclass
class EvalReport:
    results: list[CaseResult]
    runs_per_case: int
    thresholds: Thresholds
    metrics: dict[str, float | None] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def evaluate(
    adapter: PipelineAdapter, cases: list[GoldenCase], runs: int, thresholds: Thresholds
) -> EvalReport:
    """Run every golden case `runs` times through the adapter and check the agreed thresholds."""
    results = [
        CaseResult(case, [_run_once(adapter, case, thresholds) for _ in range(runs)]) for case in cases
    ]
    report = EvalReport(results, runs, thresholds)
    report.metrics = _metrics(results)
    report.failures = _check(report.metrics, thresholds)
    return report


def _run_once(adapter: PipelineAdapter, case: GoldenCase, thresholds: Thresholds) -> RunResult:
    try:
        proposal = adapter.propose(case.application)
    except Exception as exc:  # the pipeline failing on a case is a result, not a crash
        return RunResult(proposal=None, exception=f"{type(exc).__name__}: {exc}")
    return score(case.expected, proposal, thresholds.field_tolerance)


def _metrics(results: list[CaseResult]) -> dict[str, float | None]:
    all_runs = [r for res in results for r in res.runs]
    by_id = {res.case.id: res for res in results}
    pairs = [(res, by_id[res.case.counterfactual_of]) for res in results if res.case.counterfactual_of]
    agreeing = sum(majority_decision(a.runs) == majority_decision(b.runs) for a, b in pairs)
    return {
        "error_rate": rate(sum(r.wrong for r in all_runs), len(all_runs)),
        "unflagged_error_rate": rate(sum(r.unflagged_error for r in all_runs), len(all_runs)),
        "consistency": rate(sum(res.consistent for res in results), len(results)),
        "fairness_agreement": rate(agreeing, len(pairs)) if pairs else None,
    }


def _check(metrics: dict[str, float | None], t: Thresholds) -> list[str]:
    checks = [
        ("error_rate", metrics["error_rate"] <= t.max_error_rate, f"<= {t.max_error_rate:.0%}"),
        (
            "unflagged_error_rate",
            metrics["unflagged_error_rate"] <= t.max_unflagged_error_rate,
            f"<= {t.max_unflagged_error_rate:.0%}",
        ),
        ("consistency", metrics["consistency"] >= t.min_consistency, f">= {t.min_consistency:.0%}"),
    ]
    if metrics["fairness_agreement"] is not None:
        checks.append(
            (
                "fairness_agreement",
                metrics["fairness_agreement"] >= t.min_fairness_agreement,
                f">= {t.min_fairness_agreement:.0%}",
            )
        )
    return [f"{name} {metrics[name]:.0%}, threshold {limit}" for name, ok, limit in checks if not ok]
