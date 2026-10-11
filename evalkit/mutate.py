from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from evalkit.adapter import PipelineAdapter
from evalkit.config import Thresholds
from evalkit.context import GoldenCase
from evalkit.evaluate import EvalReport, evaluate
from evalkit.faults import Fault


@dataclass(frozen=True)
class MutationResult:
    fault: Fault
    status: str  # caught | survived | uncovered
    case_ids: tuple[str, ...]
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class MutationRun:
    baseline: EvalReport
    results: tuple[MutationResult, ...]

    @property
    def passed(self) -> bool:
        return self.baseline.passed and all(r.status == "caught" for r in self.results)


def mutate(
    make_adapter: Callable[[], PipelineAdapter],
    cases: list[GoldenCase],
    faults: list[Fault],
    runs: int,
    thresholds: Thresholds,
) -> MutationRun:
    """Prove the evaluation can fail: each fault must break the thresholds on the cases covering its risk.

    The clean system has to pass first, or no result counts. Each fault gets a fresh
    adapter, so run counters and caches never leak from one fault into the next.
    """
    baseline = evaluate(make_adapter(), cases, runs, thresholds)
    if not baseline.passed:
        return MutationRun(baseline, ())
    return MutationRun(
        baseline, tuple(_check_fault(make_adapter, cases, f, runs, thresholds) for f in faults)
    )


def _check_fault(make_adapter, cases, fault: Fault, runs: int, thresholds: Thresholds) -> MutationResult:
    selected = _cases_for(fault, cases)
    if not selected:
        return MutationResult(fault, "uncovered", ())
    report = evaluate(fault.wrap(make_adapter()), selected, runs, thresholds)
    ids = tuple(c.id for c in selected)
    if report.passed:
        return MutationResult(fault, "survived", ids)
    return MutationResult(fault, "caught", ids, tuple(report.failures))


def _cases_for(fault: Fault, cases: list[GoldenCase]) -> list[GoldenCase]:
    selected = [c for c in cases if set(c.covers) & set(fault.covers)]
    # A counterfactual pair is only meaningful with both halves present.
    ids = {c.id for c in selected}
    partners = [c for c in cases if c.id not in ids and any(s.counterfactual_of == c.id for s in selected)]
    return selected + partners
