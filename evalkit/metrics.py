from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from evalkit.context import Expected
from evalkit.types import Proposal


@dataclass(frozen=True)
class RunResult:
    """One run of one case. A pipeline exception is a visible failure, so it counts as flagged."""

    proposal: Proposal | None
    exception: str | None = None
    decision_ok: bool = False
    field_errors: tuple[str, ...] = ()
    missing_flags: tuple[str, ...] = ()

    @property
    def wrong(self) -> bool:
        return (
            self.exception is not None
            or not self.decision_ok
            or bool(self.field_errors or self.missing_flags)
        )

    @property
    def flagged(self) -> bool:
        return self.exception is not None or bool(self.proposal and self.proposal.flags)

    @property
    def unflagged_error(self) -> bool:
        return self.wrong and not self.flagged


def fields_match(expected: float | None, actual: float | None, tolerance: float) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


def score(expected: Expected, proposal: Proposal, tolerance: float) -> RunResult:
    field_errors = tuple(
        f"{name}: expected {want}, got {proposal.fields.get(name)}"
        for name, want in expected.fields.items()
        if not fields_match(want, proposal.fields.get(name), tolerance)
    )
    return RunResult(
        proposal=proposal,
        decision_ok=proposal.decision == expected.decision
        and (expected.risk_rating is None or proposal.risk_rating == expected.risk_rating),
        field_errors=field_errors,
        missing_flags=tuple(f for f in expected.must_flag if f not in proposal.flags),
    )


def signature(run: RunResult) -> tuple:
    """What must stay the same between runs of one case for it to count as consistent."""
    if run.proposal is None:
        return ("exception",)
    p = run.proposal
    fields = tuple(sorted((k, None if v is None else round(v, 2)) for k, v in p.fields.items()))
    return (p.decision, p.risk_rating, fields, tuple(sorted(p.flags)))


def is_consistent(runs: list[RunResult]) -> bool:
    return len({signature(r) for r in runs}) <= 1


def majority_decision(runs: list[RunResult]) -> str | None:
    decisions = [r.proposal.decision for r in runs if r.proposal]
    return Counter(decisions).most_common(1)[0][0] if decisions else None


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0
