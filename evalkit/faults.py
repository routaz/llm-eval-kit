from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from evalkit.adapter import PipelineAdapter
from evalkit.types import Application, Proposal

Transform = Callable[[Application, Proposal, int], Proposal]
"""Changes a proposal. The int is how many times this case has been proposed before (0 on the first run)."""

Predicate = Callable[[Application], bool]


@dataclass(frozen=True)
class Fault:
    """A deliberate defect seeded into the system's answers.

    The mutation check runs the cases that cover `covers` with the fault in place and
    expects the evaluation to fail. A fault the evaluation doesn't notice means the
    golden cases or thresholds are too weak for that risk.
    """

    id: str
    covers: tuple[str, ...]
    description: str
    transform: Transform

    def wrap(self, adapter: PipelineAdapter) -> PipelineAdapter:
        return _FaultyAdapter(adapter, self.transform)


class _FaultyAdapter:
    def __init__(self, adapter: PipelineAdapter, transform: Transform):
        self._adapter = adapter
        self._transform = transform
        self._calls: Counter[str] = Counter()

    def propose(self, application: Application) -> Proposal:
        proposal = self._adapter.propose(application)
        run = self._calls[application.case_id]
        self._calls[application.case_id] += 1
        return self._transform(application, proposal, run)


def drop_flags(*names: str) -> Transform:
    """Hide the named warnings from the reviewer, or all of them when no names are given."""

    def transform(_app: Application, p: Proposal, _run: int) -> Proposal:
        return p.with_changes(flags=tuple(f for f in p.flags if names and f not in names))

    return transform


def scale_field(name: str, factor: float) -> Transform:
    def transform(_app: Application, p: Proposal, _run: int) -> Proposal:
        value = p.fields.get(name)
        if value is None:
            return p
        return p.with_changes(fields={**p.fields, name: value * factor})

    return transform


def force_decision(decision: str) -> Transform:
    def transform(_app: Application, p: Proposal, _run: int) -> Proposal:
        return p.with_changes(decision=decision)

    return transform


def unstable_decision() -> Transform:
    """Change the decision on every second run of a case, like a model that isn't deterministic."""

    def transform(_app: Application, p: Proposal, run: int) -> Proposal:
        if run % 2 == 0:
            return p
        return p.with_changes(decision="refer" if p.decision != "refer" else "approve")

    return transform


def chain(*transforms: Transform) -> Transform:
    def transform(app: Application, p: Proposal, run: int) -> Proposal:
        for t in transforms:
            p = t(app, p, run)
        return p

    return transform


def when(predicate: Predicate, transform: Transform) -> Transform:
    """Apply `transform` only to applications that match `predicate`."""

    def conditional(app: Application, p: Proposal, run: int) -> Proposal:
        return transform(app, p, run) if predicate(app) else p

    return conditional


def document_contains(text: str) -> Predicate:
    needle = text.lower()
    return lambda app: any(needle in d.text.lower() for d in app.documents)


def applicant_is(field: str, value: str) -> Predicate:
    return lambda app: app.applicant.get(field) == value


def case_is(*case_ids: str) -> Predicate:
    return lambda app: app.case_id in case_ids
