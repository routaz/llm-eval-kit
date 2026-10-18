from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

Decision = Literal["approve", "decline", "refer"]
RiskRating = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class Document:
    name: str
    text: str


@dataclass(frozen=True)
class Application:
    """One case as the evaluation sends it to the system under test."""

    case_id: str
    documents: tuple[Document, ...]
    applicant: dict[str, str] = field(default_factory=dict)
    requested_amount: float = 0.0


@dataclass(frozen=True)
class Proposal:
    """What the system under test proposes to the human reviewer.

    `flags` are the warnings shown to the reviewer. Any flag counts as the system
    admitting uncertainty, which is what separates a flagged error from an unflagged one.

    `cost_usd` and `latency_s` are optional. Set them when the model reports its own cost
    or when replaying recorded calls; otherwise the kit measures latency itself.

    `model` is optional too: the exact model that produced the proposal. It is saved with
    every report and baseline, so a result can be traced to the model behind it.
    """

    decision: Decision
    risk_rating: RiskRating
    fields: dict[str, float | None] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    cost_usd: float | None = None
    latency_s: float | None = None
    model: str | None = None

    def with_changes(self, **changes) -> Proposal:
        return replace(self, **changes)
