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
    """

    decision: Decision
    risk_rating: RiskRating
    fields: dict[str, float | None] = field(default_factory=dict)
    flags: tuple[str, ...] = ()

    def with_changes(self, **changes) -> Proposal:
        return replace(self, **changes)
