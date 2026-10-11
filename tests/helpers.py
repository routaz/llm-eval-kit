from __future__ import annotations

from evalkit.context import Expected, GoldenCase
from evalkit.types import Application, Document, Proposal


def case(
    case_id: str = "c1",
    decision: str = "approve",
    fields: dict | None = None,
    must_flag: tuple[str, ...] = (),
    covers: tuple[str, ...] = ("R1",),
    counterfactual_of: str | None = None,
    applicant: dict | None = None,
) -> GoldenCase:
    return GoldenCase(
        id=case_id,
        covers=covers,
        description="",
        application=Application(case_id, (Document("doc.txt", "text"),), applicant or {}),
        expected=Expected(decision, None, fields or {}, must_flag),
        counterfactual_of=counterfactual_of,
    )


class FixedAdapter:
    """Returns a prepared proposal per case id; raises for ids it doesn't know."""

    def __init__(self, proposals: dict[str, Proposal]):
        self.proposals = proposals

    def propose(self, application: Application) -> Proposal:
        return self.proposals[application.case_id]


def proposal(decision: str = "approve", fields: dict | None = None, flags: tuple[str, ...] = ()) -> Proposal:
    return Proposal(decision, "low", fields or {}, flags)
