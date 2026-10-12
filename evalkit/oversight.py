from __future__ import annotations

import csv
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evalkit.adapter import PipelineAdapter
from evalkit.context import GoldenCase
from evalkit.faults import Fault
from evalkit.types import Proposal

DECISIONS = ("approve", "decline", "refer")


@dataclass(frozen=True)
class QueueItem:
    """One case to put in front of reviewers, and the answer key that stays with QA."""

    item_id: str
    kind: str  # seeded: the proposal is known to be wrong | control: the proposal is correct
    case_id: str
    fault_id: str | None
    proposal: Proposal
    correct_decision: str


@dataclass
class OversightPlan:
    items: list[QueueItem]
    unused_faults: list[str] = field(default_factory=list)


def build_plan(
    make_adapter: Callable[[], PipelineAdapter],
    cases: list[GoldenCase],
    faults: list[Fault],
    seed: int = 0,
) -> OversightPlan:
    """Known-wrong proposals to seed into a review queue, mixed with correct ones as controls.

    A seed is a fault applied to a golden case that it covers, kept only when it changes
    the decision: a reviewer can only show they caught an error by deciding differently.
    Controls are the clean system's correct proposals; without them, a reviewer who
    overrides everything would score a perfect catch rate.
    """
    seeded: list[tuple[GoldenCase, Fault, Proposal]] = []
    unused = []
    for fault in faults:
        found = False
        for case in cases:
            if not set(case.covers) & set(fault.covers):
                continue
            proposal = fault.wrap(make_adapter()).propose(case.application)
            if proposal.decision != case.expected.decision:
                seeded.append((case, fault, proposal))
                found = True
        if not found:
            unused.append(fault.id)

    adapter = make_adapter()
    controls = [
        (case, proposal)
        for case in cases
        if (proposal := adapter.propose(case.application)).decision == case.expected.decision
    ]

    drafts = [("seeded", c, f.id, p) for c, f, p in seeded] + [("control", c, None, p) for c, p in controls]
    random.Random(seed).shuffle(drafts)
    items = [
        QueueItem(f"item-{n:03d}", kind, case.id, fault_id, proposal, case.expected.decision)
        for n, (kind, case, fault_id, proposal) in enumerate(drafts, 1)
    ]
    return OversightPlan(items, unused)


def queue_rows(plan: OversightPlan, cases: list[GoldenCase]) -> list[dict]:
    """What the review system needs to show each item. Nothing here says which items are seeded."""
    by_id = {c.id: c for c in cases}
    rows = []
    for item in plan.items:
        app = by_id[item.case_id].application
        rows.append(
            {
                "item_id": item.item_id,
                "requested_amount": app.requested_amount,
                "applicant": app.applicant,
                "documents": [asdict(d) for d in app.documents],
                "proposal": {
                    "decision": item.proposal.decision,
                    "risk_rating": item.proposal.risk_rating,
                    "fields": item.proposal.fields,
                    "flags": list(item.proposal.flags),
                },
            }
        )
    return rows


def answer_key(plan: OversightPlan) -> list[dict]:
    return [
        {
            "item_id": i.item_id,
            "kind": i.kind,
            "case_id": i.case_id,
            "fault_id": i.fault_id,
            "proposed_decision": i.proposal.decision,
            "correct_decision": i.correct_decision,
        }
        for i in plan.items
    ]


@dataclass
class OversightScore:
    reviews: int
    unreviewed: list[str]
    catch_rate: float | None
    wrong_approval_rate: float | None
    false_alarm_rate: float | None
    by_fault: dict[str, tuple[int, int]]  # fault id -> (caught, seen)


def read_decisions(path: Path, key: list[dict]) -> list[tuple[str, str]]:
    """Rows of `item_id,decision`. Several rows for one item are several reviews of it.

    There is deliberately no reviewer column: results are for the team, never per person.
    """
    known = {k["item_id"] for k in key}
    rows, problems = [], []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["item_id", "decision"]:
            raise ValueError(f"{path.name}: expected the header 'item_id,decision', got {reader.fieldnames}")
        for line, row in enumerate(reader, 2):
            item, decision = row["item_id"].strip(), row["decision"].strip().lower()
            if item not in known:
                problems.append(f"line {line}: unknown item {item!r}")
            elif decision not in DECISIONS:
                problems.append(
                    f"line {line}: decision must be one of {', '.join(DECISIONS)}, got {decision!r}"
                )
            else:
                rows.append((item, decision))
    if problems:
        raise ValueError("\n".join(problems))
    return rows


def score_decisions(key: list[dict], decisions: list[tuple[str, str]]) -> OversightScore:
    """Caught means the reviewer did not accept a known-wrong proposal as it stood."""
    by_id = {k["item_id"]: k for k in key}
    seeded = [(by_id[i], d) for i, d in decisions if by_id[i]["kind"] == "seeded"]
    controls = [(by_id[i], d) for i, d in decisions if by_id[i]["kind"] == "control"]
    approvals = [(k, d) for k, d in seeded if k["proposed_decision"] == "approve"]

    by_fault: dict[str, tuple[int, int]] = {}
    for k, d in seeded:
        caught, seen = by_fault.get(k["fault_id"], (0, 0))
        by_fault[k["fault_id"]] = (caught + (d != k["proposed_decision"]), seen + 1)

    reviewed = {i for i, _ in decisions}
    return OversightScore(
        reviews=len(decisions),
        unreviewed=[k["item_id"] for k in key if k["item_id"] not in reviewed],
        catch_rate=_share(sum(d != k["proposed_decision"] for k, d in seeded), len(seeded)),
        wrong_approval_rate=_share(sum(d == "approve" for _, d in approvals), len(approvals)),
        false_alarm_rate=_share(sum(d != k["proposed_decision"] for k, d in controls), len(controls)),
        by_fault=dict(sorted(by_fault.items())),
    )


def _share(count: int, total: int) -> float | None:
    return count / total if total else None
