from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

from examples.loan_pipeline.ai import AIClient, build_prompt
from examples.loan_pipeline.documents import Doc
from examples.loan_pipeline.parsing import AIOutputError, Extraction, parse_extraction

TERM_MONTHS = 60
INCOME_MISMATCH = 0.10

log = logging.getLogger("loan_pipeline")


@dataclass(frozen=True)
class Proposal:
    decision: str
    risk_rating: str
    debt_to_income: float | None
    extraction: Extraction
    flags: tuple[str, ...]


@dataclass
class Case:
    id: str
    applicant: dict[str, str]
    requested_amount: float
    documents: list[Doc] = field(default_factory=list)
    status: str = "received"  # received | proposed | needs_attention | decided
    problem: str | None = None
    proposal: Proposal | None = None
    decision: dict | None = None


class LoanPipeline:
    def __init__(self, ai: AIClient, retries: int = 2):
        self.ai = ai
        self.retries = retries
        self.cases: dict[str, Case] = {}
        self.audit: list[dict] = []
        self._ids = itertools.count(1)

    def open_case(self, applicant: dict[str, str], documents: list[Doc], requested_amount: float) -> Case:
        case = Case(f"case-{next(self._ids):04d}", dict(applicant), requested_amount, list(documents))
        self.cases[case.id] = case
        self._record(case, "opened", documents=len(documents))
        return case

    def add_documents(self, case_id: str, documents: list[Doc]) -> Case:
        case = self.cases[case_id]
        case.documents.extend(documents)
        self._record(case, "documents_added", documents=len(documents))
        return case

    def needs_attention(self, case_id: str, problem: str) -> Case:
        case = self.cases[case_id]
        case.status, case.problem = "needs_attention", problem
        self._record(case, "needs_attention", problem=problem)
        return case

    def analyse(self, case_id: str) -> Case:
        """Ask the model for figures, then decide in code. A bad answer goes to a person, not a proposal."""
        case = self.cases[case_id]
        prompt = build_prompt(case.documents)
        try:
            extraction = parse_extraction(self._complete(prompt, case))
        except AIOutputError as err:
            return self.needs_attention(case.id, f"AI output rejected: {err}")
        except TimeoutError:
            return self.needs_attention(case.id, f"AI timed out after {self.retries + 1} attempts")
        except Exception as err:  # the model's server is down, the CLI failed, ...
            return self.needs_attention(case.id, f"AI call failed: {type(err).__name__}: {err}")
        case.proposal = propose(extraction, case.requested_amount)
        case.status, case.problem = "proposed", None
        self._record(case, "proposed", decision=case.proposal.decision, flags=list(case.proposal.flags))
        return case

    def record_decision(self, case_id: str, reviewer: str, decision: str, reason: str = "") -> Case:
        case = self.cases[case_id]
        ai_decision = case.proposal.decision if case.proposal else None
        case.decision = {
            "reviewer": reviewer,
            "decision": decision,
            "reason": reason,
            "ai_decision": ai_decision,
            "override": ai_decision is not None and decision != ai_decision,
        }
        case.status = "decided"
        self._record(case, "decided", **case.decision)
        return case

    def _complete(self, prompt: str, case: Case) -> str:
        for attempt in range(1, self.retries + 2):
            try:
                return self.ai.complete(prompt)
            except TimeoutError:
                log.warning("AI timeout on %s (attempt %d)", case.id, attempt)
                if attempt > self.retries:
                    raise
        raise AssertionError("unreachable")

    def _record(self, case: Case, event: str, **details) -> None:
        self.audit.append({"case": case.id, "event": event, **details})
        log.info("%s %s", case.id, event)


def propose(extraction: Extraction, requested_amount: float) -> Proposal:
    """Business rules live in code, so they can be tested exactly. The model only supplies inputs."""
    flags = list(extraction.flags)
    net, debts, stated = (
        extraction.monthly_net_income,
        extraction.monthly_debt_payments,
        extraction.stated_monthly_income,
    )
    if net and stated is not None and abs(stated - net) / net > INCOME_MISMATCH:
        flags.append("income_mismatch")
    if not net:
        return Proposal("refer", "high", None, extraction, tuple(flags))

    dti = debt_to_income(net, debts or 0.0, requested_amount)
    if dti > 0.5:
        decision = "decline"
    elif dti > 0.35 or flags:
        decision = "refer"
    else:
        decision = "approve"
    rating = "low" if dti < 0.25 else "medium" if dti < 0.4 else "high"
    return Proposal(decision, rating, dti, extraction, tuple(flags))


def debt_to_income(net_income: float, debt_payments: float, requested_amount: float) -> float:
    return (debt_payments + requested_amount / TERM_MONTHS) / net_income
