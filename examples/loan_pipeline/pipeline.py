from __future__ import annotations

import hashlib
import itertools
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from examples.loan_pipeline.ai import PROMPT_VERSION, AIClient, build_prompt
from examples.loan_pipeline.audit import explain
from examples.loan_pipeline.documents import Doc
from examples.loan_pipeline.parsing import AIOutputError, Extraction, parse_extraction

TERM_MONTHS = 60
INCOME_MISMATCH = 0.10
DECLINE_ABOVE = 0.5
REFER_ABOVE = 0.35
# Recorded with every proposal, so a decision made under old thresholds can still be explained.
RULES = {
    "term_months": TERM_MONTHS,
    "income_mismatch": INCOME_MISMATCH,
    "decline_above": DECLINE_ABOVE,
    "refer_above": REFER_ABOVE,
}

log = logging.getLogger("loan_pipeline")


@dataclass(frozen=True)
class Proposal:
    decision: str
    risk_rating: str
    debt_to_income: float | None
    extraction: Extraction
    flags: tuple[str, ...]
    rule: str = ""  # which business rule decided, in words


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
    """Holds cases and their audit trail.

    With `audit_to`, every audit event is also appended to that JSONL file, so a decision can be
    explained after the process has gone (see `examples.loan_pipeline.audit`).
    """

    def __init__(self, ai: AIClient, retries: int = 2, audit_to: Path | None = None):
        self.ai = ai
        self.retries = retries
        self.audit_to = audit_to
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

    def needs_attention(self, case_id: str, problem: str, **details) -> Case:
        case = self.cases[case_id]
        case.status, case.problem = "needs_attention", problem
        self._record(case, "needs_attention", problem=problem, **details)
        return case

    def analyse(self, case_id: str) -> Case:
        """Ask the model for figures, then decide in code. A bad answer goes to a person, not a proposal."""
        case = self.cases[case_id]
        prompt = build_prompt(case.documents)
        inputs = self._inputs(case, prompt)
        try:
            extraction = parse_extraction(self._complete(prompt, case))
        except AIOutputError as err:
            return self.needs_attention(case.id, f"AI output rejected: {err}", **inputs)
        except TimeoutError:
            return self.needs_attention(case.id, f"AI timed out after {self.retries + 1} attempts", **inputs)
        except Exception as err:  # the model's server is down, the CLI failed, ...
            return self.needs_attention(case.id, f"AI call failed: {type(err).__name__}: {err}", **inputs)
        p = case.proposal = propose(extraction, case.requested_amount)
        case.status, case.problem = "proposed", None
        self._record(
            case,
            "proposed",
            decision=p.decision,
            risk_rating=p.risk_rating,
            flags=list(p.flags),
            rule=p.rule,
            figures={
                "monthly_net_income": extraction.monthly_net_income,
                "monthly_debt_payments": extraction.monthly_debt_payments,
                "stated_monthly_income": extraction.stated_monthly_income,
                "requested_amount": case.requested_amount,
                "debt_to_income": None if p.debt_to_income is None else round(p.debt_to_income, 4),
            },
            rules=RULES,
            **inputs,
        )
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

    def explain(self, case_id: str) -> str:
        """Why this case came out the way it did, in words, from the audit trail."""
        return explain(self.audit, case_id)

    def _complete(self, prompt: str, case: Case) -> str:
        for attempt in range(1, self.retries + 2):
            try:
                return self.ai.complete(prompt)
            except TimeoutError:
                log.warning("AI timeout on %s (attempt %d)", case.id, attempt)
                if attempt > self.retries:
                    raise
        raise AssertionError("unreachable")

    def _inputs(self, case: Case, prompt: str) -> dict:
        """What the model was given: which model, which prompt, which documents.

        Documents are recorded by name and hash, never by text, so the audit holds no document
        content. `prompt_sha256` is the key recordings are stored under, so a recorded model
        answer can be found from the audit.
        """
        return {
            "model": getattr(self.ai, "model", None),
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": sha256(prompt),
            "documents": [{"name": d.name, "sha256": sha256(d.text)} for d in case.documents],
        }

    def _record(self, case: Case, event: str, **details) -> None:
        entry = {"case": case.id, "event": event, "at": datetime.now(UTC).isoformat(timespec="seconds")}
        entry.update(details)
        self.audit.append(entry)
        if self.audit_to:
            self.audit_to.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_to.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
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
        return Proposal("refer", "high", None, extraction, tuple(flags), "no net income -> refer")

    dti = debt_to_income(net, debts or 0.0, requested_amount)
    if dti > DECLINE_ABOVE:
        decision, rule = "decline", f"debt-to-income {dti:.2f} > {DECLINE_ABOVE:.2f} -> decline"
    elif dti > REFER_ABOVE:
        decision, rule = "refer", f"debt-to-income {dti:.2f} > {REFER_ABOVE:.2f} -> refer"
    elif flags:
        decision, rule = "refer", f"flags {', '.join(flags)} -> refer"
    else:
        decision, rule = "approve", f"debt-to-income {dti:.2f} <= {REFER_ABOVE:.2f}, no flags -> approve"
    rating = "low" if dti < 0.25 else "medium" if dti < 0.4 else "high"
    return Proposal(decision, rating, dti, extraction, tuple(flags), rule)


def debt_to_income(net_income: float, debt_payments: float, requested_amount: float) -> float:
    return (debt_payments + requested_amount / TERM_MONTHS) / net_income


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
