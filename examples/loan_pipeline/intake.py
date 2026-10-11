from __future__ import annotations

import email
import re
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.policy import default
from email.utils import parseaddr

from examples.loan_pipeline.documents import Doc
from examples.loan_pipeline.pipeline import LoanPipeline

READABLE = ("text/plain",)


@dataclass(frozen=True)
class IntakeResult:
    """What happened to one email. Every email ends in exactly one of these, never in silence."""

    status: str  # opened | attached | duplicate | needs_attention | rejected
    case_id: str | None
    problem: str | None = None


@dataclass
class Inbox:
    pipeline: LoanPipeline
    rejected: list[IntakeResult] = field(default_factory=list)
    _cases_by_message: dict[str, str] = field(default_factory=dict)

    def receive(self, raw: bytes) -> IntakeResult:
        msg: EmailMessage = email.message_from_bytes(raw, policy=default)
        message_id = (msg["Message-ID"] or "").strip()
        sender_name, sender_address = parseaddr(msg["From"] or "")
        if not message_id or not sender_address:
            result = IntakeResult("rejected", None, "missing Message-ID or From header")
            self.rejected.append(result)
            return result

        if message_id in self._cases_by_message:
            return IntakeResult("duplicate", self._cases_by_message[message_id])

        readable, unreadable = _attachments(msg)
        reply_to = (msg["In-Reply-To"] or "").strip()
        if reply_to in self._cases_by_message:
            case = self.pipeline.add_documents(self._cases_by_message[reply_to], readable)
            self._cases_by_message[message_id] = case.id
            return self._check(case.id, readable, unreadable, "attached")

        case = self.pipeline.open_case(
            {"name": sender_name, "email": sender_address}, readable, _requested_amount(msg)
        )
        self._cases_by_message[message_id] = case.id
        return self._check(case.id, readable, unreadable, "opened")

    def _check(self, case_id: str, readable: list[Doc], unreadable: list[str], status: str) -> IntakeResult:
        if unreadable:
            problem = f"unreadable attachments: {', '.join(unreadable)}"
        elif not self.pipeline.cases[case_id].documents:
            problem = "no attachments"
        else:
            return IntakeResult(status, case_id)
        self.pipeline.needs_attention(case_id, problem)
        return IntakeResult("needs_attention", case_id, problem)


def _attachments(msg: EmailMessage) -> tuple[list[Doc], list[str]]:
    readable: list[Doc] = []
    unreadable: list[str] = []
    for part in msg.iter_attachments():
        name = part.get_filename() or "unnamed"
        if part.get_content_type() not in READABLE:
            unreadable.append(name)
            continue
        try:
            readable.append(Doc(name, part.get_content()))
        except (LookupError, UnicodeDecodeError):
            unreadable.append(name)
    return readable, unreadable


def _requested_amount(msg: EmailMessage) -> float:
    body = msg.get_body(preferencelist=("plain",))
    text = body.get_content() if body else ""
    match = re.search(r"requested amount\s*:\s*([0-9][0-9 ,.]*)", text, re.I)
    return float(match.group(1).replace(" ", "").replace(",", "").rstrip(".")) if match else 0.0
