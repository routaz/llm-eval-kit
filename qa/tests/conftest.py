from __future__ import annotations

import json
from email.message import EmailMessage

import pytest

from examples.loan_pipeline.intake import Inbox
from examples.loan_pipeline.pipeline import LoanPipeline


class ScriptedAI:
    """Answers with prepared responses in order, and records every prompt it was sent."""

    def __init__(self, *responses: str | Exception):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response


def answer(net=3200, debts=250, stated=3200, flags=()) -> str:
    return json.dumps(
        {
            "monthly_net_income": net,
            "monthly_debt_payments": debts,
            "stated_monthly_income": stated,
            "flags": list(flags),
        }
    )


def make_email(
    message_id: str = "<m1@example.test>",
    sender: str = "Anna Virtanen <anna@example.test>",
    body: str = "Requested amount: 12000",
    attachments: tuple[tuple[str, str, bytes], ...] = (("payslip.txt", "text/plain", b"Net pay: 3,200.00"),),
    in_reply_to: str | None = None,
) -> bytes:
    msg = EmailMessage()
    msg["Message-ID"] = message_id
    msg["From"] = sender
    msg["To"] = "loans@example.test"
    msg["Subject"] = "Loan application"
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    msg.set_content(body)
    for name, ctype, data in attachments:
        maintype, subtype = ctype.split("/")
        if maintype == "text":
            msg.add_attachment(data.decode(), subtype=subtype, filename=name)
        else:
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)
    return msg.as_bytes()


@pytest.fixture
def ai() -> ScriptedAI:
    return ScriptedAI(answer())


@pytest.fixture
def pipeline(ai) -> LoanPipeline:
    return LoanPipeline(ai)


@pytest.fixture
def inbox(pipeline) -> Inbox:
    return Inbox(pipeline)
