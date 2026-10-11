import pytest

from qa.tests.conftest import make_email

pytestmark = pytest.mark.covers("R6")


def test_one_email_opens_exactly_one_case(inbox):
    result = inbox.receive(make_email())

    assert result.status == "opened"
    assert list(inbox.pipeline.cases) == [result.case_id]
    assert inbox.pipeline.cases[result.case_id].requested_amount == 12000


def test_a_resent_email_does_not_open_a_second_case(inbox):
    first = inbox.receive(make_email())
    again = inbox.receive(make_email())

    assert again.status == "duplicate"
    assert again.case_id == first.case_id
    assert len(inbox.pipeline.cases) == 1


def test_a_late_document_sent_as_a_reply_joins_the_existing_case(inbox):
    first = inbox.receive(make_email())
    reply = inbox.receive(
        make_email(
            message_id="<m2@example.test>",
            in_reply_to="<m1@example.test>",
            attachments=(("credit_report.txt", "text/plain", b"Total monthly payments: 250"),),
        )
    )

    assert reply.status == "attached"
    assert reply.case_id == first.case_id
    names = [d.name for d in inbox.pipeline.cases[first.case_id].documents]
    assert names == ["payslip.txt", "credit_report.txt"]


def test_an_unreadable_attachment_is_a_visible_problem_not_a_silent_drop(inbox):
    result = inbox.receive(make_email(attachments=(("scan.pdf", "application/pdf", b"%PDF-1.7 binary"),)))

    assert result.status == "needs_attention"
    case = inbox.pipeline.cases[result.case_id]
    assert case.status == "needs_attention"
    assert "scan.pdf" in case.problem


def test_an_email_without_attachments_still_becomes_a_case_someone_will_see(inbox):
    result = inbox.receive(make_email(attachments=()))

    assert result.status == "needs_attention"
    assert inbox.pipeline.cases[result.case_id].problem == "no attachments"


def test_an_email_that_cannot_be_traced_is_rejected_and_kept_on_record(inbox):
    result = inbox.receive(make_email(message_id=""))

    assert result.status == "rejected"
    assert inbox.rejected == [result]
    assert inbox.pipeline.cases == {}
