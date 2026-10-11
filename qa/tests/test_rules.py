import pytest

from examples.loan_pipeline.ai import EXTRACTION_SCHEMA, FLAG_CODES, build_prompt
from examples.loan_pipeline.parsing import Extraction
from examples.loan_pipeline.pipeline import debt_to_income, propose

pytestmark = pytest.mark.covers("R1")


def extraction(net=3000.0, debts=0.0, stated=None, flags=()):
    return Extraction(net, debts, stated, tuple(flags))


def test_debt_to_income_is_calculated_in_code_over_a_60_month_term():
    assert debt_to_income(3000, 700, 21000) == pytest.approx(0.35)


@pytest.mark.parametrize(
    ("debts", "decision", "rating"),
    [
        (100, "approve", "low"),  # 0.10
        (700, "approve", "medium"),  # 0.35 exactly: the rule is "above 35% refers"
        (710, "refer", "medium"),
        (1150, "refer", "high"),  # 0.50 exactly
        (1160, "decline", "high"),
    ],
)
def test_decision_and_rating_follow_the_thresholds(debts, decision, rating):
    proposal = propose(extraction(debts=debts), requested_amount=21000)

    assert (proposal.decision, proposal.risk_rating) == (decision, rating)


def test_any_warning_turns_an_approve_into_a_refer():
    proposal = propose(extraction(flags=["credit_report_missing"]), requested_amount=6000)

    assert proposal.decision == "refer"


def test_missing_income_never_produces_an_approve():
    proposal = propose(extraction(net=None), requested_amount=1000)

    assert (proposal.decision, proposal.risk_rating) == ("refer", "high")


def test_reviewer_notes_are_shown_but_do_not_force_a_refer():
    noted = Extraction(3000.0, 0.0, None, (), ("only one payslip provided",))

    assert propose(noted, requested_amount=6000).decision == "approve"


@pytest.mark.covers("R2")
def test_the_prompt_names_every_flag_code_the_rules_rely_on():
    prompt = build_prompt([])

    assert all(code in prompt for code in FLAG_CODES)
    assert EXTRACTION_SCHEMA["properties"]["flags"]["items"]["enum"] == list(FLAG_CODES)


@pytest.mark.covers("R2")
def test_stated_income_far_from_the_payslip_is_flagged():
    assert "income_mismatch" in propose(extraction(stated=3400), 6000).flags
    assert "income_mismatch" not in propose(extraction(stated=3250), 6000).flags
