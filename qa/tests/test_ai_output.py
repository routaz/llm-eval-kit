import pytest

from examples.loan_pipeline.documents import Doc
from examples.loan_pipeline.parsing import AIOutputError, to_amount
from qa.tests.conftest import ScriptedAI, answer

DOCS = [Doc("payslip.txt", "Net pay: 3,200.00")]


def analyse_with(pipeline, response):
    pipeline.ai = ScriptedAI(response)
    case = pipeline.open_case({"name": "Test"}, DOCS, 12000)
    return pipeline.analyse(case.id)


@pytest.mark.covers("R1")
@pytest.mark.parametrize(
    ("response", "problem"),
    [
        ('{"monthly_net_income": 3200, "monthly_debt', "not valid JSON"),
        ("Sure! Here are the figures you asked for.", "not valid JSON"),
        ('["not", "an", "object"]', "expected a JSON object"),
        ('{"monthly_net_income": 3200, "flags": []}', "missing keys"),
        (answer(net="about three thousand"), "not an amount"),
        (answer(net=-3200), "negative"),
        (answer().replace('"flags": []', '"flags": "none"'), "flags must be a list"),
    ],
)
def test_a_bad_ai_answer_goes_to_a_person_never_into_a_proposal(pipeline, response, problem):
    case = analyse_with(pipeline, response)

    assert case.status == "needs_attention"
    assert case.proposal is None
    assert problem in case.problem


@pytest.mark.covers("R1")
@pytest.mark.parametrize(
    ("written", "amount"),
    [
        (4500, 4500.0),
        ("4500", 4500.0),
        ("4 500 €", 4500.0),
        ("4,500.00", 4500.0),
        ("4 500,50", 4500.5),
        (None, None),
    ],
)
def test_amounts_written_the_way_people_write_them_are_read_correctly(written, amount):
    assert to_amount(written) == amount


@pytest.mark.covers("R1")
def test_an_amount_that_is_not_a_number_is_refused_not_guessed():
    with pytest.raises(AIOutputError):
        to_amount(True)


@pytest.mark.covers("R6")
def test_a_timeout_is_retried_and_the_case_survives(pipeline):
    case = analyse_with(pipeline, TimeoutError())
    assert case.status == "needs_attention"
    assert "timed out after 3 attempts" in case.problem
    assert len(pipeline.ai.prompts) == 3

    pipeline.ai = ScriptedAI(TimeoutError(), answer())
    case = pipeline.analyse(case.id)
    assert case.status == "proposed"
