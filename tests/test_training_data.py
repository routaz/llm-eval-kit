import json
from pathlib import Path

from evalkit.context import load_cases
from evalkit.leakage import find_leaks
from examples.loan_pipeline.ai import KeywordModel
from examples.loan_pipeline.parsing import parse_extraction
from training.generate import generate

GOLDEN = Path(__file__).parents[1] / "qa" / "golden" / "cases"
ROWS = generate(400, seed=11)


def test_no_golden_case_leaks_into_the_training_data():
    texts = ["\n".join(m["content"] for m in row["messages"]) for row in ROWS]

    assert find_leaks(load_cases(GOLDEN), texts) == []


def test_every_label_matches_what_the_documents_say():
    model = KeywordModel()
    for row in ROWS:
        _, user, assistant = row["messages"]
        label = parse_extraction(assistant["content"])
        read = parse_extraction(model.complete(user["content"]))
        assert (label.monthly_net_income, label.monthly_debt_payments, label.stated_monthly_income) == (
            read.monthly_net_income,
            read.monthly_debt_payments,
            read.stated_monthly_income,
        )
        assert sorted(label.flags) == sorted(read.flags)


def test_every_warning_the_model_must_learn_appears():
    flags = {f for row in ROWS for f in json.loads(row["messages"][2]["content"])["flags"]}

    assert flags == {
        "net_income_missing",
        "payslip_missing",
        "credit_report_missing",
        "suspicious_instruction",
    }
