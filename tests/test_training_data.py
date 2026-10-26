import json
from pathlib import Path

from evalkit.context import load_cases
from evalkit.leakage import find_leaks
from examples.loan_pipeline.ai import KeywordModel, build_prompt
from examples.loan_pipeline.parsing import parse_extraction
from training.generate import FIRST_NAMES, LAST_NAMES, generate

GOLDEN = Path(__file__).parents[1] / "qa" / "golden" / "cases"
ROWS = generate(400, seed=11)


def test_no_golden_case_leaks_into_the_training_data():
    texts = ["\n".join(m["content"] for m in row["messages"]) for row in ROWS]

    assert find_leaks(load_cases(GOLDEN), texts) == []


def test_a_golden_case_as_the_pipeline_prompts_it_is_a_leak():
    """Training rows built from logged prompts carry redacted documents; they must still be caught."""
    cases = load_cases(GOLDEN)
    for case in cases:
        prompt = build_prompt(list(case.application.documents), tuple(case.application.applicant.values()))
        name = case.application.applicant["name"]
        assert any(name in d.text for d in case.application.documents)
        assert name not in prompt  # redacted, so the golden text is no longer in the prompt verbatim
        assert find_leaks([case], [prompt]), case.id


def test_training_prompts_are_redacted_like_the_pipeline_s():
    """A model trained on names it will never see in production learns the wrong prompt."""
    for row in ROWS:
        prompt = row["messages"][1]["content"]
        assert "[APPLICANT]" in prompt
        assert not [n for n in FIRST_NAMES + LAST_NAMES if n in prompt]


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
