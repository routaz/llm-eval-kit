import hashlib

import pytest

from examples.loan_pipeline.ai import PROMPT_VERSION
from examples.loan_pipeline.audit import explain, read_audit
from examples.loan_pipeline.documents import Doc
from examples.loan_pipeline.parsing import Extraction
from examples.loan_pipeline.pipeline import RULES, LoanPipeline, propose
from qa.model_gate import ModelGate
from qa.tests.conftest import ScriptedAI, answer

DOCS = [Doc("payslip.txt", "Net pay: 3,200"), Doc("credit_report.txt", "Total monthly payments: 250")]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def named(model: str, *responses) -> ScriptedAI:
    ai = ScriptedAI(*responses)
    ai.model = model
    return ai


def event(pipeline, case_id, name):
    return next(e for e in pipeline.audit if e["case"] == case_id and e["event"] == name)


@pytest.mark.covers("R3")
def test_a_proposal_records_the_model_prompt_and_documents_it_came_from():
    ai = named("model-under-test", answer())
    pipeline = LoanPipeline(ModelGate(ai))
    case = pipeline.open_case({}, DOCS, 12000)
    pipeline.analyse(case.id)

    proposed = event(pipeline, case.id, "proposed")
    assert proposed["model"] == "model-under-test"
    assert proposed["prompt_version"] == PROMPT_VERSION
    assert proposed["prompt_sha256"] == sha(ai.prompts[-1])
    assert proposed["documents"] == [{"name": d.name, "sha256": sha(d.text)} for d in DOCS]


@pytest.mark.covers("R3")
def test_a_proposal_records_the_figures_and_the_rule_that_decided_it(pipeline):
    pipeline.ai = ScriptedAI(answer(net=3200, debts=1500, stated=3200))
    case = pipeline.open_case({}, DOCS, 12000)
    pipeline.analyse(case.id)

    proposed = event(pipeline, case.id, "proposed")
    assert proposed["decision"] == "decline"
    assert proposed["figures"]["monthly_debt_payments"] == 1500
    assert proposed["figures"]["debt_to_income"] == pytest.approx(0.5313, abs=1e-4)
    assert proposed["rule"] == "debt-to-income 0.53 > 0.50 -> decline"
    assert proposed["rules"] == RULES


@pytest.mark.parametrize(
    ("net", "debts", "flags", "rule"),
    [
        (None, 0, (), "no net income -> refer"),
        (3200, 1500, (), "debt-to-income 0.53 > 0.50 -> decline"),
        (3200, 1000, (), "debt-to-income 0.38 > 0.35 -> refer"),
        (3200, 250, ("credit_report_missing",), "flags credit_report_missing -> refer"),
        (3200, 250, (), "debt-to-income 0.14 <= 0.35, no flags -> approve"),
    ],
)
@pytest.mark.covers("R3")
def test_every_branch_of_the_rules_says_why_it_decided(net, debts, flags, rule):
    proposal = propose(Extraction(net, debts, net, tuple(flags)), 12000)

    assert proposal.rule == rule
    assert proposal.decision == rule.rsplit("-> ", 1)[1]


@pytest.mark.covers("R3")
def test_a_case_sent_to_a_person_still_records_which_model_failed():
    pipeline = LoanPipeline(named("model-under-test", "not json"))
    case = pipeline.open_case({}, DOCS, 12000)
    pipeline.analyse(case.id)

    attention = event(pipeline, case.id, "needs_attention")
    assert attention["model"] == "model-under-test"
    assert attention["prompt_version"] == PROMPT_VERSION


@pytest.mark.covers("R3")
def test_every_audit_event_is_timestamped(pipeline):
    case = pipeline.open_case({}, DOCS[:1], 12000)
    pipeline.add_documents(case.id, DOCS[1:])
    pipeline.analyse(case.id)
    pipeline.record_decision(case.id, "reviewer-1", "approve")

    assert [e["event"] for e in pipeline.audit] == ["opened", "documents_added", "proposed", "decided"]
    assert all(e["at"].endswith("+00:00") for e in pipeline.audit)


@pytest.mark.covers("R3")
def test_the_explanation_names_model_documents_rule_and_the_persons_override():
    pipeline = LoanPipeline(named("model-under-test", answer()))
    case = pipeline.open_case({}, DOCS, 12000)
    pipeline.analyse(case.id)
    pipeline.record_decision(case.id, "reviewer-1", "decline", reason="income looks seasonal")

    text = pipeline.explain(case.id)
    for expected in (
        "AI-assisted proposal: approve",
        "model: model-under-test",
        f"prompt version {PROMPT_VERSION}",
        "payslip.txt",
        "credit_report.txt",
        "debt-to-income 0.14 <= 0.35, no flags -> approve",
        'decided decline by reviewer-1, overriding the proposed approve: "income looks seasonal"',
    ):
        assert expected in text


@pytest.mark.covers("R3")
def test_a_decision_can_be_explained_from_the_audit_file_after_the_pipeline_is_gone(tmp_path):
    path = tmp_path / "audit.jsonl"
    pipeline = LoanPipeline(named("model-under-test", answer()), audit_to=path)
    case = pipeline.open_case({}, DOCS, 12000)
    pipeline.analyse(case.id)
    pipeline.record_decision(case.id, "reviewer-1", "approve")
    in_memory = pipeline.explain(case.id)
    del pipeline

    assert explain(read_audit(path), case.id) == in_memory
    assert "agreeing with the proposal, no reason given" in in_memory


@pytest.mark.covers("R3")
def test_explaining_an_unknown_case_is_an_error_not_an_empty_story(pipeline):
    with pytest.raises(KeyError):
        pipeline.explain("case-9999")
