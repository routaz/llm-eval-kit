from evalkit.config import Thresholds
from evalkit.faults import (
    Fault,
    applicant_is,
    case_is,
    chain,
    document_contains,
    drop_flags,
    force_decision,
    scale_field,
    unstable_decision,
    when,
)
from evalkit.mutate import mutate
from tests.helpers import FixedAdapter, case, proposal

APP = case(applicant={"name": "A"}).application


def test_transforms_change_only_what_they_say():
    p = proposal("refer", {"income": 1000.0, "debts": None}, ("a", "b"))

    assert drop_flags()(APP, p, 0).flags == ()
    assert drop_flags("a")(APP, p, 0).flags == ("b",)
    assert scale_field("income", 1.5)(APP, p, 0).fields == {"income": 1500.0, "debts": None}
    assert scale_field("debts", 2)(APP, p, 0) == p
    assert force_decision("approve")(APP, p, 0).decision == "approve"
    assert chain(force_decision("approve"), drop_flags())(APP, p, 0) == p.with_changes(
        decision="approve", flags=()
    )


def test_unstable_decision_changes_every_second_run():
    t = unstable_decision()

    assert [t(APP, proposal("approve"), run).decision for run in range(4)] == [
        "approve",
        "refer",
        "approve",
        "refer",
    ]


def test_conditional_faults_hit_only_matching_applications():
    by_name = when(applicant_is("name", "A"), force_decision("decline"))
    by_text = when(document_contains("TEXT"), force_decision("decline"))
    other = case(applicant={"name": "B"}).application

    by_case = when(case_is("c1"), force_decision("decline"))
    elsewhere = case("c2").application

    assert by_name(APP, proposal(), 0).decision == "decline"
    assert by_name(other, proposal(), 0).decision == "approve"
    assert by_text(APP, proposal(), 0).decision == "decline"
    assert by_case(APP, proposal(), 0).decision == "decline"
    assert by_case(elsewhere, proposal(), 0).decision == "approve"


def make_adapter():
    return FixedAdapter({"c1": proposal(), "c2": proposal("decline"), "pair": proposal()})


CASES = [
    case("c1"),
    case("c2", decision="decline", covers=("R2",)),
    case("pair", covers=("R5",), counterfactual_of="c1"),
]


def test_a_fault_the_evaluation_notices_is_caught():
    run = mutate(make_adapter, CASES, [Fault("f", ("R1",), "", force_decision("refer"))], 1, Thresholds())

    assert run.passed
    assert run.results[0].status == "caught"
    assert run.results[0].case_ids == ("c1",)


def test_a_fault_the_evaluation_misses_survives():
    harmless = Fault("f", ("R1",), "", drop_flags())
    run = mutate(make_adapter, CASES, [harmless], 1, Thresholds())

    assert not run.passed
    assert run.results[0].status == "survived"


def test_a_fault_for_a_risk_without_cases_is_uncovered():
    run = mutate(make_adapter, CASES, [Fault("f", ("R9",), "", force_decision("refer"))], 1, Thresholds())

    assert run.results[0].status == "uncovered"
    assert not run.passed


def test_a_counterfactual_case_brings_its_partner_along():
    fault = Fault("f", ("R5",), "", when(lambda app: app.case_id == "pair", force_decision("decline")))
    run = mutate(make_adapter, CASES, [fault], 1, Thresholds(max_error_rate=1, max_unflagged_error_rate=1))

    assert run.results[0].case_ids == ("pair", "c1")
    assert run.results[0].status == "caught"


def test_nothing_counts_when_the_clean_system_already_fails():
    broken = lambda: FixedAdapter({"c1": proposal("decline"), "c2": proposal("decline"), "pair": proposal()})  # noqa: E731
    run = mutate(broken, CASES, [Fault("f", ("R1",), "", drop_flags())], 1, Thresholds())

    assert not run.passed
    assert run.results == ()
