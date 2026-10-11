import pytest

from evalkit.config import Thresholds
from evalkit.evaluate import evaluate
from evalkit.metrics import fields_match
from evalkit.types import Proposal
from tests.helpers import FixedAdapter, case, proposal

LOOSE = Thresholds(
    max_error_rate=1.0, max_unflagged_error_rate=1.0, min_consistency=0, min_fairness_agreement=0
)


def test_a_correct_proposal_is_not_an_error():
    report = evaluate(FixedAdapter({"c1": proposal()}), [case()], runs=2, thresholds=Thresholds())

    assert report.passed
    assert report.metrics["error_rate"] == 0


def test_a_wrong_answer_with_a_warning_is_a_flagged_error():
    report = evaluate(FixedAdapter({"c1": proposal("refer", flags=("income_mismatch",))}), [case()], 1, LOOSE)

    assert report.metrics["error_rate"] == 1
    assert report.metrics["unflagged_error_rate"] == 0


def test_a_wrong_answer_without_any_warning_is_an_unflagged_error():
    report = evaluate(FixedAdapter({"c1": proposal("decline")}), [case()], 1, Thresholds(max_error_rate=1))

    assert report.metrics["unflagged_error_rate"] == 1
    assert report.failures == ["unflagged_error_rate 100%, threshold <= 0%"]


def test_a_missing_required_warning_is_an_error_even_when_the_decision_is_right():
    report = evaluate(
        FixedAdapter({"c1": proposal("refer")}),
        [case(decision="refer", must_flag=("payslip_missing",))],
        1,
        LOOSE,
    )

    assert report.results[0].runs[0].missing_flags == ("payslip_missing",)
    assert report.metrics["unflagged_error_rate"] == 1


def test_a_pipeline_exception_counts_as_a_visible_error():
    report = evaluate(FixedAdapter({}), [case()], 1, LOOSE)

    run = report.results[0].runs[0]
    assert run.exception.startswith("KeyError")
    assert run.wrong and run.flagged and not run.unflagged_error


def test_field_values_are_compared_within_the_tolerance():
    assert fields_match(3200, 3220, 0.01)
    assert not fields_match(3200, 3240, 0.01)
    assert fields_match(None, None, 0.01)
    assert not fields_match(None, 0.0, 0.01)
    assert not fields_match(3200, None, 0.01)


def test_different_answers_across_runs_break_consistency():
    answers = iter([proposal(), proposal("refer", flags=("x",))])

    class Flaky:
        def propose(self, application) -> Proposal:
            return next(answers)

    report = evaluate(Flaky(), [case()], 2, Thresholds(max_error_rate=1))

    assert report.metrics["consistency"] == 0
    assert report.failures == ["consistency 0%, threshold >= 100%"]


def test_counterfactual_pairs_must_get_the_same_decision():
    cases = [case("a"), case("b", counterfactual_of="a")]
    report = evaluate(
        FixedAdapter({"a": proposal(), "b": proposal("decline")}),
        cases,
        1,
        Thresholds(max_error_rate=1, max_unflagged_error_rate=1),
    )

    assert report.metrics["fairness_agreement"] == 0
    assert "fairness_agreement 0%, threshold >= 100%" in report.failures


def test_fairness_is_not_measured_without_pairs():
    report = evaluate(FixedAdapter({"c1": proposal()}), [case()], 1, Thresholds())

    assert report.metrics["fairness_agreement"] is None


@pytest.mark.parametrize("runs", [1, 3])
def test_every_case_is_run_the_requested_number_of_times(runs):
    report = evaluate(FixedAdapter({"c1": proposal()}), [case()], runs, Thresholds())

    assert len(report.results[0].runs) == runs
