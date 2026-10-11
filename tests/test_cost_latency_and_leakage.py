import json
from dataclasses import replace

import pytest

from evalkit.adapter import StopEvaluation
from evalkit.config import Thresholds
from evalkit.evaluate import evaluate
from evalkit.leakage import find_leaks, read_rows
from evalkit.types import Application, Document
from tests.helpers import FixedAdapter, case, proposal


def test_reported_cost_and_latency_are_totalled_and_averaged():
    priced = proposal().with_changes(cost_usd=0.02, latency_s=3.0)
    report = evaluate(FixedAdapter({"c1": priced}), [case()], 2, Thresholds())

    assert report.metrics["total_cost_usd"] == pytest.approx(0.04)
    assert report.metrics["mean_latency_s"] == 3.0
    assert report.metrics["max_latency_s"] == 3.0


def test_latency_is_measured_when_the_adapter_does_not_report_it():
    report = evaluate(FixedAdapter({"c1": proposal()}), [case()], 1, Thresholds())

    assert report.results[0].runs[0].latency_s >= 0
    assert report.metrics["total_cost_usd"] is None


def test_stop_evaluation_halts_the_run_instead_of_failing_a_case():
    class Budgeted:
        def propose(self, application: Application):
            raise StopEvaluation("budget spent")

    with pytest.raises(StopEvaluation):
        evaluate(Budgeted(), [case()], 1, Thresholds())


PAYSLIP = "Payslip, September 2026\nEmployee: Anna Virtanen\nNet pay: 3,200.00"


def golden_with(text: str):
    return replace(case(), application=Application("c1", (Document("payslip.txt", text),)))


def test_a_golden_document_inside_training_data_is_a_leak():
    row = "Extract the figures.\n=== DOCUMENT ===\n" + PAYSLIP.upper().replace("\n", "\n\n")

    assert find_leaks([golden_with(PAYSLIP)], [row]) == ["c1: payslip.txt"]
    assert find_leaks([golden_with(PAYSLIP)], [row.replace("ANNA", "ELSA")]) == []


def test_short_boilerplate_is_not_counted_as_a_leak():
    assert find_leaks([golden_with("Net pay: 3,200")], ["Net pay: 3,200"]) == []


def test_training_rows_are_read_from_any_jsonl_shape(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {"messages": [{"role": "user", "content": PAYSLIP}, {"role": "assistant", "content": "{}"}]}
        )
        + "\n\n",
        encoding="utf-8",
    )

    [row] = read_rows(path)
    assert PAYSLIP in row
