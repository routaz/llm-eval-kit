from pathlib import Path

import pytest

from evalkit.context import ContextError, Risk, load_cases, load_risks
from evalkit.coverage import build_coverage
from evalkit.faults import Fault, drop_flags
from tests.helpers import case

CASE = """
id: {id}
covers: [{covers}]
description: test
application:
  documents: [{{name: a.txt, text: hello}}]
expected:
  decision: {decision}
"""


def write(folder: Path, name: str, text: str) -> None:
    (folder / name).write_text(text, encoding="utf-8")


def test_valid_cases_load(tmp_path):
    write(tmp_path, "a.yaml", CASE.format(id="a", covers="R1", decision="approve"))

    [loaded] = load_cases(tmp_path, {"R1"})
    assert loaded.id == "a"
    assert loaded.application.documents[0].text == "hello"


def test_schema_errors_name_the_file_and_field(tmp_path):
    write(tmp_path, "bad.yaml", CASE.format(id="a", covers="R1", decision="maybe"))

    with pytest.raises(ContextError) as err:
        load_cases(tmp_path)
    assert err.value.problems == [
        "bad.yaml: expected/decision: 'maybe' is not one of ['approve', 'decline', 'refer']"
    ]


def test_cross_file_problems_are_all_reported(tmp_path):
    write(tmp_path, "a.yaml", CASE.format(id="a", covers="R9", decision="approve"))
    write(
        tmp_path, "b.yaml", CASE.format(id="a", covers="R1", decision="approve") + "counterfactual_of: zz\n"
    )

    with pytest.raises(ContextError) as err:
        load_cases(tmp_path, {"R1"})
    assert err.value.problems == [
        "a: counterfactual_of unknown case zz",
        "a: covers unknown risk R9",
        "duplicate case id a",
    ]


def test_risks_are_sorted_by_rank_and_ids_must_be_unique(tmp_path):
    path = tmp_path / "risks.yaml"
    path.write_text(
        "risks:\n"
        "  - {id: R2, rank: 2, title: b, layer: [evaluation]}\n"
        "  - {id: R1, rank: 1, title: a, layer: [pipeline]}\n",
        encoding="utf-8",
    )
    assert [r.id for r in load_risks(path)] == ["R1", "R2"]

    path.write_text(
        path.read_text(encoding="utf-8") + "  - {id: R1, rank: 3, title: c, layer: [pipeline]}\n", "utf-8"
    )
    with pytest.raises(ContextError):
        load_risks(path)


RISKS = [
    Risk("R1", 1, "covered", ("evaluation",)),
    Risk("R2", 2, "manual", ("monitoring",), manual="dashboards"),
    Risk("R3", 3, "nothing", ("pipeline",)),
]


def test_coverage_combines_cases_tests_and_faults():
    report = build_coverage(
        RISKS,
        [case("c1", covers=("R1",))],
        {"test_a": ["R1"], "test_b": []},
        [Fault("f1", ("R1",), "", drop_flags())],
    )

    assert [(r.risk.id, r.status) for r in report.rows] == [
        ("R1", "covered"),
        ("R2", "manual"),
        ("R3", "missing"),
    ]
    assert (report.rows[0].cases, report.rows[0].tests, report.rows[0].faults) == (["c1"], ["test_a"], ["f1"])
    assert report.untagged_tests == ["test_b"]
    assert report.problems(strict=False) == []
    assert report.problems(strict=True) == ["R3 nothing: no golden case or test"]


def test_coverage_reports_ids_that_match_no_risk():
    report = build_coverage(RISKS, [], {"test_a": ["R7"]}, [])

    assert report.problems(strict=False) == ["test_a covers unknown risk R7"]
