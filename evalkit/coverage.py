from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from evalkit.context import GoldenCase, Risk
from evalkit.faults import Fault


@dataclass
class RiskCoverage:
    risk: Risk
    cases: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    faults: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.cases or self.tests:
            return "covered"
        return "manual" if self.risk.manual else "missing"


@dataclass
class CoverageReport:
    rows: list[RiskCoverage]
    unknown: list[str]
    untagged_tests: list[str]

    def problems(self, strict: bool) -> list[str]:
        found = list(self.unknown)
        if strict:
            found += [
                f"{r.risk.id} {r.risk.title}: no golden case or test"
                for r in self.rows
                if r.status == "missing"
            ]
        return found


def build_coverage(
    risks: list[Risk], cases: list[GoldenCase], test_covers: dict[str, list[str]], faults: list[Fault]
) -> CoverageReport:
    rows = {r.id: RiskCoverage(r) for r in risks}
    unknown: list[str] = []

    def add(ids, owner: str, bucket: str) -> None:
        for risk_id in ids:
            if risk_id in rows:
                getattr(rows[risk_id], bucket).append(owner)
            else:
                unknown.append(f"{owner} covers unknown risk {risk_id}")

    for case in cases:
        add(case.covers, case.id, "cases")
    for test, ids in test_covers.items():
        add(ids, test, "tests")
    for fault in faults:
        add(fault.covers, fault.id, "faults")
    untagged = sorted(t for t, ids in test_covers.items() if not ids)
    return CoverageReport(list(rows.values()), unknown, untagged)


class _CoversCollector:
    def __init__(self) -> None:
        self.found: dict[str, list[str]] = {}

    def pytest_collection_modifyitems(self, items) -> None:
        for item in items:
            self.found[item.nodeid] = [i for m in item.iter_markers("covers") for i in m.args]


def collect_test_covers(paths: tuple[Path, ...]) -> dict[str, list[str]]:
    """Read `@pytest.mark.covers(...)` from the tests without running them."""
    if not paths:
        return {}
    collector = _CoversCollector()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = pytest.main(["--collect-only", "-q", *map(str, paths)], plugins=[collector])
    if code not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        raise RuntimeError(f"pytest could not collect the tests:\n{out.getvalue()}")
    return collector.found
