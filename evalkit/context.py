from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from evalkit.types import Application, Document


@dataclass(frozen=True)
class Risk:
    id: str
    rank: int
    title: str
    layer: tuple[str, ...]
    manual: str | None = None


@dataclass(frozen=True)
class Expected:
    decision: str
    risk_rating: str | None
    fields: dict[str, float | None]
    must_flag: tuple[str, ...]


@dataclass(frozen=True)
class GoldenCase:
    id: str
    covers: tuple[str, ...]
    description: str
    application: Application
    expected: Expected
    counterfactual_of: str | None = None


class ContextError(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


def _schema(name: str) -> Draft202012Validator:
    text = resources.files("evalkit.schemas").joinpath(name).read_text(encoding="utf-8")
    return Draft202012Validator(json.loads(text))


def _validate(data, schema: str, where: str) -> list[str]:
    return [
        f"{where}: {'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}"
        for e in _schema(schema).iter_errors(data)
    ]


def load_risks(path: Path) -> list[Risk]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    problems = _validate(data, "risks.schema.json", path.name)
    if problems:
        raise ContextError(problems)
    risks = [Risk(r["id"], r["rank"], r["title"], tuple(r["layer"]), r.get("manual")) for r in data["risks"]]
    ids = [r.id for r in risks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ContextError([f"{path.name}: duplicate risk ids {', '.join(duplicates)}"])
    return sorted(risks, key=lambda r: r.rank)


def load_cases(folder: Path, risk_ids: set[str] | None = None) -> list[GoldenCase]:
    """Load every `*.yaml` golden case in `folder`, validated against the schema and each other."""
    problems: list[str] = []
    cases: list[GoldenCase] = []
    for path in sorted(folder.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        errors = _validate(data, "case.schema.json", path.name)
        if errors:
            problems += errors
            continue
        cases.append(_to_case(data))

    ids = [case.id for case in cases]
    for case in cases:
        if ids.count(case.id) > 1:
            problems.append(f"duplicate case id {case.id}")
        if risk_ids is not None:
            problems += [f"{case.id}: covers unknown risk {risk}" for risk in case.covers if risk not in risk_ids]
        if case.counterfactual_of and case.counterfactual_of not in ids:
            problems.append(f"{case.id}: counterfactual_of unknown case {case.counterfactual_of}")
    if not cases and not problems:
        problems.append(f"no golden cases in {folder}")
    if problems:
        raise ContextError(sorted(set(problems)))
    return cases


def _to_case(data: dict) -> GoldenCase:
    app = data["application"]
    exp = data["expected"]
    return GoldenCase(
        id=data["id"],
        covers=tuple(data["covers"]),
        description=data["description"],
        counterfactual_of=data.get("counterfactual_of"),
        application=Application(
            case_id=data["id"],
            documents=tuple(Document(d["name"], d["text"]) for d in app["documents"]),
            applicant=dict(app.get("applicant", {})),
            requested_amount=float(app.get("requested_amount", 0)),
        ),
        expected=Expected(
            decision=exp["decision"],
            risk_rating=exp.get("risk_rating"),
            fields=dict(exp.get("fields", {})),
            must_flag=tuple(exp.get("must_flag", [])),
        ),
    )
