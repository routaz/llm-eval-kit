from __future__ import annotations

import json
import re
from pathlib import Path

from evalkit.context import GoldenCase

MIN_CHARS = 40
# What a redacted applicant detail looks like in training text: [APPLICANT], <name>, {postcode}, ***, XXX.
# Another person's name doesn't count: that is a different document, not a redacted copy.
PLACEHOLDER = r"(?:\[[^\]]{1,30}\]|<[^>]{1,30}>|\{[^}]{1,30}\}|[*#_x\u2588]{2,})"
GAP = rf"\s?(?:{PLACEHOLDER}\s?)*"


def find_leaks(cases: list[GoldenCase], rows: list[str], min_chars: int = MIN_CHARS) -> list[str]:
    """Golden documents that appear in training data, as `case-id: document name`.

    A model trained on its own test cases scores well by memory, not skill. Text is
    compared after lower-casing and collapsing whitespace; documents shorter than
    `min_chars` are skipped, since short boilerplate matches by chance.

    Training rows are often built from real prompts, where the applicant's name and postcode
    have been redacted. So the values on a case's applicant record (and each word of a
    multi-word one) may appear as a `PLACEHOLDER` or not at all; the rest of the document,
    at least `min_chars` of it, must match exactly.
    """
    corpus = [_normalise(row) for row in rows]
    leaks = []
    for case in cases:
        for doc in case.application.documents:
            pattern = _pattern(_normalise(doc.text), case.application.applicant, min_chars)
            if pattern and any(pattern.search(row) for row in corpus):
                leaks.append(f"{case.id}: {doc.name}")
    return leaks


def _pattern(text: str, applicant: dict, min_chars: int) -> re.Pattern | None:
    """`text` as a pattern in which each applicant value may be replaced, or None if too little is left."""
    values = set()
    for value in applicant.values():
        value = _normalise(str(value))
        if value:
            values.add(value)
            values.update(w for w in value.split() if len(w) > 1)
    if not values:
        return re.compile(re.escape(text)) if len(text) >= min_chars else None
    alternatives = "|".join(re.escape(v) for v in sorted(values, key=len, reverse=True))
    fixed = [part.strip() for part in re.split(rf"(?<!\w)(?:{alternatives})(?!\w)", text)]
    if sum(len(part) for part in fixed) < min_chars:
        return None
    return re.compile(GAP.join(re.escape(part) for part in fixed))


def read_rows(path: Path) -> list[str]:
    """One string per JSONL line: every string value in the record, joined."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append("\n".join(_strings(json.loads(line))))
    return rows


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())
