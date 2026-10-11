from __future__ import annotations

import json
from pathlib import Path

from evalkit.context import GoldenCase

MIN_CHARS = 40


def find_leaks(cases: list[GoldenCase], rows: list[str], min_chars: int = MIN_CHARS) -> list[str]:
    """Golden documents that appear in training data, as `case-id: document name`.

    A model trained on its own test cases scores well by memory, not skill. Text is
    compared after lower-casing and collapsing whitespace; documents shorter than
    `min_chars` are skipped, since short boilerplate matches by chance.
    """
    corpus = [_normalise(row) for row in rows]
    leaks = []
    for case in cases:
        for doc in case.application.documents:
            text = _normalise(doc.text)
            if len(text) >= min_chars and any(text in row for row in corpus):
                leaks.append(f"{case.id}: {doc.name}")
    return leaks


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
