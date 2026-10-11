from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Doc:
    name: str
    text: str
