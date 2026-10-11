from __future__ import annotations

import json
import re
from dataclasses import dataclass

FIELDS = ("monthly_net_income", "monthly_debt_payments", "stated_monthly_income")


class AIOutputError(Exception):
    """The model's answer can't be trusted as data. The case goes to a person instead."""


@dataclass(frozen=True)
class Extraction:
    monthly_net_income: float | None
    monthly_debt_payments: float | None
    stated_monthly_income: float | None
    flags: tuple[str, ...]


def parse_extraction(raw: str) -> Extraction:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        raise AIOutputError(f"not valid JSON ({err.msg})") from err
    if not isinstance(data, dict):
        raise AIOutputError("expected a JSON object")
    missing = [k for k in (*FIELDS, "flags") if k not in data]
    if missing:
        raise AIOutputError(f"missing keys: {', '.join(missing)}")
    flags = data["flags"]
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        raise AIOutputError("flags must be a list of strings")
    values = {k: to_amount(data[k], k) for k in FIELDS}
    return Extraction(**values, flags=tuple(flags))


def to_amount(value, name: str = "value") -> float | None:
    """Accept amounts the way people write them: 4500, "4 500 €", "4,500.00", "4 500,50"."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise AIOutputError(f"{name} is not an amount: {value!r}")
    if isinstance(value, int | float):
        amount = float(value)
    elif isinstance(value, str):
        amount = _parse_text_amount(value, name)
    else:
        raise AIOutputError(f"{name} is not an amount: {value!r}")
    if amount < 0:
        raise AIOutputError(f"{name} is negative: {amount}")
    return amount


def _parse_text_amount(text: str, name: str) -> float:
    cleaned = re.sub(r"[\s €$£]|EUR", "", text.strip(), flags=re.I)
    if not re.fullmatch(r"[0-9][0-9.,]*", cleaned):
        raise AIOutputError(f"{name} is not an amount: {text!r}")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        whole, _, decimals = cleaned.rpartition(",")
        cleaned = f"{whole.replace(',', '')}.{decimals}" if len(decimals) == 2 else cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError as err:
        raise AIOutputError(f"{name} is not an amount: {text!r}") from err
