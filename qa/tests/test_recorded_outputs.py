import json
from pathlib import Path

import pytest

from examples.loan_pipeline.ai import FLAG_CODES
from examples.loan_pipeline.parsing import parse_extraction

RECORDINGS = sorted((Path(__file__).parents[1] / "recordings").glob("*.jsonl"))
ANSWERS = [
    pytest.param(json.loads(line)["response"], id=f"{path.stem}:{n}")
    for path in RECORDINGS
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
    if line.strip()
]


@pytest.mark.covers("R1")
@pytest.mark.skipif(
    not ANSWERS, reason="no recordings yet; record with make_claude_adapter or make_soup_adapter"
)
@pytest.mark.parametrize("response", ANSWERS)
def test_every_recorded_real_answer_parses_and_uses_only_defined_flags(response):
    extraction = parse_extraction(response)

    assert set(extraction.flags) <= set(FLAG_CODES)
