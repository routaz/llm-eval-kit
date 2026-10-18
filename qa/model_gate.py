from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from evalkit.adapter import StopEvaluation
from examples.loan_pipeline.ai import AIClient


class CallBudgetSpent(StopEvaluation):
    pass


def prompt_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class ModelGate:
    """Sits between the pipeline and its model: caps calls, records answers, tallies cost and latency.

    `max_calls` stops the evaluation before a paid model is called once more than you
    allowed, retries included. With `record_to`, every answer is appended to a JSONL file
    that `ReplayClient` can play back without calling the model.
    """

    def __init__(self, inner: AIClient, max_calls: int | None = None, record_to: Path | None = None):
        self.inner = inner
        self.max_calls = max_calls
        self.record_to = record_to
        self.calls = 0
        self.reset_tally()

    @property
    def model(self) -> str | None:
        return getattr(self.inner, "model", None)

    def reset_tally(self) -> None:
        self.cost_usd: float | None = None
        self.latency_s = 0.0

    def complete(self, prompt: str) -> str:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise CallBudgetSpent(
                f"call budget of {self.max_calls} spent. Use --cases and --runs to run less, "
                "or raise EVALKIT_MAX_CALLS on purpose."
            )
        self.calls += 1
        started = time.perf_counter()
        response = self.inner.complete(prompt)
        latency = getattr(self.inner, "last_latency_s", None) or time.perf_counter() - started
        cost = getattr(self.inner, "last_cost_usd", None)
        self.latency_s += latency
        if cost is not None:
            self.cost_usd = (self.cost_usd or 0.0) + cost
        if self.record_to:
            self._record(prompt, response, cost, latency)
        return response

    def _record(self, prompt: str, response: str, cost: float | None, latency: float) -> None:
        self.record_to.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "prompt_sha256": prompt_key(prompt),
            "response": response,
            "cost_usd": cost,
            "latency_s": latency,
        }
        with self.record_to.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


class ReplayClient:
    """Plays back recorded answers. Several recordings of one prompt are returned in turn,
    so consistency across runs is measured on what the real model actually said."""

    def __init__(self, path: Path):
        self.path = path
        self.model = f"replay:{path.stem}"
        self.answers: dict[str, list[dict]] = defaultdict(list)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                self.answers[entry["prompt_sha256"]].append(entry)
        self._served: dict[str, int] = defaultdict(int)
        self.last_cost_usd: float | None = None
        self.last_latency_s: float | None = None

    def complete(self, prompt: str) -> str:
        key = prompt_key(prompt)
        if key not in self.answers:
            raise LookupError(f"no recording for this prompt in {self.path.name}; record it first")
        recorded = self.answers[key]
        entry = recorded[self._served[key] % len(recorded)]
        self._served[key] += 1
        self.last_cost_usd = entry["cost_usd"]
        self.last_latency_s = entry["latency_s"]
        return entry["response"]
