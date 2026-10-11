import pytest

from qa.model_gate import CallBudgetSpent, ModelGate, ReplayClient


class PricedModel:
    def __init__(self):
        self.answers = iter(["first", "second", "third"])
        self.last_cost_usd = 0.01
        self.last_latency_s = 2.0

    def complete(self, prompt: str) -> str:
        return next(self.answers)


def test_the_call_budget_stops_before_the_model_is_called_again():
    model = PricedModel()
    gate = ModelGate(model, max_calls=2)
    gate.complete("a")
    gate.complete("b")

    with pytest.raises(CallBudgetSpent):
        gate.complete("c")
    assert next(model.answers) == "third"


def test_cost_and_latency_are_tallied_until_reset():
    gate = ModelGate(PricedModel())
    gate.complete("a")
    gate.complete("a")

    assert (gate.cost_usd, gate.latency_s) == (pytest.approx(0.02), 4.0)
    gate.reset_tally()
    assert (gate.cost_usd, gate.latency_s) == (None, 0.0)


def test_recorded_answers_replay_in_order_with_their_cost(tmp_path):
    path = tmp_path / "model.jsonl"
    recorder = ModelGate(PricedModel(), record_to=path)
    recorder.complete("prompt A")
    recorder.complete("prompt A")
    recorder.complete("prompt B")

    replay = ModelGate(ReplayClient(path))
    assert [replay.complete("prompt A") for _ in range(3)] == ["first", "second", "first"]
    assert replay.complete("prompt B") == "third"
    assert replay.cost_usd == pytest.approx(0.04)


def test_replaying_a_prompt_that_was_never_recorded_says_so(tmp_path):
    path = tmp_path / "model.jsonl"
    ModelGate(PricedModel(), record_to=path).complete("prompt A")

    with pytest.raises(LookupError, match="no recording"):
        ReplayClient(path).complete("prompt B")
