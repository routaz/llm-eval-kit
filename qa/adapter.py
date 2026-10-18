from __future__ import annotations

import os
from pathlib import Path

from evalkit import Application, Proposal
from examples.loan_pipeline.ai import AIClient, KeywordModel
from examples.loan_pipeline.clients import ClaudeCodeClient, OpenAICompatibleClient
from examples.loan_pipeline.documents import Doc
from examples.loan_pipeline.pipeline import LoanPipeline
from qa.model_gate import ModelGate, ReplayClient

RECORDINGS = Path(__file__).parent / "recordings"


class LoanPipelineAdapter:
    def __init__(self, ai: AIClient | None = None):
        self.gate = ai if isinstance(ai, ModelGate) else ModelGate(ai or KeywordModel())
        self.pipeline = LoanPipeline(self.gate)

    def propose(self, application: Application) -> Proposal:
        self.gate.reset_tally()
        docs = [Doc(d.name, d.text) for d in application.documents]
        case = self.pipeline.open_case(application.applicant, docs, application.requested_amount)
        case = self.pipeline.analyse(case.id)
        if case.proposal is None:
            raise RuntimeError(case.problem)
        p = case.proposal
        return Proposal(
            decision=p.decision,
            risk_rating=p.risk_rating,
            fields={
                "monthly_net_income": p.extraction.monthly_net_income,
                "monthly_debt_payments": p.extraction.monthly_debt_payments,
            },
            flags=p.flags,
            cost_usd=self.gate.cost_usd,
            latency_s=self.gate.latency_s,
            model=self.gate.model,
        )


def make_adapter() -> LoanPipelineAdapter:
    """The deterministic stand-in model. Free, offline, used in CI."""
    return LoanPipelineAdapter()


def make_claude_adapter() -> LoanPipelineAdapter:
    """Claude through your own Claude Code login, capped at EVALKIT_MAX_CALLS calls (default 5).

    Every answer is recorded, so one small paid run can be replayed for free afterwards.
    """
    model = os.environ.get("EVALKIT_CLAUDE_MODEL", "claude-opus-5-5")
    gate = ModelGate(
        ClaudeCodeClient(model=model),
        max_calls=int(os.environ.get("EVALKIT_MAX_CALLS", "5")),
        record_to=RECORDINGS / f"{model}.jsonl",
    )
    return LoanPipelineAdapter(gate)


def make_soup_adapter() -> LoanPipelineAdapter:
    """A local model served by `soup serve`, at SOUP_URL (default http://localhost:8000/v1)."""
    model = os.environ.get("SOUP_MODEL", "output")
    client = OpenAICompatibleClient(os.environ.get("SOUP_URL", "http://localhost:8000/v1"), model)
    return LoanPipelineAdapter(ModelGate(client, record_to=RECORDINGS / f"soup-{model}.jsonl"))


def make_replay_adapter() -> LoanPipelineAdapter:
    """Replays a recording from qa/recordings/ (EVALKIT_REPLAY, default claude-opus-5-5) with no calls."""
    name = os.environ.get("EVALKIT_REPLAY", "claude-opus-5-5")
    return LoanPipelineAdapter(ReplayClient(RECORDINGS / f"{name}.jsonl"))
