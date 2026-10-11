from __future__ import annotations

from evalkit import Application, Proposal
from examples.loan_pipeline.ai import AIClient, KeywordModel
from examples.loan_pipeline.documents import Doc
from examples.loan_pipeline.pipeline import LoanPipeline


class LoanPipelineAdapter:
    def __init__(self, ai: AIClient | None = None):
        self.pipeline = LoanPipeline(ai or KeywordModel())

    def propose(self, application: Application) -> Proposal:
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
        )


def make_adapter() -> LoanPipelineAdapter:
    return LoanPipelineAdapter()
