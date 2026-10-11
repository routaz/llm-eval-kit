from __future__ import annotations

from typing import Protocol

from evalkit.types import Application, Proposal


class PipelineAdapter(Protocol):
    """The only code a project must write to point the kit at its system.

    `propose` sends one application through the real pipeline, including the real
    model, and returns the proposal a reviewer would see. Raise an exception if the
    pipeline cannot produce a proposal; the evaluation counts that case as an error.
    """

    def propose(self, application: Application) -> Proposal: ...
