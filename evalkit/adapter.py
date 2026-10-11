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


class StopEvaluation(BaseException):
    """Raise from an adapter to halt the whole run, e.g. when a call budget is spent.

    Any other exception counts as a failed case; this one stops the evaluation and the
    CLI reports why, so a budget limit never shows up as a model failure. Like
    KeyboardInterrupt it is a BaseException, so it passes through the system's own
    `except Exception` handlers without the system knowing about the kit.
    """
