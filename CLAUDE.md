# Working rules for agents

This repo tests an AI-assisted decision pipeline in three layers. `evalkit/` is the reusable kit;
`qa/` is the project-specific side; `examples/loan_pipeline/` is the demo system under test.

## Before you finish any change

```bash
uv run ruff format . && uv run ruff check .
uv run pytest
uv run evalkit check && uv run evalkit coverage --strict
uv run evalkit eval && uv run evalkit mutate
```

## Ground rules

- `qa/context/risks.yaml` and the `expected:` blocks in `qa/golden/cases/` are the source of truth.
  Expected results are set by domain experts. Never change one to match what the system does; report the
  disagreement instead.
- Every Layer 1 test carries `@pytest.mark.covers("R<n>")`. Every golden case and fault names its risks.
- Layer 1 tests never call a real model. Use `ScriptedAI` from `qa/tests/conftest.py`.
- Never make a red evaluation green by loosening a threshold in `evalkit.toml`. Thresholds change only by
  agreement with the business.
- A fault that survives `evalkit mutate` means a golden case is too weak. Strengthen or add the case;
  don't delete the fault.
- Keep personal data out of logs, prompts and fixtures beyond what a case needs. Use made-up names.
- Paid models spend the user's plan. Never run `make_claude_adapter` without being asked, and then only
  with `--cases`, `--runs 1` and a small `EVALKIT_MAX_CALLS`. Use `make_replay_adapter` for everything else.
- Training data comes from `training/generate.py` only. Run `evalkit leakcheck` on it before any training.
