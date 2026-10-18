from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

LOWER_IS_BETTER = ("error_rate", "unflagged_error_rate")
HIGHER_IS_BETTER = ("consistency", "fairness_agreement")
FOR_INFORMATION = ("mean_latency_s", "total_cost_usd")


def summarise(report: dict, label: str | None = None) -> dict:
    """The part of an eval.json worth keeping as a baseline: metrics and each case's result."""
    return {
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": label,
        "adapter": report["adapter"],
        "models": report.get("models", []),
        "git_commit": report.get("git_commit"),
        "runs_per_case": report["runs_per_case"],
        "metrics": report["metrics"],
        "cases": {c["id"]: c["status"] for c in report["cases"]},
    }


def git_commit() -> str | None:
    """The commit the code was at, with `+dirty` for uncommitted changes. None outside git."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("+dirty" if dirty.strip() else "")


def save(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def latest_baseline(path: Path, adapter: str) -> dict | None:
    if not path.exists():
        return None
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matching = [e for e in entries if e["adapter"] == adapter]
    return matching[-1] if matching else None


@dataclass
class Drift:
    metrics: list[tuple[str, float | None, float | None, bool]] = field(default_factory=list)
    comparable: bool = True
    regressed: list[str] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)
    not_compared: list[str] = field(default_factory=list)
    models_changed: tuple[list[str], list[str]] | None = None

    @property
    def worse(self) -> bool:
        return bool(self.regressed) or any(w for *_, w in self.metrics)


def compare(baseline: dict, current: dict, tolerance: float) -> Drift:
    """What got worse since the baseline.

    Metrics are compared only when both runs covered the same cases; otherwise a rate
    could move just because the mix changed. Cases present in both are always compared:
    one that passed before and doesn't now is a regression, whatever the averages say.
    """
    before, after = baseline["cases"], current["cases"]
    drift = Drift(comparable=set(before) == set(after))
    old_models, new_models = baseline.get("models"), current.get("models")
    if old_models is not None and new_models is not None and old_models != new_models:
        drift.models_changed = (old_models, new_models)
    for name in (*LOWER_IS_BETTER, *HIGHER_IS_BETTER, *FOR_INFORMATION):
        old, new = baseline["metrics"].get(name), current["metrics"].get(name)
        worse = False
        if drift.comparable and old is not None and new is not None:
            if name in LOWER_IS_BETTER:
                worse = new > old + tolerance
            elif name in HIGHER_IS_BETTER:
                worse = new < old - tolerance
        drift.metrics.append((name, old, new, worse))
    for case_id in sorted(set(before) | set(after)):
        if case_id not in before or case_id not in after:
            drift.not_compared.append(case_id)
        elif before[case_id] == "ok" and after[case_id] != "ok":
            drift.regressed.append(case_id)
        elif before[case_id] != "ok" and after[case_id] == "ok":
            drift.fixed.append(case_id)
    return drift
