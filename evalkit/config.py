from __future__ import annotations

import importlib
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from evalkit.adapter import PipelineAdapter

CONFIG_FILE = "evalkit.toml"


@dataclass(frozen=True)
class Thresholds:
    max_error_rate: float = 0.1
    max_unflagged_error_rate: float = 0.0
    min_consistency: float = 1.0
    min_fairness_agreement: float = 1.0
    field_tolerance: float = 0.01
    drift_tolerance: float = 0.0
    min_catch_rate: float | None = None


@dataclass(frozen=True)
class Config:
    root: Path
    risks: Path
    cases: Path
    adapter: str
    faults: str | None = None
    tests: tuple[Path, ...] = ()
    history: Path | None = None
    runs: int = 3
    report_dir: Path = Path("reports")
    thresholds: Thresholds = field(default_factory=Thresholds)

    def load_adapter(self, ref: str | None = None) -> PipelineAdapter:
        """Build the adapter named in evalkit.toml, or `ref` ('module:factory') to try another model."""
        return _load_object(self.root, ref or self.adapter)()

    def load_faults(self) -> list:
        if not self.faults:
            return []
        return list(_load_object(self.root, self.faults))


def load_config(root: Path | None = None) -> Config:
    """Read `evalkit.toml` from `root` (default: the current directory)."""
    root = (root or Path.cwd()).resolve()
    path = root / CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(f"No {CONFIG_FILE} in {root}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    paths = data.get("paths", {})
    ev = data.get("eval", {})
    return Config(
        root=root,
        risks=root / paths["risks"],
        cases=root / paths["cases"],
        tests=tuple(root / p for p in paths.get("tests", [])),
        history=root / paths["history"] if "history" in paths else None,
        adapter=ev["adapter"],
        faults=ev.get("faults"),
        runs=int(ev.get("runs", 3)),
        report_dir=root / ev.get("report_dir", "reports"),
        thresholds=Thresholds(**data.get("thresholds", {})),
    )


def _load_object(root: Path, ref: str):
    module_name, _, attr = ref.partition(":")
    if not attr:
        raise ValueError(f"Expected 'module:attribute', got {ref!r}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return getattr(importlib.import_module(module_name), attr)
