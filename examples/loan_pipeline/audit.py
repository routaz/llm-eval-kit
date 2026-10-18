"""Explain one decision afterwards, from the audit trail alone.

    uv run python -m examples.loan_pipeline.audit audit.jsonl case-0001

The explanation answers what a reviewer, auditor or applicant may ask later: what the system
was given, which model and prompt read it, which rule decided, and what the person did.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def read_audit(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def explain(events: list[dict], case_id: str) -> str:
    mine = [e for e in events if e["case"] == case_id]
    if not mine:
        raise KeyError(f"no audit events for {case_id}")
    lines = [case_id]
    for e in mine:
        lines.append(f"  {e.get('at', '(no time)')}  {_headline(e)}")
        lines += [f"      {detail}" for detail in _details(e)]
    return "\n".join(lines)


def _headline(e: dict) -> str:
    event = e["event"]
    if event == "opened":
        return f"opened with {e.get('documents', 0)} documents"
    if event == "documents_added":
        return f"{e.get('documents', 0)} documents added"
    if event == "proposed":
        return f"AI-assisted proposal: {e['decision']} (risk {e['risk_rating']})"
    if event == "needs_attention":
        return f"sent to a person: {e['problem']}"
    if event == "decided":
        who, final, ai = e["reviewer"], e["decision"], e.get("ai_decision")
        if ai is None:
            how = "with no AI proposal"
        elif e.get("override"):
            how = f"overriding the proposed {ai}"
        else:
            how = "agreeing with the proposal"
        reason = f': "{e["reason"]}"' if e.get("reason") else ", no reason given"
        return f"decided {final} by {who}, {how}{reason}"
    return event


def _details(e: dict) -> list[str]:
    out = []
    if "model" in e:
        out.append(
            f"model: {e['model'] or 'unknown'}, prompt version {e['prompt_version']}, "
            f"prompt sha256 {e['prompt_sha256'][:12]}"
        )
    if e.get("documents") and isinstance(e["documents"], list):
        docs = ", ".join(f"{d['name']} (sha256 {d['sha256'][:12]})" for d in e["documents"])
        out.append(f"documents read: {docs}")
    if "figures" in e:
        out.append("figures: " + ", ".join(f"{k} {_num(v)}" for k, v in e["figures"].items()))
    if e.get("flags"):
        out.append(f"flags shown to the reviewer: {', '.join(e['flags'])}")
    if e.get("rule"):
        rules = ", ".join(f"{k} {v}" for k, v in e.get("rules", {}).items())
        out.append(f"rule: {e['rule']}" + (f" (rules in force: {rules})" if rules else ""))
    return out


def _num(value) -> str:
    return "-" if value is None else f"{value:g}"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m examples.loan_pipeline.audit AUDIT.jsonl CASE_ID")
        return 2
    try:
        print(explain(read_audit(Path(args[0])), args[1]))
    except KeyError as err:
        print(err.args[0])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
