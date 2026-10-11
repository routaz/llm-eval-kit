from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

from examples.loan_pipeline.ai import EXTRACTION_SCHEMA

SYSTEM_PROMPT = "You extract figures from loan application documents. Reply with the JSON object only."


class ClaudeCodeClient:
    """Calls Claude through the Claude Code CLI, using the signed-in user's own plan.

    Each call is one turn with no tools, a short system prompt in place of Claude Code's
    default one, and the extraction schema enforced. It runs in an empty temporary folder
    so no project CLAUDE.md is read. After each call, `last_cost_usd` and `last_latency_s`
    hold what the CLI reported.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        command: Sequence[str] | None = None,
        timeout_s: float = 180,
    ):
        self.model = model
        self.command = list(command) if command else [shutil.which("claude") or "claude"]
        self.timeout_s = timeout_s
        self.last_cost_usd: float | None = None
        self.last_latency_s: float | None = None

    def complete(self, prompt: str) -> str:
        args = [
            *self.command,
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--tools",
            "",
            "--system-prompt",
            SYSTEM_PROMPT,
            "--json-schema",
            json.dumps(EXTRACTION_SCHEMA),
            "--no-session-persistence",
            "--strict-mcp-config",
        ]
        started = time.perf_counter()
        with tempfile.TemporaryDirectory() as empty_dir:
            try:
                done = subprocess.run(
                    args,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    cwd=empty_dir,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired as err:
                raise TimeoutError(f"claude did not answer within {self.timeout_s} s") from err
        try:
            data = json.loads(done.stdout)
        except json.JSONDecodeError as err:
            raise RuntimeError(f"claude exited {done.returncode}: {done.stderr.strip()[:300]}") from err
        if data.get("is_error"):
            raise RuntimeError(f"claude reported an error: {str(data.get('result'))[:300]}")
        self.last_cost_usd = data.get("total_cost_usd")
        duration_ms = data.get("duration_ms")
        self.last_latency_s = duration_ms / 1000 if duration_ms else time.perf_counter() - started
        structured = data.get("structured_output")
        return json.dumps(structured) if structured is not None else str(data.get("result", ""))


class OpenAICompatibleClient:
    """Calls a local model served with an OpenAI-compatible API, such as `soup serve`.

    Uses `POST {base_url}/chat/completions` at temperature 0. Local inference has no
    per-call price, so `last_cost_usd` is 0.
    """

    def __init__(self, base_url: str, model: str, timeout_s: float = 180, max_tokens: int = 512):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.last_cost_usd: float | None = None
        self.last_latency_s: float | None = None

    def complete(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": self.max_tokens,
            }
        ).encode()
        request = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read())
        except TimeoutError as err:
            raise TimeoutError(f"no answer from {self.url} within {self.timeout_s} s") from err
        except urllib.error.URLError as err:
            raise RuntimeError(f"cannot reach {self.url}: {err.reason}") from err
        self.last_latency_s = time.perf_counter() - started
        self.last_cost_usd = 0.0
        return data["choices"][0]["message"]["content"]
