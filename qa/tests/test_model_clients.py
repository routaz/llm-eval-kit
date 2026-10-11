import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from examples.loan_pipeline.clients import ClaudeCodeClient, OpenAICompatibleClient
from examples.loan_pipeline.documents import Doc
from examples.loan_pipeline.pipeline import LoanPipeline
from qa.tests.conftest import answer


def fake_claude(tmp_path, reply: dict | str):
    script = tmp_path / "fake_claude.py"
    seen = tmp_path / "seen.json"
    body = reply if isinstance(reply, str) else json.dumps(reply)
    script.write_text(
        "import json, sys\n"
        "call = {'argv': sys.argv[1:], 'prompt': sys.stdin.read()}\n"
        f"open({str(seen)!r}, 'w', encoding='utf-8').write(json.dumps(call))\n"
        f"sys.stdout.write({body!r})\n",
        encoding="utf-8",
    )
    return ClaudeCodeClient(model="claude-opus-5-5", command=[sys.executable, str(script)]), seen


REPLY = {
    "type": "result",
    "is_error": False,
    "result": "",
    "structured_output": {
        "monthly_net_income": 3200,
        "monthly_debt_payments": 250,
        "stated_monthly_income": 3200,
        "flags": [],
    },
    "total_cost_usd": 0.0123,
    "duration_ms": 1500,
}


@pytest.mark.covers("R1")
def test_claude_is_called_once_with_no_tools_and_the_schema_enforced(tmp_path):
    client, seen = fake_claude(tmp_path, REPLY)

    text = client.complete("PROMPT TEXT")

    call = json.loads(seen.read_text(encoding="utf-8"))
    assert call["prompt"] == "PROMPT TEXT"
    argv = call["argv"]
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5"
    assert argv[argv.index("--tools") + 1] == ""
    assert json.loads(argv[argv.index("--json-schema") + 1])["required"]
    assert json.loads(text)["monthly_net_income"] == 3200
    assert (client.last_cost_usd, client.last_latency_s) == (0.0123, 1.5)


@pytest.mark.covers("R1")
def test_a_claude_error_reaches_the_pipeline_as_an_error_not_as_figures(tmp_path):
    client, _ = fake_claude(tmp_path, {**REPLY, "is_error": True, "result": "usage limit reached"})

    with pytest.raises(RuntimeError, match="usage limit reached"):
        client.complete("prompt")


@pytest.mark.covers("R1")
def test_claude_output_that_is_not_json_is_an_error(tmp_path):
    client, _ = fake_claude(tmp_path, "Not logged in")

    with pytest.raises(RuntimeError, match="claude exited"):
        client.complete("prompt")


class StubServer(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):
        StubServer.received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        body = json.dumps({"choices": [{"message": {"content": answer()}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_model():
    server = HTTPServer(("127.0.0.1", 0), StubServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    StubServer.received = []
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


@pytest.mark.covers("R1")
def test_a_local_model_answer_flows_through_the_pipeline(local_model):
    pipeline = LoanPipeline(OpenAICompatibleClient(local_model, "output"))
    case = pipeline.open_case({}, [Doc("payslip.txt", "Net pay: 3,200")], 12000)

    assert pipeline.analyse(case.id).proposal.decision == "approve"
    request = StubServer.received[0]
    assert (request["model"], request["temperature"]) == ("output", 0)
    assert request["messages"][-1]["content"].endswith("Net pay: 3,200\n=== END ===")


@pytest.mark.covers("R6")
def test_an_unreachable_model_sends_the_case_to_a_person():
    pipeline = LoanPipeline(OpenAICompatibleClient("http://127.0.0.1:9/v1", "output", timeout_s=2))
    case = pipeline.open_case({}, [Doc("payslip.txt", "Net pay: 3,200")], 12000)

    pipeline.analyse(case.id)

    assert case.status == "needs_attention"
    assert case.problem.startswith("AI call failed: RuntimeError: cannot reach")
    assert case.proposal is None
