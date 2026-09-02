"""Tests for capx.llm.client's OpenAI-SDK-backed query functions.

Spins up a tiny local HTTP server that mimics an OpenAI-API-compatible
endpoint (both the Chat Completions and Responses surfaces) so we can verify
payload construction, response parsing, env-var fallback, and retry behavior
without depending on any real provider.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import pytest

from capx.llm.client import ModelQueryArgs, query_model


class _FakeHandler(BaseHTTPRequestHandler):
    # Overridden per-test via class attributes set by the fixture.
    responses: list[tuple[int, dict]] = []
    received: list[dict] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        self.received.append(
            {
                "path": self.path,
                "body": body,
                "headers": dict(self.headers),
            }
        )
        status, payload = self.responses.pop(0)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, format, *args):  # noqa: A002
        pass


@pytest.fixture()
def fake_server():
    _FakeHandler.responses = []
    _FakeHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _FakeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield base_url, _FakeHandler
    finally:
        server.shutdown()
        thread.join()


CHAT_COMPLETION_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello from chat"},
            "finish_reason": "stop",
        }
    ],
}

RESPONSES_API_RESPONSE = {
    "id": "resp-1",
    "object": "response",
    "created_at": 0,
    "model": "test-model",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg-1",
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "hello from responses", "annotations": []}
            ],
        }
    ],
}


def test_query_model_chat_wire_payload_and_parsing(fake_server):
    base_url, handler = fake_server
    handler.responses = [(200, CHAT_COMPLETION_RESPONSE)]

    args = ModelQueryArgs(model="test-model", server_url=base_url, wire="chat", max_tokens=123)
    out = query_model(args, [{"role": "user", "content": "hi"}])

    assert out["content"] == "hello from chat"
    req = handler.received[0]
    assert req["path"] == "/chat/completions"
    assert req["body"]["model"] == "test-model"
    assert req["body"]["max_tokens"] == 123
    # temperature/reasoning_effort must NOT be sent unless explicitly set
    assert "temperature" not in req["body"]
    assert "reasoning_effort" not in req["body"]


def test_query_model_chat_wire_optional_params_included_when_set(fake_server):
    base_url, handler = fake_server
    handler.responses = [(200, CHAT_COMPLETION_RESPONSE)]

    args = ModelQueryArgs(
        model="test-model",
        server_url=base_url,
        wire="chat",
        temperature=0.7,
        reasoning_effort="medium",
    )
    query_model(args, [{"role": "user", "content": "hi"}])

    body = handler.received[0]["body"]
    assert body["temperature"] == 0.7
    assert body["reasoning_effort"] == "medium"


def test_query_model_responses_wire(fake_server):
    base_url, handler = fake_server
    handler.responses = [(200, RESPONSES_API_RESPONSE)]

    args = ModelQueryArgs(model="test-model", server_url=base_url, wire="responses")
    prompt = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    out = query_model(args, prompt)

    assert out["content"] == "hello from responses"
    assert out["reasoning"] is None
    req = handler.received[0]
    assert req["path"] == "/responses"
    # prompt content should have been converted to responses-api input_text
    assert req["body"]["input"][0]["content"][0]["type"] == "input_text"


def test_query_model_api_key_header(fake_server):
    base_url, handler = fake_server
    handler.responses = [(200, CHAT_COMPLETION_RESPONSE)]

    args = ModelQueryArgs(model="test-model", server_url=base_url, wire="chat", api_key="secret-key")
    query_model(args, [{"role": "user", "content": "hi"}])

    assert handler.received[0]["headers"]["Authorization"] == "Bearer secret-key"


def test_query_model_env_fallback(fake_server):
    base_url, handler = fake_server
    handler.responses = [(200, CHAT_COMPLETION_RESPONSE)]

    args = ModelQueryArgs(model="test-model")  # server_url/api_key/wire all unset
    with mock.patch.dict(
        "os.environ",
        {"OPENAI_BASE_URL": base_url, "OPENAI_API_KEY": "env-key", "CAPX_LLM_WIRE": "chat"},
    ):
        query_model(args, [{"role": "user", "content": "hi"}])

    assert handler.received[0]["headers"]["Authorization"] == "Bearer env-key"


def test_query_model_retries_on_5xx_then_succeeds(fake_server):
    base_url, handler = fake_server
    handler.responses = [
        (500, {"error": {"message": "boom"}}),
        (200, CHAT_COMPLETION_RESPONSE),
    ]

    args = ModelQueryArgs(model="test-model", server_url=base_url, wire="chat")
    with mock.patch("time.sleep") as mock_sleep:
        out = query_model(args, [{"role": "user", "content": "hi"}])

    assert out["content"] == "hello from chat"
    assert len(handler.received) == 2
    mock_sleep.assert_called_once()
