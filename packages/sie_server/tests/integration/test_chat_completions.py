"""Integration tests for the ``/v1/chat/completions`` gateway endpoint.

These tests require a running Rust gateway + at least one worker subscribed
to the JetStream pool that has the target model loaded. They are skipped
unless ``SIE_GATEWAY_URL`` points to a reachable gateway.

Run via ``mise run test -- -i`` after starting a local gateway with::

    mise run gateway-build -- -r
    target/release/sie-gateway --nats-url ... &
    mise run serve -- -m Qwen/Qwen3-4B-Instruct-2507

Acceptance criteria:

* official OpenAI Python SDK round-trip
* unsupported field rejection with OpenAI envelope (``frequency_penalty``,
  ``response_format`` grammar boundary)
* unknown model surfaces as 404 with OpenAI envelope
* ``/v1/generate`` shares the OpenAI envelope on errors (criterion 7)
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

_GATEWAY_URL_ENV = "SIE_GATEWAY_URL"
_CHAT_MODEL_ENV = "SIE_CHAT_MODEL"


def _gateway_url() -> str:
    url = os.environ.get(_GATEWAY_URL_ENV)
    if not url:
        pytest.skip(f"set {_GATEWAY_URL_ENV} to run chat-completions integration tests")
    return url.rstrip("/")


def _chat_model() -> str:
    return os.environ.get(_CHAT_MODEL_ENV, "Qwen/Qwen3-4B-Instruct-2507")


@pytest.fixture(scope="module")
def gateway_url() -> str:
    return _gateway_url()


def test_official_openai_python_sdk_round_trip(gateway_url: str) -> None:
    """OpenAI Python SDK against the gateway returns a valid completion."""
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=f"{gateway_url}/v1", api_key="sk-not-used")
    resp = client.chat.completions.create(
        model=_chat_model(),
        messages=[{"role": "user", "content": "Say hi in one word."}],
        max_completion_tokens=16,
    )
    assert resp.choices, "expected at least one choice"
    msg = resp.choices[0].message
    assert msg.role == "assistant"
    assert msg.content, "expected non-empty assistant content"
    assert resp.usage.completion_tokens > 0


def test_frequency_penalty_nonzero_accepted(gateway_url: str) -> None:
    """Penalties in OpenAI's ``[-2.0, 2.0]`` range are validated by the
    gateway and forwarded to the SGLang sampler (``ChatRequestParams``
    plumbing in ``packages/sie_gateway/src/handlers/proxy.rs``). Prior
    the gateway used to 400-reject non-zero values; the regression
    guard here flips to "request must succeed" so we don't silently
    re-introduce the reject.
    """
    body = {
        "model": _chat_model(),
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 8,
        "frequency_penalty": 0.5,
    }
    r = httpx.post(f"{gateway_url}/v1/chat/completions", json=body, timeout=30.0)
    assert r.status_code == 200, r.text


def test_response_format_json_object_accepted(gateway_url: str) -> None:
    """Loose ``json_object`` mode is now accepted and backed by a
    built-in generic JSON schema (``{type: object, additionalProperties:
    true}``) labelled ``"json_object"`` in cache observability. Prior
    behaviour returned 400; this regression guard flips it.
    """
    body = {
        "model": _chat_model(),
        "messages": [
            {"role": "user", "content": "Return a JSON object with key 'ok' set to true."},
        ],
        "max_completion_tokens": 32,
        "response_format": {"type": "json_object"},
    }
    r = httpx.post(f"{gateway_url}/v1/chat/completions", json=body, timeout=30.0)
    assert r.status_code == 200, r.text


def test_response_format_json_schema_round_trip(gateway_url: str) -> None:
    """Grammar acceptance criterion: OpenAI SDK with
    ``response_format.type='json_schema'`` returns a ``ChatCompletion``
    whose ``content`` parses as the schema.
    """
    import json as _json

    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=f"{gateway_url}/v1", api_key="sk-not-used")
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["answer", "rationale"],
        "additionalProperties": False,
    }
    resp = client.chat.completions.create(
        model=_chat_model(),
        messages=[
            {"role": "system", "content": "Always respond using the JSON schema provided."},
            {"role": "user", "content": "What is 6 times 7? Reply as JSON."},
        ],
        max_completion_tokens=128,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "math_response",
                "strict": True,
                "schema": schema,
            },
        },
    )
    assert resp.choices, "expected at least one choice"
    content = resp.choices[0].message.content
    assert isinstance(content, str)
    assert content, "expected non-empty content"
    parsed = _json.loads(content)
    assert "answer" in parsed
    assert "rationale" in parsed
    assert isinstance(parsed["answer"], (int, float))


def test_response_format_pathological_depth_returns_400(gateway_url: str) -> None:
    """§5.10 — schema deeper than the safety cap returns 400
    ``grammar_invalid`` (``invalid_request`` code) *before* the gateway
    publishes any inference work. The exact ``param`` path mentions
    ``grammar.json_schema``.
    """
    # Build a schema with depth well past MAX_SCHEMA_DEPTH (16).
    leaf: dict = {"type": "string"}
    for _ in range(25):
        leaf = {"type": "object", "properties": {"x": leaf}}
    body = {
        "model": _chat_model(),
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 8,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "deep", "schema": leaf},
        },
    }
    r = httpx.post(f"{gateway_url}/v1/chat/completions", json=body, timeout=30.0)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["param"].startswith("grammar.json_schema")


def test_unknown_model_returns_404_openai_shape(gateway_url: str) -> None:
    body = {
        "model": "definitely-not-a-real-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 8,
    }
    r = httpx.post(f"{gateway_url}/v1/chat/completions", json=body, timeout=30.0)
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["type"] == "model_not_found"
    assert err["code"] == "model_not_found"
    assert err["param"] == "model"


def test_generate_endpoint_returns_openai_error_envelope(gateway_url: str) -> None:
    """Acceptance criterion 7 — OpenAI-compatible envelope on /v1/generate too."""
    body = {
        "prompt": "Hi",
        "max_new_tokens": 8,
        "wat": True,  # unknown field — parser ignores, but model lookup gives a clean test
    }
    # Use a deliberately-missing model so we hit the OpenAI envelope on a 404.
    safe_model = "definitely-not-a-real-model"
    r = httpx.post(
        f"{gateway_url}/v1/generate/{safe_model}",
        json=body,
        timeout=30.0,
    )
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["type"] == "model_not_found"
    assert err["code"] == "model_not_found"
