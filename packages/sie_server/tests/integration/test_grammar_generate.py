"""Grammar integration tests: SIE-native ``/v1/generate/{model}`` with
``grammar``.

Skipped unless ``SIE_GATEWAY_URL`` is set. Mirrors the acceptance
criteria for structured outputs:

* §5.7 — moderate JSON Schema returns JSON conforming to the schema.
* §5.9 — two identical-schema requests show exactly 1 compile + 1 hit
  in the worker's Prometheus metrics.
* §5.10 — a pathologically nested schema returns 400 ``grammar_invalid``
  *before* any compile metric ticks.

The chat-side OpenAI SDK acceptance test (§5.8) lives in
``test_chat_completions.py`` next to the chat-completions fixtures.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

pytestmark = pytest.mark.integration

_GATEWAY_URL_ENV = "SIE_GATEWAY_URL"
_WORKER_METRICS_URL_ENV = "SIE_WORKER_METRICS_URL"
_GEN_MODEL_ENV = "SIE_GEN_MODEL"


def _gateway_url() -> str:
    url = os.environ.get(_GATEWAY_URL_ENV)
    if not url:
        pytest.skip(f"set {_GATEWAY_URL_ENV} to run grammar generate integration tests")
    return url.rstrip("/")


def _gen_model() -> str:
    # SIE-safe path form (double underscore separator, matching the
    # gateway's /v1/generate/{model} contract).
    return os.environ.get(_GEN_MODEL_ENV, "Qwen__Qwen3-4B-Instruct-2507")


def _worker_metrics() -> str | None:
    return os.environ.get(_WORKER_METRICS_URL_ENV)


@pytest.fixture(scope="module")
def gateway_url() -> str:
    return _gateway_url()


# -----------------------------------------------------------------------------
# §5.7 — JSON Schema round-trip
# -----------------------------------------------------------------------------


def test_generate_with_json_schema_returns_conforming_json(gateway_url: str) -> None:
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "year": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "year"],
        "additionalProperties": False,
    }
    body = {
        "prompt": ("Return JSON describing a fictional book with a title, a year, and 1-3 tags."),
        "max_new_tokens": 128,
        "grammar": {"json_schema": schema},
    }
    r = httpx.post(
        f"{gateway_url}/v1/generate/{_gen_model()}",
        json=body,
        timeout=60.0,
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    text = payload.get("text") or payload.get("choices", [{}])[0].get("text", "")
    assert text, f"unexpected response shape: {payload!r}"
    parsed = json.loads(text)
    assert "title" in parsed
    assert isinstance(parsed["title"], str)
    assert "year" in parsed
    assert isinstance(parsed["year"], int)


# -----------------------------------------------------------------------------
# §5.10 — safety cap fast-fails before compile
# -----------------------------------------------------------------------------


def _read_metric(text: str, name: str, labels: dict[str, str] | None = None) -> float | None:
    """Parse a single Prometheus counter / histogram count line.

    Returns ``None`` when the metric (or label combination) is absent.
    """
    if labels is None:
        labels = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not line.startswith(name):
            continue
        if "{" in line:
            label_part = line[line.index("{") + 1 : line.index("}")]
            parsed: dict[str, str] = {}
            for entry in label_part.split(","):
                if "=" not in entry:
                    continue
                k, v = entry.split("=", 1)
                parsed[k.strip()] = v.strip().strip('"')
            if not all(parsed.get(k) == v for k, v in labels.items()):
                continue
        elif labels:
            continue
        try:
            return float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            return None
    return None


def test_pathological_schema_rejects_before_compile(gateway_url: str) -> None:
    """A deeply-nested schema → 400 ``invalid_request`` and no worker
    compile activity. Confirms the gateway is the authority for safety
    caps.
    """
    metrics_url = _worker_metrics()
    before: float | None = None
    if metrics_url:
        m = httpx.get(metrics_url, timeout=10.0).text
        before = _read_metric(m, "sie_worker_grammar_compile_seconds_count") or 0.0

    deep: dict = {"type": "string"}
    for _ in range(25):
        deep = {"type": "object", "properties": {"nested": deep}}
    body = {
        "prompt": "Hi",
        "max_new_tokens": 8,
        "grammar": {"json_schema": deep},
    }
    r = httpx.post(
        f"{gateway_url}/v1/generate/{_gen_model()}",
        json=body,
        timeout=10.0,
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["param"].startswith("grammar.json_schema")

    if metrics_url and before is not None:
        m = httpx.get(metrics_url, timeout=10.0).text
        after = _read_metric(m, "sie_worker_grammar_compile_seconds_count") or 0.0
        assert after == before, f"worker observed a compile after a safety-cap reject: {before} -> {after}"


# -----------------------------------------------------------------------------
# §5.9 — identical schema → 1 compile + 1 cache hit
# -----------------------------------------------------------------------------


def test_cache_hit_after_first_compile(gateway_url: str) -> None:
    metrics_url = _worker_metrics()
    if not metrics_url:
        pytest.skip(f"set {_WORKER_METRICS_URL_ENV} to assert the worker cache counters")

    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    body = {
        "prompt": 'Reply with JSON: {"value": "ok"}',
        "max_new_tokens": 32,
        "grammar": {"json_schema": schema},
    }

    def _send() -> None:
        r = httpx.post(
            f"{gateway_url}/v1/generate/{_gen_model()}",
            json=body,
            timeout=60.0,
        )
        assert r.status_code == 200, r.text

    # Snapshot, send twice, snapshot. The worker should observe one
    # additional compile observation and (at least) one additional
    # cache hit. Tests run within a single worker so the labels match.
    before = httpx.get(metrics_url, timeout=10.0).text
    misses_before = _read_metric(before, "sie_worker_grammar_cache_misses_total") or 0.0
    hits_before = _read_metric(before, "sie_worker_grammar_cache_hits_total") or 0.0

    _send()
    _send()

    after = httpx.get(metrics_url, timeout=10.0).text
    misses_after = _read_metric(after, "sie_worker_grammar_cache_misses_total") or 0.0
    hits_after = _read_metric(after, "sie_worker_grammar_cache_hits_total") or 0.0

    assert misses_after - misses_before == 1.0, (
        f"expected exactly one new compile, observed {misses_after - misses_before}"
    )
    assert hits_after - hits_before >= 1.0, f"expected at least one cache hit, observed {hits_after - hits_before}"


# -----------------------------------------------------------------------------
# Mutual exclusivity (cross-cuts §5.5 unit test but worth an integration probe)
# -----------------------------------------------------------------------------


def test_mutex_violation_rejects_at_gateway(gateway_url: str) -> None:
    body = {
        "prompt": "Hi",
        "max_new_tokens": 8,
        "grammar": {"json_schema": {"type": "object"}, "regex": "[a-z]+"},
    }
    r = httpx.post(
        f"{gateway_url}/v1/generate/{_gen_model()}",
        json=body,
        timeout=10.0,
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["param"] == "grammar"
    assert "mutually exclusive" in err["message"]
