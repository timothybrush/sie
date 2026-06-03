"""Tests for the direct ``/v1/generate/{model}`` HTTP route (walking-skeleton local-dev path).

Mirrors :mod:`tests.api.test_score` but targets the new local-dev path that
calls the adapter directly (no NATS, no gateway). The gateway-side handler
``proxy_generate`` is covered by Rust inline tests in
``packages/sie_gateway/src/handlers/proxy.rs``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.adapters._generation_base import GenerationAdapter, GenerationChunk
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters.base import ModelCapabilities, ModelDims
from sie_server.api.generate import router as generate_router
from sie_server.config.model import (
    GenerateCapabilities,
    GenerateTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.registry import ModelRegistry


class _FakeGenAdapter(GenerationAdapter):
    """Minimal in-memory GenerationAdapter for route tests."""

    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    def __init__(self) -> None:
        self._device = None
        self.last_call: dict | None = None

    def load(self, device: str) -> None:  # pragma: no cover — registry-mocked
        self._device = device

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(inputs=["text"], outputs=["tokens"])

    @property
    def dims(self) -> ModelDims:
        return ModelDims()

    # The terminal finish_reason the fake yields; tests flip this to
    # exercise the route's error/cancelled → non-200 mapping (BUG: a
    # terminal error/cancelled chunk must NOT become an HTTP 200).
    finish_reason: str = "stop"

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        self.last_call = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "seed": seed,
            "logit_bias": logit_bias,
        }
        # Yield one delta + a terminal chunk so the local-dev route can
        # drain the iterator into the walking-skeleton-shaped aggregate response.
        yield GenerationChunk(text_delta=f"echo:{prompt}", is_first=True)
        yield GenerationChunk(
            text_delta="",
            done=True,
            finish_reason=self.finish_reason,  # type: ignore[arg-type]
            prompt_tokens=len(prompt.split()),
            completion_tokens=2,
        )


def _make_config() -> ModelConfig:
    return ModelConfig(
        sie_id="Qwen/Qwen3-4B-Instruct",
        hf_id="Qwen/Qwen3-4B-Instruct",
        tasks=Tasks(
            generate=GenerateTask(
                context_length=32768,
                max_output_tokens=4096,
                capabilities=GenerateCapabilities(),
            ),
        ),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                max_batch_tokens=16384,
                kv_budget_tokens=8192,
            ),
        },
    )


@pytest.fixture
def fake_adapter() -> _FakeGenAdapter:
    return _FakeGenAdapter()


@pytest.fixture
def registry(fake_adapter: _FakeGenAdapter) -> MagicMock:
    reg = MagicMock(spec=ModelRegistry)
    reg.has_model.return_value = True
    reg.is_loaded.return_value = True
    reg.is_loading.return_value = False
    reg.is_unloading.return_value = False
    reg.is_failed.return_value = False
    reg.get_failure.return_value = None
    reg.get.return_value = fake_adapter
    reg.get_config.return_value = _make_config()
    reg.device = "cpu"
    reg.engine_config = None
    # Required by ``ensure_loaded`` short-circuit when already loaded.
    return reg


@pytest.fixture
def client(registry: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(generate_router)
    app.state.registry = registry
    return TestClient(app)


class TestGenerateEndpoint:
    def test_happy_path_returns_text_finish_reason_usage(
        self, client: TestClient, fake_adapter: _FakeGenAdapter
    ) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hello", "max_new_tokens": 32, "temperature": 0.7, "top_p": 0.9},
        )
        assert response.status_code == 200
        data = response.json()
        # Response echoes the canonical (slash-form) model id, not the raw
        # ``__``-form path param, so it round-trips with what the SDK sent.
        assert data["model"] == "Qwen/Qwen3-4B-Instruct"
        assert data["text"] == "echo:Hello"
        assert data["finish_reason"] == "stop"
        assert data["usage"]["completion_tokens"] == 2
        assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + 2

        # Adapter received the parsed sampling params verbatim.
        assert fake_adapter.last_call == {
            "prompt": "Hello",
            "max_new_tokens": 32,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": None,
            "seed": None,
            "logit_bias": None,
        }

    def test_registry_lookup_uses_denormalized_slash_key(self, client: TestClient, registry: MagicMock) -> None:
        # Regression: the registry keys on the canonical slash ``sie_id``
        # (``ModelConfig.name``), so the ``__`` path segment must be
        # denormalized before lookup or every real model 404s.
        response = client.post(
            "/v1/generate/Qwen__Qwen3.5-4B",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 200
        registry.has_model.assert_called_with("Qwen/Qwen3.5-4B")
        registry.get_config.assert_called_with("Qwen/Qwen3.5-4B")
        registry.get.assert_called_with("Qwen/Qwen3.5-4B")

    def test_slash_in_model_path_returns_400_with_suggestion(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen/Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 400
        body = response.json()
        # The suggested SIE-safe id should appear in the message.
        assert "Qwen__Qwen3-4B-Instruct" in body["detail"]["message"]

    def test_unknown_model_returns_404(self, client: TestClient, registry: MagicMock) -> None:
        registry.has_model.return_value = False
        response = client.post(
            "/v1/generate/unknown__model",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 404

    def test_missing_prompt_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"max_new_tokens": 8},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["param"] == "prompt"

    def test_zero_max_new_tokens_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 0},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["param"] == "max_new_tokens"

    def test_max_new_tokens_exceeds_cap_returns_400(self, client: TestClient) -> None:
        # The config caps at 4096; ask for 5000.
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 5000},
        )
        assert response.status_code == 400
        body = response.json()
        assert body["detail"]["code"] == "context_exceeded"
        assert body["detail"]["param"] == "max_new_tokens"

    def test_unsupported_field_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "tools": []},
        )
        assert response.status_code == 400
        body = response.json()
        assert body["detail"]["code"] == "unsupported_field"
        assert body["detail"]["param"] == "tools"

    def test_non_generation_adapter_returns_400(self, client: TestClient, registry: MagicMock) -> None:
        # Registry returns a non-GenerationAdapter (e.g. an embedding adapter).
        registry.get.return_value = MagicMock(spec=[])  # plain MagicMock — not GenerationAdapter
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 400

    def test_model_loading_returns_503(self, client: TestClient, registry: MagicMock) -> None:
        registry.is_loading.return_value = True
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 503

    def test_stop_must_be_list_of_strings(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "stop": "not-a-list"},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["param"] == "stop"

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            ("temperature", "0.7"),
            ("temperature", True),
            ("top_p", "0.9"),
            ("top_p", False),
        ],
    )
    def test_sampling_params_must_be_json_numbers(self, client: TestClient, param: str, value: object) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, param: value},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["param"] == param

    def test_response_model_is_canonical_slash_id(self, client: TestClient) -> None:
        # The request path uses the SIE-safe ``__`` form, but the response
        # ``model`` field must be the canonical slash id so it round-trips
        # with what the SDK sent.
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 200
        assert response.json()["model"] == "Qwen/Qwen3-4B-Instruct"

    def test_oversized_prompt_returns_413(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        # Shrink the cap so the test doesn't have to build a 4 MiB string.
        monkeypatch.setattr("sie_server.api.generate._MAX_PROMPT_BYTES", 16)
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "x" * 64, "max_new_tokens": 8},
        )
        assert response.status_code == 413
        body = response.json()
        assert body["detail"]["param"] == "prompt"
        assert body["detail"]["code"] == "INPUT_TOO_LONG"

    # ── BUG 12: penalty / grammar validation (gateway parity) ──────────

    @pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
    @pytest.mark.parametrize("value", [999, -999, "x", True])
    def test_penalty_out_of_range_or_wrong_type_returns_400(
        self, client: TestClient, field: str, value: object
    ) -> None:
        """BUG 12: penalties must be validated identically to the gateway —
        finite number in [-2.0, 2.0]; reject out-of-range / string / bool.
        Previously these were whitelisted but never validated → 200.
        """
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, field: value},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == field

    @pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_penalty_nan_inf_returns_400(self, client: TestClient, field: str, literal: str) -> None:
        """NaN / inf (non-finite) penalties reject with 400 (gateway parity)."""
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            data=f'{{"prompt": "Hi", "max_new_tokens": 8, "{field}": {literal}}}',
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == field

    @pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
    def test_valid_penalty_is_accepted(self, client: TestClient, field: str) -> None:
        """A valid penalty (0.5) is accepted (200) and dropped, not forwarded."""
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, field: 0.5},
        )
        assert response.status_code == 200, response.text

    def test_grammar_malformed_returns_400(self, client: TestClient) -> None:
        """BUG 12: a grammar not matching the basic gateway shape (must be a
        JSON object with exactly one of json_schema/regex/ebnf) → 400.
        """
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "grammar": "not-an-object"},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == "grammar"

    def test_grammar_mutually_exclusive_variants_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={
                "prompt": "Hi",
                "max_new_tokens": 8,
                "grammar": {"regex": "[a-z]+", "ebnf": 'root ::= "x"'},
            },
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == "grammar"

    def test_grammar_valid_regex_is_accepted(self, client: TestClient) -> None:
        """A well-shaped grammar is accepted (200) — it's dropped by the dev
        route, but it must not be rejected.
        """
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "grammar": {"regex": "[a-z]+"}},
        )
        assert response.status_code == 200, response.text

    def test_prompt_at_cap_is_accepted(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        # A prompt exactly at the byte cap is allowed (boundary check).
        monkeypatch.setattr("sie_server.api.generate._MAX_PROMPT_BYTES", 16)
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "x" * 16, "max_new_tokens": 8},
        )
        assert response.status_code == 200

    # ── FIX 4: adapter-supported fields (seed / logit_bias / logprobs /
    # top_logprobs) are whitelisted and validated, not 400'd as
    # unsupported_field ──────────────────────────────────────────────

    def test_seed_is_accepted_and_forwarded(self, client: TestClient, fake_adapter: _FakeGenAdapter) -> None:
        """``seed`` is whitelisted (the adapter forwards it) and reaches the
        adapter — previously a schema-compliant ``seed`` body 400'd as
        ``unsupported_field``.
        """
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "seed": 1234},
        )
        assert response.status_code == 200, response.text
        assert fake_adapter.last_call is not None
        assert fake_adapter.last_call["seed"] == 1234

    @pytest.mark.parametrize("value", ["x", 1.5, True])
    def test_seed_wrong_type_returns_400(self, client: TestClient, value: object) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "seed": value},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == "seed"

    def test_logit_bias_is_accepted_and_forwarded(self, client: TestClient, fake_adapter: _FakeGenAdapter) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "logit_bias": {"123": 1.5, "456": -2.0}},
        )
        assert response.status_code == 200, response.text
        assert fake_adapter.last_call is not None
        assert fake_adapter.last_call["logit_bias"] == {"123": 1.5, "456": -2.0}

    @pytest.mark.parametrize(
        "value",
        [
            "not-an-object",
            {"abc": 1.0},  # non-integer key
            {"123": 999.0},  # out of [-100, 100]
            {"123": "x"},  # non-numeric value
        ],
    )
    def test_logit_bias_malformed_returns_400(self, client: TestClient, value: object) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "logit_bias": value},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == "logit_bias"

    def test_logprobs_is_accepted_and_dropped(self, client: TestClient) -> None:
        """``logprobs`` is validated then dropped (the blocking shape has no
        logprob field), so a valid request still 200s.
        """
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "logprobs": True, "top_logprobs": 5},
        )
        assert response.status_code == 200, response.text

    def test_logprobs_wrong_type_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "logprobs": "yes"},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == "logprobs"

    @pytest.mark.parametrize("value", [-1, 21, 1.5, True])
    def test_top_logprobs_out_of_range_returns_400(self, client: TestClient, value: object) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "logprobs": True, "top_logprobs": value},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == "top_logprobs"

    def test_top_logprobs_requires_logprobs_true(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "top_logprobs": 5},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == "top_logprobs"

    # ── FIX 5: a terminal finish_reason of error / cancelled must NOT be
    # an HTTP 200 with partial text ──────────────────────────────────

    def test_terminal_error_finish_reason_returns_500(self, client: TestClient, fake_adapter: _FakeGenAdapter) -> None:
        """A stream that ends with ``finish_reason="error"`` (adapter caught
        an upstream failure and surfaced it as a terminal chunk rather than
        raising) must map to HTTP 500, not a 200 with partial text.
        """
        fake_adapter.finish_reason = "error"
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 500, response.text
        assert response.json()["detail"]["code"] == "inference_error"

    def test_terminal_cancelled_finish_reason_returns_503(
        self, client: TestClient, fake_adapter: _FakeGenAdapter
    ) -> None:
        """A stream that ends with ``finish_reason="cancelled"`` must map to a
        non-2xx (503), not a 200 with partial text.
        """
        fake_adapter.finish_reason = "cancelled"
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "generation_cancelled"

    def test_terminal_stop_finish_reason_still_returns_200(self, client: TestClient) -> None:
        """Sanity: the normal ``stop`` terminator is unaffected by the
        error/cancelled mapping and still 200s.
        """
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 200, response.text
