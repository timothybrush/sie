"""Unit tests for core.runtime_options.merge_runtime_options.

This is the single merge used by BOTH the single-server HTTP path
(api.options.resolve_runtime_options) and the cluster queue worker
(queue_executor.process_encode_batch). Regression coverage for #1489: the
worker path historically forwarded raw SDK options and dropped profile
``adapter_options.runtime`` defaults (query_template / default_instruction /
pooling / normalize) for every queued request.
"""

from __future__ import annotations

import pytest
from sie_server.config.model import ModelConfig
from sie_server.core.runtime_options import apply_generation_runtime_options, merge_runtime_options


def _embedder_config() -> ModelConfig:
    """An instruction-tuned embedder whose prompt lives in profile runtime."""
    return ModelConfig.model_validate(
        {
            "sie_id": "test/instruct-embedder",
            "hf_id": "test/instruct-embedder",
            "inputs": {"text": True},
            "tasks": {"encode": {"dense": {"dim": 8}}},
            "max_sequence_length": 512,
            "profiles": {
                "default": {
                    "max_batch_tokens": 8192,
                    "adapter_path": "sie_server.adapters.sglang.embedding:SGLangEmbeddingAdapter",
                    "adapter_options": {
                        "runtime": {
                            "pooling": "last_token",
                            "normalize": True,
                            "query_template": "Instruct: {instruction}\nQuery: {text}",
                            "default_instruction": "Given a query, retrieve relevant passages",
                        },
                    },
                },
                "alt": {
                    "max_batch_tokens": 8192,
                    "adapter_path": "sie_server.adapters.sglang.embedding:SGLangEmbeddingAdapter",
                    "adapter_options": {"runtime": {"query_template": "alt: {text}"}},
                },
            },
        }
    )


def test_merges_profile_runtime_under_request_options() -> None:
    """Profile runtime defaults appear even when the request only sends is_query."""
    config = _embedder_config()
    merged = merge_runtime_options(config, {"is_query": True})

    assert merged["query_template"] == "Instruct: {instruction}\nQuery: {text}"
    assert merged["default_instruction"] == "Given a query, retrieve relevant passages"
    assert merged["pooling"] == "last_token"
    assert merged["normalize"] is True
    # Request-supplied key is preserved alongside the merged defaults.
    assert merged["is_query"] is True


def test_request_options_win_over_runtime_defaults() -> None:
    """Per-request overrides take precedence over profile runtime defaults."""
    config = _embedder_config()
    merged = merge_runtime_options(config, {"query_template": "custom: {text}", "normalize": False})

    assert merged["query_template"] == "custom: {text}"
    assert merged["normalize"] is False


def test_none_request_options_yield_profile_defaults() -> None:
    """An empty/None request still receives the profile's runtime defaults."""
    config = _embedder_config()
    merged = merge_runtime_options(config, None)

    assert merged["query_template"] == "Instruct: {instruction}\nQuery: {text}"
    assert merged["default_instruction"] == "Given a query, retrieve relevant passages"


def test_profile_key_selects_profile_and_is_consumed() -> None:
    """The 'profile' key chooses the profile and is not forwarded to the adapter."""
    config = _embedder_config()
    merged = merge_runtime_options(config, {"profile": "alt", "is_query": True})

    assert merged["query_template"] == "alt: {text}"
    assert "profile" not in merged
    assert merged["is_query"] is True


def test_unknown_profile_raises_value_error() -> None:
    """An unknown profile name surfaces as ValueError (handled by callers)."""
    config = _embedder_config()
    with pytest.raises(ValueError, match="nope"):
        merge_runtime_options(config, {"profile": "nope"})


def _generation_config() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "sie_id": "test/generator",
            "hf_id": "test/generator",
            "inputs": {"text": True},
            "tasks": {"generate": {"context_length": 4096, "max_output_tokens": 512}},
            "max_sequence_length": 4096,
            "profiles": {
                "default": {
                    "max_batch_tokens": 4096,
                    "kv_budget_tokens": 2048,
                    "adapter_path": "sie_server.adapters.fake.adapter:FakeAdapter",
                    "adapter_options": {
                        "runtime": {
                            "default_sampling": {"temperature": 0.7, "top_p": 0.8},
                            "stop_tokens": ["</s>"],
                            "overall_timeout_s": 60,
                        }
                    },
                }
            },
        }
    )


def test_generation_runtime_defaults_apply_below_typed_fields() -> None:
    resolved = apply_generation_runtime_options(
        _generation_config(),
        {"profile": "default"},
        {"prompt": "hi", "temperature": 1.0, "stop": ["DONE"]},
    )

    assert resolved["temperature"] == 1.0
    assert resolved["top_p"] == 0.8
    assert resolved["stop"] == ["DONE", "</s>"]
    assert "profile" not in resolved


def test_generation_request_runtime_overrides_profile_defaults() -> None:
    resolved = apply_generation_runtime_options(
        _generation_config(),
        {"default_sampling": {"temperature": 0.2}},
        {"prompt": "hi"},
    )

    assert resolved["temperature"] == 0.2
    assert resolved["top_p"] == 0.8


def test_generation_non_default_profile_requires_model_variant_identity() -> None:
    with pytest.raises(ValueError, match="model:profile"):
        apply_generation_runtime_options(
            _generation_config(),
            {"profile": "fast"},
            {"prompt": "hi"},
        )


def test_generation_unknown_option_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported generation option"):
        apply_generation_runtime_options(
            _generation_config(),
            {"not_executable": True},
            {"prompt": "hi"},
        )


@pytest.mark.parametrize(
    "sampling",
    [
        {"temperature": "0.7"},
        {"top_p": None},
        {"presence_penalty": float("inf")},
        {"top_k": True},
        {"min_new_tokens": -1},
    ],
)
def test_generation_invalid_sampling_option_fails_closed(sampling: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="invalid value"):
        apply_generation_runtime_options(
            _generation_config(),
            {"default_sampling": sampling},
            {"prompt": "hi"},
        )


def test_generation_non_finite_timeout_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive number"):
        apply_generation_runtime_options(_generation_config(), {"overall_timeout_s": float("inf")}, {"prompt": "hi"})
