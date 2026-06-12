"""Tests for the pool isolation validator.

A worker's pool may hold either generation models OR non-generation
models (encode/score/extract) — not both. The validator runs at
config-load and at :meth:`ModelRegistry.add_config` time.
"""

from __future__ import annotations

import pytest
from sie_server.config.model import (
    EmbeddingDim,
    EncodeTask,
    GenerateCapabilities,
    GenerateTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.pool_isolation import (
    PoolIsolationError,
    is_generation_model,
    validate_pool_isolation,
)
from sie_server.core.registry import ModelRegistry


def _gen_config(sie_id: str = "Qwen/Qwen3-4B-Instruct-2507") -> ModelConfig:
    return ModelConfig(
        sie_id=sie_id,
        hf_id=sie_id,
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


def _encode_config(sie_id: str = "BAAI/bge-m3") -> ModelConfig:
    return ModelConfig(
        sie_id=sie_id,
        hf_id=sie_id,
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=1024))),
        profiles={
            "default": ProfileConfig(adapter_path="mod:Encoder", max_batch_tokens=8192),
        },
    )


class TestIsGenerationModel:
    def test_gen_model_detected(self) -> None:
        assert is_generation_model(_gen_config()) is True

    def test_encode_model_not_gen(self) -> None:
        assert is_generation_model(_encode_config()) is False


class TestValidatePoolIsolation:
    def test_gen_into_non_gen_pool_rejects(self) -> None:
        with pytest.raises(PoolIsolationError) as exc:
            validate_pool_isolation(
                candidate_name="Qwen/Qwen3-4B-Instruct-2507",
                candidate_config=_gen_config(),
                existing_configs={"BAAI/bge-m3": _encode_config()},
                pool_name="p1",
            )
        msg = str(exc.value)
        assert "Qwen/Qwen3-4B-Instruct-2507" in msg
        assert "BAAI/bge-m3" in msg
        assert "p1" in msg
        assert "SIE_POOL" in msg

    def test_non_gen_into_gen_pool_rejects(self) -> None:
        with pytest.raises(PoolIsolationError) as exc:
            validate_pool_isolation(
                candidate_name="BAAI/bge-m3",
                candidate_config=_encode_config(),
                existing_configs={"Qwen/Qwen3-4B-Instruct-2507": _gen_config()},
                pool_name="p1",
            )
        msg = str(exc.value)
        assert "Qwen/Qwen3-4B-Instruct-2507" in msg
        assert "BAAI/bge-m3" in msg

    def test_mixed_pool_allowed_when_fairness_enabled(self, caplog) -> None:
        # With fairness opted-in, a mixed pool is intended: warn, don't raise.
        import logging

        with caplog.at_level(logging.WARNING):
            validate_pool_isolation(
                candidate_name="Qwen/Qwen3-4B-Instruct-2507",
                candidate_config=_gen_config(),
                existing_configs={"BAAI/bge-m3": _encode_config()},
                pool_name="p1",
                fairness_enabled=True,
            )
        assert any("fairness" in r.message.lower() for r in caplog.records)

    def test_gen_into_gen_pool_accepts(self) -> None:
        validate_pool_isolation(
            candidate_name="qwen-b",
            candidate_config=_gen_config("qwen-b"),
            existing_configs={"qwen-a": _gen_config("qwen-a")},
            pool_name="p1",
        )

    def test_non_gen_into_non_gen_pool_accepts(self) -> None:
        validate_pool_isolation(
            candidate_name="enc-b",
            candidate_config=_encode_config("enc-b"),
            existing_configs={"enc-a": _encode_config("enc-a")},
            pool_name="p1",
        )

    def test_re_registration_same_name_accepted(self) -> None:
        """Hot reload of the same model should not trip the validator."""
        validate_pool_isolation(
            candidate_name="qwen-a",
            candidate_config=_gen_config("qwen-a"),
            existing_configs={"qwen-a": _gen_config("qwen-a")},
            pool_name="p1",
        )


class TestRegistryHook:
    def test_add_config_gen_then_encode_rejects(self, tmp_path: object) -> None:
        registry = ModelRegistry(pool_name="p1")
        registry.add_config(_gen_config())
        with pytest.raises(PoolIsolationError):
            registry.add_config(_encode_config())

    def test_add_config_encode_then_gen_rejects(self) -> None:
        registry = ModelRegistry(pool_name="p1")
        registry.add_config(_encode_config())
        with pytest.raises(PoolIsolationError):
            registry.add_config(_gen_config())

    def test_add_config_no_pool_skips_validator(self) -> None:
        """When pool_name is None (tests/no-cluster), validator is skipped."""
        registry = ModelRegistry()
        registry.add_config(_gen_config())
        registry.add_config(_encode_config())  # should not raise

    def test_add_config_same_task_class_ok(self) -> None:
        registry = ModelRegistry(pool_name="p1")
        registry.add_config(_gen_config("qwen-a"))
        registry.add_config(_gen_config("qwen-b"))  # should not raise

    def test_failed_add_does_not_mutate_state(self) -> None:
        registry = ModelRegistry(pool_name="p1")
        registry.add_config(_gen_config())
        try:
            registry.add_config(_encode_config())
        except PoolIsolationError:
            pass
        # bge-m3 should not have been added
        assert "BAAI/bge-m3" not in registry.model_names
