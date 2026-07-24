from __future__ import annotations

from pathlib import Path

import yaml
from sie_server.config.model import ModelConfig

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "lightonai__GTE-ModernColBERT-v1.yaml"


def _config() -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(_MODEL_PATH.read_text()))


def test_default_and_muvera_preserve_published_retrieval_lengths() -> None:
    config = _config()

    for name in ("default", "muvera"):
        profile = config.resolve_profile(name)
        assert profile.loadtime["query_prefix"] == "[Q] "
        assert profile.loadtime["doc_prefix"] == "[D] "
        assert profile.loadtime["doc_punctuation_skiplist"] is True
        assert profile.runtime["query_max_length"] == 48
        assert profile.runtime["max_seq_length"] == 300


def test_long_context_changes_only_document_length() -> None:
    config = _config()
    default = config.resolve_profile("default")
    long_context = config.resolve_profile("long_context")

    assert long_context.loadtime == default.loadtime
    assert long_context.runtime == dict(default.runtime) | {"max_seq_length": 8192}
