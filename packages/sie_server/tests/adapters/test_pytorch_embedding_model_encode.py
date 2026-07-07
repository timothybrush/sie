"""use_model_encode delegation in PyTorchEmbeddingAdapter (#1692).

Models like NV-Embed-v2 ship a checkpoint-native ``encode(prompts,
instruction=...)`` implementing their full recipe (instruction prepend +
EOS append, right padding, internal latent-attention pooling with the
instruction masked out of the mean). These tests pin the delegation
contract: raw texts go to the model, the rendered template prefix rides
the ``instruction`` kwarg, and the adapter only normalizes on top.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter
from sie_server.types.inputs import Item


def _loaded_adapter(**kwargs: object) -> tuple[PyTorchEmbeddingAdapter, MagicMock]:
    adapter = PyTorchEmbeddingAdapter("test-model", use_model_encode=True, **kwargs)
    model = MagicMock()

    def fake_encode(texts, instruction="", max_length=None):
        return torch.arange(len(texts) * 4, dtype=torch.float32).reshape(len(texts), 4) + 1.0

    model.encode.side_effect = fake_encode
    adapter._model = model
    adapter._tokenizer = MagicMock()
    adapter._device = "cpu"
    adapter._dense_dim = 4
    return adapter, model


class TestUseModelEncode:
    def test_queries_pass_rendered_prefix_as_instruction(self) -> None:
        adapter, model = _loaded_adapter()

        out = adapter.encode(
            [Item(text="what is entropy"), Item(text="define enthalpy")],
            output_types=["dense"],
            is_query=True,
            instruction="Given a question, retrieve relevant documents",
            options={"query_template": "Instruct: {instruction}\nQuery: {text}"},
        )

        args, kwargs = model.encode.call_args
        assert args[0] == ["what is entropy", "define enthalpy"]
        assert kwargs["instruction"] == ("Instruct: Given a question, retrieve relevant documents\nQuery: ")
        assert out.batch_size == 2
        assert out.dense is not None
        assert out.dense.shape == (2, 4)

    def test_documents_get_empty_instruction(self) -> None:
        adapter, model = _loaded_adapter()

        adapter.encode(
            [Item(text="a passage")],
            output_types=["dense"],
            is_query=False,
            options={"query_template": "Instruct: {instruction}\nQuery: {text}"},
        )

        args, kwargs = model.encode.call_args
        assert args[0] == ["a passage"]
        assert kwargs["instruction"] == ""

    def test_instruction_without_template_still_passes_through(self) -> None:
        """No template configured -> the caller-provided instruction is still
        forwarded (mirrors _format_texts' fallback), not silently dropped.
        """
        adapter, model = _loaded_adapter()

        adapter.encode(
            [Item(text="q")],
            output_types=["dense"],
            is_query=True,
            instruction="search the docs:",
        )

        _, kwargs = model.encode.call_args
        assert kwargs["instruction"] == "search the docs:"

    def test_normalize_applied_on_top(self) -> None:
        adapter, _ = _loaded_adapter()

        out = adapter.encode(
            [Item(text="q")],
            output_types=["dense"],
            is_query=True,
            options={"normalize": True},
        )

        assert out.dense is not None
        np.testing.assert_allclose(np.linalg.norm(out.dense, axis=-1), 1.0, rtol=1e-5)

    def test_rejects_non_prefix_template(self) -> None:
        adapter, _ = _loaded_adapter()

        with pytest.raises(ValueError, match="use_model_encode"):
            adapter.encode(
                [Item(text="q")],
                output_types=["dense"],
                is_query=True,
                options={"query_template": "{text} </s>"},
            )

    def test_missing_model_encode_raises(self) -> None:
        adapter, model = _loaded_adapter()
        del model.encode
        model.mock_add_spec([])  # spec without encode
        adapter._model = model

        with pytest.raises(RuntimeError, match="use_model_encode"):
            adapter.encode([Item(text="q")], output_types=["dense"], is_query=True)
