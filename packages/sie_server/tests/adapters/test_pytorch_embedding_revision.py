from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter


class TestPyTorchEmbeddingAdapterRevision:
    """Loader contract: PyTorchEmbeddingAdapter accepts and forwards revision.

    Regression coverage for #876, where google/embeddinggemma-300m failed to
    load because the adapter rejected the revision= kwarg the loader passes
    when ModelConfig.hf_revision is set.
    """

    def _patched_load(self, adapter: PyTorchEmbeddingAdapter, hidden_size: int = 8) -> tuple[MagicMock, MagicMock]:
        """Drive adapter.load("cpu") with mocked HF loaders. Returns (tokenizer_cls, model_cls)."""
        mock_model = MagicMock()
        mock_model.config.hidden_size = hidden_size
        mock_model.to.return_value = mock_model

        mock_tokenizer = MagicMock()

        with (
            patch("sie_server.adapters.pytorch_embedding.AutoTokenizer") as mock_tok_cls,
            patch("sie_server.adapters.pytorch_embedding.AutoModel") as mock_mdl_cls,
        ):
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer
            mock_mdl_cls.from_pretrained.return_value = mock_model
            adapter.load("cpu")
            return mock_tok_cls, mock_mdl_cls

    def test_accepts_revision_kwarg(self) -> None:
        """Construction with revision= must not raise (loader contract)."""
        adapter = PyTorchEmbeddingAdapter(
            "BAAI/bge-base-en-v1.5",
            pooling="cls",
            revision="abc123",
        )
        assert adapter._revision == "abc123"

    def test_load_forwards_revision_to_both_loaders(self) -> None:
        adapter = PyTorchEmbeddingAdapter(
            "BAAI/bge-base-en-v1.5",
            pooling="cls",
            revision="abc123",
        )
        mock_tok_cls, mock_mdl_cls = self._patched_load(adapter)

        tok_kwargs = mock_tok_cls.from_pretrained.call_args.kwargs
        mdl_kwargs = mock_mdl_cls.from_pretrained.call_args.kwargs
        assert tok_kwargs["revision"] == "abc123"
        assert mdl_kwargs["revision"] == "abc123"

    def test_load_without_revision_omits_kwarg(self) -> None:
        """Default (revision=None) must not pass revision= to from_pretrained."""
        adapter = PyTorchEmbeddingAdapter(
            "BAAI/bge-base-en-v1.5",
            pooling="cls",
        )
        mock_tok_cls, mock_mdl_cls = self._patched_load(adapter)

        assert "revision" not in mock_tok_cls.from_pretrained.call_args.kwargs
        assert "revision" not in mock_mdl_cls.from_pretrained.call_args.kwargs

    def test_load_forwards_trust_remote_code(self) -> None:
        """Sanity: shared_kwargs still threads trust_remote_code unchanged."""
        adapter = PyTorchEmbeddingAdapter(
            "BAAI/bge-base-en-v1.5",
            pooling="cls",
            trust_remote_code=True,
            revision="abc123",
        )
        mock_tok_cls, mock_mdl_cls = self._patched_load(adapter)

        assert mock_tok_cls.from_pretrained.call_args.kwargs["trust_remote_code"] is True
        assert mock_mdl_cls.from_pretrained.call_args.kwargs["trust_remote_code"] is True

    def test_accepts_dense_dim_kwarg(self) -> None:
        """Construction with dense_dim= must not raise (loader contract)."""
        adapter = PyTorchEmbeddingAdapter(
            "google/embeddinggemma-300m",
            pooling="last_token",
            dense_dim=768,
        )
        assert adapter._configured_dense_dim == 768
        assert adapter.dims.dense == 768

    def test_load_accepts_matching_dense_dim(self) -> None:
        adapter = PyTorchEmbeddingAdapter(
            "google/embeddinggemma-300m",
            pooling="last_token",
            dense_dim=768,
        )

        self._patched_load(adapter, hidden_size=768)

        assert adapter.dims.dense == 768

    def test_load_rejects_dense_dim_mismatch(self) -> None:
        adapter = PyTorchEmbeddingAdapter(
            "google/embeddinggemma-300m",
            pooling="last_token",
            dense_dim=768,
        )

        with pytest.raises(ValueError, match="configured dense_dim=768, model hidden_size=1024"):
            self._patched_load(adapter, hidden_size=1024)
