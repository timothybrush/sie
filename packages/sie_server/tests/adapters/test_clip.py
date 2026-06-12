from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sie_server.adapters.clip import CLIPAdapter
from sie_server.types.inputs import Item

# Create a random generator for tests
_RNG = np.random.default_rng(42)


class TestCLIPAdapter:
    """Tests for CLIPAdapter with mocked model."""

    @pytest.fixture
    def mock_clip_model(self) -> MagicMock:
        """Create a mock CLIPModel."""
        mock = MagicMock()
        # Mock config with projection_dim
        mock.config.projection_dim = 512
        # Mock get_text_features and get_image_features
        mock.get_text_features.return_value = MagicMock(
            __getitem__=lambda self, idx: MagicMock(
                float=lambda: MagicMock(
                    cpu=lambda: MagicMock(numpy=lambda: _RNG.standard_normal(512).astype(np.float32))
                )
            )
        )
        mock.get_image_features.return_value = MagicMock(
            mean=lambda dim, keepdim: MagicMock(
                __getitem__=lambda self, idx: MagicMock(
                    float=lambda: MagicMock(
                        cpu=lambda: MagicMock(numpy=lambda: _RNG.standard_normal(512).astype(np.float32))
                    )
                )
            )
        )
        return mock

    @pytest.fixture
    def mock_clip_processor(self) -> MagicMock:
        """Create a mock CLIPProcessor."""
        mock = MagicMock()
        # Return dict-like object for processor outputs
        mock.return_value = {"pixel_values": MagicMock(), "input_ids": MagicMock()}
        return mock

    @pytest.fixture
    def adapter(self) -> CLIPAdapter:
        """Create an adapter instance."""
        return CLIPAdapter(
            "openai/clip-vit-base-patch32",
            normalize=True,
            compute_precision="float16",
        )

    def test_capabilities(self, adapter: CLIPAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["dense"]

    def test_dims_before_load(self, adapter: CLIPAdapter) -> None:
        """Dims returns None dense before load (BaseAdapter reads from spec)."""
        dims = adapter.dims
        assert dims.dense is None

    def test_accepts_dense_dim_kwarg(self) -> None:
        """Construction with dense_dim= must not raise (loader contract)."""
        adapter = CLIPAdapter(
            "openai/clip-vit-base-patch32",
            dense_dim=512,
        )
        assert adapter._configured_dense_dim == 512
        assert adapter.dims.dense == 512

    def test_encode_before_load_raises(self, adapter: CLIPAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["dense"])

    def test_encode_without_input_raises(self, adapter: CLIPAdapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["dense"])

    def test_validate_output_types(self, adapter: CLIPAdapter) -> None:
        """Only dense output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["multivector"])

    @patch("transformers.CLIPModel.from_pretrained")
    @patch("transformers.CLIPProcessor.from_pretrained")
    def test_load(
        self,
        mock_processor_from_pretrained: MagicMock,
        mock_model_from_pretrained: MagicMock,
        adapter: CLIPAdapter,
        mock_clip_model: MagicMock,
        mock_clip_processor: MagicMock,
    ) -> None:
        """Load initializes the model."""
        mock_model_from_pretrained.return_value = mock_clip_model
        mock_processor_from_pretrained.return_value = mock_clip_processor

        adapter.load("cpu")

        mock_model_from_pretrained.assert_called_once()
        mock_processor_from_pretrained.assert_called_once()
        assert adapter.dims.dense == 512

    @patch("transformers.CLIPModel.from_pretrained")
    @patch("transformers.CLIPProcessor.from_pretrained")
    def test_load_accepts_matching_dense_dim(
        self,
        mock_processor_from_pretrained: MagicMock,
        mock_model_from_pretrained: MagicMock,
        mock_clip_model: MagicMock,
        mock_clip_processor: MagicMock,
    ) -> None:
        """Load accepts catalog dense_dim when it matches projection_dim."""
        adapter = CLIPAdapter("openai/clip-vit-base-patch32", dense_dim=512)
        mock_model_from_pretrained.return_value = mock_clip_model
        mock_processor_from_pretrained.return_value = mock_clip_processor

        adapter.load("cpu")

        assert adapter.dims.dense == 512

    @patch("transformers.CLIPModel.from_pretrained")
    @patch("transformers.CLIPProcessor.from_pretrained")
    def test_load_rejects_dense_dim_mismatch(
        self,
        mock_processor_from_pretrained: MagicMock,
        mock_model_from_pretrained: MagicMock,
        mock_clip_model: MagicMock,
        mock_clip_processor: MagicMock,
    ) -> None:
        """Load rejects catalog dense_dim when it differs from projection_dim."""
        adapter = CLIPAdapter("openai/clip-vit-base-patch32", dense_dim=768)
        mock_model_from_pretrained.return_value = mock_clip_model
        mock_processor_from_pretrained.return_value = mock_clip_processor

        with pytest.raises(ValueError, match="configured dense_dim=768, model projection_dim=512"):
            adapter.load("cpu")

    @patch("sie_server.adapters.clip.torch")
    def test_unload(self, mock_torch: MagicMock, adapter: CLIPAdapter) -> None:
        """Unload clears the model."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._dense_dim = 512

        adapter.unload()

        assert adapter._model is None
        assert adapter._processor is None
        assert adapter._dense_dim is None
        assert adapter.dims.dense is None
