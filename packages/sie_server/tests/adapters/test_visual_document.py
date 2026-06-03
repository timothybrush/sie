"""Tests for visual document retrieval adapters (ColPali, ColQwen2, NemoColEmbed).

These adapters encode document images into multi-vector representations
for late interaction retrieval using MaxSim scoring.
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.adapters.colpali import ColPaliAdapter
from sie_server.adapters.colqwen2 import ColQwen2Adapter
from sie_server.adapters.colqwen3 import ColQwen3Adapter
from sie_server.adapters.nemo_colembed import NemoColEmbedAdapter
from sie_server.types.inputs import Item


class TestColPaliAdapter:
    """Tests for ColPaliAdapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> ColPaliAdapter:
        """Create an adapter instance."""
        return ColPaliAdapter(
            "vidore/colpali-v1.3-hf",
            normalize=True,
            compute_precision="float32",
        )

    def test_capabilities(self, adapter: ColPaliAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_has_default(self, adapter: ColPaliAdapter) -> None:
        """Dims returns default value before load."""
        dims = adapter.dims
        assert dims.multivector == 128  # ColPali default

    def test_encode_before_load_raises(self, adapter: ColPaliAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_encode_without_input_raises(self, adapter: ColPaliAdapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["multivector"])

    def test_validate_output_types(self, adapter: ColPaliAdapter) -> None:
        """Only multivector output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])


class TestColQwen2Adapter:
    """Tests for ColQwen2Adapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> ColQwen2Adapter:
        """Create an adapter instance."""
        return ColQwen2Adapter(
            "vidore/colqwen2.5-v0.2",
            normalize=True,
            compute_precision="float16",
        )

    def test_capabilities(self, adapter: ColQwen2Adapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_has_default(self, adapter: ColQwen2Adapter) -> None:
        """Dims returns default value before load."""
        dims = adapter.dims
        assert dims.multivector == 128  # ColQwen2 default

    def test_encode_before_load_raises(self, adapter: ColQwen2Adapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_encode_without_input_raises(self, adapter: ColQwen2Adapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["multivector"])

    def test_validate_output_types(self, adapter: ColQwen2Adapter) -> None:
        """Only multivector output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])


class TestColQwen3Adapter:
    """Tests for ColQwen3Adapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> ColQwen3Adapter:
        """Create an adapter instance."""
        return ColQwen3Adapter(
            "TomoroAI/tomoro-colqwen3-embed-4b",
            normalize=True,
            compute_precision="bfloat16",
        )

    def test_capabilities(self, adapter: ColQwen3Adapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_has_default(self, adapter: ColQwen3Adapter) -> None:
        """Dims returns default value before load."""
        dims = adapter.dims
        assert dims.multivector == 320  # ColQwen3 default

    def test_encode_before_load_raises(self, adapter: ColQwen3Adapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_encode_without_input_raises(self, adapter: ColQwen3Adapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["multivector"])

    def test_validate_output_types(self, adapter: ColQwen3Adapter) -> None:
        """Only multivector output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

    def test_encode_mixed_batch_preserves_order(self, adapter: ColQwen3Adapter) -> None:
        """Mixed text/image items round-trip in input order with one mv per item."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._multivector_dim = 4

        # Stamp each output with a unique value so we can assert which item it came from.
        def fake_load_images(item: Any) -> list[Any]:
            return [f"img-{item.text}"] * (1 if item.images else 0)

        def fake_encode_images(images: list[Any]) -> list[np.ndarray]:
            return [np.full((2, 4), float(i + 100), dtype=np.float32) for i, _ in enumerate(images)]

        def fake_encode_text(text: str) -> np.ndarray:
            return np.full((1, 4), float(hash(text) % 1000), dtype=np.float32)

        adapter._load_images = fake_load_images  # type: ignore[method-assign]
        adapter._encode_images = fake_encode_images  # type: ignore[method-assign]
        adapter._encode_text = fake_encode_text  # type: ignore[method-assign]

        items = [
            Item(text="a", images=[{"data": b"x", "format": "png"}]),
            Item(text="b"),
            Item(text="c", images=[{"data": b"y", "format": "png"}]),
            Item(text="d"),
        ]
        out = adapter.encode(items, output_types=["multivector"])
        assert out.batch_size == len(items)
        assert out.multivector is not None
        assert len(out.multivector) == len(items)

        # Image items got the per-image stamp (100, 101) in input order; text items got hash stamps.
        assert out.multivector[0][0, 0] == 100.0
        assert out.multivector[1][0, 0] == float(hash("b") % 1000)
        assert out.multivector[2][0, 0] == 101.0
        assert out.multivector[3][0, 0] == float(hash("d") % 1000)

    def test_encode_multi_image_item_concatenates_seq_dim(self, adapter: ColQwen3Adapter) -> None:
        """A single item with N images yields one mv with seq = sum of per-image seqs."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._multivector_dim = 4

        adapter._load_images = lambda item: ["img-1", "img-2", "img-3"]  # type: ignore[method-assign]
        per_image = [
            np.full((3, 4), 1.0, dtype=np.float32),
            np.full((5, 4), 2.0, dtype=np.float32),
            np.full((7, 4), 3.0, dtype=np.float32),
        ]
        adapter._encode_images = lambda images: per_image  # type: ignore[method-assign]

        items = [Item(images=[{"data": b"x", "format": "png"}] * 3)]
        out = adapter.encode(items, output_types=["multivector"])
        assert out.multivector is not None
        assert len(out.multivector) == 1
        mv = out.multivector[0]
        assert mv.shape == (3 + 5 + 7, 4)
        # Concatenation order is per-image order: 3 rows of 1.0, 5 rows of 2.0, 7 rows of 3.0.
        assert mv[0, 0] == 1.0
        assert mv[3, 0] == 2.0
        assert mv[8, 0] == 3.0


class TestNemoColEmbedV2Config:
    """Tests for NemoColEmbedAdapter v2 configuration (Qwen3-VL backbone, token_dim=2560)."""

    def test_v2_token_dim_constructor(self) -> None:
        """V2 adapter accepts token_dim=2560 and stores it on _multivector_dim."""
        adapter = NemoColEmbedAdapter(
            "nvidia/nemotron-colembed-vl-4b-v2",
            token_dim=2560,
            normalize=True,
        )
        # The class-level spec dim is fixed at 128 (v1) but the per-instance
        # _multivector_dim must reflect the v2 token_dim.
        assert adapter._multivector_dim == 2560

    def test_v2_default_compute_precision(self) -> None:
        """V2 adapter inherits the bf16 default."""
        adapter = NemoColEmbedAdapter(
            "nvidia/nemotron-colembed-vl-4b-v2",
            token_dim=2560,
        )
        assert adapter._compute_precision == "bfloat16"


class TestNemoColEmbedAdapter:
    """Tests for NemoColEmbedAdapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> NemoColEmbedAdapter:
        """Create an adapter instance."""
        return NemoColEmbedAdapter(
            "nvidia/llama-nemoretriever-colembed-3b-v1",
            normalize=True,
            compute_precision="bfloat16",
        )

    def test_capabilities(self, adapter: NemoColEmbedAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_has_default(self, adapter: NemoColEmbedAdapter) -> None:
        """Dims returns default value before load."""
        dims = adapter.dims
        assert dims.multivector == 128  # NemoColEmbed default

    def test_encode_before_load_raises(self, adapter: NemoColEmbedAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_validate_output_types(self, adapter: NemoColEmbedAdapter) -> None:
        """Only multivector output type is supported."""
        # Mock model as loaded
        adapter._model = MagicMock()
        adapter._device = "cpu"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])


class TestNemoColEmbedPreprocessor:
    """Tests for NemoColEmbedPreprocessor infrastructure."""

    def test_preprocessor_class_exists(self) -> None:
        """NemoColEmbedPreprocessor class is defined."""
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        assert NemoColEmbedPreprocessor is not None

    def test_preprocessor_payload_class_exists(self) -> None:
        """NemoColEmbedPayload dataclass is defined."""
        from sie_server.core.prepared import NemoColEmbedPayload

        assert NemoColEmbedPayload is not None

    def test_adapter_has_preprocessor_method(self) -> None:
        """NemoColEmbedAdapter has get_preprocessor method."""
        adapter = NemoColEmbedAdapter(
            "nvidia/llama-nemoretriever-colembed-3b-v1",
            normalize=True,
        )
        # get_preprocessor returns CharCountPreprocessor for cost estimation
        preprocessor = adapter.get_preprocessor()
        assert preprocessor is not None

    def test_adapter_processor_created_on_load(self) -> None:
        """NemoColEmbedAdapter creates _processor after load().

        Note: Full _create_processor() test requires loaded model with
        tokenizer/config attributes. This test just verifies the interface.
        """
        adapter = NemoColEmbedAdapter(
            "nvidia/llama-nemoretriever-colembed-3b-v1",
            normalize=True,
        )
        # Before load, _processor should be None
        assert adapter._processor is None


class TestVLMCudaCacheClearing:
    """Tests that VLM adapters contain torch.cuda.empty_cache() after inference.

    This prevents GPU memory accumulation (OOM) on L4 22GB GPUs when
    encoding many document images in sequence.
    """

    def test_colqwen2_encode_images_has_empty_cache(self) -> None:
        """ColQwen2 _encode_images source contains empty_cache call."""
        import inspect

        source = inspect.getsource(ColQwen2Adapter._encode_images)
        assert "torch.cuda.empty_cache()" in source, (
            "ColQwen2._encode_images must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_colqwen3_encode_images_has_empty_cache(self) -> None:
        """ColQwen3 _encode_images source contains empty_cache call."""
        import inspect

        source = inspect.getsource(ColQwen3Adapter._encode_images)
        assert "torch.cuda.empty_cache()" in source, (
            "ColQwen3._encode_images must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_colpali_encode_prepared_batch_has_empty_cache(self) -> None:
        """ColPali _encode_prepared_batch source contains empty_cache call."""
        import inspect

        source = inspect.getsource(ColPaliAdapter._encode_prepared_batch)
        assert "torch.cuda.empty_cache()" in source, (
            "ColPali._encode_prepared_batch must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_colpali_encode_images_has_empty_cache(self) -> None:
        """ColPali _encode_images (single-item fallback) source contains empty_cache call."""
        import inspect

        source = inspect.getsource(ColPaliAdapter._encode_images)
        assert "torch.cuda.empty_cache()" in source, (
            "ColPali._encode_images must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_colqwen2_encode_images_batched_has_empty_cache(self) -> None:
        """ColQwen2 _encode_images_batched (document-path) source contains empty_cache call."""
        import inspect

        source = inspect.getsource(ColQwen2Adapter._encode_images_batched)
        assert "torch.cuda.empty_cache()" in source, (
            "ColQwen2._encode_images_batched must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_nemo_colembed_encode_images_has_empty_cache(self) -> None:
        """NemoColEmbed _encode_images source contains empty_cache call."""
        import inspect

        source = inspect.getsource(NemoColEmbedAdapter._encode_images)
        assert "torch.cuda.empty_cache()" in source, (
            "NemoColEmbed._encode_images must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_nemo_colembed_preprocessed_has_empty_cache(self) -> None:
        """NemoColEmbed _encode_images_preprocessed source contains empty_cache call."""
        import inspect

        source = inspect.getsource(NemoColEmbedAdapter._encode_images_preprocessed)
        assert "torch.cuda.empty_cache()" in source, (
            "NemoColEmbed._encode_images_preprocessed must call torch.cuda.empty_cache() to prevent OOM"
        )
