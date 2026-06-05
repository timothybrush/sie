from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sie_server.types.inputs import Item


class TestFlorence2Adapter:
    """Tests for Florence2Adapter with mocked model."""

    @pytest.fixture
    def mock_florence2_model(self) -> MagicMock:
        """Create a mock Florence2 model."""
        mock = MagicMock()
        # Mock generate method
        mock.generate.return_value = MagicMock()
        mock.dtype = MagicMock()
        return mock

    @pytest.fixture
    def mock_florence2_processor(self) -> MagicMock:
        """Create a mock Florence2 processor."""
        mock = MagicMock()
        # Return dict-like object for processor outputs
        mock.return_value = {
            "pixel_values": MagicMock(),
            "input_ids": MagicMock(),
        }
        # Mock batch_decode
        mock.batch_decode.return_value = ["<s><OCR_WITH_REGION>text</s>"]
        # Mock post_process_generation
        mock.post_process_generation.return_value = {
            "<OCR_WITH_REGION>": {
                "quad_boxes": [[10.0, 10.0, 100.0, 10.0, 100.0, 50.0, 10.0, 50.0]],
                "labels": ["Hello World"],
            }
        }
        return mock

    @pytest.fixture
    def adapter(self) -> Florence2Adapter:
        """Create an adapter instance."""
        from sie_server.adapters.florence2 import Florence2Adapter

        return Florence2Adapter(
            "microsoft/Florence-2-base",
            default_task="<OCR_WITH_REGION>",
            compute_precision="float16",
        )

    def test_capabilities(self, adapter: Florence2Adapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["image"]
        assert caps.outputs == ["json"]

    def test_dims(self, adapter: Florence2Adapter) -> None:
        """Adapter reports empty dimensions (extraction model)."""
        dims = adapter.dims
        assert dims.dense is None
        assert dims.sparse is None
        assert dims.multivector is None

    def test_encode_raises_not_implemented(self, adapter: Florence2Adapter) -> None:
        """Encode raises NotImplementedError."""
        items = [Item(text="hello")]
        with pytest.raises(NotImplementedError, match="does not support encode"):
            adapter.encode(items, output_types=["dense"])

    def test_extract_before_load_raises(self, adapter: Florence2Adapter) -> None:
        """Extract before load raises error."""
        from sie_server.types.inputs import ImageInput

        items = [Item(images=[ImageInput(data=b"fake", format="jpeg")])]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.extract(items)

    def test_quad_to_bbox_conversion(self, adapter: Florence2Adapter) -> None:
        """Quad box is correctly converted to bbox."""
        # Quad box: 4 corners (x1,y1,x2,y2,x3,y3,x4,y4)
        quad_box = [10.0, 10.0, 100.0, 10.0, 100.0, 50.0, 10.0, 50.0]
        image_size = (200, 100)

        bbox = adapter._quad_to_bbox(quad_box, image_size)

        # Expected: [10/200, 10/100, 100/200, 50/100] = [0.05, 0.1, 0.5, 0.5]
        assert len(bbox) == 4
        assert abs(bbox[0] - 0.05) < 1e-6
        assert abs(bbox[1] - 0.1) < 1e-6
        assert abs(bbox[2] - 0.5) < 1e-6
        assert abs(bbox[3] - 0.5) < 1e-6

    def test_normalize_bbox(self, adapter: Florence2Adapter) -> None:
        """Bbox is correctly normalized."""
        bbox = [50.0, 25.0, 150.0, 75.0]
        image_size = (200, 100)

        norm_bbox = adapter._normalize_bbox(bbox, image_size)

        assert len(norm_bbox) == 4
        assert abs(norm_bbox[0] - 0.25) < 1e-6
        assert abs(norm_bbox[1] - 0.25) < 1e-6
        assert abs(norm_bbox[2] - 0.75) < 1e-6
        assert abs(norm_bbox[3] - 0.75) < 1e-6

    def test_build_prompt_basic_task(self, adapter: Florence2Adapter) -> None:
        """Build prompt returns task token for basic tasks."""
        prompt = adapter._build_prompt("<OCR>", labels=None, instruction=None)
        assert prompt == "<OCR>"

    def test_build_prompt_free_text_instruction_uses_docvqa(self, adapter: Florence2Adapter) -> None:
        """A free-text instruction is answered via DocVQA, not appended to the task.

        Regression for #1053: appending an instruction to a region/OCR task token
        makes Florence-2's processor raise ("task token should be the only token").
        """
        prompt = adapter._build_prompt("<OCR_WITH_REGION>", labels=None, instruction="What is the title?")
        assert prompt == "<DocVQA>What is the title?"

    def test_build_prompt_instruction_with_task_token_used_verbatim(self, adapter: Florence2Adapter) -> None:
        """An instruction that already carries a task token is used verbatim (no double token)."""
        prompt = adapter._build_prompt("<DocVQA>", labels=None, instruction="<DocVQA> What is the title?")
        assert prompt == "<DocVQA>What is the title?"

    def test_build_prompt_phrase_grounding_with_labels(self, adapter: Florence2Adapter) -> None:
        """Build prompt appends labels for phrase grounding."""
        prompt = adapter._build_prompt(
            "<CAPTION_TO_PHRASE_GROUNDING>",
            labels=["person", "car"],
            instruction=None,
        )
        assert prompt == "<CAPTION_TO_PHRASE_GROUNDING>person, car"

    def test_convert_output_ocr_with_region(self, adapter: Florence2Adapter) -> None:
        """Convert output handles OCR_WITH_REGION format."""
        parsed = {
            "<OCR_WITH_REGION>": {
                "quad_boxes": [[0.0, 0.0, 100.0, 0.0, 100.0, 50.0, 0.0, 50.0]],
                "labels": ["Hello"],
            }
        }
        image_size = (100, 100)

        entities, objects = adapter._convert_output(parsed, "<OCR_WITH_REGION>", image_size)

        assert len(entities) == 1
        assert entities[0]["text"] == "Hello"
        assert entities[0]["label"] == "text"
        assert entities[0]["score"] == 1.0
        assert entities[0]["bbox"] is not None
        assert len(objects) == 0

    def test_convert_output_object_detection(self, adapter: Florence2Adapter) -> None:
        """Convert output handles OD format as DetectedObject."""
        parsed = {
            "<OD>": {
                "bboxes": [[10.0, 20.0, 80.0, 90.0]],
                "labels": ["car"],
            }
        }
        image_size = (100, 100)

        entities, objects = adapter._convert_output(parsed, "<OD>", image_size)

        assert len(entities) == 0
        assert len(objects) == 1
        assert objects[0]["label"] == "car"
        assert objects[0]["score"] == 1.0
        assert objects[0]["bbox"] is not None

    def test_extract_threads_docvqa_task_into_postprocessing(
        self,
        adapter: Florence2Adapter,
        mock_florence2_model: MagicMock,
        mock_florence2_processor: MagicMock,
    ) -> None:
        """A free-text instruction makes extract() post-process as DocVQA, not OCR.

        Regression for #1053: the configured task is <OCR_WITH_REGION>, but an
        instruction must switch both the prompt and the post-processing to DocVQA.
        """
        import io

        from PIL import Image as PILImage
        from sie_server.types.inputs import ImageInput

        mock_florence2_processor.batch_decode.return_value = ["<s><DocVQA>1.8 to 5.5 V</s>"]
        mock_florence2_processor.post_process_generation.return_value = {"<DocVQA>": "1.8 to 5.5 V"}
        adapter._model = mock_florence2_model
        adapter._processor = mock_florence2_processor
        adapter._device = "cpu"

        buf = io.BytesIO()
        PILImage.new("RGB", (8, 8), "white").save(buf, format="JPEG")
        items = [Item(images=[ImageInput(data=buf.getvalue(), format="jpeg")])]
        out = adapter.extract(items, instruction="What is the operating voltage range?")

        # Post-processing must be asked for the DocVQA task, not the configured OCR task.
        _, kwargs = mock_florence2_processor.post_process_generation.call_args
        assert kwargs["task"] == "<DocVQA>"
        assert out.entities[0][0]["text"] == "1.8 to 5.5 V"
        assert out.entities[0][0]["label"] == "answer"


class TestResolveFlorence2Prompt:
    """Tests for the shared Florence-2 prompt/task resolver."""

    def test_no_instruction_keeps_configured_task(self) -> None:
        from sie_server.core.preprocessor.vision import resolve_florence2_prompt

        assert resolve_florence2_prompt("<OCR_WITH_REGION>", None, None) == (
            "<OCR_WITH_REGION>",
            "<OCR_WITH_REGION>",
        )

    def test_free_text_instruction_becomes_docvqa(self) -> None:
        from sie_server.core.preprocessor.vision import resolve_florence2_prompt

        # Even though the configured task is OCR, a free-text question is DocVQA.
        assert resolve_florence2_prompt("<OCR_WITH_REGION>", None, "What is the title?") == (
            "<DocVQA>What is the title?",
            "<DocVQA>",
        )

    def test_instruction_with_task_token_is_verbatim(self) -> None:
        from sie_server.core.preprocessor.vision import resolve_florence2_prompt

        # No double task token, leading/inner whitespace normalised.
        assert resolve_florence2_prompt("<DocVQA>", None, "<DocVQA> What is the title?") == (
            "<DocVQA>What is the title?",
            "<DocVQA>",
        )
        assert resolve_florence2_prompt("<OCR_WITH_REGION>", None, "<CAPTION>") == ("<CAPTION>", "<CAPTION>")

    def test_phrase_grounding_appends_labels(self) -> None:
        from sie_server.core.preprocessor.vision import resolve_florence2_prompt

        assert resolve_florence2_prompt("<CAPTION_TO_PHRASE_GROUNDING>", ["person", "car"], None) == (
            "<CAPTION_TO_PHRASE_GROUNDING>person, car",
            "<CAPTION_TO_PHRASE_GROUNDING>",
        )
