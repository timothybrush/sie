from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sie_server.adapters._generation_base import GenerationChunk
from sie_server.adapters.sglang_vision_extract.adapter import SGLangVisionExtractAdapter
from sie_server.types.inputs import ImageInput, InvalidMediaError, Item


class _FakeProcessor:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        self.messages = messages
        assert add_generation_prompt is True
        assert tokenize is False
        return "rendered prompt"


class _FakeAsyncClient:
    def __init__(self) -> None:
        self.closed_on_loop: asyncio.AbstractEventLoop | None = None

    async def aclose(self) -> None:
        self.closed_on_loop = asyncio.get_running_loop()


def _image(marker: str) -> ImageInput:
    return ImageInput(data=marker.encode(), format="png")


@pytest.fixture
def adapter() -> SGLangVisionExtractAdapter:
    instance = SGLangVisionExtractAdapter(
        "lightonai/LightOnOCR-2-1B",
        max_concurrent_requests=2,
        system_prompt="You are an OCR engine. Return the markdown representation of the document.",
    )
    instance._processor = _FakeProcessor()
    instance._server_url = "http://sglang.test"
    return instance


def test_capabilities_are_extract_only(adapter: SGLangVisionExtractAdapter) -> None:
    assert adapter.capabilities.inputs == ["image"]
    assert adapter.capabilities.outputs == ["json"]
    assert adapter.get_preprocessor() is None


def test_device_factory_does_not_apply_text_only_mlx_fallback() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        SGLangVisionExtractAdapter.create_for_device(
            "mps",
            model_name_or_path="lightonai/LightOnOCR-2-1B",
        )

    instance = SGLangVisionExtractAdapter.create_for_device(
        "cuda:0",
        model_name_or_path="lightonai/LightOnOCR-2-1B",
    )
    assert isinstance(instance, SGLangVisionExtractAdapter)


def test_build_prompt_preserves_instruction(adapter: SGLangVisionExtractAdapter) -> None:
    assert adapter._build_prompt("Extract tables only") == "rendered prompt"
    assert adapter._processor.messages == [
        {
            "role": "system",
            "content": "You are an OCR engine. Return the markdown representation of the document.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Extract tables only"},
            ],
        },
    ]


def test_build_prompt_omits_system_and_uses_model_default_prompt() -> None:
    instance = SGLangVisionExtractAdapter(
        "zai-org/GLM-OCR",
        default_prompt="Text Recognition:",
    )
    instance._processor = _FakeProcessor()

    prompt, label = instance._resolve_prompt_and_label(None, {})
    assert prompt == "Text Recognition:"
    assert label == "markdown"
    assert instance._build_prompt(prompt) == "rendered prompt"
    assert instance._processor.messages == [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]


def test_task_prompt_instruction_override_and_label_mapping() -> None:
    instance = SGLangVisionExtractAdapter(
        "PaddlePaddle/PaddleOCR-VL-1.5",
        default_task="ocr",
        task_prompts={"ocr": "OCR:", "spotting": "Spotting:"},
        task_labels={"spotting": "spotting"},
    )

    assert instance._resolve_prompt_and_label(None, {}) == ("OCR:", "markdown")
    assert instance._resolve_prompt_and_label(None, {"task": "spotting"}) == ("Spotting:", "spotting")
    assert instance._resolve_prompt_and_label("Custom", {"task": "ocr"}) == ("Custom", "markdown")
    assert instance._resolve_prompt_and_label("", {"task": "ocr"}) == ("OCR:", "markdown")
    with pytest.raises(ValueError, match="must be one of"):
        instance._resolve_prompt_and_label(None, {"task": "unknown"})


@pytest.mark.asyncio
async def test_extract_async_refills_bounded_requests_and_preserves_order(
    adapter: SGLangVisionExtractAdapter,
) -> None:
    active = 0
    peak_active = 0
    third_started = asyncio.Event()
    release_first = asyncio.Event()

    async def generate(
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        images: list[ImageInput],
        **_: Any,
    ) -> AsyncIterator[GenerationChunk]:
        nonlocal active, peak_active
        assert prompt == "rendered prompt"
        assert max_new_tokens == 32
        assert temperature == 0.0
        assert top_p == 1.0
        active += 1
        peak_active = max(peak_active, active)
        marker = images[0]["data"].decode()
        if marker == "one":
            await release_first.wait()
        elif marker == "three":
            third_started.set()
        active -= 1
        yield GenerationChunk(
            text_delta=marker,
            done=True,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=1,
        )

    adapter.generate = generate  # ty: ignore[invalid-assignment]
    pending = asyncio.create_task(
        adapter._extract_async(
            "rendered prompt",
            [_image("one"), _image("two"), _image("three")],
            max_new_tokens=32,
        )
    )
    await asyncio.wait_for(third_started.wait(), timeout=1)
    release_first.set()
    results = await pending

    assert [result.text for result in results] == ["one", "two", "three"]
    assert peak_active == 2


def test_extract_requires_loaded_request_loop(adapter: SGLangVisionExtractAdapter) -> None:
    with pytest.raises(RuntimeError, match="Model not loaded"):
        adapter.extract([Item(images=[_image("page")])])


def test_extract_rejects_missing_images_and_beam_search(adapter: SGLangVisionExtractAdapter) -> None:
    adapter._request_loop = MagicMock()
    with pytest.raises(ValueError, match="requires image input"):
        adapter.extract([Item(text="no image")])
    with pytest.raises(ValueError, match="greedy decoding only"):
        adapter.extract([Item(images=[_image("page")])], options={"num_beams": 2})


def test_count_input_images_matches_consumed_first_image(adapter: SGLangVisionExtractAdapter) -> None:
    items = [Item(images=[_image("one"), _image("ignored")]), Item(text="none")]
    assert adapter.count_input_images(items) == [1, 0]


@pytest.mark.parametrize("images", [None, [], [_image("one"), _image("two")]])
def test_page_metering_rejects_non_single_image_arrays(
    images: list[ImageInput] | None,
) -> None:
    instance = SGLangVisionExtractAdapter(
        "lightonai/LightOnOCR-2-1B",
        meter_pages=True,
    )
    instance._processor = _FakeProcessor()
    instance._request_loop = MagicMock()

    with pytest.raises(InvalidMediaError, match="exactly one image"):
        instance.extract([Item(images=images)])


def test_page_metering_replaces_image_units_and_stamps_successes() -> None:
    instance = SGLangVisionExtractAdapter(
        "lightonai/LightOnOCR-2-1B",
        meter_pages=True,
    )
    instance._processor = _FakeProcessor()
    instance._request_loop = MagicMock()
    items = [Item(images=[_image("page")])]
    future = MagicMock()
    future.result.return_value = [MagicMock(text="# page")]

    def submit(coroutine: Any, loop: Any) -> MagicMock:
        del loop
        coroutine.close()
        return future

    with patch(
        "sie_server.adapters.sglang_vision_extract.adapter.asyncio.run_coroutine_threadsafe",
        side_effect=submit,
    ):
        out = instance.extract(items)

    assert instance.count_input_images(items) is None
    assert out.pages == [1]
    assert out.entities[0][0]["text"] == "# page"


def test_load_resolves_processor_to_pinned_snapshot() -> None:
    from transformers import AutoProcessor

    instance = SGLangVisionExtractAdapter(
        "lightonai/LightOnOCR-2-1B",
        revision="abc123",
        processor_use_fast=False,
    )
    processor = MagicMock()

    with (
        patch(
            "sie_server.adapters.sglang_vision_extract.adapter.snapshot_download",
            return_value="/models/lighton-snapshot",
        ) as snapshot,
        patch.object(AutoProcessor, "from_pretrained") as from_pretrained,
        patch("sie_server.adapters.sglang_vision_extract.adapter.SGLangGenerationAdapter.load") as engine_load,
        patch.object(instance, "_start_request_loop") as start_loop,
    ):
        from_pretrained.return_value = processor
        instance.load("cuda:0")

    snapshot.assert_called_once_with("lightonai/LightOnOCR-2-1B", revision="abc123")
    from_pretrained.assert_called_once_with(
        "/models/lighton-snapshot",
        trust_remote_code=True,
        use_fast=False,
    )
    engine_load.assert_called_once_with("cuda:0")
    start_loop.assert_called_once_with()
    assert instance._processor is processor


def test_resolve_processor_dir_preserves_local_path(tmp_path: Path) -> None:
    instance = SGLangVisionExtractAdapter(tmp_path)

    with patch("sie_server.adapters.sglang_vision_extract.adapter.snapshot_download") as snapshot:
        assert instance._resolve_processor_dir() == str(tmp_path)

    snapshot.assert_not_called()


@pytest.mark.asyncio
async def test_aclose_client_runs_on_dedicated_request_loop(adapter: SGLangVisionExtractAdapter) -> None:
    client = _FakeAsyncClient()
    adapter._http_client = client  # ty: ignore[invalid-assignment]
    adapter._start_request_loop()
    request_loop = adapter._request_loop

    try:
        await adapter.aclose_client()
        assert client.closed_on_loop is request_loop
        assert adapter._http_client is None
    finally:
        adapter.unload()
