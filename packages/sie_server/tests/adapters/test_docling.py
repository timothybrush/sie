from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sie_server.adapters.docling import DoclingAdapter
from sie_server.types.inputs import Item


def _make_adapter(*, ocr_factory: Any = None) -> tuple[DoclingAdapter, MagicMock]:
    """Build a loaded adapter whose `_make_converter` returns mocks.

    Returns the adapter and the patched factory so tests can inspect calls.
    Each invocation of the factory yields a *new* MagicMock-backed converter;
    the adapter caches them via `_get_converter` so within a single
    `ocr_enabled` value only the first extract triggers a build.
    """
    adapter = DoclingAdapter()
    adapter._loaded = True

    factory = MagicMock(name="_make_converter")

    def _new_converter(*, ocr_enabled: bool) -> MagicMock:
        if ocr_enabled and ocr_factory is not None:
            return ocr_factory(ocr_enabled=ocr_enabled)
        tag = "ocr" if ocr_enabled else "default"
        return _stub_converter(tag)

    factory.side_effect = _new_converter
    adapter._make_converter = factory  # type: ignore[method-assign]
    return adapter, factory


def _stub_converter(tag: str) -> MagicMock:
    """Return a MagicMock that mimics DocumentConverter.convert() return shape."""
    converter = MagicMock(name=f"DocumentConverter[{tag}]")

    def _convert(stream: Any) -> MagicMock:
        result = MagicMock()
        result.document.export_to_text.return_value = f"text:{tag}"
        result.document.export_to_markdown.return_value = f"# md:{tag}"
        result.document.export_to_dict.return_value = {"name": stream.name}
        return result

    converter.convert.side_effect = _convert
    return converter


class TestDoclingExtract:
    def test_returns_text_markdown_and_document(self) -> None:
        adapter, _ = _make_adapter()

        out = adapter.extract([Item(document={"data": b"%PDF-1.4 fake", "format": "pdf"})])

        assert out.batch_size == 1
        assert out.entities == [[]]
        assert out.data is not None
        assert out.data[0]["text"] == "text:default"
        assert out.data[0]["markdown"] == "# md:default"
        assert out.data[0]["document"] == {"name": "document.pdf"}

    def test_format_hint_drives_stream_name(self) -> None:
        adapter, _ = _make_adapter()

        out = adapter.extract([Item(document={"data": b"<html></html>", "format": "html"})])

        assert out.data is not None
        assert out.data[0]["document"] == {"name": "document.html"}

    def test_missing_format_falls_back_to_generic_name(self) -> None:
        adapter, _ = _make_adapter()

        out = adapter.extract([Item(document={"data": b"raw"})])

        assert out.data is not None
        assert out.data[0]["document"] == {"name": "document"}

    def test_non_document_item_yields_per_item_error(self) -> None:
        adapter, factory = _make_adapter()

        out = adapter.extract([Item(text="just text, no document")])

        assert out.data is not None
        assert "error" in out.data[0]
        assert out.data[0]["error"]  # non-empty
        # No converter is constructed for a malformed item
        factory.assert_not_called()

    def test_image_only_item_routes_to_converter(self) -> None:
        adapter, factory = _make_adapter()

        out = adapter.extract([Item(images=[{"data": b"\x89PNG\r\n\x1a\nfake", "format": "png"}])])

        assert out.data is not None
        assert out.data[0]["text"] == "text:default"
        # Stream name should reflect the image format hint
        assert out.data[0]["document"] == {"name": "document.png"}
        factory.assert_called_once_with(ocr_enabled=False)

    def test_image_without_format_defaults_to_png_name(self) -> None:
        adapter, _ = _make_adapter()

        out = adapter.extract([Item(images=[{"data": b"\x89PNG"}])])

        assert out.data is not None
        # When no format is supplied, the adapter defaults to "png" so Docling
        # has a usable extension hint.
        assert out.data[0]["document"] == {"name": "document.png"}

    def test_document_wins_when_both_document_and_image_present(self) -> None:
        adapter, _ = _make_adapter()

        out = adapter.extract(
            [
                Item(
                    document={"data": b"%PDF-1.4 real", "format": "pdf"},
                    images=[{"data": b"\x89PNG ignored", "format": "png"}],
                )
            ]
        )

        assert out.data is not None
        # Document path uses ``document.pdf`` stream name; image path would
        # have produced ``document.png``.
        assert out.data[0]["document"] == {"name": "document.pdf"}

    def test_neither_document_nor_image_yields_per_item_error(self) -> None:
        adapter, factory = _make_adapter()

        out = adapter.extract([Item(text="just text")])

        assert out.data is not None
        assert "error" in out.data[0]
        factory.assert_not_called()

    def test_empty_images_list_yields_per_item_error(self) -> None:
        adapter, factory = _make_adapter()

        out = adapter.extract([Item(images=[])])

        assert out.data is not None
        assert "error" in out.data[0]
        factory.assert_not_called()

    def test_per_item_failure_does_not_poison_batch(self) -> None:
        adapter = DoclingAdapter()
        adapter._loaded = True

        converter = MagicMock(name="DocumentConverter[shared]")

        def _good_result(name: str, stream_name: str) -> MagicMock:
            r = MagicMock()
            r.document.export_to_text.return_value = f"text:{name}"
            r.document.export_to_markdown.return_value = f"# md:{name}"
            r.document.export_to_dict.return_value = {"name": stream_name}
            return r

        calls = {"n": 0}

        def _convert_side_effect(stream: Any, **_kwargs: Any) -> MagicMock:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return _good_result(f"doc{calls['n']}", stream.name)

        converter.convert.side_effect = _convert_side_effect
        factory = MagicMock(name="_make_converter", return_value=converter)
        adapter._make_converter = factory  # type: ignore[method-assign]

        out = adapter.extract(
            [
                Item(document={"data": b"a", "format": "pdf"}),
                Item(document={"data": b"b", "format": "pdf"}),
                Item(document={"data": b"c", "format": "pdf"}),
            ]
        )

        assert out.data is not None
        statuses = [d.get("error", "ok") for d in out.data]
        assert statuses == ["ok", "boom", "ok"]
        assert factory.call_count == 1  # converter built once, not three times

    def test_extract_before_load_raises(self) -> None:
        adapter = DoclingAdapter()
        with pytest.raises(RuntimeError, match="load"):
            adapter.extract([Item(document={"data": b"x"})])

    def test_ocr_opt_in_passes_flag_to_factory(self) -> None:
        adapter, factory = _make_adapter()

        adapter.extract([Item(document={"data": b"x", "format": "pdf"})], options={"ocr": True})
        adapter.extract([Item(document={"data": b"x", "format": "pdf"})], options={"ocr": True})

        # Cache means we built once even though we extracted twice.
        assert factory.call_count == 1
        assert factory.call_args.kwargs == {"ocr_enabled": True}

    def test_ocr_default_off_passes_false(self) -> None:
        adapter, factory = _make_adapter()

        adapter.extract([Item(document={"data": b"x", "format": "pdf"})])

        assert factory.call_count == 1
        assert factory.call_args.kwargs == {"ocr_enabled": False}

    def test_batch_reuses_cached_converter(self) -> None:
        adapter, factory = _make_adapter()

        items = [Item(document={"data": b"x", "format": "pdf"}) for _ in range(3)]
        out = adapter.extract(items)

        assert out.batch_size == 3
        # One cached converter shared across the whole batch.
        assert factory.call_count == 1
        assert factory.call_args.kwargs == {"ocr_enabled": False}

    def test_extract_caches_converter_across_calls(self) -> None:
        adapter, factory = _make_adapter()
        adapter.extract([Item(document={"data": b"x", "format": "pdf"})])
        adapter.extract([Item(document={"data": b"y", "format": "pdf"})])
        assert factory.call_count == 1

    def test_extract_caches_per_ocr_key(self) -> None:
        adapter, factory = _make_adapter()
        adapter.extract([Item(document={"data": b"x", "format": "pdf"})])  # ocr=False
        adapter.extract([Item(document={"data": b"y", "format": "pdf"})], options={"ocr": True})  # ocr=True
        assert factory.call_count == 2
        adapter.extract([Item(document={"data": b"z", "format": "pdf"})])  # ocr=False (cached)
        assert factory.call_count == 2


class TestDoclingSpec:
    def test_capabilities(self) -> None:
        adapter = DoclingAdapter()
        assert adapter.capabilities.inputs == ["document", "image"]
        assert adapter.capabilities.outputs == ["json"]

    def test_unload_clears_converter_cache(self) -> None:
        adapter, _factory = _make_adapter()
        adapter.extract([Item(document={"data": b"x", "format": "pdf"})])
        assert adapter._converters  # populated
        assert adapter._loaded

        adapter.unload()  # must not raise
        assert adapter._converters == {}
        assert not adapter._loaded

        with pytest.raises(RuntimeError, match=r"load.* before extract"):
            adapter.extract([Item(document={"data": b"x", "format": "pdf"})])


class TestDoclingMakeConverter:
    def test_make_converter_no_ocr_passes_do_ocr_false(self) -> None:
        # Docling's PdfPipelineOptions defaults do_ocr=True, so we must pass
        # do_ocr=False explicitly on the default path — otherwise the `ocr`
        # profile is a no-op vs the default profile.
        adapter = DoclingAdapter()
        assert adapter._device is None

        with (
            patch("docling.document_converter.DocumentConverter") as mock_cls,
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions") as mock_opts,
            patch("docling.document_converter.PdfFormatOption") as mock_fmt_opt,
            patch("docling.document_converter.ImageFormatOption"),
        ):
            adapter._make_converter(ocr_enabled=False)

        mock_opts.assert_called_once_with(do_ocr=False)
        mock_fmt_opt.assert_called_once()
        mock_cls.assert_called_once()
        assert "format_options" in mock_cls.call_args.kwargs

    def test_make_converter_ocr_passes_pdf_pipeline_options(self) -> None:
        adapter = DoclingAdapter()

        with (
            patch("docling.document_converter.DocumentConverter") as mock_cls,
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions") as mock_opts,
            patch("docling.document_converter.PdfFormatOption") as mock_fmt_opt,
            patch("docling.document_converter.ImageFormatOption"),
        ):
            adapter._make_converter(ocr_enabled=True)

        mock_opts.assert_called_once_with(do_ocr=True)
        mock_fmt_opt.assert_called_once()
        mock_cls.assert_called_once()
        assert "format_options" in mock_cls.call_args.kwargs

    def test_make_converter_no_ocr_uses_accelerator_options_when_device_set(self) -> None:
        adapter = DoclingAdapter()
        adapter._device = "cuda"

        with (
            patch("docling.document_converter.DocumentConverter") as mock_cls,
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions") as mock_opts,
            patch("docling.datamodel.accelerator_options.AcceleratorOptions") as mock_accel,
            patch("docling.document_converter.PdfFormatOption") as mock_fmt_opt,
            patch("docling.document_converter.ImageFormatOption"),
        ):
            adapter._make_converter(ocr_enabled=False)

        mock_accel.assert_called_once_with(device="cuda")
        kwargs = mock_opts.call_args.kwargs
        assert kwargs["accelerator_options"] is mock_accel.return_value
        assert kwargs["do_ocr"] is False
        mock_fmt_opt.assert_called_once()
        mock_cls.assert_called_once()
        assert "format_options" in mock_cls.call_args.kwargs

    def test_make_converter_ocr_threads_accelerator_options(self) -> None:
        adapter = DoclingAdapter()
        adapter._device = "cuda:0"

        with (
            patch("docling.document_converter.DocumentConverter") as mock_cls,
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions") as mock_opts,
            patch("docling.datamodel.accelerator_options.AcceleratorOptions") as mock_accel,
            patch("docling.document_converter.PdfFormatOption"),
            patch("docling.document_converter.ImageFormatOption"),
        ):
            adapter._make_converter(ocr_enabled=True)

        mock_accel.assert_called_once_with(device="cuda:0")
        kwargs = mock_opts.call_args.kwargs
        assert kwargs.get("do_ocr") is True
        assert kwargs.get("accelerator_options") is mock_accel.return_value
        mock_cls.assert_called_once()

    @pytest.mark.parametrize("device", ["cuda", "cuda:0", "mps", "cpu", "auto"])
    def test_make_converter_passes_known_device_strings_through(self, device: str) -> None:
        adapter = DoclingAdapter()
        adapter._device = device

        with (
            patch("docling.document_converter.DocumentConverter"),
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions"),
            patch("docling.datamodel.accelerator_options.AcceleratorOptions") as mock_accel,
            patch("docling.document_converter.PdfFormatOption"),
            patch("docling.document_converter.ImageFormatOption"),
        ):
            adapter._make_converter(ocr_enabled=True)

        mock_accel.assert_called_once_with(device=device)

    def test_make_converter_invalid_device_falls_back_to_auto(self) -> None:
        adapter = DoclingAdapter()
        adapter._device = "definitely-not-a-device"

        with (
            patch("docling.document_converter.DocumentConverter"),
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions"),
            patch("docling.datamodel.accelerator_options.AcceleratorOptions") as mock_accel,
            patch("docling.document_converter.PdfFormatOption"),
            patch("docling.document_converter.ImageFormatOption"),
        ):
            mock_accel.side_effect = [ValueError("invalid device"), MagicMock()]
            adapter._make_converter(ocr_enabled=True)

        actual = [c.kwargs for c in mock_accel.call_args_list]
        assert actual == [
            {"device": "definitely-not-a-device"},
            {"device": "auto"},
        ]


class TestDoclingOcrPrecedence:
    """The DoclingAdapter only sees the merged ``options`` dict produced by
    ``resolve_runtime_options`` (api/options.py); it does not know about
    profiles. These tests pin down the contract that the merged dict drives
    ``ocr_enabled`` regardless of where the value came from.
    """

    def test_profile_runtime_ocr_true_makes_extract_ocr_default(self) -> None:
        """When profile resolution sets options={'ocr': True}, the adapter
        builds an OCR-enabled converter even with no per-request overrides.
        """
        adapter, factory = _make_adapter()

        # Simulates what the worker passes through after the 'ocr' profile is
        # resolved server-side: profile.runtime.ocr=true becomes options.ocr=true
        # before reaching the adapter.
        adapter.extract(
            [Item(document={"data": b"x", "format": "pdf"})],
            options={"ocr": True},
        )

        assert factory.call_count == 1
        assert factory.call_args.kwargs == {"ocr_enabled": True}

    def test_request_ocr_false_overrides_profile_ocr_true(self) -> None:
        """Per-request override wins. ``resolve_runtime_options`` emits the
        already-merged dict, so the adapter sees options={'ocr': False}
        even if the profile would have defaulted ocr=True.
        """
        adapter, factory = _make_adapter()

        adapter.extract(
            [Item(document={"data": b"x", "format": "pdf"})],
            options={"ocr": False},
        )

        assert factory.call_count == 1
        assert factory.call_args.kwargs == {"ocr_enabled": False}

    def test_request_ocr_true_works_without_profile_default(self) -> None:
        """Regression: per-request ocr=true keeps working when no profile
        default is set (i.e., the historical 'default' profile path).
        """
        adapter, factory = _make_adapter()

        adapter.extract(
            [Item(document={"data": b"x", "format": "pdf"})],
            options={"ocr": True},
        )

        assert factory.call_count == 1
        assert factory.call_args.kwargs == {"ocr_enabled": True}
