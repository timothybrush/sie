from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import is_document_input, is_image_input

if TYPE_CHECKING:
    from sie_server.types.inputs import Item


logger = logging.getLogger(__name__)


_ERR_REQUIRES_DOCUMENT = "Docling requires Item.document or Item.images[0] with raw bytes"

# Minimal one-page PDF used to warm Docling's layout/table model downloads
# during load() so the first user request doesn't pay the download latency.
_TINY_PDF_BYTES = (
    b"%PDF-1.1\n%\xc2\xa5\xc2\xb1\xc3\xab\n\n1 0 obj\n  << /Type /Catalog\n     /Pages 2 0 R\n  >>\n"
    b"endobj\n\n2 0 obj\n  << /Type /Pages\n     /Kids [3 0 R]\n     /Count 1\n     /MediaBox [0 0 300 144]\n  >>\n"
    b"endobj\n\n3 0 obj\n  <<  /Type /Page\n      /Parent 2 0 R\n      /Resources\n       <<"
    b" /Font\n           <<\n             /F1\n              << /Type /Font\n                 /Subtype /Type1\n"
    b"                 /BaseFont /Times-Roman\n              >>\n           >>\n       >>\n      /Contents 4 0 R\n  >>\n"
    b"endobj\n\n4 0 obj\n  << /Length 55 >>\nstream\n  BT\n    /F1 18 Tf\n    0 0 Td\n    (Hello, world!) Tj\n  ET\nendstream\n"
    b"endobj\n\nxref\n0 5\n0000000000 65535 f \n0000000018 00000 n \n0000000077 00000 n \n0000000178 00000 n \n0000000457 00000 n \n"
    b"trailer\n  <<  /Root 1 0 R\n      /Size 5\n  >>\nstartxref\n565\n%%EOF\n"
)


class DoclingAdapter(BaseAdapter):
    """Composite-document parser backed by Docling's `DocumentConverter`.

    Supports PDF, DOCX, HTML, and other formats Docling auto-detects from bytes.
    The adapter is package-backed (no HF/local weights) — Docling ships its own
    layout / table-structure models which are downloaded on first use. We pre-warm
    those during ``load()`` so user requests don't pay that latency.

    Result shape (per item, in ``ExtractOutput.data``):

        {
            "text": "...",          # plain text rendering
            "markdown": "...",      # Markdown rendering (preserves tables, headings)
            "document": {...},      # full DoclingDocument JSON for downstream chunkers
        }

    OCR is disabled by default for speed and predictability. Pass
    ``options={"ocr": True}`` per request to enable it.

    Concurrency: one ``DocumentConverter`` is cached per ``ocr_enabled`` value
    on the adapter instance. ``self._device`` is set once in ``load()`` and is
    stable for the adapter's lifetime, so the effective cache key is
    ``(self._device, ocr_enabled)`` and at most two converters ever exist per
    adapter instance. Cross-request serialization is provided by
    ``ModelWorker._inference_executor`` (max_workers=1), so the cache itself
    does not need a lock. Items within one batch are processed serially
    (rather than via a per-item thread pool) to sidestep the converter's known
    thread-safety issue (https://github.com/docling-project/docling/issues/115);
    at GPU-bound concurrency the upstream worker is already saturating the
    device, so intra-batch parallelism does not buy real throughput.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("document", "image"),
        outputs=("json",),
        unload_fields=(),
    )

    def __init__(
        self,
        model_name_or_path: str | None = None,  # unused; Docling is package-backed
        *,
        compute_precision: str | None = None,  # unused; device is threaded via load()
        **kwargs: Any,
    ) -> None:
        _ = (model_name_or_path, compute_precision, kwargs)
        self._loaded = False
        self._device: str | None = None
        self._converters: dict[bool, Any] = {}

    def load(self, device: str) -> None:
        self._device = device
        # Pre-warm: triggers Docling's lazy download of layout/table models so
        # the first real request doesn't block on a multi-hundred-MB pull.
        # Models cache globally, so subsequent per-task converters are cheap.
        try:
            warm_converter = self._get_converter(ocr_enabled=False)
            self._convert_bytes(warm_converter, _TINY_PDF_BYTES, format_hint="pdf")
            # Also build the OCR-enabled converter so the first ocr-profile
            # request doesn't pay layout+OCR model-init latency.
            self._get_converter(ocr_enabled=True)
        except Exception:
            logger.exception("Docling pre-warm failed; first real request may be slow")
        self._loaded = True

    def unload(self) -> None:
        self._converters.clear()
        self._loaded = False
        super().unload()

    def extract(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        prepared_items: list[Any] | None = None,
    ) -> ExtractOutput:
        _ = (labels, output_schema, instruction, prepared_items)
        if not self._loaded:
            msg = "DoclingAdapter.load() must be called before extract()"
            raise RuntimeError(msg)

        ocr_enabled = bool(options and options.get("ocr"))
        results = self._run_extract(items, ocr_enabled=ocr_enabled)

        return ExtractOutput(
            entities=[[] for _ in items],
            data=results,
            batch_size=len(items),
        )

    def _run_extract(self, items: list[Item], *, ocr_enabled: bool) -> list[dict[str, Any]]:
        """Run extract per-item, serially.

        Items are processed one at a time so we can share a single cached
        DocumentConverter (see class docstring). At GPU-bound concurrency the
        worker-level inference executor is already saturating the device, so
        intra-batch parallelism does not buy real throughput.
        """
        return [self._extract_one(item, ocr_enabled=ocr_enabled) for item in items]

    def _extract_one(self, item: Item, *, ocr_enabled: bool) -> dict[str, Any]:
        # Prefer document when both are provided: PDF/DOCX/HTML carry layout
        # that Docling's pipeline exploits; an image is the rasterized
        # fallback for callers that only have a page render.
        document = item.document
        if is_document_input(document):
            payload = document["data"]
            format_hint = document.get("format")
        else:
            images = item.images or []
            first_image = images[0] if images else None
            if not is_image_input(first_image):
                return {"error": _ERR_REQUIRES_DOCUMENT}
            payload = first_image["data"]
            format_hint = first_image.get("format") or "png"
        try:
            converter = self._get_converter(ocr_enabled=ocr_enabled)
            return self._convert_bytes(converter, payload, format_hint=format_hint)
        except Exception as e:  # noqa: BLE001 - per-item failure must not poison the batch
            logger.warning("Docling extract failed for item id=%s: %s", item.id, e)
            return {"error": str(e)}

    def _get_converter(self, *, ocr_enabled: bool) -> Any:
        """Return the cached DocumentConverter for this ocr_enabled value, building lazily on first use."""
        cached = self._converters.get(ocr_enabled)
        if cached is not None:
            return cached
        converter = self._make_converter(ocr_enabled=ocr_enabled)
        self._converters[ocr_enabled] = converter
        return converter

    def _convert_bytes(self, converter: Any, data: bytes, *, format_hint: str | None) -> dict[str, Any]:
        from docling.datamodel.base_models import DocumentStream  # ty: ignore[unresolved-import]

        # Docling auto-detects format from bytes; the hint becomes the source name
        # (used for logging + extension-based fallback when sniffing is ambiguous).
        source_name = f"document.{format_hint}" if format_hint else "document"
        stream = DocumentStream(name=source_name, stream=io.BytesIO(data))
        result = converter.convert(stream)
        doc = result.document
        return {
            "text": doc.export_to_text(),
            "markdown": doc.export_to_markdown(),
            "document": doc.export_to_dict(),
        }

    def _make_converter(self, *, ocr_enabled: bool) -> Any:
        """Build a fresh DocumentConverter. Callers should usually go through _get_converter() for caching.

        Threads self._device through Docling's AcceleratorOptions so layout, table,
        and OCR models actually run on the configured device. Without this, Docling
        silently defaults to CPU regardless of how SIE was launched.
        """
        from docling.document_converter import DocumentConverter  # ty: ignore[unresolved-import]

        accelerator_options = self._build_accelerator_options()

        from docling.datamodel.base_models import InputFormat  # ty: ignore[unresolved-import]
        from docling.datamodel.pipeline_options import PdfPipelineOptions  # ty: ignore[unresolved-import]
        from docling.document_converter import ImageFormatOption, PdfFormatOption  # ty: ignore[unresolved-import]

        # Pass do_ocr explicitly on both paths. Docling's PdfPipelineOptions defaults
        # do_ocr=True, so an unset default would silently OCR every PDF and make the
        # `ocr` profile a no-op vs. the default profile.
        pdf_kwargs: dict[str, Any] = {"do_ocr": ocr_enabled}
        if accelerator_options is not None:
            pdf_kwargs["accelerator_options"] = accelerator_options
        pdf_opts = PdfPipelineOptions(**pdf_kwargs)
        # Reuse the same PdfPipelineOptions for IMAGE input: Docling's image
        # pipeline shares the layout/OCR model stack with the PDF pipeline,
        # and re-using the option object keeps device/ocr settings in lock-step.
        # ImageFormatOption attaches the ImageDocumentBackend so Docling doesn't
        # auto-correct + deprecation-warn on every image request.
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pdf_opts),
            }
        )

    def _build_accelerator_options(self) -> Any:
        """Translate self._device into a Docling AcceleratorOptions, or None."""
        if not self._device:
            return None
        from docling.datamodel.accelerator_options import AcceleratorOptions  # ty: ignore[unresolved-import]

        try:
            return AcceleratorOptions(device=str(self._device))
        except Exception as e:  # noqa: BLE001 - pydantic validation; fall back to auto
            logger.warning(
                "Docling: invalid device %r, falling back to 'auto' (%s)",
                self._device,
                e,
            )
            try:
                return AcceleratorOptions(device="auto")
            except Exception:
                logger.exception("Docling: failed to build AcceleratorOptions even with 'auto'")
                return None
