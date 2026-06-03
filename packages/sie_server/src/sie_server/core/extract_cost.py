"""Per-item batching cost for the extract path.

The cost surface is consumed by ``BatchFormer`` (see
``core/batcher.py``) when deciding how many extract items can pack
into one forward pass. Two distinct adapter shapes feed this:

1. **Encoder-only** (GLiNER, GLiClass, …) — runtime is dominated by
   the input pass. Char/byte count is a faithful proxy.

2. **Decoder OCR** (LightOnOCR, GLM-OCR, PaddleOCR-VL, …) — runtime
   grows with both input *and* generated output, because KV cache
   inflates as tokens are decoded. Char/byte count under-counts the
   output side, so the batcher over-packs and the GPU stalls or
   OOMs. This is the bug behind issue #33.

The fix: callers operating on decoder-OCR adapters pass
``decoder_max_output_tokens`` so the cost reflects the worst-case
KV growth during decode. Encoder-only callers pass nothing — default
zero preserves the existing behaviour exactly.

The output-token uplift is intentionally additive (not multiplicative)
on the input cost. The two quantities have different units in this
function (byte count for documents, tokens for the decoder output);
adding them is approximation, not exact accounting. It is good
*enough* for the BatchFormer's packing decision — the goal is to stop
the batcher from packing 64 large images into one batch, not to
predict GPU runtime to the millisecond. A future refinement could
introduce per-adapter cost calibration; the current surface
deliberately keeps the API simple so all decoder-OCR adapters can
adopt it without coordinating on a calibration table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sie_server.core.prepared import ExtractPreparedItem
from sie_server.types.inputs import is_document_input

if TYPE_CHECKING:
    from sie_server.types.inputs import Item


def extract_item_cost(
    item: Item,
    *,
    decoder_max_output_tokens: int = 0,
) -> int:
    """Return the batching cost for a single extract item.

    For document items the input cost is the byte size of the raw
    document so that very large PDFs/DOCX inputs do not get bundled
    into a single batch with other heavy items. For text items the
    input cost is the character count, which matches the historical
    behavior used by GLiNER/GLiClass adapters.

    Args:
        item: The extract item.
        decoder_max_output_tokens: Worst-case generated-output token
            count for decoder-OCR adapters. Added to the input cost to
            reflect KV-cache growth during decode. Encoder-only callers
            (GLiNER, GLiClass) leave this at 0 — backward-compatible
            with the pre-#33 behaviour. Decoder-OCR callers should
            pass the model's configured ``max_output_tokens`` (typically
            from ``model_config.tasks.extract.max_output_tokens`` or
            the equivalent generate-task field).
    """
    document = item.document
    if is_document_input(document):
        input_cost = len(document["data"])
    elif item.text:
        input_cost = len(item.text)
    else:
        input_cost = 0
    # ``decoder_max_output_tokens`` is in tokens; ``input_cost`` is in
    # bytes/chars. Adding them is intentional approximation — see the
    # module docstring. Negative values are silently clamped to zero so
    # a config typo doesn't subtract from the input cost.
    return input_cost + max(decoder_max_output_tokens, 0)


def build_extract_prepared_items(
    items: list[Item],
    *,
    decoder_max_output_tokens: int = 0,
) -> list[ExtractPreparedItem]:
    """Build PreparedItems for a batch of extract items.

    ``decoder_max_output_tokens`` is forwarded to :func:`extract_item_cost`
    for each item — see that function's docstring for the rationale.
    Decoder-OCR call sites must pass this; encoder-only call sites
    leave it at the default and get the pre-#33 byte-count behaviour.
    """
    return [
        ExtractPreparedItem(
            cost=extract_item_cost(item, decoder_max_output_tokens=decoder_max_output_tokens),
            original_index=i,
        )
        for i, item in enumerate(items)
    ]
