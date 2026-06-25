"""MCP-backed document offload jobs adapted from the PR #1336 sie_tools skills."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from sie_mcp.config import DEFAULT_EXTRACT_MODEL, DEFAULT_PII_MODEL

SUMMARY_CHUNK_CHARS = 12_000
SUMMARY_MAP_MAX_TOKENS = 400
SUMMARY_REDUCE_MAX_TOKENS = 700
SUMMARY_REDUCE_BATCH_SIZE = 12

# Defaults for direct offload callers; the MCP tools pass the env-overridable
# config values (SIE_MCP_EXTRACT_MODEL / SIE_MCP_PII_MODEL) instead.
EXTRACT_MODEL = DEFAULT_EXTRACT_MODEL
PII_MODEL = DEFAULT_PII_MODEL

DEFAULT_PII_LABELS = [
    "person",
    "email",
    "phone number",
    "address",
    "social security number",
    "credit card number",
    "passport number",
    "date of birth",
    "ip address",
]

REQUEST_TIMEOUT_S = 600.0
MIN_PII_SCORE = 0.3

# Bounded chunking for the GLiNER extract/redact path. GLiNER runs on a single
# text item per call (no internal chunking), so a large resolved document is
# split into overlapping windows, run per-window, and the entities merged with
# offsets shifted back to the original text. The overlap keeps an entity that
# straddles a window edge fully present in at least one window; a detection that
# touches an interior window edge (likely truncated) is dropped in favor of the
# neighbor window's full span. EXTRACT_MAX_CHARS bounds the fan-out so one
# request cannot spawn an unbounded number of (un-batched, ~1 rps) GLiNER calls.
EXTRACT_CHUNK_CHARS = 8_000
EXTRACT_OVERLAP_CHARS = 256
EXTRACT_MAX_CHARS = 1_000_000

_MAP_PROMPT = (
    "Summarize the following document section into at most 10 dense bullet "
    "points. Keep concrete facts, names, numbers, and decisions; drop "
    "boilerplate.\n\nSECTION:\n{chunk}\n\nBULLET SUMMARY:"
)

_REDUCE_PROMPT = (
    "The following are bullet summaries of consecutive sections of one "
    "document. Merge them into a single coherent markdown summary with a "
    "short opening paragraph and a bullet list of the key points. Do not "
    "invent facts.\n\nSECTION SUMMARIES:\n{parts}\n\nFINAL SUMMARY:"
)

_SINGLE_PROMPT = (
    "Summarize the following document as markdown: a short opening paragraph "
    "followed by a bullet list of the key points. Keep concrete facts, "
    "names, numbers, and decisions. Do not invent facts.\n\nDOCUMENT:\n{text}\n\nSUMMARY:"
)


class OffloadError(ValueError):
    """Raised when an MCP offload request is malformed or returns unusable output."""


def chunk_text(text: str, chunk_chars: int = SUMMARY_CHUNK_CHARS) -> list[str]:
    """Split text on paragraph boundaries into chunks of at most ``chunk_chars``."""
    if chunk_chars <= 0:
        raise OffloadError("chunk_chars must be positive")
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in text.split("\n\n"):
        while len(paragraph) > chunk_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            chunks.append(paragraph[:chunk_chars])
            paragraph = paragraph[chunk_chars:]
        if current_len + len(paragraph) + 2 > chunk_chars and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(paragraph)
        current_len += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))
    return [chunk for chunk in chunks if chunk.strip()]


def estimate_tokens_saved(input_chars: int, artifact_chars: int) -> int:
    """Estimate context tokens avoided when the agent reads the smaller artifact."""
    return max(input_chars // 4 - artifact_chars // 4, 0)


async def summarize_document(
    client: Any,
    *,
    content: str,
    model: str,
    gpu: str | None = None,
    max_output_tokens: int = SUMMARY_REDUCE_MAX_TOKENS,
) -> dict[str, Any]:
    """Summarize content on the SIE cluster using the PR #1336 map-reduce flow."""
    if not content.strip():
        raise OffloadError("content contains no text to summarize")

    chunks = chunk_text(content, SUMMARY_CHUNK_CHARS)
    reduce_tokens = max(1, max_output_tokens)
    map_tokens = min(SUMMARY_MAP_MAX_TOKENS, reduce_tokens)
    reduce_calls = 0
    reduction_rounds = 0

    if len(chunks) == 1:
        result = await client.generate(
            model,
            _SINGLE_PROMPT.format(text=chunks[0]),
            max_new_tokens=reduce_tokens,
            gpu=gpu,
            wait_for_capacity=True,
            provision_timeout_s=REQUEST_TIMEOUT_S,
        )
        summary = _generated_text(result)
    else:
        summaries: list[str] = []
        for chunk in chunks:
            result = await client.generate(
                model,
                _MAP_PROMPT.format(chunk=chunk),
                max_new_tokens=map_tokens,
                gpu=gpu,
                wait_for_capacity=True,
                provision_timeout_s=REQUEST_TIMEOUT_S,
            )
            summaries.append(_generated_text(result))
        summary, reduce_calls, reduction_rounds = await _reduce_summaries(
            client,
            summaries,
            model=model,
            gpu=gpu,
            max_new_tokens=reduce_tokens,
        )

    summary = summary.strip()
    if not summary:
        raise OffloadError("model returned an empty summary")

    return {
        "summary": summary,
        "metadata": {
            "input_chars": len(content),
            "summary_chars": len(summary),
            "chunks": len(chunks),
            "reduce_calls": reduce_calls,
            "reduction_rounds": reduction_rounds,
            "token_savings_estimate": estimate_tokens_saved(len(content), len(summary)),
        },
    }


async def _reduce_summaries(
    client: Any,
    summaries: list[str],
    *,
    model: str,
    gpu: str | None,
    max_new_tokens: int,
) -> tuple[str, int, int]:
    """Reduce map summaries recursively so no prompt contains every section."""
    if not summaries:
        raise OffloadError("no summaries to reduce")

    batch_size = max(2, SUMMARY_REDUCE_BATCH_SIZE)
    current = summaries
    reduce_calls = 0
    reduction_rounds = 0
    while len(current) > 1:
        reduction_rounds += 1
        next_round: list[str] = []
        for batch in _batches(current, batch_size):
            if len(batch) == 1:
                next_round.append(batch[0])
                continue
            parts = "\n\n".join(f"## Section {i}\n{part}" for i, part in enumerate(batch, start=1))
            result = await client.generate(
                model,
                _REDUCE_PROMPT.format(parts=parts),
                max_new_tokens=max_new_tokens,
                gpu=gpu,
                wait_for_capacity=True,
                provision_timeout_s=REQUEST_TIMEOUT_S,
            )
            reduce_calls += 1
            next_round.append(_generated_text(result))
        if len(next_round) >= len(current):
            raise OffloadError("summary reduction did not make progress")
        current = next_round
    return current[0], reduce_calls, reduction_rounds


def _batches(items: Sequence[str], batch_size: int) -> list[list[str]]:
    return [list(items[index : index + batch_size]) for index in range(0, len(items), batch_size)]


def _generated_text(result: Mapping[str, Any]) -> str:
    text = result.get("text")
    if not isinstance(text, str):
        raise OffloadError("generation result did not include text")
    return text


def _extract_windows(content: str, chunk_chars: int, overlap: int) -> list[tuple[int, str]]:
    """Overlapping ``(base_offset, window)`` tuples covering ``content`` exactly."""
    if len(content) <= chunk_chars:
        return [(0, content)]
    step = max(1, chunk_chars - overlap)
    windows: list[tuple[int, str]] = []
    index = 0
    length = len(content)
    while index < length:
        windows.append((index, content[index : index + chunk_chars]))
        if index + chunk_chars >= length:
            break
        index += step
    return windows


async def _gliner_entities(
    client: Any,
    *,
    content: str,
    labels: Sequence[str],
    model: str,
    gpu: str | None,
) -> list[dict[str, Any]]:
    """Run GLiNER over ``content``, chunking large inputs and merging offset-adjusted spans.

    GLiNER is called on one text item per request, so a document larger than
    ``EXTRACT_CHUNK_CHARS`` is split into overlapping windows; each window's
    entities are shifted back to the original offsets and de-duplicated. A
    detection clipped at an interior window edge is dropped (its overlapping
    neighbor holds the full span). Inputs over ``EXTRACT_MAX_CHARS`` are rejected
    rather than fanned into an unbounded number of GLiNER calls.
    """
    if len(content) > EXTRACT_MAX_CHARS:
        raise OffloadError(
            f"content is {len(content)} characters, over the {EXTRACT_MAX_CHARS}-character "
            "extraction limit; summarize or section the document first, or send a smaller piece"
        )

    async def _run(text: str) -> list[Mapping[str, Any]]:
        result = await client.extract(
            model,
            {"text": text},
            labels=list(labels),
            gpu=gpu,
            wait_for_capacity=True,
            provision_timeout_s=REQUEST_TIMEOUT_S,
        )
        return list(result.get("entities", []))

    windows = _extract_windows(content, EXTRACT_CHUNK_CHARS, EXTRACT_OVERLAP_CHARS)
    if len(windows) == 1:
        return [dict(entity) for entity in await _run(content)]

    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    last = len(windows) - 1
    for position, (base, window) in enumerate(windows):
        window_len = len(window)
        for entity in await _run(window):
            start = entity.get("start")
            end = entity.get("end")
            # Drop detections clipped at an interior window edge; the overlapping
            # neighbor window holds the full, untruncated span.
            if position != 0 and start == 0:
                continue
            if position != last and end == window_len:
                continue
            shifted = dict(entity)
            if start is not None:
                shifted["start"] = int(start) + base
            if end is not None:
                shifted["end"] = int(end) + base
            key = (shifted.get("start"), shifted.get("end"), shifted.get("label"), shifted.get("text"))
            existing = merged.get(key)
            if existing is None or float(shifted.get("score", 0.0)) > float(existing.get("score", 0.0)):
                merged[key] = shifted
    return sorted(
        merged.values(),
        key=lambda entity: (entity["start"] if entity.get("start") is not None else 0, str(entity.get("label", ""))),
    )


async def extract_entities(
    client: Any,
    *,
    content: str,
    labels: list[str],
    model: str = EXTRACT_MODEL,
    gpu: str | None = None,
) -> dict[str, Any]:
    """Extract zero-shot entities and return a compact table."""
    if not content.strip():
        raise OffloadError("content contains no text to extract from")
    labels = [label.strip() for label in labels if label.strip()]
    if not labels:
        raise OffloadError("at least one entity label is required")

    entities = await _gliner_entities(client, content=content, labels=labels, model=model, gpu=gpu)
    table = entities_table(entities, labels)
    return {
        "entities": entities,
        "markdown_table": table,
        "metadata": {
            "input_chars": len(content),
            "entity_count": len(entities),
            "label_counts": dict(Counter(entity.get("label", "") for entity in entities)),
            "token_savings_estimate": estimate_tokens_saved(len(content), len(table)),
        },
    }


def entities_table(entities: Sequence[Mapping[str, Any]], labels: list[str]) -> str:
    """Render extracted entities as a markdown table."""
    lines = ["| Entity | Label | Score |", "| --- | --- | --- |"]
    for entity in entities:
        text = str(entity.get("text", ""))
        label = str(entity.get("label", ""))
        score = float(entity.get("score", 0.0))
        lines.append(f"| {text} | {label} | {score:.2f} |")
    header = f"# Extracted entities ({', '.join(labels)})\n"
    return header + "\n" + "\n".join(lines) + "\n"


async def redact_pii(
    client: Any,
    *,
    content: str,
    labels: list[str] | None = None,
    model: str = PII_MODEL,
    gpu: str | None = None,
) -> dict[str, Any]:
    """Redact PII spans and return only the redacted text plus counts."""
    if not content.strip():
        raise OffloadError("content contains no text to redact")
    resolved_labels = labels or DEFAULT_PII_LABELS

    entities = await _gliner_entities(client, content=content, labels=resolved_labels, model=model, gpu=gpu)
    redacted, placeholder_map, span_count = redact_text(content, entities)
    return {
        "redacted_text": redacted,
        "metadata": {
            "span_count": span_count,
            "label_counts": _label_counts_from_placeholders(placeholder_map),
            "token_savings_estimate": estimate_tokens_saved(len(content), len(redacted)),
            "pii_map_returned": False,
        },
    }


def _placeholder_base(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", label.upper()).strip("_")


def redact_text(text: str, entities: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, str], int]:
    """Replace detected spans with placeholders, keeping original values server-side only."""
    spans = [
        entity
        for entity in entities
        if entity.get("start") is not None
        and entity.get("end") is not None
        and float(entity.get("score", 1.0)) >= MIN_PII_SCORE
    ]
    spans.sort(key=lambda entity: (-float(entity.get("score", 0.0)), int(entity["start"])))

    chosen: list[Mapping[str, Any]] = []
    taken: list[tuple[int, int]] = []
    for span in spans:
        start, end = int(span["start"]), int(span["end"])
        if any(start < taken_end and end > taken_start for taken_start, taken_end in taken):
            continue
        chosen.append(span)
        taken.append((start, end))

    placeholder_map: dict[str, str] = {}
    label_counters: dict[str, int] = {}
    assignments: dict[tuple[str, str], str] = {}
    for span in chosen:
        start, end = int(span["start"]), int(span["end"])
        label = str(span["label"])
        key = (text[start:end], label)
        if key not in assignments:
            base = _placeholder_base(label)
            label_counters[base] = label_counters.get(base, 0) + 1
            placeholder = f"[{base}_{label_counters[base]}]"
            assignments[key] = placeholder
            placeholder_map[placeholder] = key[0]

    redacted = text
    for span in sorted(chosen, key=lambda entity: int(entity["start"]), reverse=True):
        start, end = int(span["start"]), int(span["end"])
        key = (text[start:end], str(span["label"]))
        redacted = redacted[:start] + assignments[key] + redacted[end:]

    return redacted, placeholder_map, len(chosen)


def _label_counts_from_placeholders(placeholder_map: Mapping[str, str]) -> dict[str, int]:
    label_counts: dict[str, int] = {}
    for placeholder in placeholder_map:
        base = placeholder.strip("[]").rsplit("_", 1)[0].lower().replace("_", " ")
        label_counts[base] = label_counts.get(base, 0) + 1
    return label_counts
