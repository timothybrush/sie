from __future__ import annotations

from typing import Any

import pytest
from sie_mcp import offload


class _FakeOffloadClient:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.extract_calls: list[dict[str, Any]] = []

    async def generate(self, model: str, prompt: str, **kwargs: Any) -> dict[str, str]:
        self.generate_calls.append({"model": model, "prompt": prompt, **kwargs})
        return {"text": f"summary {len(self.generate_calls)}"}

    async def extract(self, model: str, items: dict[str, str], **kwargs: Any) -> dict[str, Any]:
        self.extract_calls.append({"model": model, "items": items, **kwargs})
        if model == offload.PII_MODEL:
            text = items["text"]
            return {
                "entities": [
                    {
                        "text": "Alice Example",
                        "label": "person",
                        "start": text.index("Alice Example"),
                        "end": text.index("Alice Example") + len("Alice Example"),
                        "score": 0.97,
                    },
                    {
                        "text": "alice@example.com",
                        "label": "email",
                        "start": text.index("alice@example.com"),
                        "end": text.index("alice@example.com") + len("alice@example.com"),
                        "score": 0.98,
                    },
                ]
            }
        return {
            "entities": [
                {"text": "Alice Example", "label": "person", "score": 0.93},
                {"text": "Superlinked", "label": "organization", "score": 0.88},
            ]
        }


async def test_summarize_document_uses_map_reduce_for_large_content() -> None:
    client = _FakeOffloadClient()
    text = ("alpha " * 1200) + "\n\n" + ("bravo " * 1200)

    result = await offload.summarize_document(
        client,
        content=text,
        model="gen",
        gpu="l4",
        max_output_tokens=128,
    )

    assert result["summary"] == "summary 3"
    assert result["metadata"]["chunks"] == 2
    assert result["metadata"]["token_savings_estimate"] > 0
    assert [call["model"] for call in client.generate_calls] == ["gen", "gen", "gen"]
    assert all(call["gpu"] == "l4" for call in client.generate_calls)


async def test_summarize_document_reduces_many_chunks_hierarchically() -> None:
    client = _FakeOffloadClient()
    chunk_count = offload.SUMMARY_REDUCE_BATCH_SIZE + 1
    text = "\n\n".join("x" * offload.SUMMARY_CHUNK_CHARS for _ in range(chunk_count))

    result = await offload.summarize_document(
        client,
        content=text,
        model="gen",
        max_output_tokens=128,
    )

    reduce_prompts = [call["prompt"] for call in client.generate_calls if "SECTION SUMMARIES:" in call["prompt"]]
    assert result["metadata"]["chunks"] == chunk_count
    assert result["metadata"]["reduce_calls"] == 2
    assert result["metadata"]["reduction_rounds"] == 2
    assert result["summary"] == f"summary {chunk_count + 2}"
    assert len(client.generate_calls) == chunk_count + 2
    assert all(prompt.count("## Section") <= offload.SUMMARY_REDUCE_BATCH_SIZE for prompt in reduce_prompts)


async def test_extract_entities_returns_entities_and_markdown_table() -> None:
    client = _FakeOffloadClient()

    result = await offload.extract_entities(
        client,
        content="Alice Example works at Superlinked.",
        labels=["person", "organization"],
        model="ner",
    )

    assert result["metadata"]["entity_count"] == 2
    assert result["entities"][0]["text"] == "Alice Example"
    assert "| Alice Example | person | 0.93 |" in result["markdown_table"]
    assert client.extract_calls[0]["labels"] == ["person", "organization"]


async def test_redact_pii_returns_redacted_text_without_original_map() -> None:
    client = _FakeOffloadClient()

    result = await offload.redact_pii(
        client,
        content="Alice Example can be reached at alice@example.com.",
    )

    assert result["redacted_text"] == "[PERSON_1] can be reached at [EMAIL_1]."
    assert result["metadata"]["span_count"] == 2
    assert result["metadata"]["label_counts"] == {"person": 1, "email": 1}
    assert result["metadata"]["pii_map_returned"] is False
    assert "Alice Example" not in result["redacted_text"]


def test_redact_text_drops_overlapping_lower_score_spans() -> None:
    redacted, placeholder_map, count = offload.redact_text(
        "Alice Example",
        [
            {"label": "person", "start": 0, "end": 5, "score": 0.5},
            {"label": "person", "start": 0, "end": 13, "score": 0.9},
        ],
    )

    assert redacted == "[PERSON_1]"
    assert placeholder_map == {"[PERSON_1]": "Alice Example"}
    assert count == 1


@pytest.mark.parametrize(
    ("content", "labels"),
    [
        ("", ["person"]),
        ("content", []),
    ],
)
async def test_extract_entities_rejects_bad_input(content: str, labels: list[str]) -> None:
    with pytest.raises(offload.OffloadError):
        await offload.extract_entities(_FakeOffloadClient(), content=content, labels=labels)


class _WindowAwareClient:
    """Returns each known phrase present in the window with start/end relative to
    that window, mimicking GLiNER being called per chunk.
    """

    def __init__(self, phrases: list[tuple[str, str]]) -> None:
        self.phrases = phrases
        self.windows_seen: list[str] = []

    async def extract(self, model: str, items: dict[str, str], **kwargs: Any) -> dict[str, Any]:
        text = items["text"]
        self.windows_seen.append(text)
        entities: list[dict[str, Any]] = []
        for phrase, label in self.phrases:
            idx = text.find(phrase)
            while idx != -1:
                entities.append({"text": phrase, "label": label, "start": idx, "end": idx + len(phrase), "score": 0.9})
                idx = text.find(phrase, idx + 1)
        return {"entities": entities}


async def test_extract_entities_chunks_large_content_and_shifts_offsets() -> None:
    phrase = "Acme Corporation"
    # Past the first window, so it is only found in a later chunk.
    content = ("x" * (offload.EXTRACT_CHUNK_CHARS + 300)) + " " + phrase + " " + ("y" * 200)
    abs_start = content.index(phrase)
    assert abs_start > offload.EXTRACT_CHUNK_CHARS

    client = _WindowAwareClient([(phrase, "organization")])
    result = await offload.extract_entities(client, content=content, labels=["organization"], model="gliner")

    assert len(client.windows_seen) > 1  # actually chunked
    matches = [e for e in result["entities"] if e["text"] == phrase]
    assert len(matches) == 1
    assert matches[0]["start"] == abs_start
    assert matches[0]["end"] == abs_start + len(phrase)


async def test_extract_entities_dedupes_phrase_in_window_overlap() -> None:
    phrase = "Globex Inc"
    # Inside the overlap region, so it is seen in two adjacent windows.
    pos = offload.EXTRACT_CHUNK_CHARS - offload.EXTRACT_OVERLAP_CHARS + 20
    content = ("x" * pos) + phrase + ("z" * 1000)
    client = _WindowAwareClient([(phrase, "organization")])

    result = await offload.extract_entities(client, content=content, labels=["organization"], model="gliner")

    assert len(client.windows_seen) > 1
    matches = [e for e in result["entities"] if e["text"] == phrase]
    assert len(matches) == 1
    assert matches[0]["start"] == pos


async def test_redact_pii_chunks_large_content_and_redacts_full_span() -> None:
    ssn = "123-45-6789"
    content = ("x" * (offload.EXTRACT_CHUNK_CHARS + 200)) + " SSN: " + ssn + " end"
    client = _WindowAwareClient([(ssn, "social security number")])

    result = await offload.redact_pii(client, content=content, labels=["social security number"], model="pii")

    assert len(client.windows_seen) > 1
    assert ssn not in result["redacted_text"]
    assert "[SOCIAL_SECURITY_NUMBER_1]" in result["redacted_text"]
    assert result["metadata"]["span_count"] == 1


@pytest.mark.parametrize("tool", ["extract", "redact"])
async def test_extraction_rejects_content_over_cap(tool: str) -> None:
    client = _FakeOffloadClient()
    oversized = "z" * (offload.EXTRACT_MAX_CHARS + 1)
    with pytest.raises(offload.OffloadError, match="extraction limit"):
        if tool == "extract":
            await offload.extract_entities(client, content=oversized, labels=["person"], model="g")
        else:
            await offload.redact_pii(client, content=oversized, labels=["person"], model="g")
