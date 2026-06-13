from __future__ import annotations

from typing import Any

from sie_server.adapters.qwen3_vl_reranker import _build_reranker_conversation

_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def _render_user_image_tokens(messages: list[dict[str, Any]]) -> str:
    """Render the image-token behavior relevant to Qwen3-VL-Reranker's template."""
    tokens = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                tokens.append(_IMAGE_PLACEHOLDER)
    return "".join(tokens)


def test_document_image_is_in_user_message_for_vision_token_rendering() -> None:
    """Document images render in user content so Qwen emits image placeholders."""
    doc_image: Any = object()

    conversation = _build_reranker_conversation(
        query_text="solid propellant rocket nozzle cross-section",
        doc_image=doc_image,
        instruction="Retrieve images or text relevant to the user's query.",
    )

    assert [message["role"] for message in conversation] == ["system", "user"]
    assert _render_user_image_tokens(conversation).count(_IMAGE_PLACEHOLDER) == 1
    assert [
        part["image"]
        for message in conversation
        if message["role"] == "user"
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image"
    ] == [doc_image]


def test_non_empty_query_with_image_document_places_image_after_document_marker() -> None:
    """Document images stay under the document section, not the query section."""
    doc_image: Any = object()

    conversation = _build_reranker_conversation(
        query_text="solid propellant rocket nozzle cross-section",
        doc_image=doc_image,
        instruction="Retrieve images or text relevant to the user's query.",
    )

    user_content = conversation[1]["content"]
    text_parts = [part["text"] for part in user_content if isinstance(part, dict) and part.get("type") == "text"]
    document_marker_index = next(
        idx
        for idx, part in enumerate(user_content)
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text") == "\n<Document>:"
    )
    image_index = next(
        idx
        for idx, part in enumerate(user_content)
        if isinstance(part, dict) and part.get("type") == "image" and part.get("image") is doc_image
    )

    assert text_parts == [
        "<Instruct>: Retrieve images or text relevant to the user's query.",
        "<Query>:",
        "solid propellant rocket nozzle cross-section",
        "\n<Document>:",
    ]
    assert image_index > document_marker_index
    assert _render_user_image_tokens(conversation).count(_IMAGE_PLACEHOLDER) == 1


def test_empty_query_or_document_side_uses_null_placeholder() -> None:
    """Empty query or document sides use the upstream reranker NULL sentinel."""
    conversation = _build_reranker_conversation(
        doc_text="Cross-section drawing of a solid propellant rocket motor nozzle.",
        instruction="Retrieve relevant documents for the query.",
    )

    user_content = conversation[1]["content"]
    rendered_text = [part["text"] for part in user_content if isinstance(part, dict) and part.get("type") == "text"]

    assert rendered_text == [
        "<Instruct>: Retrieve relevant documents for the query.",
        "<Query>:",
        "NULL",
        "\n<Document>:",
        "Cross-section drawing of a solid propellant rocket motor nozzle.",
    ]
