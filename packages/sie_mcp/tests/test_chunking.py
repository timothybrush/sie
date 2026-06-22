import pytest
from sie_mcp.chunking import chunk_documents, chunk_text


def test_windows_cover_text_with_overlap() -> None:
    windows = chunk_text("abcdefghij", window=4, overlap=1)

    # stride = window - overlap = 3: starts at 0, 3, 6; the 6-window already
    # reaches the end so no further window is emitted.
    assert [start for start, _ in windows] == [0, 3, 6]
    assert [piece for _, piece in windows] == ["abcd", "defg", "ghij"]
    # Overlap means neighbours share a character ("d", "g").
    assert windows[0][1][-1] == windows[1][1][0]


def test_single_window_when_text_shorter_than_window() -> None:
    assert chunk_text("short", window=100, overlap=10) == [(0, "short")]


def test_empty_text_yields_no_windows() -> None:
    assert chunk_text("", window=10, overlap=2) == []


def test_rejects_overlap_not_smaller_than_window() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("abc", window=4, overlap=4)


def test_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window"):
        chunk_text("abc", window=0, overlap=0)


def test_chunk_documents_assigns_unique_ids_and_doc_index() -> None:
    chunks = chunk_documents(["abcdef", "uvwxyz"], window=3, overlap=0)

    assert [c.id for c in chunks] == ["d0-c0", "d0-c1", "d1-c0", "d1-c1"]
    assert [c.doc_index for c in chunks] == [0, 0, 1, 1]
    assert [c.text for c in chunks] == ["abc", "def", "uvw", "xyz"]
    assert [c.start for c in chunks] == [0, 3, 0, 3]


def test_chunk_documents_skips_empty_documents_but_keeps_doc_index() -> None:
    chunks = chunk_documents(["", "   ", "abc"], window=10, overlap=0)

    # Empty/whitespace docs contribute nothing; the real doc keeps its index (2).
    assert [c.doc_index for c in chunks] == [2]
    assert chunks[0].text == "abc"


def test_large_document_chunks_into_many() -> None:
    # A document far larger than one window must split into multiple chunks.
    chunks = chunk_documents(["x" * 10_000], window=1000, overlap=100)

    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)
