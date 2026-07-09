"""Pure helpers for the files surface (``client.files``) — no transport.

The files API is the OpenAI-compatible file store. ``client.files.upload``
binds to ``POST /v1/files``; this module owns the transport-free piece — turning
a path / bytes / file-like into ``(content, filename)`` — so it stays
unit-testable without a gateway (mirroring ``sie_sdk.jobs``).
"""

from __future__ import annotations

from pathlib import Path
from typing import IO


def resolve_upload(
    file: str | Path | bytes | bytearray | IO[bytes],
    filename: str | None = None,
) -> tuple[bytes, str]:
    """Normalize an upload argument to ``(content_bytes, filename)``.

    Accepts a filesystem path (``str``/``Path``), raw ``bytes``, or a binary
    file-like object (anything with ``.read()``) — the same inputs OpenAI's
    ``files.create(file=...)`` accepts. ``filename`` overrides the derived name;
    a file-like's ``.name`` (which may be a full path) is reduced to its
    basename.

    Raises:
        TypeError: If ``file`` is none of the accepted shapes.
    """
    if isinstance(file, (bytes, bytearray)):
        return bytes(file), filename or "upload.jsonl"
    if isinstance(file, (str, Path)):
        path = Path(file)
        return path.read_bytes(), filename or path.name
    read = getattr(file, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        name = filename or getattr(file, "name", None) or "upload.jsonl"
        return bytes(data), Path(str(name)).name
    msg = f"file must be a path, bytes, or a binary file-like object, got {type(file).__name__}"
    raise TypeError(msg)
