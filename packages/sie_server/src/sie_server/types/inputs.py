"""Input types for SIE Server API (wire format).

These types define the structure of items received over the wire after msgpack
deserialization. The SDK converts flexible Python types (PIL.Image, numpy arrays,
file paths) to these wire format types before transport.

Using TypedDict for zero runtime overhead - validation is done manually where needed.
"""

from typing import Any, TypedDict, TypeGuard, cast

import msgspec


class ImageInput(TypedDict, total=False):
    """Image input for multimodal models (wire format).

    On the wire, images are sent as bytes with format hint.

    Attributes:
        data: Image data as bytes.
        format: Image format hint: 'jpeg', 'png', etc. Inferred if not provided.
    """

    data: bytes
    format: str | None


class AudioInput(TypedDict, total=False):
    """Audio input for audio models (wire format).

    On the wire, audio is sent as bytes with format and sample rate metadata.

    Attributes:
        data: Audio data as bytes.
        format: Audio format: 'wav', 'mp3', etc.
        sample_rate: Sample rate in Hz.
    """

    data: bytes
    format: str | None
    sample_rate: int | None


class VideoInput(TypedDict, total=False):
    """Video input for video models (wire format).

    On the wire, video is sent as bytes with format hint.

    Attributes:
        data: Video data as bytes.
        format: Video format: 'mp4', 'webm', etc.
    """

    data: bytes
    format: str | None


class DocumentInput(TypedDict, total=False):
    """Document input for composite-document extractors (wire format).

    On the wire, documents are sent as bytes with a format hint
    (e.g., 'pdf', 'docx', 'html'). The hint is advisory — adapters may
    still sniff the bytes when format is missing or unrecognized.

    Attributes:
        data: Document bytes (raw file content).
        format: Document format hint: 'pdf', 'docx', 'html', etc.
    """

    data: bytes
    format: str | None


class Item(msgspec.Struct):
    """A single item to encode, score, or extract from.

    All fields are optional. Models accept text-only, image-only, or multimodal
    items depending on their capabilities.
    """

    id: str | None = None
    text: str | None = None
    # These reference the typed *Input TypedDicts (data: bytes) rather than
    # dict[str, Any] so msgspec base64-decodes the `data` field on the JSON
    # path, matching the msgpack path. See issue #1026.
    images: list[ImageInput] | None = None
    audio: AudioInput | None = None
    video: VideoInput | None = None
    document: DocumentInput | None = None
    metadata: dict[str, Any] | None = None


# =============================================================================
# Type Guards
# =============================================================================


def is_image_input(obj: Any) -> TypeGuard[ImageInput]:
    """Check if obj is a valid ImageInput dict.

    Args:
        obj: Object to validate.

    Returns:
        True if obj is a dict with 'data' key containing bytes.
    """
    return isinstance(obj, dict) and "data" in obj and isinstance(obj.get("data"), bytes)


def is_audio_input(obj: Any) -> TypeGuard[AudioInput]:
    """Check if obj is a valid AudioInput dict.

    Args:
        obj: Object to validate.

    Returns:
        True if obj is a dict with 'data' key containing bytes.
    """
    return isinstance(obj, dict) and "data" in obj and isinstance(obj.get("data"), bytes)


def is_video_input(obj: Any) -> TypeGuard[VideoInput]:
    """Check if obj is a valid VideoInput dict.

    Args:
        obj: Object to validate.

    Returns:
        True if obj is a dict with 'data' key containing bytes.
    """
    return isinstance(obj, dict) and "data" in obj and isinstance(obj.get("data"), bytes)


def is_document_input(obj: Any) -> TypeGuard[DocumentInput]:
    """Check if obj is a valid DocumentInput dict.

    Args:
        obj: Object to validate.

    Returns:
        True if obj is a dict with 'data' key containing bytes.
    """
    return isinstance(obj, dict) and "data" in obj and isinstance(obj.get("data"), bytes)


def is_item(obj: Any) -> TypeGuard[Item | dict[str, Any]]:
    """Check if obj is a valid Item or Item-like dict.

    Args:
        obj: Object to validate.

    Returns:
        True if obj is an Item Struct or a dict.
    """
    return isinstance(obj, (dict, Item))


# =============================================================================
# Validated media access
# =============================================================================


class InvalidMediaError(ValueError):
    """A media input's ``data`` field is missing or not bytes.

    Subclasses ``ValueError`` on purpose so both request paths surface it as a
    structured ``INVALID_INPUT`` (HTTP 400) rather than a generic 500:

    - HTTP / in-process: the endpoints' ``except ValueError`` routes it through
      ``InferenceErrorHandler.handle_value_error`` (see ``api/encode.py``).
    - Queue / sidecar: ``queue_executor._inference_exception_outcome`` maps it
      to ``ErrorCode.INVALID_INPUT`` before the sidecar publishes the result.

    Without this, an un-decoded base64 ``str`` slipping past the wire boundary
    raised ``TypeError: a bytes-like object is required, not 'str'`` deep inside
    a preprocessor/adapter — a generic 500, and one trigger for a malformed
    tensor reaching a CUDA kernel. See issue #1026.
    """


def media_bytes(media: object, *, kind: str = "media") -> bytes:
    """Return the validated ``data`` bytes from a media input mapping.

    Every image/video/document consumer relies on the wire contract that
    ``data`` is raw ``bytes``. msgspec only enforces that on a typed ``bytes``
    field, and only on decode paths that run through it — the queue path builds
    ``Item`` from a plain dict and bypasses that check. This is the single
    enforcement point all consumers funnel through, turning any contract
    violation into a clean :class:`InvalidMediaError` at the point of use.

    Args:
        media: The media input mapping (e.g. an :class:`ImageInput`).
        kind: Human label used in the error message ("image", "document", ...).

    Returns:
        The ``data`` as ``bytes`` (``bytearray``/``memoryview`` are coerced).

    Raises:
        InvalidMediaError: If ``media`` is not a mapping, lacks ``data``, or
            ``data`` is not a bytes-like object.
    """
    if not isinstance(media, dict):
        raise InvalidMediaError(f"{kind} input must be a mapping with a 'data' field")
    mapping = cast("dict[str, Any]", media)
    if "data" not in mapping:
        raise InvalidMediaError(f"{kind} input must be a mapping with a 'data' field")
    data = mapping["data"]
    if isinstance(data, bytes):
        return data
    if isinstance(data, (bytearray, memoryview)):
        return bytes(data)
    raise InvalidMediaError(
        f"{kind} data must be bytes, got {type(data).__name__} "
        "(base64 JSON strings must be decoded to bytes before inference)"
    )
