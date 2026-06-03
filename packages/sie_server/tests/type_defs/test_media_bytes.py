"""Tests for the validated media accessor (defense-in-depth for #1026).

`media_bytes` is the single point every image/video/document consumer funnels
through. It must accept bytes-like payloads and reject anything else — most
importantly an un-decoded base64 ``str`` — with an :class:`InvalidMediaError`
that both request paths map to a structured ``INVALID_INPUT`` (HTTP 400)
instead of a generic 500.
"""

import pytest
from sie_server.types.inputs import InvalidMediaError, media_bytes


class TestMediaBytesAccepts:
    """Valid bytes-like payloads pass through unchanged."""

    def test_bytes(self) -> None:
        assert media_bytes({"data": b"\x89PNG"}) == b"\x89PNG"

    def test_bytes_with_format(self) -> None:
        assert media_bytes({"data": b"img", "format": "png"}, kind="image") == b"img"

    def test_empty_bytes(self) -> None:
        # Empty is still bytes; content validation happens downstream (PIL).
        assert media_bytes({"data": b""}) == b""

    def test_bytearray_is_coerced(self) -> None:
        out = media_bytes({"data": bytearray(b"abc")})
        assert out == b"abc"
        assert isinstance(out, bytes)

    def test_memoryview_is_coerced(self) -> None:
        out = media_bytes({"data": memoryview(b"abc")})
        assert out == b"abc"
        assert isinstance(out, bytes)


class TestMediaBytesRejects:
    """Contract violations raise InvalidMediaError (a ValueError)."""

    def test_str_data_is_rejected(self) -> None:
        """The exact #1026 failure mode: base64 str that was never decoded."""
        with pytest.raises(InvalidMediaError, match="data must be bytes, got str"):
            media_bytes({"data": "aGVsbG8="}, kind="image")

    def test_error_is_a_value_error(self) -> None:
        """ValueError subclassing is what routes it to HTTP 400 INVALID_INPUT."""
        assert issubclass(InvalidMediaError, ValueError)

    def test_kind_appears_in_message(self) -> None:
        with pytest.raises(InvalidMediaError, match="document data must be bytes"):
            media_bytes({"data": 123}, kind="document")

    def test_none_data_is_rejected(self) -> None:
        with pytest.raises(InvalidMediaError):
            media_bytes({"data": None})

    def test_int_data_is_rejected(self) -> None:
        with pytest.raises(InvalidMediaError):
            media_bytes({"data": 123})

    def test_missing_data_key_is_rejected(self) -> None:
        with pytest.raises(InvalidMediaError, match="must be a mapping with a 'data' field"):
            media_bytes({"format": "png"})

    def test_non_mapping_is_rejected(self) -> None:
        with pytest.raises(InvalidMediaError):
            media_bytes(None)
        with pytest.raises(InvalidMediaError):
            media_bytes(b"raw bytes")
        with pytest.raises(InvalidMediaError):
            media_bytes("string")


class TestWorkerErrorClassification:
    """The queue/sidecar path maps InvalidMediaError to INVALID_INPUT, not inference_error."""

    def test_invalid_media_maps_to_invalid_input(self) -> None:
        from sie_server.ipc_types import EncodeBatchItem
        from sie_server.queue_executor import _inference_exception_outcome
        from sie_server.types.responses import ErrorCode

        outcome = _inference_exception_outcome(
            EncodeBatchItem(
                work_item_id="wi-1",
                request_id="req-1",
                item_index=0,
                total_items=1,
                timestamp=0.0,
                item={},
                output_types=["dense"],
            ),
            InvalidMediaError("image data must be bytes, got str"),
        )

        assert outcome.disposition == "publish_error_and_ack"
        assert outcome.error_code == ErrorCode.INVALID_INPUT.value
        assert outcome.error is not None
        assert "must be bytes" in outcome.error

    def test_generic_error_still_maps_to_inference_error(self) -> None:
        from sie_server.ipc_types import EncodeBatchItem
        from sie_server.queue_executor import _inference_exception_outcome

        outcome = _inference_exception_outcome(
            EncodeBatchItem(
                work_item_id="wi-1",
                request_id="req-1",
                item_index=0,
                total_items=1,
                timestamp=0.0,
                item={},
                output_types=["dense"],
            ),
            RuntimeError("boom"),
        )

        assert outcome.disposition == "publish_error_and_ack"
        assert outcome.error_code == "inference_error"
