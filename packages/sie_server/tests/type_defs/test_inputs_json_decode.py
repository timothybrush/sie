"""Wire-format decoding tests for media inputs (regression for #1026).

`POST /v1/{encode,score,extract}/{model}` with `Content-Type: application/json`
and a base64-encoded image must base64-decode the inner `data` field into
`bytes`, exactly like the msgpack path. msgspec only base64-decodes a JSON
string when the target field is *typed* `bytes` — so ``Item`` must reference the
``*Input`` TypedDicts (whose ``data`` is ``bytes``) rather than
``dict[str, Any]``. With ``Any``, the string flows straight into the
preprocessor's ``io.BytesIO(...)`` and raises
``TypeError: a bytes-like object is required, not 'str'``.
"""

import base64

import msgspec
import pytest
from sie_server.types.inputs import Item, is_image_input
from sie_server.types.requests import EncodeRequest, ExtractRequest, ScoreRequest

# A tiny but real 1x1 PNG so the bytes are non-trivial and round-trip cleanly.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")


def test_json_image_data_decodes_to_bytes() -> None:
    """Repro for #1026: JSON image `data` must be base64-decoded to bytes, not str."""
    body = msgspec.json.encode({"items": [{"id": "t", "images": [{"data": _PNG_B64, "format": "png"}]}]})

    request = msgspec.json.decode(body, type=EncodeRequest)

    img = request.items[0].images[0]
    assert isinstance(img["data"], bytes), "JSON image data must be base64-decoded to bytes"
    assert img["data"] == _PNG_BYTES
    # The guard every preprocessor relies on must accept the decoded payload.
    assert is_image_input(img)


def test_msgpack_image_data_stays_bytes() -> None:
    """The msgpack path already works; lock it in so the fix keeps parity."""
    body = msgspec.msgpack.encode({"items": [{"id": "t", "images": [{"data": _PNG_BYTES, "format": "png"}]}]})

    request = msgspec.msgpack.decode(body, type=EncodeRequest)

    img = request.items[0].images[0]
    assert isinstance(img["data"], bytes)
    assert img["data"] == _PNG_BYTES


@pytest.mark.parametrize("request_type", [EncodeRequest, ExtractRequest])
def test_json_image_data_decodes_to_bytes_all_request_types(request_type: type) -> None:
    """Every items-bearing request type must base64-decode image data on JSON."""
    body = msgspec.json.encode({"items": [{"id": "t", "images": [{"data": _PNG_B64, "format": "png"}]}]})

    request = msgspec.json.decode(body, type=request_type)

    assert isinstance(request.items[0].images[0]["data"], bytes)


def test_json_score_query_and_items_image_data_decodes_to_bytes() -> None:
    """ScoreRequest carries images on both `query` and `items`."""
    body = msgspec.json.encode(
        {
            "query": {"id": "q", "images": [{"data": _PNG_B64, "format": "png"}]},
            "items": [{"id": "t", "images": [{"data": _PNG_B64, "format": "png"}]}],
        }
    )

    request = msgspec.json.decode(body, type=ScoreRequest)

    assert isinstance(request.query.images[0]["data"], bytes)
    assert isinstance(request.items[0].images[0]["data"], bytes)


def test_json_audio_video_document_data_decodes_to_bytes() -> None:
    """The sibling media fields share the same latent bug; lock the contract."""
    body = msgspec.json.encode(
        {
            "items": [
                {
                    "id": "t",
                    "audio": {"data": _PNG_B64, "format": "wav", "sample_rate": 16000},
                    "video": {"data": _PNG_B64, "format": "mp4"},
                    "document": {"data": _PNG_B64, "format": "pdf"},
                }
            ]
        }
    )

    item: Item = msgspec.json.decode(body, type=EncodeRequest).items[0]

    assert isinstance(item.audio["data"], bytes)
    assert isinstance(item.video["data"], bytes)
    assert isinstance(item.document["data"], bytes)
