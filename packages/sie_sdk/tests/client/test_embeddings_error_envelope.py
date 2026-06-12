"""Regression: SIE SDK must parse the OpenAI ``{error:{…}}`` envelope.

``/v1/embeddings`` returns the OpenAI-shaped error envelope
(``{"error": {"message", "type", "param", "code"}}``) on
*every* error path, not the SIE-native ``{"detail": {"code", "message"}}``.
A SIE SDK user hitting ``/v1/embeddings`` directly must still get a typed
:class:`RequestError` / :class:`ServerError` with the parsed ``code`` and
``message`` — the shared parser already prefers ``error`` over ``detail``,
and this test locks that behaviour in so it cannot silently regress.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sie_sdk.client._shared import get_error_code, get_error_detail, handle_error
from sie_sdk.client.errors import RequestError, ServerError


def _openai_error_resp(
    status_code: int,
    *,
    code: str,
    err_type: str,
    message: str,
    param: str | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = {"error": {"message": message, "type": err_type, "param": param, "code": code}}
    return resp


def test_get_error_detail_reads_openai_envelope() -> None:
    resp = _openai_error_resp(
        400, code="invalid_request", err_type="invalid_request_error", message="bad", param="model"
    )
    detail = get_error_detail(resp)
    assert detail is not None
    assert detail["code"] == "invalid_request"
    assert detail["type"] == "invalid_request_error"
    assert detail["param"] == "model"
    assert get_error_code(resp) == "invalid_request"


def test_handle_error_4xx_openai_envelope_raises_request_error() -> None:
    resp = _openai_error_resp(
        400,
        code="invalid_request",
        err_type="invalid_request_error",
        message='field "model" is required',
        param="model",
    )
    with pytest.raises(RequestError) as excinfo:
        handle_error(resp)
    assert excinfo.value.code == "invalid_request"
    assert excinfo.value.status_code == 400
    assert str(excinfo.value) == 'field "model" is required'


def test_handle_error_5xx_openai_envelope_raises_server_error() -> None:
    # Inner /v1/encode 503 re-surfaced through embeddings as a translated
    # OpenAI envelope: server_error / transport_failure.
    resp = _openai_error_resp(503, code="transport_failure", err_type="server_error", message="queue unavailable")
    with pytest.raises(ServerError) as excinfo:
        handle_error(resp)
    assert excinfo.value.code == "transport_failure"
    assert excinfo.value.status_code == 503
