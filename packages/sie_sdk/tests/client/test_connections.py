"""client.connections namespace — control-plane request-shape assertions (mocked)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_sdk import SIEAsyncClient, SIEClient

GW = "http://gw.test:8080"
CP = "http://cp.test:9000"
KEY = "sk-sie-testkey"


def _resp(status: int, body: Any) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.headers = {"content-type": "application/json"}
    r.content = json.dumps(body).encode()
    r.json.return_value = body
    return r


class _FakeAio:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self.content = json.dumps(body).encode()
        self.headers = {"content-type": "application/json"}
        self._body = body

    def json(self) -> Any:
        return self._body


def test_add_posts_to_control_plane_with_secret() -> None:
    created = {"org": "acme", "account_id": 7, "id": 1, "type": "postgres", "name": "wh", "created_at": 1.0}
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(201, created))
        client = SIEClient(GW, api_key=KEY, control_plane_url=CP, org="acme")
        out = client.connections.add(name="wh", type="postgres", secret="postgres://u:p@h/db")  # noqa: S106
        assert out["name"] == "wh"
        method, url = mock_client.return_value.request.call_args.args
        body = mock_client.return_value.request.call_args.kwargs["json"]
        assert method == "POST"
        assert url == f"{CP}/internal/orgs/acme/connections"
        assert body == {"type": "postgres", "name": "wh", "secret": "postgres://u:p@h/db"}
        client.close()


def test_list_returns_connections_array() -> None:
    payload = {"org": "acme", "account_id": 7, "connections": [{"id": 1, "type": "postgres", "name": "wh"}]}
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(200, payload))
        client = SIEClient(GW, api_key=KEY, control_plane_url=CP, org="acme")
        conns = client.connections.list()
        assert [c["name"] for c in conns] == ["wh"]
        assert mock_client.return_value.request.call_args.args == ("GET", f"{CP}/internal/orgs/acme/connections")
        client.close()


def test_revoke_deletes_named_connection() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(
            return_value=_resp(200, {"org": "acme", "name": "wh", "state": "revoked"})
        )
        client = SIEClient(GW, api_key=KEY, control_plane_url=CP, org="acme")
        out = client.connections.revoke("wh")
        assert out["state"] == "revoked"
        assert mock_client.return_value.request.call_args.args == ("DELETE", f"{CP}/internal/orgs/acme/connections/wh")
        client.close()


def test_connections_without_control_plane_raises() -> None:
    with patch("sie_sdk.client.sync.httpx.Client"):
        client = SIEClient(GW, api_key=KEY)
        with pytest.raises(ValueError, match="control_plane_url"):
            client.connections.list()
        client.close()


def test_connections_without_org_raises() -> None:
    with patch("sie_sdk.client.sync.httpx.Client"):
        client = SIEClient(GW, api_key=KEY, control_plane_url=CP)
        with pytest.raises(ValueError, match="org"):
            client.connections.list()
        client.close()


@pytest.mark.asyncio
async def test_async_add_and_revoke() -> None:
    client = SIEAsyncClient(GW, api_key=KEY, control_plane_url=CP, org="acme")
    created = {"org": "acme", "id": 1, "type": "postgres", "name": "wh"}
    client._post = AsyncMock(return_value=_FakeAio(201, created))
    client._delete = AsyncMock(return_value=_FakeAio(200, {"name": "wh", "state": "revoked"}))
    out = await client.connections.add(name="wh", type="postgres", secret="dsn")  # noqa: S106
    assert out["name"] == "wh"
    assert client._post.call_args.args[0] == f"{CP}/internal/orgs/acme/connections"
    assert json.loads(client._post.call_args.kwargs["data"]) == {"type": "postgres", "name": "wh", "secret": "dsn"}
    revoked = await client.connections.revoke("wh")
    assert revoked["state"] == "revoked"
    assert client._delete.call_args.args[0] == f"{CP}/internal/orgs/acme/connections/wh"
    await client.close()
