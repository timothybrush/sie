"""client.jobs namespace — request-shape assertions against a mocked transport."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest
from sie_sdk import SIEAsyncClient, SIEClient

GW = "http://gw.test:8080"
KEY = "sk-sie-testkey"

_SUBMIT_RESP = {
    "id": "job-1",
    "object": "job",
    "operation": "encode",
    "model": "BAAI/bge-m3",
    "state": "queued",
    "total_items": 2,
    "chunks": 1,
    "preflight": {"estimated_credits": 64},
}


def _resp(status: int, body: Any) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.headers = {"content-type": "application/json"}
    payload = json.dumps(body).encode()
    r.content = payload
    r.json.return_value = body
    return r


class _FakeAio:
    """Minimal stand-in for the async client's _AioResponse."""

    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self.content = json.dumps(body).encode()
        self.headers = {"content-type": "application/json"}
        self._body = body

    def json(self) -> Any:
        return self._body


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


def test_submit_inline_maps_items_and_posts_v1_jobs() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(201, _SUBMIT_RESP))
        client = SIEClient(GW, api_key=KEY)
        result = client.jobs.submit(source=["a", "b"], model="BAAI/bge-m3")
        assert result["id"] == "job-1"
        method, url = mock_client.return_value.request.call_args.args
        body = mock_client.return_value.request.call_args.kwargs["json"]
        assert (method, url) == ("POST", "/v1/jobs")
        assert body == {"operation": "encode", "model": "BAAI/bge-m3", "items": [{"text": "a"}, {"text": "b"}]}
        client.close()


def test_submit_connector_job_body() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(201, _SUBMIT_RESP))
        client = SIEClient(GW, api_key=KEY)
        client.jobs.submit(
            source="postgres://warehouse?query=x",
            model="BAAI/bge-m3",
            sink="s3://out/vecs",
        )
        body = mock_client.return_value.request.call_args.kwargs["json"]
        assert body["src"] == "postgres://warehouse?query=x"
        assert body["connection"] == "warehouse"
        assert body["sink"] == "s3://out/vecs"
        assert body["sink_connection"] == "out"
        client.close()


def test_submit_field_map_job_body() -> None:
    """field_map + output_field ride the wire; upload:// derives no connection."""
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(201, _SUBMIT_RESP))
        client = SIEClient(GW, api_key=KEY)
        client.jobs.submit(
            source="upload://file-abc?format=csv",
            model="BAAI/bge-m3",
            sink="upload://file-out",
            field_map={"id_field": "doc_id", "input_field": "text", "carry": ["source_url"], "input_type": "text"},
            output_field="embedding",
        )
        body = mock_client.return_value.request.call_args.kwargs["json"]
        assert body["src"] == "upload://file-abc?format=csv"
        assert body["sink"] == "upload://file-out"
        assert "connection" not in body  # internal scheme: OUR store, no org connection
        assert body["field_map"] == {
            "id_field": "doc_id",
            "input_field": "text",
            "carry": ["source_url"],
            "input_type": "text",
        }
        assert body["output_field"] == "embedding"
        client.close()


def test_submit_score_options_query_rides_the_wire() -> None:
    """Op inputs ride `options` (op matrix): a score job's query reaches the body."""
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(201, _SUBMIT_RESP))
        client = SIEClient(GW, api_key=KEY)
        client.jobs.submit(
            source="postgres://warehouse?query=x",
            model="BAAI/bge-m3",
            operation="score",
            sink="postgres://warehouse?table=scores",
            options={"query": "rank these documents"},
        )
        body = mock_client.return_value.request.call_args.kwargs["json"]
        assert body["operation"] == "score"
        assert body["options"] == {"query": "rank these documents"}
        client.close()


def test_get_and_cancel_hit_expected_urls() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(200, {"id": "job-1", "state": "cancelled"}))
        client = SIEClient(GW, api_key=KEY)
        client.jobs.get("job-1")
        assert mock_client.return_value.request.call_args.args == ("GET", "/v1/jobs/job-1")
        out = client.jobs.cancel("job-1")
        assert mock_client.return_value.request.call_args.args == ("POST", "/v1/jobs/job-1/cancel")
        assert out["state"] == "cancelled"
        client.close()


def test_list_returns_data_array() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(
            return_value=_resp(200, {"object": "list", "data": [{"id": "job-1"}, {"id": "job-2"}]})
        )
        client = SIEClient(GW, api_key=KEY)
        jobs = client.jobs.list()
        assert [j["id"] for j in jobs] == ["job-1", "job-2"]
        assert mock_client.return_value.request.call_args.args == ("GET", "/v1/jobs")
        client.close()


def test_results_reads_local_refs_and_decodes(tmp_path: Any) -> None:
    ref = tmp_path / "chunk0.msgpack"
    ref.write_bytes(
        msgpack.packb(
            [
                {
                    "success": True,
                    "id": str(i),
                    "units": {"input_tokens": 5},
                    "result_msgpack": msgpack.packb(
                        {"dense": {"dims": 4, "values": [0.1, 0.2, 0.3, 0.4]}}, use_bin_type=True
                    ),
                }
                for i in range(3)
            ],
            use_bin_type=True,
        )
    )
    job = {
        "id": "job-1",
        "state": "succeeded",
        "total_items": 3,
        "settled_credits": 15,
        "output": {"kind": "refs", "chunks": [{"seq": 0, "items": 3, "state": "succeeded", "ref": str(ref)}]},
    }
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(200, job))
        client = SIEClient(GW, api_key=KEY)
        results = client.jobs.results("job-1")
        assert results["retrieved"] == 3
        assert results["dims"] == 4
        assert results["items"][0]["dense"].shape == (4,)
        client.close()


# ---------------------------------------------------------------------------
# async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_submit_serializes_json_body() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock(return_value=_FakeAio(201, _SUBMIT_RESP))
    result = await client.jobs.submit(source=["a"], model="m")
    assert result["id"] == "job-1"
    call = client._post.call_args
    assert call.args[0] == "/v1/jobs"
    assert json.loads(call.kwargs["data"]) == {"operation": "encode", "model": "m", "items": [{"text": "a"}]}
    await client.close()


@pytest.mark.asyncio
async def test_async_submit_floors_long_running_timeout() -> None:
    """jobs.submit floors the per-call timeout to 120s (parity with the sync client)."""
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock(return_value=_FakeAio(201, _SUBMIT_RESP))
    await client.jobs.submit(source=["a"], model="m")
    assert client._post.call_args.kwargs["timeout_s"] == max(client._timeout, 120.0)
    await client.close()


@pytest.mark.asyncio
async def test_async_submit_field_map_body() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock(return_value=_FakeAio(201, _SUBMIT_RESP))
    await client.jobs.submit(
        source="postgres://wh?query=select id, body, source_url from docs",
        model="BAAI/bge-m3",
        sink="postgres://wh?table=doc_vectors",
        field_map={"id_field": "id", "input_field": "body", "carry": ["source_url"]},
        output_field="embedding",
    )
    body = json.loads(client._post.call_args.kwargs["data"])
    assert body["field_map"] == {"id_field": "id", "input_field": "body", "carry": ["source_url"]}
    assert body["output_field"] == "embedding"
    assert body["connection"] == "wh"
    await client.close()


@pytest.mark.asyncio
async def test_async_submit_extract_options_labels_body() -> None:
    """Async parity: an extract job's labels/output_schema ride `options` as-is."""
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock(return_value=_FakeAio(201, _SUBMIT_RESP))
    await client.jobs.submit(
        source=["some text"],
        model="urchade/gliner_small-v2.1",
        operation="extract",
        options={"labels": ["PERSON", "ORG"], "output_schema": {"type": "object"}},
    )
    body = json.loads(client._post.call_args.kwargs["data"])
    assert body["operation"] == "extract"
    assert body["options"] == {"labels": ["PERSON", "ORG"], "output_schema": {"type": "object"}}
    await client.close()


@pytest.mark.asyncio
async def test_async_list_and_cancel() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._get = AsyncMock(return_value=_FakeAio(200, {"object": "list", "data": [{"id": "job-9"}]}))
    client._post = AsyncMock(return_value=_FakeAio(200, {"id": "job-9", "state": "cancelled"}))
    assert [j["id"] for j in await client.jobs.list()] == ["job-9"]
    out = await client.jobs.cancel("job-9")
    assert out["state"] == "cancelled"
    assert client._post.call_args.args[0] == "/v1/jobs/job-9/cancel"
    await client.close()
