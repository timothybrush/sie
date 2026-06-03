from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import numpy as np
import pytest
from sie_server.core.inference_output import ExtractOutput, ScoreOutput
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.types import WorkerResult
from sie_server.ipc_types import (
    EncodeBatchItem,
    ExtractBatchItem,
    ProcessEncodeBatchRequest,
    ProcessExtractBatchRequest,
    ProcessScoreBatchRequest,
    ScoreBatchItem,
)
from sie_server.queue_executor import QueueExecutor


def _make_registry(*, loaded: bool = True, loading: bool = False) -> MagicMock:
    reg = MagicMock()
    reg.model_names = ["test/model"]
    reg.device = "cpu"
    reg.is_loaded.return_value = loaded
    reg.is_loading.return_value = loading
    reg.get_config.return_value = MagicMock()
    return reg


def _encode_item(
    *,
    wiid: str = "req-1.0",
    item: dict | None = None,
    output_types: list[str] | None = None,
    instruction: str | None = None,
    is_query: bool = False,
    options: dict | None = None,
    item_index: int = 0,
) -> EncodeBatchItem:
    return EncodeBatchItem(
        work_item_id=wiid,
        request_id=wiid.split(".", maxsplit=1)[0],
        item_index=item_index,
        total_items=1,
        timestamp=time.time(),
        item=item or {"text": "hello"},
        output_types=output_types,
        instruction=instruction,
        is_query=is_query,
        options=options,
    )


def _score_item() -> ScoreBatchItem:
    return ScoreBatchItem(
        work_item_id="req-1.0",
        request_id="req-1",
        item_index=0,
        total_items=1,
        timestamp=time.time(),
        query_item={"text": "q"},
        score_items=[{"text": "a", "id": "doc-a"}],
    )


def _extract_item() -> ExtractBatchItem:
    return ExtractBatchItem(
        work_item_id="req-1.0",
        request_id="req-1",
        item_index=0,
        total_items=1,
        timestamp=time.time(),
        item={"text": "Alice works at Acme."},
        labels=["person"],
    )


# -----------------------------------------------------------------------------
# ensure_model_ready
# -----------------------------------------------------------------------------


class TestEnsureModelReady:
    @pytest.mark.asyncio
    async def test_loaded_returns_ready(self) -> None:
        reg = _make_registry(loaded=True)
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "ready"
        reg.start_load_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_already_loading_returns_in_progress(self) -> None:
        reg = _make_registry(loaded=False, loading=True)
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "loading_in_progress"
        reg.start_load_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_triggers_new_load(self) -> None:
        reg = _make_registry(loaded=False, loading=False)
        reg.start_load_async = AsyncMock(return_value=True)
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "loading_started"
        reg.start_load_async.assert_awaited_once_with("test/model", "cpu")

    @pytest.mark.asyncio
    async def test_start_load_already_running_returns_in_progress(self) -> None:
        reg = _make_registry(loaded=False, loading=False)
        reg.start_load_async = AsyncMock(return_value=False)  # already loading/loaded in another task
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "loading_in_progress"

    @pytest.mark.asyncio
    async def test_unknown_model_returns_retry_later(self) -> None:
        reg = _make_registry(loaded=False, loading=False)
        reg.start_load_async = AsyncMock(side_effect=KeyError("test/model"))
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "retry_later"


class TestGetBatchBudget:
    """Per-model batch budget advertised to the Rust queue consumer via
    EnsureModelReadyResponse.
    """

    def test_returns_max_batch_requests_when_worker_has_batch_config(self) -> None:
        reg = _make_registry()
        worker = MagicMock()
        worker._batch_config = MagicMock(max_batch_requests=37)
        reg.get_worker.return_value = worker
        ex = QueueExecutor(reg)
        assert ex.get_batch_budget("test/model") == 37

    def test_returns_none_when_worker_missing(self) -> None:
        reg = _make_registry()
        reg.get_worker.return_value = None
        ex = QueueExecutor(reg)
        assert ex.get_batch_budget("test/model") is None

    def test_returns_none_when_worker_has_no_batch_config(self) -> None:
        reg = _make_registry()
        worker = object()  # no `_batch_config` attribute
        reg.get_worker.return_value = worker
        ex = QueueExecutor(reg)
        assert ex.get_batch_budget("test/model") is None

    def test_returns_none_when_registry_raises(self) -> None:
        reg = _make_registry()
        reg.get_worker.side_effect = KeyError("test/model")
        ex = QueueExecutor(reg)
        assert ex.get_batch_budget("test/model") is None

    def test_returns_none_when_budget_is_non_int(self) -> None:
        # Protect the wire: a misconfigured BatchConfig with a non-int
        # max_batch_requests should not leak a MagicMock / string / None
        # into the IPC response where msgspec would choke on it.
        reg = _make_registry()
        worker = MagicMock()
        worker._batch_config = MagicMock(max_batch_requests="64")  # str, not int
        reg.get_worker.return_value = worker
        ex = QueueExecutor(reg)
        assert ex.get_batch_budget("test/model") is None

    def test_returns_none_when_budget_is_non_positive(self) -> None:
        reg = _make_registry()
        worker = MagicMock()
        worker._batch_config = MagicMock(max_batch_requests=0)
        reg.get_worker.return_value = worker
        ex = QueueExecutor(reg)
        assert ex.get_batch_budget("test/model") is None


# -----------------------------------------------------------------------------
# process_encode_batch
# -----------------------------------------------------------------------------


class TestProcessEncodeBatch:
    @pytest.mark.asyncio
    async def test_single_item_publish_and_ack(self) -> None:
        reg = _make_registry()
        ex = QueueExecutor(reg)

        fake_outputs = [{"dense": [0.1, 0.2]}]
        fake_timing = RequestTiming()

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=(fake_outputs, fake_timing),
        ) as mock_encode:
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(model_id="test/model", items=[_encode_item()])
            )

        assert len(outcome.outcomes) == 1
        item = outcome.outcomes[0]
        assert item.disposition == "publish_and_ack"
        assert item.result_msgpack is not None
        inner = msgpack.unpackb(item.result_msgpack, raw=False)
        assert inner == {"dense": [0.1, 0.2]}
        mock_encode.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_model_evicted_naks_all(self) -> None:
        reg = _make_registry()
        reg.get_config.side_effect = KeyError("test/model")
        ex = QueueExecutor(reg)

        outcome = await ex.process_encode_batch(
            ProcessEncodeBatchRequest(
                model_id="test/model",
                items=[_encode_item(wiid="a.0"), _encode_item(wiid="b.0", item_index=1)],
            )
        )
        assert len(outcome.outcomes) == 2
        assert all(o.disposition == "nak_retry" for o in outcome.outcomes)
        assert all((o.nak_delay_ms or 0) > 0 for o in outcome.outcomes)

    @pytest.mark.asyncio
    async def test_pipeline_exception_error_outcome(self) -> None:
        reg = _make_registry()
        ex = QueueExecutor(reg)

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(model_id="test/model", items=[_encode_item()])
            )

        assert len(outcome.outcomes) == 1
        assert outcome.outcomes[0].disposition == "publish_error_and_ack"
        assert outcome.outcomes[0].error_code == "inference_error"
        assert "boom" in (outcome.outcomes[0].error or "")

    @pytest.mark.asyncio
    async def test_pipeline_oom_naks_without_publishing_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_OOM_NAK_DELAY_S", "11.0")
        reg = _make_registry()
        ex = QueueExecutor(reg)

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB"),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(model_id="test/model", items=[_encode_item()])
            )

        assert len(outcome.outcomes) == 1
        assert outcome.outcomes[0].disposition == "nak_retry"
        assert outcome.outcomes[0].nak_delay_ms == 11_000
        assert outcome.outcomes[0].error_code is None
        assert outcome.outcomes[0].error is None

    @pytest.mark.asyncio
    async def test_heterogeneous_batch_groups_by_params(self) -> None:
        """Items with different (output_types, instruction, is_query, options) run as separate sub-batches."""
        reg = _make_registry()
        ex = QueueExecutor(reg)

        call_count = 0
        observed_output_types: list[list[str]] = []

        async def fake_run_encode(**kwargs):
            nonlocal call_count
            call_count += 1
            observed_output_types.append(kwargs["output_types"])
            n = len(kwargs["items"])
            return [{"dense": [float(call_count)]} for _ in range(n)], RequestTiming()

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new=AsyncMock(side_effect=fake_run_encode),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id="test/model",
                    items=[
                        _encode_item(wiid="a.0", output_types=["dense"]),
                        _encode_item(wiid="b.0", output_types=["dense"], item_index=1),
                        _encode_item(wiid="c.0", output_types=["sparse"], item_index=2),
                    ],
                )
            )

        assert call_count == 2
        assert sorted([tuple(ot) for ot in observed_output_types]) == [("dense",), ("sparse",)]
        assert len(outcome.outcomes) == 3
        assert all(o.disposition == "publish_and_ack" for o in outcome.outcomes)
        # Outcomes preserve input order regardless of grouping.
        assert [o.work_item_id for o in outcome.outcomes] == ["a.0", "b.0", "c.0"]

    @pytest.mark.asyncio
    async def test_output_ordering_matches_input(self) -> None:
        reg = _make_registry()
        ex = QueueExecutor(reg)

        fake_outputs = [{"dense": [float(i)]} for i in range(3)]
        fake_timing = RequestTiming()

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=(fake_outputs, fake_timing),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id="test/model",
                    items=[
                        _encode_item(wiid="a.0", item_index=0),
                        _encode_item(wiid="b.0", item_index=1),
                        _encode_item(wiid="c.0", item_index=2),
                    ],
                )
            )

        assert [o.work_item_id for o in outcome.outcomes] == ["a.0", "b.0", "c.0"]

    @pytest.mark.asyncio
    async def test_records_ipc_batch_shape_metric(self) -> None:
        """process_encode_batch emits a fragmentation observation with
        items==N and sub_groups==distinct(output_types, instruction, ...).

        Guards the invariant that dashboards reading
        `sie_ipc_batch_sub_groups / sie_ipc_batch_items` will correctly
        reflect how many separate GPU forward passes the IPC batch
        split into — that's the entire reason the metric exists.
        """
        reg = _make_registry()
        ex = QueueExecutor(reg)

        async def fake_run_encode(**kwargs):
            n = len(kwargs["items"])
            return [{"dense": [0.0]} for _ in range(n)], RequestTiming()

        with (
            patch(
                "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
                new=AsyncMock(side_effect=fake_run_encode),
            ),
            patch("sie_server.queue_executor.record_ipc_batch_shape") as mock_record,
        ):
            await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id="test/model",
                    items=[
                        _encode_item(wiid="a.0", output_types=["dense"]),
                        _encode_item(wiid="b.0", output_types=["dense"], item_index=1),
                        _encode_item(wiid="c.0", output_types=["dense"], item_index=2),
                        _encode_item(wiid="d.0", output_types=["sparse"], item_index=3),
                        _encode_item(wiid="e.0", output_types=["sparse"], item_index=4),
                    ],
                )
            )

        mock_record.assert_called_once()
        kwargs = mock_record.call_args.kwargs
        assert kwargs["model"] == "test/model"
        assert kwargs["endpoint"] == "encode"
        assert kwargs["total_items"] == 5
        assert sorted(kwargs["sub_group_sizes"]) == [2, 3]

    @pytest.mark.asyncio
    async def test_ipc_batch_shape_recorded_even_when_pipeline_fails(self) -> None:
        """Shape metric is observed before any forward pass runs, so
        failed batches still populate the fragmentation histogram.
        This keeps dashboards honest under error storms (otherwise
        they'd under-report IPC batch volume).
        """
        reg = _make_registry()
        ex = QueueExecutor(reg)

        with (
            patch(
                "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ),
            patch("sie_server.queue_executor.record_ipc_batch_shape") as mock_record,
        ):
            await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id="test/model",
                    items=[_encode_item(wiid="a.0"), _encode_item(wiid="b.0", item_index=1)],
                )
            )

        mock_record.assert_called_once()
        kwargs = mock_record.call_args.kwargs
        assert kwargs["total_items"] == 2
        assert kwargs["sub_group_sizes"] == [2]


# -----------------------------------------------------------------------------
# process_score_batch
# -----------------------------------------------------------------------------


class TestProcessScoreBatch:
    @pytest.mark.asyncio
    async def test_single_item_publish_and_ack(self) -> None:
        reg = _make_registry()
        worker = AsyncMock()
        score_output = ScoreOutput(scores=np.array([0.9, 0.1], dtype=np.float32))
        wr = WorkerResult(output=score_output, timing=RequestTiming())
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(wr)
        worker.submit_score = AsyncMock(return_value=fut)
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)
        outcome = await ex.process_score_batch(
            ProcessScoreBatchRequest(
                model_id="test/model",
                items=[
                    ScoreBatchItem(
                        work_item_id="req-1.0",
                        request_id="req-1",
                        item_index=0,
                        total_items=1,
                        timestamp=time.time(),
                        query_item={"text": "q"},
                        score_items=[{"text": "a", "id": "doc-a"}, {"text": "b", "id": "doc-b"}],
                    )
                ],
            )
        )

        assert len(outcome.outcomes) == 1
        o = outcome.outcomes[0]
        assert o.disposition == "publish_and_ack"
        # Score output is Rust-frameable. The legacy ``result_msgpack``
        # is None; the parallel arrays land on ``raw_output.score`` and
        # ``sie_server_sidecar::output::build_score_payload`` does
        # the descending stable sort + rank assignment byte-identically.
        assert o.result_msgpack is None
        assert o.raw_output is not None
        assert o.raw_output.score is not None
        assert o.raw_output.score.item_ids == ["doc-a", "doc-b"]
        assert o.raw_output.score.scores == pytest.approx([0.9, 0.1])

    @pytest.mark.asyncio
    async def test_model_evicted_naks_individual_item(self) -> None:
        reg = _make_registry()
        reg.start_worker = AsyncMock(side_effect=KeyError("test/model"))
        ex = QueueExecutor(reg)

        outcome = await ex.process_score_batch(
            ProcessScoreBatchRequest(
                model_id="test/model",
                items=[
                    ScoreBatchItem(
                        work_item_id="req-1.0",
                        request_id="req-1",
                        item_index=0,
                        total_items=1,
                        timestamp=time.time(),
                        query_item={"text": "q"},
                        score_items=[{"text": "a"}],
                    )
                ],
            )
        )
        assert outcome.outcomes[0].disposition == "nak_retry"

    @pytest.mark.asyncio
    async def test_score_oom_naks_without_publishing_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_OOM_NAK_DELAY_S", "12.0")
        reg = _make_registry()
        worker = AsyncMock()
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_exception(RuntimeError("MPS backend out of memory"))
        worker.submit_score = AsyncMock(return_value=fut)
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)
        outcome = await ex.process_score_batch(ProcessScoreBatchRequest(model_id="test/model", items=[_score_item()]))

        assert outcome.outcomes[0].disposition == "nak_retry"
        assert outcome.outcomes[0].nak_delay_ms == 12_000
        assert outcome.outcomes[0].error_code is None
        assert outcome.outcomes[0].error is None

    @pytest.mark.asyncio
    async def test_records_ipc_batch_shape(self) -> None:
        """Score submits one task per IPC item, so the fragmentation
        metric must report sub_groups == items. Any deviation means
        the queue executor changed its dispatch strategy without
        updating the dashboard semantics.
        """
        reg = _make_registry()
        worker = AsyncMock()
        score_output = ScoreOutput(scores=np.array([0.5], dtype=np.float32))
        wr = WorkerResult(output=score_output, timing=RequestTiming())
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(wr)
        worker.submit_score = AsyncMock(return_value=fut)
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)

        with patch("sie_server.queue_executor.record_ipc_batch_shape") as mock_record:
            await ex.process_score_batch(
                ProcessScoreBatchRequest(
                    model_id="test/model",
                    items=[
                        ScoreBatchItem(
                            work_item_id=f"req-{i}.0",
                            request_id=f"req-{i}",
                            item_index=0,
                            total_items=1,
                            timestamp=time.time(),
                            query_item={"text": "q"},
                            score_items=[{"text": "a"}],
                        )
                        for i in range(4)
                    ],
                )
            )

        mock_record.assert_called_once()
        kwargs = mock_record.call_args.kwargs
        assert kwargs["endpoint"] == "score"
        assert kwargs["total_items"] == 4
        assert kwargs["sub_group_sizes"] == [1, 1, 1, 1]


# -----------------------------------------------------------------------------
# process_extract_batch
# -----------------------------------------------------------------------------


class TestProcessExtractBatch:
    @pytest.mark.asyncio
    async def test_single_item_publish_and_ack(self) -> None:
        reg = _make_registry()
        worker = AsyncMock()
        extract_output = ExtractOutput(
            entities=[[{"text": "Alice", "label": "person", "score": 0.99, "start": 0, "end": 5}]]
        )
        wr = WorkerResult(output=extract_output, timing=RequestTiming())
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(wr)
        worker.submit_extract = AsyncMock(return_value=fut)
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)
        outcome = await ex.process_extract_batch(
            ProcessExtractBatchRequest(
                model_id="test/model",
                items=[
                    ExtractBatchItem(
                        work_item_id="req-1.0",
                        request_id="req-1",
                        item_index=0,
                        total_items=1,
                        timestamp=time.time(),
                        item={"text": "Alice works at Acme."},
                        labels=["person"],
                    )
                ],
            )
        )

        assert outcome.outcomes[0].disposition == "publish_and_ack"
        inner = msgpack.unpackb(outcome.outcomes[0].result_msgpack, raw=False)
        assert "entities" in inner

    @pytest.mark.asyncio
    async def test_model_evicted_naks(self) -> None:
        reg = _make_registry()
        reg.start_worker = AsyncMock(side_effect=RuntimeError("evicted"))
        ex = QueueExecutor(reg)
        outcome = await ex.process_extract_batch(
            ProcessExtractBatchRequest(
                model_id="test/model",
                items=[
                    ExtractBatchItem(
                        work_item_id="req-1.0",
                        request_id="req-1",
                        item_index=0,
                        total_items=1,
                        timestamp=time.time(),
                        item={"text": "x"},
                    )
                ],
            )
        )
        assert outcome.outcomes[0].disposition == "nak_retry"

    @pytest.mark.asyncio
    async def test_extract_oom_naks_without_publishing_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_OOM_NAK_DELAY_S", "13.0")
        reg = _make_registry()
        worker = AsyncMock()
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_exception(RuntimeError("CUDA out of memory"))
        worker.submit_extract = AsyncMock(return_value=fut)
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)
        outcome = await ex.process_extract_batch(
            ProcessExtractBatchRequest(model_id="test/model", items=[_extract_item()])
        )

        assert outcome.outcomes[0].disposition == "nak_retry"
        assert outcome.outcomes[0].nak_delay_ms == 13_000
        assert outcome.outcomes[0].error_code is None
        assert outcome.outcomes[0].error is None

    @pytest.mark.asyncio
    async def test_records_ipc_batch_shape(self) -> None:
        """Extract is also per-item dispatch; same semantics as score."""
        reg = _make_registry()
        worker = AsyncMock()
        extract_output = ExtractOutput(entities=[[]])
        wr = WorkerResult(output=extract_output, timing=RequestTiming())
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(wr)
        worker.submit_extract = AsyncMock(return_value=fut)
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)

        with patch("sie_server.queue_executor.record_ipc_batch_shape") as mock_record:
            await ex.process_extract_batch(
                ProcessExtractBatchRequest(
                    model_id="test/model",
                    items=[
                        ExtractBatchItem(
                            work_item_id=f"req-{i}.0",
                            request_id=f"req-{i}",
                            item_index=0,
                            total_items=1,
                            timestamp=time.time(),
                            item={"text": "x"},
                        )
                        for i in range(3)
                    ],
                )
            )

        mock_record.assert_called_once()
        kwargs = mock_record.call_args.kwargs
        assert kwargs["endpoint"] == "extract"
        assert kwargs["total_items"] == 3
        assert kwargs["sub_group_sizes"] == [1, 1, 1]


# -----------------------------------------------------------------------------
# _wrap_encode_output: queue path MUST match HTTP wire shape for every
# output_type the SDK understands. Otherwise HTTP and queue consumers
# see different encodings for the same model.
# -----------------------------------------------------------------------------


class TestWrapEncodeOutput:
    @staticmethod
    def _config(
        *,
        dense_dim: int | None = None,
        sparse_dim: int | None = None,
        multivector_dim: int | None = None,
    ) -> MagicMock:
        encode_task = MagicMock()
        encode_task.dense = MagicMock(dim=dense_dim) if dense_dim is not None else None
        encode_task.sparse = MagicMock(dim=sparse_dim) if sparse_dim is not None else None
        encode_task.multivector = MagicMock(dim=multivector_dim) if multivector_dim is not None else None
        cfg = MagicMock()
        cfg.tasks = MagicMock(encode=encode_task)
        return cfg

    def test_dense_float32_wraps_with_dims_and_dtype(self) -> None:
        from sie_server.queue_executor import _wrap_encode_output

        arr = np.zeros(768, dtype=np.float32)
        wrapped = _wrap_encode_output({"dense": arr}, self._config(dense_dim=768))
        assert wrapped["dense"]["dims"] == 768
        assert wrapped["dense"]["dtype"] == "float32"
        assert np.array_equal(wrapped["dense"]["values"], arr)

    def test_sparse_wraps_like_http_format_sparse(self) -> None:
        from sie_server.queue_executor import _wrap_encode_output

        indices = np.array([1, 7, 42], dtype=np.int64)
        values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        wrapped = _wrap_encode_output(
            {"sparse": {"indices": indices, "values": values}},
            self._config(sparse_dim=30522),
        )
        sp = wrapped["sparse"]
        assert sp["dims"] == 30522
        assert sp["dtype"] == "float32"
        assert np.array_equal(sp["indices"], indices)
        assert np.array_equal(sp["values"], values)

    def test_multivector_wraps_with_token_dims_and_num_tokens(self) -> None:
        from sie_server.queue_executor import _wrap_encode_output

        arr = np.zeros((12, 128), dtype=np.float32)
        wrapped = _wrap_encode_output({"multivector": arr}, self._config(multivector_dim=128))
        mv = wrapped["multivector"]
        assert mv["token_dims"] == 128
        assert mv["num_tokens"] == 12
        assert mv["dtype"] == "float32"
        assert np.array_equal(mv["values"], arr)

    def test_binary_multivector_sets_dtype_binary(self) -> None:
        from sie_server.queue_executor import _wrap_encode_output

        arr = np.zeros((7, 16), dtype=np.uint8)  # 128-dim packed into 16 bytes/token
        wrapped = _wrap_encode_output({"multivector": arr}, self._config(multivector_dim=128))
        mv = wrapped["multivector"]
        assert mv["token_dims"] == 128
        assert mv["num_tokens"] == 7
        assert mv["dtype"] == "binary"
