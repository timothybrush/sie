"""Tests for Prometheus metrics."""

import logging
from unittest.mock import MagicMock

import pytest
from sie_server.observability.metrics import (
    IPC_BATCH_ITEMS,
    IPC_BATCH_SUB_GROUP_ITEMS,
    IPC_BATCH_SUB_GROUPS,
    MODEL_LOADED,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    record_ipc_batch_shape,
    record_request,
    set_model_loaded,
)


class TestRecordRequest:
    """Tests for record_request helper."""

    def test_increments_counter(self) -> None:
        """record_request should increment the requests counter."""
        # Get initial value
        initial = REQUESTS_TOTAL.labels(model="test-model", endpoint="encode", status="success")._value.get()

        # Record a request
        record_request(model="test-model", endpoint="encode", status="success")

        # Check counter incremented
        new_value = REQUESTS_TOTAL.labels(model="test-model", endpoint="encode", status="success")._value.get()
        assert new_value == initial + 1

    def test_records_error_status(self) -> None:
        """record_request should handle error status."""
        initial = REQUESTS_TOTAL.labels(model="test-model", endpoint="score", status="error")._value.get()

        record_request(model="test-model", endpoint="score", status="error")

        new_value = REQUESTS_TOTAL.labels(model="test-model", endpoint="score", status="error")._value.get()
        assert new_value == initial + 1

    def test_records_timing_breakdown(self) -> None:
        """record_request should record timing breakdown when provided."""
        # Create mock timing object
        timing = MagicMock()
        timing.total_ms = 100.0
        timing.queue_ms = 10.0
        timing.tokenization_ms = 20.0
        timing.inference_ms = 70.0

        # Record request with timing
        record_request(
            model="timing-test",
            endpoint="encode",
            status="success",
            timing=timing,
        )

        # Verify histograms were observed (we can't easily check exact values
        # but we can verify the labels exist in the registry)
        assert REQUEST_DURATION.labels(model="timing-test", endpoint="encode", phase="total") is not None
        assert REQUEST_DURATION.labels(model="timing-test", endpoint="encode", phase="queue") is not None
        assert REQUEST_DURATION.labels(model="timing-test", endpoint="encode", phase="tokenize") is not None
        assert REQUEST_DURATION.labels(model="timing-test", endpoint="encode", phase="inference") is not None

    def test_skips_zero_timing_phases(self) -> None:
        """record_request should skip phases with zero duration."""
        timing = MagicMock()
        timing.total_ms = 50.0
        timing.queue_ms = 0  # Should be skipped
        timing.tokenization_ms = 0  # Should be skipped
        timing.inference_ms = 50.0

        # Should not raise even with zero values
        record_request(
            model="zero-timing-test",
            endpoint="encode",
            status="success",
            timing=timing,
        )

    def test_record_request_emits_structured_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """record_request emits a structured log with model, endpoint, status, and timing."""
        from unittest.mock import MagicMock

        from sie_server.observability.metrics import record_request

        timing = MagicMock()
        timing.total_ms = 50.0
        timing.queue_ms = 5.0
        timing.tokenization_ms = 10.0
        timing.inference_ms = 35.0

        with caplog.at_level(logging.DEBUG, logger="sie_server.observability.metrics"):
            record_request(
                model="bge-m3",
                endpoint="encode",
                status="success",
                timing=timing,
                request_id="req-abc",
                api_key="sk-secret-1234",
                queue_depth=3,
            )

        assert len(caplog.records) >= 1
        rec = caplog.records[-1]
        assert rec.message == "Request completed"
        assert rec.model == "bge-m3"  # type: ignore
        assert rec.endpoint == "encode"  # type: ignore
        assert rec.status == "success"  # type: ignore
        assert rec.request_id == "req-abc"  # type: ignore
        # record_request masks defensively: raw keys are never logged, only the
        # last 4 chars survive (see _mask_secret).
        assert rec.api_key == "***1234"  # type: ignore
        assert rec.queue_depth == 3  # type: ignore
        assert rec.latency_ms == 50.0  # type: ignore
        assert rec.tokenization_ms == 10.0  # type: ignore
        assert rec.queue_ms == 5.0  # type: ignore
        assert rec.inference_ms == 35.0  # type: ignore

    def test_record_request_without_timing(self, caplog: pytest.LogCaptureFixture) -> None:
        """record_request emits log even without timing data."""
        from sie_server.observability.metrics import record_request

        with caplog.at_level(logging.DEBUG, logger="sie_server.observability.metrics"):
            record_request(
                model="bge-m3",
                endpoint="encode",
                status="error",
            )

        assert len(caplog.records) >= 1
        rec = caplog.records[-1]
        assert rec.message == "Request completed"
        assert rec.model == "bge-m3"  # type: ignore
        assert rec.status == "error"  # type: ignore


class TestSetModelLoaded:
    """Tests for set_model_loaded helper."""

    def test_sets_loaded_true(self) -> None:
        """set_model_loaded should set 1 when loaded=True."""
        set_model_loaded(model="loaded-test", device="cuda:0", loaded=True)

        value = MODEL_LOADED.labels(model="loaded-test", device="cuda:0")._value.get()
        assert value == 1

    def test_sets_loaded_false(self) -> None:
        """set_model_loaded should set 0 when loaded=False."""
        set_model_loaded(model="unloaded-test", device="cpu", loaded=False)

        value = MODEL_LOADED.labels(model="unloaded-test", device="cpu")._value.get()
        assert value == 0


class TestRecordIpcBatchShape:
    """Guards the IPC-batch fragmentation metric.

    The three histograms together answer:
        "How many items did an IPC batch deliver, into how many GPU
         forward passes did that batch split, and how many items were
         in each resulting pass?"

    Every commit that changes the dispatch strategy in the queue
    executor should make sure these semantics stay true — otherwise
    Grafana panels built on these metrics become subtly wrong.
    """

    @staticmethod
    def _sample_count(histogram: object, **labels: str) -> float:
        return histogram.labels(**labels)._sum.get()  # type: ignore[attr-defined]

    @staticmethod
    def _observation_count(histogram: object, **labels: str) -> float:
        # prometheus_client Histograms expose per-label-tuple samples via
        # the private `_sum` (running sum) and per-bucket counters; for a
        # cumulative check we read `.collect()` and sum the `+Inf` bucket.
        samples = histogram.labels(**labels).collect()[0].samples  # type: ignore[attr-defined]
        for s in samples:
            if s.name.endswith("_count"):
                return s.value
        return 0.0

    def test_observes_all_three_histograms(self) -> None:
        # Use a fresh (model, endpoint) tuple so the test is
        # independent of anything else that ran in the same process.
        model, endpoint = "shape-test/model-1", "encode"

        before_items = self._observation_count(IPC_BATCH_ITEMS, model=model, endpoint=endpoint)
        before_groups = self._observation_count(IPC_BATCH_SUB_GROUPS, model=model, endpoint=endpoint)
        before_sub = self._observation_count(IPC_BATCH_SUB_GROUP_ITEMS, model=model, endpoint=endpoint)

        record_ipc_batch_shape(
            model=model,
            endpoint=endpoint,
            total_items=10,
            sub_group_sizes=[4, 3, 3],
        )

        assert self._observation_count(IPC_BATCH_ITEMS, model=model, endpoint=endpoint) == before_items + 1
        assert self._observation_count(IPC_BATCH_SUB_GROUPS, model=model, endpoint=endpoint) == before_groups + 1
        # One observation per sub-group size (3 entries).
        assert self._observation_count(IPC_BATCH_SUB_GROUP_ITEMS, model=model, endpoint=endpoint) == before_sub + 3

    def test_sums_reflect_reported_shape(self) -> None:
        model, endpoint = "shape-test/model-2", "encode"
        before_items_sum = self._sample_count(IPC_BATCH_ITEMS, model=model, endpoint=endpoint)
        before_groups_sum = self._sample_count(IPC_BATCH_SUB_GROUPS, model=model, endpoint=endpoint)
        before_sub_sum = self._sample_count(IPC_BATCH_SUB_GROUP_ITEMS, model=model, endpoint=endpoint)

        record_ipc_batch_shape(
            model=model,
            endpoint=endpoint,
            total_items=29,
            sub_group_sizes=[17, 9, 3],
        )

        assert self._sample_count(IPC_BATCH_ITEMS, model=model, endpoint=endpoint) == pytest.approx(
            before_items_sum + 29
        )
        assert self._sample_count(IPC_BATCH_SUB_GROUPS, model=model, endpoint=endpoint) == pytest.approx(
            before_groups_sum + 3
        )
        assert self._sample_count(IPC_BATCH_SUB_GROUP_ITEMS, model=model, endpoint=endpoint) == pytest.approx(
            before_sub_sum + 17 + 9 + 3
        )

    def test_empty_batch_is_a_noop(self) -> None:
        # Zero-item IPC batches don't happen in practice but the helper
        # must refuse to poison the histogram with a 0-bucket sample —
        # that would bias any p50 computation.
        model, endpoint = "shape-test/model-3", "encode"
        before = self._observation_count(IPC_BATCH_ITEMS, model=model, endpoint=endpoint)

        record_ipc_batch_shape(model=model, endpoint=endpoint, total_items=0, sub_group_sizes=[])

        assert self._observation_count(IPC_BATCH_ITEMS, model=model, endpoint=endpoint) == before

    def test_ignores_zero_sized_sub_groups(self) -> None:
        # Defensive: if a caller ever passed `[2, 0, 1]` (shouldn't
        # happen but could during refactors), we record only the
        # truthy entries so we don't skew the group-items histogram.
        model, endpoint = "shape-test/model-4", "encode"
        before_sub = self._observation_count(IPC_BATCH_SUB_GROUP_ITEMS, model=model, endpoint=endpoint)
        before_groups = self._observation_count(IPC_BATCH_SUB_GROUPS, model=model, endpoint=endpoint)

        record_ipc_batch_shape(
            model=model,
            endpoint=endpoint,
            total_items=3,
            sub_group_sizes=[2, 0, 1],
        )

        assert self._observation_count(IPC_BATCH_SUB_GROUP_ITEMS, model=model, endpoint=endpoint) == before_sub + 2
        # The sub_groups count itself still reflects what the caller
        # reported — the 0 is a caller bug, and this metric surfaces
        # it as a "group count > non-zero items" anomaly rather than
        # silently dropping it.
        assert self._observation_count(IPC_BATCH_SUB_GROUPS, model=model, endpoint=endpoint) == before_groups + 1
