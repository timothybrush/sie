"""Tests for EncodeHandler.format_output response filtering.

Regression coverage for #1430: the muvera profile drives ColBERT as a dense
encoder. The request is translated to ``output_types=["multivector"]`` for the
adapter (which cannot emit ``dense``), the muvera postprocessor then adds
``output.dense``, and the response must be filtered by the user-requested types
(which include ``dense``) — not by the translated adapter types — so the
postprocessor-produced ``dense`` survives.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sie_server.core.encode_pipeline import EncodePipeline
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.worker.handlers.encode import EncodeHandler


class TestFormatOutputResponseFiltering:
    """format_output must filter by the requested response output types."""

    def _muvera_output(self) -> EncodeOutput:
        """EncodeOutput carrying both multivector (adapter) and dense (postprocessor)."""
        multivector = [np.random.randn(5, 128).astype(np.float32) for _ in range(2)]
        dense = np.random.randn(2, 64).astype(np.float32)
        return EncodeOutput(dense=dense, multivector=multivector, batch_size=2)

    def test_dense_survives_when_requested(self) -> None:
        """Postprocessor-added dense survives when filtered by the user-requested types."""
        output = self._muvera_output()

        # The eval requests dense even though the adapter was asked for multivector.
        results = EncodeHandler.format_output(output, output_types=["dense"])

        assert len(results) == 2
        for item in results:
            assert "dense" in item
            assert "multivector" not in item

    def test_both_types_keep_dense_and_multivector(self) -> None:
        """A [dense, multivector] request keeps both the adapter and postprocessor outputs.

        Mirrors the muvera-ColBERT request that asks the adapter only for
        ``multivector`` while the user requested both: the final filter runs on the
        user-requested ``["dense", "multivector"]`` and must keep both. Under the
        pre-fix ``["multivector"]``-only filter the postprocessor dense was dropped.
        """
        output = self._muvera_output()

        results = EncodeHandler.format_output(output, output_types=["dense", "multivector"])

        assert len(results) == 2
        for item in results:
            assert "dense" in item
            assert "multivector" in item

    def test_multivector_filter_drops_dense(self) -> None:
        """Filtering by the translated adapter types alone drops the postprocessor dense.

        This is the pre-fix behaviour and the reason #1430 raised KeyError 'dense':
        the translated ['multivector'] was reused for the final response filter.
        """
        output = self._muvera_output()

        results = EncodeHandler.format_output(output, output_types=["multivector"])

        for item in results:
            assert "multivector" in item
            assert "dense" not in item

    def test_none_includes_all_available(self) -> None:
        """output_types=None keeps every populated field."""
        output = self._muvera_output()

        results = EncodeHandler.format_output(output, output_types=None)

        for item in results:
            assert "dense" in item
            assert "multivector" in item


class TestRunEncodeResponseOutputTypesPlumbing:
    """run_encode must filter the final response by response_output_types.

    Locks in the integration the #1430 fix wires up: the adapter is asked only
    for ``multivector`` (``output_types``), the muvera postprocessor adds
    ``dense``, and the response is filtered by the user-requested
    ``response_output_types`` (``["dense"]``) — so the postprocessor dense
    reaches the caller. Drives the real ``run_encode`` direct-adapter path with
    only the adapter/postprocessor boundaries mocked (no model weights).
    """

    @pytest.mark.asyncio
    async def test_response_output_types_reaches_final_filter(self) -> None:
        adapter_output = EncodeOutput(
            multivector=[np.random.randn(5, 128).astype(np.float32)],
            batch_size=1,
        )

        def fake_post_process(
            _self: Any, is_query: bool, options: dict[str, Any], encode_output: EncodeOutput
        ) -> float:
            # Simulate the muvera postprocessor adding dense onto the adapter output.
            encode_output.dense = np.random.randn(1, 64).astype(np.float32)
            return 0.0

        registry = MagicMock()
        registry.get.return_value = MagicMock()

        with (
            patch.object(EncodePipeline, "_prepare_batch", new=AsyncMock(return_value=None)),
            patch.object(EncodeHandler, "encode", return_value=adapter_output),
            patch.object(EncodeHandler, "post_process", new=fake_post_process),
        ):
            results, _timing = await EncodePipeline.run_encode(
                registry=registry,
                model="muvera-colbert",
                items=[MagicMock()],
                output_types=["multivector"],
                instruction=None,
                config=MagicMock(),
                is_query=True,
                options={},
                response_output_types=["dense"],
            )

        assert len(results) == 1
        assert "dense" in results[0]
        assert "multivector" not in results[0]
