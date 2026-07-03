"""Wire-dtype parity between the queue path and the HTTP path.

The queue path (`queue_executor._wrap_encode_output`) and the HTTP path
(`api/encode._format_dense` / `_format_multivector`) must tag the wire `dtype`
identically for the same array — a client that round-trips a request over HTTP
vs the queue/cluster edge otherwise decodes the bytes differently.

Regression guard for #1603: a linear `uint8` quantization (full dimensionality)
was tagged `"binary"` on the queue path but `"uint8"` over HTTP, because
`_NP_DTYPE_MAP` hard-coded `uint8 -> "binary"`. Only genuinely bit-packed uint8
(`shape < dim`) is `"binary"`.
"""

from __future__ import annotations

import numpy as np
import pytest
from sie_server.api.encode import _format_dense, _format_multivector
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.queue_executor import _wrap_encode_output

_DENSE_DIM = 8
_MV_DIM = 16


def _config() -> ModelConfig:
    return ModelConfig(
        sie_id="m",
        hf_id="org/m",
        tasks=Tasks(
            encode=EncodeTask(
                dense=EmbeddingDim(dim=_DENSE_DIM),
                multivector=EmbeddingDim(dim=_MV_DIM),
            )
        ),
        profiles={"default": ProfileConfig(adapter_path="t:A", max_batch_tokens=8192)},
    )


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        (np.arange(_DENSE_DIM, dtype=np.uint8), "uint8"),  # linear uint8, full dim
        (np.zeros(_DENSE_DIM // 8, dtype=np.uint8), "binary"),  # bit-packed, shape < dim
        (np.arange(_DENSE_DIM, dtype=np.int8), "int8"),
        (np.zeros(_DENSE_DIM, dtype=np.float32), "float32"),
    ],
    ids=["linear-uint8", "packed-binary", "int8", "float32"],
)
def test_dense_dtype_tag_matches_between_queue_and_http(arr: np.ndarray, expected: str) -> None:
    config = _config()
    http_dtype = _format_dense(arr, config)["dtype"]
    queue_dtype = _wrap_encode_output({"dense": arr.copy()}, config)["dense"]["dtype"]

    assert http_dtype == expected
    assert queue_dtype == expected
    assert queue_dtype == http_dtype, f"queue tagged {queue_dtype!r}, HTTP tagged {http_dtype!r}"


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        (np.zeros((5, _MV_DIM), dtype=np.uint8), "uint8"),  # linear uint8, full token dim
        (np.zeros((5, _MV_DIM // 8), dtype=np.uint8), "binary"),  # bit-packed, shape[1] < dim
        (np.zeros((5, _MV_DIM), dtype=np.int8), "int8"),
    ],
    ids=["linear-uint8", "packed-binary", "int8"],
)
def test_multivector_dtype_tag_matches_between_queue_and_http(arr: np.ndarray, expected: str) -> None:
    config = _config()
    http_dtype = _format_multivector(arr, config)["dtype"]
    queue_dtype = _wrap_encode_output({"multivector": arr.copy()}, config)["multivector"]["dtype"]

    assert http_dtype == expected
    assert queue_dtype == expected
    assert queue_dtype == http_dtype, f"queue tagged {queue_dtype!r}, HTTP tagged {http_dtype!r}"
