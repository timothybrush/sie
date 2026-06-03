"""Rust output-framing byte-identity: Python legacy path == Rust shaper.

This test establishes the equivalence:

    legacy_python_bytes(input) == rust_shaped_bytes(input)

by pinning **both** sides to the same hex goldens. The matching Rust
assertion lives in crate ``sie_server_sidecar`` at repo path
``packages/sie_server_sidecar/src/output/mod.rs`` as ``GOLDEN_DENSE_123`` /
``GOLDEN_SCORE_AB`` — if either language's
test breaks, the mismatch is immediately visible and the
non-destructive output-framing contract (same wire bytes either way) is
violated before any production traffic sees drift.

Update protocol: if you change the Python wire shape here, update
**both** goldens in lockstep. Never silently re-baseline just one
side.
"""

from __future__ import annotations

import msgpack
import msgpack_numpy
import numpy as np
from sie_server.queue_executor import _wrap_encode_output

msgpack_numpy.patch()

# Must match ``sie_server_sidecar::output::tests::GOLDEN_DENSE_123``.
GOLDEN_DENSE_123 = bytes(
    [
        0x81,
        0xA5,
        0x64,
        0x65,
        0x6E,
        0x73,
        0x65,
        0x83,
        0xA4,
        0x64,
        0x69,
        0x6D,
        0x73,
        0x03,
        0xA5,
        0x64,
        0x74,
        0x79,
        0x70,
        0x65,
        0xA7,
        0x66,
        0x6C,
        0x6F,
        0x61,
        0x74,
        0x33,
        0x32,
        0xA6,
        0x76,
        0x61,
        0x6C,
        0x75,
        0x65,
        0x73,
        0x85,
        0xC4,
        0x02,
        0x6E,
        0x64,
        0xC3,
        0xC4,
        0x04,
        0x74,
        0x79,
        0x70,
        0x65,
        0xA3,
        0x3C,
        0x66,
        0x34,
        0xC4,
        0x04,
        0x6B,
        0x69,
        0x6E,
        0x64,
        0xC4,
        0x00,
        0xC4,
        0x05,
        0x73,
        0x68,
        0x61,
        0x70,
        0x65,
        0x91,
        0x03,
        0xC4,
        0x04,
        0x64,
        0x61,
        0x74,
        0x61,
        0xC4,
        0x0C,
        0x00,
        0x00,
        0x80,
        0x3F,
        0x00,
        0x00,
        0x00,
        0x40,
        0x00,
        0x00,
        0x40,
        0x40,
    ],
)


# Must match ``sie_server_sidecar::output::tests::GOLDEN_SCORE_AB``.
GOLDEN_SCORE_AB = bytes(
    [
        0x92,
        0x83,
        0xA7,
        0x69,
        0x74,
        0x65,
        0x6D,
        0x5F,
        0x69,
        0x64,
        0xA1,
        0x61,
        0xA5,
        0x73,
        0x63,
        0x6F,
        0x72,
        0x65,
        0xCB,
        0x3F,
        0xEC,
        0xCC,
        0xCC,
        0xC0,
        0x00,
        0x00,
        0x00,
        0xA4,
        0x72,
        0x61,
        0x6E,
        0x6B,
        0x00,
        0x83,
        0xA7,
        0x69,
        0x74,
        0x65,
        0x6D,
        0x5F,
        0x69,
        0x64,
        0xA1,
        0x62,
        0xA5,
        0x73,
        0x63,
        0x6F,
        0x72,
        0x65,
        0xCB,
        0x3F,
        0xE0,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0xA4,
        0x72,
        0x61,
        0x6E,
        0x6B,
        0x01,
    ],
)


class _FakeDenseConfig:
    """Minimal stand-in for the production encode config — only the
    fields ``_wrap_encode_output`` actually reads.
    """

    class _Tasks:
        class _Encode:
            class _Dense:
                dim = 3

            dense = _Dense()

        encode = _Encode()

    tasks = _Tasks()


def test_legacy_dense_matches_rust_golden() -> None:
    """Legacy Python path (``_wrap_encode_output`` + ``msgpack.packb``
    with ``msgpack_numpy`` active) must produce the exact bytes the
    Rust ``build_dense_payload`` shaper asserts against.
    """
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    wrapped = _wrap_encode_output({"dense": arr}, _FakeDenseConfig())
    assert isinstance(wrapped["dense"], dict)
    bytes_out = msgpack.packb(wrapped, use_bin_type=True)
    assert bytes_out == GOLDEN_DENSE_123, (
        "Python legacy dense bytes diverged from the Rust shaper golden — "
        "update BOTH sides in lockstep (crate sie_server_sidecar, see "
        "packages/sie_server_sidecar/src/output/mod.rs::tests::GOLDEN_DENSE_123)."
    )


def test_legacy_score_matches_rust_golden() -> None:
    """Legacy Python path (manual sort + dict list + ``msgpack.packb``)
    must produce the same bytes as ``build_score_payload`` emits for
    the identical input.
    """
    scores_raw = np.array([0.9, 0.5], dtype=np.float32)
    item_ids = ["a", "b"]

    scored = sorted(
        zip(item_ids, (float(s) for s in scores_raw), strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    entries = [{"item_id": item_id, "score": sc, "rank": rank} for rank, (item_id, sc) in enumerate(scored)]
    bytes_out = msgpack.packb(entries, use_bin_type=True)
    assert bytes_out == GOLDEN_SCORE_AB, (
        "Python legacy score bytes diverged from the Rust shaper golden — "
        "update BOTH sides in lockstep (crate sie_server_sidecar, see "
        "packages/sie_server_sidecar/src/output/mod.rs::tests::GOLDEN_SCORE_AB)."
    )


def test_legacy_score_stable_ties_match_rust_stable_sort() -> None:
    """Equal scores must preserve input order on both sides (Python's
    ``list.sort`` is stable; Rust uses ``slice::sort_by`` which is
    documented stable).
    """
    scores_raw = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    item_ids = ["x", "y", "z"]

    scored = sorted(
        zip(item_ids, (float(s) for s in scores_raw), strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    entries = [{"item_id": item_id, "score": sc, "rank": rank} for rank, (item_id, sc) in enumerate(scored)]
    assert [e["item_id"] for e in entries] == ["x", "y", "z"]
    assert [e["rank"] for e in entries] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Sparse + multivector goldens for the Rust framing path.
#
# Must match:
#   * sparse (dims=None, [3,7,42] i32 / [0.5,1.5,2.5] f32)
#     → ``sie_server_sidecar::output::tests::GOLDEN_SPARSE_NIL_DIM_HEX``
#   * sparse (dims=30522, same indices/values)
#     → ``GOLDEN_SPARSE_WITH_DIM_HEX``
#   * multivector (np.arange(12, f32).reshape(3, 4))
#     → ``GOLDEN_MULTIVECTOR_3X4_HEX``
#
# Keep these in lockstep — the whole point is that divergence on
# either side surfaces on the next CI run, not in production.
# ---------------------------------------------------------------------------

GOLDEN_SPARSE_NIL_DIM = bytes.fromhex(
    "81a673706172736584a464696d73c0a56474797065a7666c6f61743332a7696e6469636573"
    "85c4026e64c3c40474797065a33c6934c4046b696e64c400c40573686170659103c4046461"
    "7461c40c03000000070000002a000000a676616c75657385c4026e64c3c40474797065a33c"
    "6634c4046b696e64c400c40573686170659103c40464617461c40c0000003f0000c03f0000"
    "2040"
)

GOLDEN_SPARSE_WITH_DIM = bytes.fromhex(
    "81a673706172736584a464696d73cd773aa56474797065a7666c6f61743332a7696e646963"
    "657385c4026e64c3c40474797065a33c6934c4046b696e64c400c40573686170659103c404"
    "64617461c40c03000000070000002a000000a676616c75657385c4026e64c3c40474797065"
    "a33c6634c4046b696e64c400c40573686170659103c40464617461c40c0000003f0000c03f"
    "00002040"
)

GOLDEN_MULTIVECTOR_3X4 = bytes.fromhex(
    "81ab6d756c7469766563746f7284aa746f6b656e5f64696d7304aa6e756d5f746f6b656e73"
    "03a56474797065a7666c6f61743332a676616c75657385c4026e64c3c40474797065a33c66"
    "34c4046b696e64c400c4057368617065920304c40464617461c430000000000000803f0000"
    "004000004040000080400000a0400000c0400000e04000000041000010410000204100003041"
)


class _FakeSparseConfig:
    class _Tasks:
        class _Encode:
            class _Sparse:
                dim = 30522

            sparse = _Sparse()

        encode = _Encode()

    tasks = _Tasks()


class _FakeSparseConfigNoDim:
    """Config with no sparse.dim — exercises the ``dims=None`` wire."""

    class _Tasks:
        class _Encode:
            sparse = None

        encode = _Encode()

    tasks = _Tasks()


class _FakeMultivectorConfig:
    class _Tasks:
        class _Encode:
            class _Multivector:
                dim = 4

            multivector = _Multivector()

        encode = _Encode()

    tasks = _Tasks()


def test_legacy_sparse_nil_dim_matches_rust_golden() -> None:
    """``_wrap_encode_output`` + ``msgpack.packb`` with ``dims=None``
    (no configured ``sparse.dim``) must produce the same bytes as
    ``build_sparse_payload(indices, values, None)``.
    """
    formatted = {
        "sparse": {
            "indices": np.array([3, 7, 42], dtype=np.int32),
            "values": np.array([0.5, 1.5, 2.5], dtype=np.float32),
        }
    }
    wrapped = _wrap_encode_output(formatted, _FakeSparseConfigNoDim())
    bytes_out = msgpack.packb(wrapped, use_bin_type=True)
    assert bytes_out == GOLDEN_SPARSE_NIL_DIM, (
        "Python legacy sparse (dims=None) bytes diverged from the Rust "
        "shaper golden — update BOTH sides in lockstep (crate sie_server_sidecar, see "
        "packages/sie_server_sidecar/src/output/mod.rs::tests::GOLDEN_SPARSE_NIL_DIM_HEX)."
    )


def test_legacy_sparse_with_dim_matches_rust_golden() -> None:
    formatted = {
        "sparse": {
            "indices": np.array([3, 7, 42], dtype=np.int32),
            "values": np.array([0.5, 1.5, 2.5], dtype=np.float32),
        }
    }
    wrapped = _wrap_encode_output(formatted, _FakeSparseConfig())
    bytes_out = msgpack.packb(wrapped, use_bin_type=True)
    assert bytes_out == GOLDEN_SPARSE_WITH_DIM, (
        "Python legacy sparse (dims=30522) bytes diverged from the Rust "
        "shaper golden — update BOTH sides in lockstep (crate sie_server_sidecar, see "
        "packages/sie_server_sidecar/src/output/mod.rs::tests::GOLDEN_SPARSE_WITH_DIM_HEX)."
    )


def test_legacy_multivector_matches_rust_golden() -> None:
    """3 tokens x 4 dims float32 multivector. Python legacy path ==
    ``build_multivector_payload(values, 3, 4)``.
    """
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    wrapped = _wrap_encode_output({"multivector": arr}, _FakeMultivectorConfig())
    bytes_out = msgpack.packb(wrapped, use_bin_type=True)
    assert bytes_out == GOLDEN_MULTIVECTOR_3X4, (
        "Python legacy multivector bytes diverged from the Rust "
        "shaper golden — update BOTH sides in lockstep (crate sie_server_sidecar, see "
        "packages/sie_server_sidecar/src/output/mod.rs::tests::GOLDEN_MULTIVECTOR_3X4_HEX)."
    )
