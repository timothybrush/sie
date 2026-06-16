from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Final, cast

import msgpack
import yaml

from sie_server.api.ws import compute_bundle_config_hash_cached
from sie_server.config.model import ModelConfig
from sie_server.core.oom import is_oom_error
from sie_server.core.prepared import ExtractPreparedItem
from sie_server.core.registry import ModelRegistry
from sie_server.core.score_cost import build_score_prepared_items
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.ipc_types import (
    ApplyModelConfigRequest,
    ApplyModelConfigResponse,
    BatchOutcome,
    DenseOutput,
    EncodeBatchItem,
    ExtractBatchItem,
    ItemOutcome,
    ModelDescriptor,
    MultivectorOutput,
    ProcessEncodeBatchRequest,
    ProcessExtractBatchRequest,
    ProcessScoreBatchRequest,
    RawOutput,
    ReadinessState,
    ReplaceModelConfigsRequest,
    ReplaceModelConfigsResponse,
    ScoreBatchItem,
    ScoreOutputRaw,
    SparseOutput,
)
from sie_server.observability.metrics import record_ipc_batch_shape
from sie_server.types.inputs import InvalidMediaError, Item
from sie_server.types.responses import ErrorCode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rust-side output framing: every adapter that emits typed RawOutput
# ---------------------------------------------------------------------------
#
# Per-request safety rules in ``_maybe_dense_raw_output`` /
# ``_maybe_sparse_raw_output`` / ``_maybe_multivector_raw_output`` emit
# ``RawOutput`` only when the adapter's output is the exact (single-key,
# float32, well-shaped) form Rust knows how to frame byte-identically;
# everything else still falls back to the legacy
# ``msgpack.packb(_wrap_encode_output(...))`` path.
#
# Safety rules:
#   * Dense: ONLY when the adapter emits a single ``dense`` output key
#     backed by a float32 ``np.ndarray``. Binary / int8 / float16 dense
#     still go through the Python-framed fallback path so nothing
#     regresses.
#   * Sparse: ONLY when the adapter emits a single ``sparse`` output
#     key with int32 indices + float32 values (the
#     ``SparseVector(indices=..., values=...)`` shape every in-tree
#     sparse adapter produces). float16 values fall back.
#   * Multivector: ONLY for a single ``multivector`` output key with
#     a float32 ``[num_tokens, token_dims]`` ndarray. Bit-packed
#     binary multivector (``shape[1] < mv_dim``) and float16 fall back.
#   * Score: always eligible — Rust mirrors the Python sort + rank
#     assignment byte-for-byte.
#   * Multi-output items (e.g. dense + sparse in one response) always
#     fall back — the wire contract is one variant per ``RawOutput``.
#   * On any shape error the Rust publisher converts the outcome into
#     ``publish_error_and_ack`` with ``error_code="raw_output_shape_error"``.
#     We never silently drop or mis-frame a request.

# ---------------------------------------------------------------------------
# Tokenizer materialisation
# ---------------------------------------------------------------------------

# Treat ``model_max_length`` ≥ this value as "unset" — HF defaults
# slow tokenizers to ``int(1e30)`` when no cap is declared. Anything
# above 1M tokens is implausible for the encoders we run; the sidecar
# falls back to its own default cap rather than truncating after a
# trillion tokens.
_TOKENIZER_MAX_SEQ_LEN_PLAUSIBILITY_CAP: Final[int] = 1_000_000
#
# The worker-sidecar can't read the adapter's HF cache directly (different
# container in the split-image world; in single-container today it's the
# same FS but the path is hard to discover from the model_id alone). On
# first ``EnsureModelReady`` we write the adapter's ``tokenizer.json``
# to a stable, per-model path inside an emptyDir-equivalent staging
# directory and ship that path to the sidecar in
# ``ModelDescriptor.tokenizer_path``. The sidecar loads from there.
#
# The directory is configurable via ``SIE_TOKENIZER_STAGING_DIR`` for
# the split-container deploy (point both containers at the same
# ``emptyDir`` mount); defaults to ``$TMPDIR/sie-tokenizers`` so unit
# tests and dev shells need no extra config.

_TOKENIZER_STAGING_DIR = Path(
    os.environ.get("SIE_TOKENIZER_STAGING_DIR") or (tempfile.gettempdir() + "/sie-tokenizers")
)


def _safe_model_id_for_path(model_id: str) -> str:
    """Map a HF-style ``org/name`` (or worse, ``Org/Name@revision``) to
    a single filesystem-safe path component. We keep the mapping
    intentionally lossy (no reverse) because the Rust side gets the
    full path back via the descriptor, not by reconstruction.
    """
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "__" for ch in model_id)


def _materialise_tokenizer(model_id: str, tokenizer: Any) -> str | None:
    """Write the canonical ``tokenizer.json`` for ``tokenizer`` to a
    per-model path under :data:`_TOKENIZER_STAGING_DIR` and return the
    absolute path. Returns ``None`` when the tokenizer is a slow (Python)
    tokenizer that doesn't expose a ``backend_tokenizer`` — the sidecar
    cannot fast-path those anyway, so there is no value in materialising
    them.

    Idempotent on repeat invocations: if the target file already exists
    with matching size, we trust the cached descriptor (the per-process
    cache in :class:`QueueExecutor` short-circuits before we get here on
    the steady-state hot path; this routine is the cold-start writer
    plus the cross-process recovery path).

    Concurrency-safe: the temporary file name embeds ``os.getpid()`` so
    two adapter processes (or two threads in the same process taking
    different identity locks) writing the same model's tokenizer don't
    stomp each other's tmp file. POSIX ``rename`` then publishes the
    final ``tokenizer.json`` atomically — readers in the sidecar never
    see a partial write.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None
    try:
        raw = backend.to_str(pretty=False)
    except Exception:  # noqa: BLE001
        logger.debug("materialise_tokenizer: backend_tokenizer.to_str failed for %s", model_id, exc_info=True)
        return None
    # ``to_str`` MUST return ``str`` on a real ``tokenizers.Tokenizer``.
    # Anything else (MagicMock, None, bytes) is treated as "no canonical
    # JSON available" — the sidecar will fall back to Python tokenisation
    # for this model.
    if not isinstance(raw, str):
        return None
    canonical = raw.encode("utf-8")

    target_dir = _TOKENIZER_STAGING_DIR / _safe_model_id_for_path(model_id)
    target_path = target_dir / "tokenizer.json"
    try:
        if target_path.is_file() and target_path.stat().st_size == len(canonical):
            # Size match is a cheap proxy for byte identity. The sidecar
            # additionally hashes on its side and reconciles against the
            # ``tokenizer_id`` on the descriptor, so a false positive
            # here just means the registry refuses the registration and
            # Python keeps tokenising — never a silent mis-frame.
            return str(target_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = target_dir / f"tokenizer.json.{os.getpid()}.tmp"
        tmp_path.write_bytes(canonical)
        tmp_path.replace(target_path)
        return str(target_path)
    except OSError:
        logger.warning(
            "materialise_tokenizer: failed to write %s — sidecar will fall back to Python tokenisation",
            target_path,
            exc_info=True,
        )
        return None


def _maybe_dense_raw_output(
    formatted: dict[str, Any],
    config: Any,
    output_types: list[str],
) -> RawOutput | None:
    """Return a ``RawOutput`` for the dense-only fast path, or ``None``
    to fall back to the Python-framed fallback path.

    All conditions must hold — the gate is deliberately strict so
    Rust-side framing only intercepts cases the Rust shaper is known
    to produce byte-identical bytes for.
    """
    import numpy as np  # noqa: PLC0415

    if output_types != ["dense"]:
        return None
    if set(formatted.keys()) - {"dense"}:
        # Sparse / multivector in the same item would need different
        # Rust framers — not in v1.
        return None
    arr = formatted.get("dense")
    if not isinstance(arr, np.ndarray):
        return None
    if arr.dtype != np.float32:
        # Binary (uint8 bit-packed), float16, int8 all need different
        # framers and/or dtype tags; Python still handles them.
        return None
    if arr.ndim != 1:
        return None

    encode_task = getattr(getattr(config, "tasks", None), "encode", None)
    dense_cfg = getattr(encode_task, "dense", None) if encode_task else None
    dense_dim = dense_cfg.dim if dense_cfg else None
    dim = int(dense_dim) if dense_dim is not None else int(arr.shape[0])
    if arr.shape[0] != dim:
        # Don't try to hide a shape mismatch; fall back and let the
        # legacy path mis-label it the same way it does today rather
        # than introduce a new error class here.
        return None

    # ``arr.tolist()`` widens each ``np.float32`` to a Python ``float``
    # (f64). The Rust side narrows back to ``f32`` on decode — this
    # round-trip is exact because every f32 has a unique f64
    # representation. The Rust shaper then widens back to f64 before
    # emitting the ``msgpack_numpy`` sentinel's raw-bytes payload,
    # matching Python's ``arr.tobytes()`` bit-for-bit.
    return RawOutput(
        dense=DenseOutput(
            values=arr.tolist(),
            dim=dim,
            # v1 policy: adapter-side normalize stays in Python.
            # Flipping this to ``True`` is a later optimisation.
            normalize=False,
        ),
    )


def _maybe_sparse_raw_output(
    formatted: dict[str, Any],
    config: Any,
    output_types: list[str],
) -> RawOutput | None:
    """Sparse-only fast path. Returns a ``RawOutput`` carrying a
    ``SparseOutput`` when every v1 invariant holds, otherwise ``None``
    so the Python-framed fallback path takes over.

    The Rust shaper emits the exact bytes that
    ``_wrap_encode_output`` packs today — see
    ``sie_server_sidecar::output::build_sparse_payload`` and the
    byte-identity tests in
    ``test_stage1d_byte_identity.py::test_sparse_legacy_matches_rust_golden``.
    """
    import numpy as np  # noqa: PLC0415

    if output_types != ["sparse"]:
        return None
    if set(formatted.keys()) - {"sparse"}:
        return None
    sparse_in = formatted.get("sparse")
    if not isinstance(sparse_in, dict):
        return None
    indices = sparse_in.get("indices")
    values = sparse_in.get("values")
    if indices is None or values is None:
        return None
    if not isinstance(indices, np.ndarray) or not isinstance(values, np.ndarray):
        return None
    if indices.ndim != 1 or values.ndim != 1:
        return None
    if indices.shape[0] != values.shape[0]:
        return None
    # Adapter-layer convention is ``np.int32`` indices; anything
    # wider / signedness-different would reshape on .tolist() but we
    # stay strict so the gate is obvious at review time.
    if indices.dtype != np.int32:
        return None
    # v1: float32 values only. float16 (the other dtype the legacy
    # path labels) stays on Python until we teach the Rust shaper
    # the ``"<f2"`` sentinel variant.
    if values.dtype != np.float32:
        return None

    encode_task = getattr(getattr(config, "tasks", None), "encode", None)
    sparse_cfg = getattr(encode_task, "sparse", None) if encode_task else None
    sparse_dim = sparse_cfg.dim if sparse_cfg else None
    dims = int(sparse_dim) if sparse_dim is not None else None

    return RawOutput(
        sparse=SparseOutput(
            # ``.tolist()`` widens np.int32 → Python int (exact) and
            # np.float32 → Python float (exact round-trip). Rust
            # rehydrates via ``Vec<i32>`` / ``Vec<f32>`` with narrowing.
            indices=indices.tolist(),
            values=values.tolist(),
            dims=dims,
        ),
    )


def _maybe_multivector_raw_output(
    formatted: dict[str, Any],
    config: Any,
    output_types: list[str],
) -> RawOutput | None:
    """Multivector-only fast path for the Rust output shaper.

    Mirrors the invariants of the ``multivector`` branch of
    ``_wrap_encode_output``:

      * Single output key == ``multivector``.
      * ``np.float32`` 2-D ``[num_tokens, token_dims]`` ndarray.
      * NOT bit-packed binary (``shape[1] < mv_dim`` with uint8
        dtype) — the binary path stays in Python for v1 because the
        Rust shaper does not know the ``"binary"`` dtype tag yet.
      * ``shape[1]`` matches the configured ``token_dims`` when the
        model exposes one, so the Rust shaper's ``num_tokens ×
        token_dims`` invariant holds without Python-side backfill.
    """
    import numpy as np  # noqa: PLC0415

    _MV_NDIM = 2  # `[num_tokens, token_dims]` — the only shape we forward.

    if output_types != ["multivector"]:
        return None
    if set(formatted.keys()) - {"multivector"}:
        return None
    arr = formatted.get("multivector")
    if not isinstance(arr, np.ndarray):
        return None
    if arr.dtype != np.float32:
        return None
    if arr.ndim != _MV_NDIM:
        return None

    encode_task = getattr(getattr(config, "tasks", None), "encode", None)
    mv_cfg = getattr(encode_task, "multivector", None) if encode_task else None
    mv_dim = mv_cfg.dim if mv_cfg else None

    num_tokens = int(arr.shape[0])
    if mv_dim is not None:
        # Bit-packed binary has shape[1] == dim/8 with uint8 dtype —
        # the dtype check above already refused uint8, but also
        # refuse a narrower float32 shape just in case an adapter
        # ever pre-flattens/truncates.
        if arr.shape[1] != int(mv_dim):
            return None
        token_dims = int(mv_dim)
    else:
        token_dims = int(arr.shape[1])

    # Values must be contiguous in C order so ``.tobytes()`` (and the
    # Rust ``values.to_le_bytes()`` equivalent) agree. ``tolist()``
    # on a 2-D ndarray returns a nested list; ``ravel()`` flattens
    # row-major first so we stay byte-compatible regardless of input
    # memory layout.
    return RawOutput(
        multivector=MultivectorOutput(
            values=arr.ravel(order="C").tolist(),
            num_tokens=num_tokens,
            token_dims=token_dims,
        ),
    )


# ---------------------------------------------------------------------------
# Wire formatting helpers
# ---------------------------------------------------------------------------

_NP_DTYPE_MAP = {"float32": "float32", "float16": "float16", "int8": "int8", "uint8": "binary"}


def _wrap_encode_output(output: dict, config: Any) -> dict:
    """Wrap raw numpy arrays from EncodeHandler.format_output into the
    DenseVector / SparseVector / MultiVector wire format the SDK expects.

    The HTTP path does this via pydantic models (see
    ``api/encode.py::_format_dense`` / ``_format_sparse`` /
    ``_format_multivector``); the queue path publishes msgpack bytes
    directly, so the same wrapping must happen here. Keeping the two
    paths in sync matters — an SDK client that round-trips via HTTP then
    via queue otherwise sees different shapes for ``sparse`` and
    ``multivector``.
    """
    import numpy as np  # noqa: PLC0415

    wrapped = dict(output)

    encode_task = getattr(config, "tasks", None)
    encode_task = getattr(encode_task, "encode", None)

    if "dense" in wrapped and isinstance(wrapped["dense"], np.ndarray):
        arr = wrapped["dense"]
        dense_cfg = getattr(encode_task, "dense", None) if encode_task else None
        dense_dim = dense_cfg.dim if dense_cfg else None

        is_binary = arr.dtype == np.uint8 and dense_dim and arr.shape[0] < dense_dim
        dims = dense_dim if dense_dim is not None else arr.shape[0]
        dtype = "binary" if is_binary else _NP_DTYPE_MAP.get(str(arr.dtype), "float32")

        wrapped["dense"] = {"dims": int(dims), "dtype": dtype, "values": arr}

    if "sparse" in wrapped and isinstance(wrapped["sparse"], dict):
        # Adapter output shape: {"indices": np.ndarray, "values": np.ndarray}
        # SDK wire shape: {"dims": int|None, "dtype": "float32"|"float16",
        #                   "indices": np.ndarray, "values": np.ndarray}
        sparse_in = wrapped["sparse"]
        indices = sparse_in.get("indices")
        values = sparse_in.get("values")
        if indices is not None and values is not None:
            if not isinstance(indices, np.ndarray):
                indices = np.asarray(indices)
            if not isinstance(values, np.ndarray):
                values = np.asarray(values)
            sparse_cfg = getattr(encode_task, "sparse", None) if encode_task else None
            sparse_dim = sparse_cfg.dim if sparse_cfg else None
            dtype = _NP_DTYPE_MAP.get(str(values.dtype), "float32")
            # Sparse only supports float{32,16}; fall back rather than
            # silently mislabel.
            if dtype not in {"float32", "float16"}:
                dtype = "float32"
            wrapped["sparse"] = {
                "dims": int(sparse_dim) if sparse_dim is not None else None,
                "dtype": dtype,
                "indices": indices,
                "values": values,
            }

    if "multivector" in wrapped and isinstance(wrapped["multivector"], np.ndarray):
        arr = wrapped["multivector"]
        mv_cfg = getattr(encode_task, "multivector", None) if encode_task else None
        mv_dim = mv_cfg.dim if mv_cfg else None
        # Binary multivector packs `dim/8` bytes per token; detect by
        # `shape[1] < mv_dim` like `_format_multivector` does.
        if arr.dtype == np.uint8 and mv_dim is not None and arr.shape[1] < mv_dim:
            token_dims = int(mv_dim)
            dtype = "binary"
        else:
            token_dims = int(mv_dim if mv_dim is not None else arr.shape[1])
            dtype = _NP_DTYPE_MAP.get(str(arr.dtype), "float32")
        wrapped["multivector"] = {
            "token_dims": token_dims,
            "num_tokens": int(arr.shape[0]),
            "dtype": dtype,
            "values": arr,
        }

    return wrapped


def _dict_to_item(d: dict) -> Item:
    """Convert an SDK wire-format dict into a server-side Item struct.

    The SDK ships ``content`` while the server expects ``text``; remap and
    drop any unknown keys so the Item constructor does not raise.
    """
    if "content" in d and "text" not in d:
        d = {**d, "text": d.pop("content")}
    known = set(Item.__struct_fields__)
    return Item(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# QueueExecutor
# ---------------------------------------------------------------------------


class QueueExecutor:
    """NATS-free execution layer fronted by the IPC server for the worker-sidecar.

    The executor owns the path from "decoded work items have arrived for model X"
    to "per-item outcomes ready to be ACKed, NAKed, or replied to". It does NOT
    own JetStream fetch, ACK/NAK, payload store fetch, or reply publish — those
    live in the worker-sidecar.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        # Per-model descriptor cache. Populated on the first
        # ``EnsureModelReady`` for a model and reused on every
        # subsequent batch's handshake (the dispatcher re-handshakes
        # per group). Keeps file I/O off the hot path:
        # ``_materialise_tokenizer`` runs once at cold start, then we
        # just hand back the cached struct. Cleared by
        # :meth:`invalidate_model_descriptor` when a model is unloaded
        # or hot-reloaded.
        self._descriptor_cache: dict[str, ModelDescriptor] = {}

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def invalidate_model_descriptor(self, model_id: str) -> None:
        """Drop the cached descriptor for ``model_id``.

        Called when a model is unloaded or hot-reloaded so the next
        ``EnsureModelReady`` re-materialises the tokeniser and the
        sidecar picks up the new ``tokenizer_id``. Safe to call for
        unknown models (no-op).
        """
        self._descriptor_cache.pop(model_id, None)

    def apply_model_config(self, req: ApplyModelConfigRequest) -> ApplyModelConfigResponse:
        """Validate and add a bundle-scoped config delta to the local registry."""
        if not req.bundle_id:
            msg = "bundle_id is required"
            raise ValueError(msg)
        if not req.model_config.strip():
            msg = "model_config is required"
            raise ValueError(msg)

        raw = yaml.safe_load(req.model_config)
        if not isinstance(raw, dict):
            msg = "model_config must decode to a YAML mapping"
            raise ValueError(msg)

        model_config = ModelConfig(**raw)
        if req.model_id and model_config.sie_id != req.model_id:
            msg = f"model_id mismatch: notification={req.model_id!r} config={model_config.sie_id!r}"
            raise ValueError(msg)

        self._registry.add_config(model_config)
        self.invalidate_model_descriptor(model_config.sie_id)
        bundle_hash = compute_bundle_config_hash_cached(self._registry, req.bundle_id)
        return ApplyModelConfigResponse(
            applied=True,
            bundle_config_hash=bundle_hash,
            config_version=int(getattr(self._registry, "_config_version", 0)),
        )

    def compute_bundle_config_hash(self, bundle_id: str) -> str:
        """Return the local registry hash for ``bundle_id``."""
        if not bundle_id:
            return ""
        return compute_bundle_config_hash_cached(self._registry, bundle_id)

    async def replace_model_configs(self, req: ReplaceModelConfigsRequest) -> ReplaceModelConfigsResponse:
        """Replace the bundle-scoped registry view from a full export snapshot."""
        if not req.bundle_id:
            msg = "bundle_id is required"
            raise ValueError(msg)

        configs: list[ModelConfig] = []
        for entry in req.models:
            if not entry.model_config.strip():
                msg = "model_config is required"
                raise ValueError(msg)

            raw = yaml.safe_load(entry.model_config)
            if not isinstance(raw, dict):
                msg = "model_config must decode to a YAML mapping"
                raise ValueError(msg)

            model_config = ModelConfig(**raw)
            if entry.model_id and model_config.sie_id != entry.model_id:
                msg = f"model_id mismatch: export={entry.model_id!r} config={model_config.sie_id!r}"
                raise ValueError(msg)
            configs.append(model_config)

        invalidated = await self._registry.replace_configs_async(configs)
        for model_id in invalidated:
            self.invalidate_model_descriptor(model_id)
        bundle_hash = compute_bundle_config_hash_cached(self._registry, req.bundle_id)
        applied_models = sorted(self._registry.get_configs_snapshot(req.bundle_id))
        return ReplaceModelConfigsResponse(
            applied=True,
            bundle_config_hash=bundle_hash,
            config_version=int(getattr(self._registry, "_config_version", 0)),
            applied_models=applied_models,
        )

    # -- Readiness ---------------------------------------------------------

    async def ensure_model_ready(self, model_id: str) -> ReadinessState:
        """Return the current readiness state for a model, triggering a load if needed.

        Mapping (for the Rust side):
        - ``ready``: continue processing
        - ``loading_started``: NAK with base delay (this call triggered a new load)
        - ``loading_in_progress``: NAK with longer delay (already loading)
        - ``retry_later``: NAK with base delay (unknown error path)
        """
        if not self._registry.has_model(model_id):
            return "retry_later"

        if self._registry.is_loaded(model_id):
            return "ready"

        if self._registry.is_loading(model_id):
            return "loading_in_progress"

        try:
            started = await self._registry.start_load_async(model_id, self._registry.device)
        except KeyError:
            # Unknown model — gateway should not be sending us work for it, but
            # be defensive: tell caller to retry rather than hard-erroring.
            return "retry_later"
        except Exception:  # noqa: BLE001
            logger.warning("ensure_model_ready: start_load_async failed for %s", model_id, exc_info=True)
            return "retry_later"

        return "loading_started" if started else "loading_in_progress"

    # -- Handshake-driven model descriptor --------------------------------

    def get_model_descriptor(self, model_id: str) -> ModelDescriptor | None:
        """Return the ``ModelDescriptor`` carried on the
        ``EnsureModelReadyResponse`` for this model, or ``None`` if the
        adapter has nothing structured to declare yet (slow tokenizer,
        image / audio adapter, model not yet loaded).

        The descriptor lets the worker-sidecar discover per-model
        capabilities at runtime; see
        ``packages/sie_server_sidecar/docs/architecture-guide.md``.

        Populates:

        * ``tokenizer_path`` — path to a sidecar-readable
          ``tokenizer.json`` materialised from
          ``preprocessor.backend_tokenizer.to_str(pretty=False)``. Stays
          ``None`` for slow tokenizers (no canonical JSON to ship) and
          for adapters that don't expose a ``TextPreprocessor`` (image /
          audio).
        * ``tokenizer_id`` — BLAKE3 (32 hex) of the canonical tokenizer
          JSON. Same value the sidecar will compute from the
          materialised file, so the two sides can verify byte-identity
          before enabling the Rust-tokenise fast path.
        * ``max_seq_len`` — ``tokenizer.model_max_length`` when sane
          (``< 10**6``; HF's ``VERY_LARGE_INTEGER`` sentinel reads as
          ``None`` here).
        * ``output_types`` — informational; left empty for now since
          per-request shape checks in ``_maybe_*_raw_output`` are the
          authoritative gate. Future work can populate from
          ``adapter.supported_output_types`` if/when adapters expose it.
        * ``supports_run_batch`` — every Python adapter does.

        Cached after the first successful build; the dispatcher
        re-handshakes on every batch and we don't want to re-stat the
        staging dir each time. Use
        :meth:`invalidate_model_descriptor` on unload / hot reload.
        """
        cached = self._descriptor_cache.get(model_id)
        if cached is not None:
            return cached

        # Gate on the model actually being loaded — calling this before
        # ``ensure_model_ready`` would race the registry. The IPC
        # server only invokes us on the ``ready`` branch so this is
        # belt-and-braces, but it also short-circuits MagicMock-based
        # tests where ``get_worker`` returns ``None`` for unknown ids.
        try:
            if self._registry.get_worker(model_id) is None:
                return None
        except (KeyError, AttributeError):
            return None

        tokenizer_path: str | None = None
        tokenizer_id: str | None = None
        max_seq_len: int | None = None

        try:
            preprocessor = self._registry.preprocessor_registry.get_preprocessor(model_id, "text")
        except Exception:  # noqa: BLE001
            preprocessor = None
        if preprocessor is not None and hasattr(preprocessor, "tokenizer_id"):
            try:
                candidate_id = preprocessor.tokenizer_id  # may be None for slow tokenizers
            except Exception:  # noqa: BLE001
                logger.debug("get_model_descriptor: tokenizer_id property raised for %s", model_id, exc_info=True)
                candidate_id = None
            # Strict type guard: only ``str`` survives. Test fixtures
            # often pass ``MagicMock``-based preprocessors whose
            # ``tokenizer_id`` would otherwise be a Mock; treating that
            # as a real id would crash msgpack later.
            if isinstance(candidate_id, str):
                tokenizer_id = candidate_id
            inner = getattr(preprocessor, "_tokenizer", None)
            if inner is not None:
                tokenizer_path = _materialise_tokenizer(model_id, inner)
                # ``model_max_length`` is set to a giant sentinel
                # (``int(1e30)``) when the tokenizer doesn't declare a
                # cap; treat anything implausibly large as "unset".
                raw_max = getattr(inner, "model_max_length", None)
                if (
                    isinstance(raw_max, int)
                    and not isinstance(raw_max, bool)
                    and 0 < raw_max < _TOKENIZER_MAX_SEQ_LEN_PLAUSIBILITY_CAP
                ):
                    max_seq_len = raw_max

        # Surface the adapter's model-default templates so the sidecar
        # can apply them before tokenising in Rust. All
        # text adapters store these as ``_query_template`` /
        # ``_doc_template`` (set in their ``__init__`` from the model
        # YAML). Image / audio adapters and adapters without text
        # templating leave them as ``None``, which keeps the sidecar
        # on the legacy "no Rust-side templating" path for that model.
        # ``MagicMock``-based test fixtures and adapters that don't
        # follow the convention silently degrade the same way.
        default_query_template: str | None = None
        default_doc_template: str | None = None
        try:
            worker = self._registry.get_worker(model_id)
            adapter = worker.adapter if worker is not None else None
        except Exception:  # noqa: BLE001
            adapter = None
        if adapter is not None:
            qt = getattr(adapter, "_query_template", None)
            if isinstance(qt, str):
                default_query_template = qt
            dt = getattr(adapter, "_doc_template", None)
            if isinstance(dt, str):
                default_doc_template = dt

        descriptor = ModelDescriptor(
            tokenizer_path=tokenizer_path,
            tokenizer_id=tokenizer_id,
            max_seq_len=max_seq_len,
            output_types=[],
            supports_run_batch=True,
            default_query_template=default_query_template,
            default_doc_template=default_doc_template,
        )
        self._descriptor_cache[model_id] = descriptor
        return descriptor

    def get_batch_budget(self, model_id: str) -> int | None:
        """Return the per-batch dispatch budget for a model, or ``None`` if
        unknown / model not loaded.

        Reads ``worker._batch_config.max_batch_requests`` when available. Queue
        consumers (Python or Rust) use this to cap how many messages for
        one model are processed per fetch batch so a hot model doesn't
        monopolise the GPU.
        """
        try:
            worker = self._registry.get_worker(model_id)
        except (KeyError, AttributeError):
            return None
        if worker is None:
            return None
        batch_config = getattr(worker, "_batch_config", None)
        if batch_config is None:
            return None
        budget = getattr(batch_config, "max_batch_requests", None)
        if isinstance(budget, int) and budget > 0:
            return budget
        return None

    # -- Encode ------------------------------------------------------------

    async def process_encode_batch(self, req: ProcessEncodeBatchRequest) -> BatchOutcome:
        """Run an encode batch through EncodePipeline and return per-item outcomes.

        Sub-grouping: items from different API requests may have different ``output_types``,
        ``instruction``, ``is_query``, or ``options`` — these cannot share a
        single ``EncodePipeline.run_encode()`` call.
        """
        from sie_server.core.encode_pipeline import EncodePipeline  # noqa: PLC0415

        model_id = req.model_id
        items = req.items

        try:
            config = self._registry.get_config(model_id)
        except KeyError:
            # Model evicted mid-batch — NAK all items so another worker (or
            # this one after re-loading) processes them.
            return BatchOutcome(outcomes=[_nak_outcome(it) for it in items])

        outcomes: dict[str, ItemOutcome] = {}

        # Group key includes `profile_id` and `bundle_config_hash` — the
        # gateway uses those to select adapter variants / postprocessors,
        # and merging items with different values would run the wrong
        # pipeline. Keep the key in sync with whatever `EncodePipeline.run_encode`
        # actually reads; add more fields here if it grows.
        groups: dict[tuple, list[EncodeBatchItem]] = {}
        for bi in items:
            output_types = tuple(bi.output_types or ["dense"])
            options_key = msgpack.packb(bi.options, use_bin_type=True) if bi.options else b""
            key = (
                output_types,
                bi.instruction,
                bi.is_query,
                options_key,
                bi.profile_id,
                bi.bundle_config_hash,
            )
            groups.setdefault(key, []).append(bi)

        # Record IPC-batch shape so operators can see the fragmentation
        # ratio (items per IPC batch vs items per GPU forward pass).
        # Emitted once per batch *after* grouping but *before* running
        # any forward pass, so batches whose inference throws are still
        # accounted for in the histogram. Model-eviction NAKs above
        # are intentionally not recorded — no real work was attempted.
        record_ipc_batch_shape(
            model=model_id,
            endpoint="encode",
            total_items=len(items),
            sub_group_sizes=[len(g) for g in groups.values()],
        )

        for group_key, group in groups.items():
            (output_types_t, instruction, is_query, options_key, _profile_id, _bundle_hash) = group_key
            output_types = list(output_types_t)
            options = msgpack.unpackb(options_key, raw=False) if options_key else {}
            server_items = [_dict_to_item(bi.item) for bi in group]
            # Collect worker-sidecar prepared_tokens aligned with
            # ``server_items``. ``None`` per item is expected for the
            # v1 safety-rule skips (`is_query`, `instruction`,
            # non-text, empty text). The pipeline accepts the mix
            # per-item: items with usable Rust bytes skip Python
            # tokenisation, items with ``None`` are tokenised in
            # Python and spliced back — see
            # ``TextPreprocessor.try_prepare_from_prepared_tokens``
            # for the hybrid policy. Whole-batch fallback only fires
            # on correctness-critical drift (tokenizer_id mismatch,
            # malformed wire shape).
            prepared_tokens_per_item = [bi.prepared_tokens for bi in group]
            if not any(pt is not None for pt in prepared_tokens_per_item):
                # Not a single fast-path candidate — pass None so the
                # pipeline doesn't even bother looking up the
                # preprocessor's cached tokenizer_id.
                prepared_tokens_per_item = None

            try:
                formatted_outputs, timing = await EncodePipeline.run_encode(
                    registry=self._registry,
                    model=model_id,
                    items=server_items,
                    output_types=output_types,
                    instruction=instruction,
                    config=config,
                    is_query=is_query,
                    options=options,
                    prepared_tokens_per_item=prepared_tokens_per_item,
                )

                if len(formatted_outputs) != len(group):
                    # Output/input length mismatch means the adapter
                    # silently dropped items. Surface as a per-item error
                    # rather than publishing empty success results.
                    logger.warning(
                        "Encode sub-batch for %s returned %d outputs for %d items — emitting per-item errors",
                        model_id,
                        len(formatted_outputs),
                        len(group),
                    )
                    for bi in group:
                        outcomes[bi.work_item_id] = _error_outcome(
                            bi,
                            "inference_error",
                            "adapter returned fewer outputs than items",
                        )
                else:
                    for idx, bi in enumerate(group):
                        raw_output: RawOutput | None = None
                        result_msgpack: bytes | None = None
                        # Dispatch by the single declared output type —
                        # the v1 wire contract is exactly one variant
                        # per ``RawOutput``. Multi-output items (no
                        # single key matches) and adapter outputs that
                        # don't pass the per-helper safety rules drop
                        # to the legacy ``_wrap_encode_output`` path
                        # below.
                        if output_types == ["dense"]:
                            raw_output = _maybe_dense_raw_output(
                                formatted_outputs[idx],
                                config,
                                output_types,
                            )
                        elif output_types == ["sparse"]:
                            raw_output = _maybe_sparse_raw_output(
                                formatted_outputs[idx],
                                config,
                                output_types,
                            )
                        elif output_types == ["multivector"]:
                            raw_output = _maybe_multivector_raw_output(
                                formatted_outputs[idx],
                                config,
                                output_types,
                            )
                        if raw_output is None:
                            output = _wrap_encode_output(formatted_outputs[idx], config)
                            result_msgpack = msgpack.packb(output, use_bin_type=True)
                        outcomes[bi.work_item_id] = ItemOutcome(
                            work_item_id=bi.work_item_id,
                            request_id=bi.request_id,
                            item_index=bi.item_index,
                            disposition="publish_and_ack",
                            result_msgpack=result_msgpack,
                            raw_output=raw_output,
                            inference_ms=timing.inference_ms,
                            tokenization_ms=timing.tokenization_ms if timing.tokenization_ms > 0 else None,
                            postprocessing_ms=timing.postprocessing_ms if timing.postprocessing_ms > 0 else None,
                        )
            except Exception as e:  # noqa: BLE001
                logger.warning("Encode sub-batch failed for model %s: %s", model_id, e)
                for bi in group:
                    outcomes[bi.work_item_id] = _inference_exception_outcome(bi, e)

        return BatchOutcome(outcomes=[outcomes[bi.work_item_id] for bi in items])

    # -- Score -------------------------------------------------------------

    async def process_score_batch(self, req: ProcessScoreBatchRequest) -> BatchOutcome:
        """Run a score batch — items are submitted concurrently so BatchFormer
        can cross-batch them inside ``ModelWorker`` (same as the legacy path).
        """
        import asyncio  # noqa: PLC0415

        model_id = req.model_id

        # Score dispatches one task per IPC item; actual GPU batching
        # happens below in BatchFormer. We still report
        # sub_groups == items so the fragmentation dashboard panel
        # treats all three endpoints uniformly (any deviation then
        # becomes a signal that BatchFormer merged less than expected
        # — surfaced via sie_gpu_batch_items in model_worker.py).
        record_ipc_batch_shape(
            model=model_id,
            endpoint="score",
            total_items=len(req.items),
            sub_group_sizes=[1] * len(req.items),
        )

        tasks = [self._process_single_score(model_id, bi) for bi in req.items]
        outcomes = await asyncio.gather(*tasks)
        return BatchOutcome(outcomes=list(outcomes))

    async def _process_single_score(self, model_id: str, bi: ScoreBatchItem) -> ItemOutcome:
        try:
            worker = await self._registry.start_worker(model_id)
        except (KeyError, RuntimeError) as e:
            logger.info("Model %s not available for score: %s — NAKing", model_id, e)
            return _nak_outcome(bi)

        query_item = _dict_to_item(bi.query_item)
        score_items = [_dict_to_item(it) for it in bi.score_items]

        try:
            timing = RequestTiming()

            timing.start_tokenization()
            prepared_items = build_score_prepared_items(query_item, score_items)
            timing.end_tokenization()

            future = await worker.submit_score(
                prepared_items=prepared_items,
                query=query_item,
                items=score_items,
                instruction=bi.instruction,
                options=bi.options or {},
                timing=timing,
            )
            worker_result = await future

            score_output = cast("Any", worker_result.output)
            raw_scores = [float(score_output.scores[i]) for i in range(score_output.batch_size)]
            item_ids: list[str] = [
                (sid if (sid := score_items[i].id) is not None else f"item-{i}") for i in range(score_output.batch_size)
            ]

            # Score output is always Rust-frameable: the Python and Rust
            # sort/rank paths produce byte-identical results (see the
            # parity test in ``test_queue_executor_stage1d.py``).
            # Rust-side framing is unconditional for score; the Python-framed fallback
            # sort+pack lives only as a doc-comment record of what
            # ``sie_server_sidecar::output::build_score_payload`` mirrors.
            raw_output: RawOutput | None = RawOutput(
                score=ScoreOutputRaw(scores=raw_scores, item_ids=item_ids),
            )
            result_msgpack: bytes | None = None

            return ItemOutcome(
                work_item_id=bi.work_item_id,
                request_id=bi.request_id,
                item_index=bi.item_index,
                disposition="publish_and_ack",
                result_msgpack=result_msgpack,
                raw_output=raw_output,
                inference_ms=timing.inference_ms,
                tokenization_ms=timing.tokenization_ms if timing.tokenization_ms > 0 else None,
                postprocessing_ms=timing.postprocessing_ms if timing.postprocessing_ms > 0 else None,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Score failed for %s: %s", bi.work_item_id, e)
            return _inference_exception_outcome(bi, e)

    # -- Extract -----------------------------------------------------------

    async def process_extract_batch(self, req: ProcessExtractBatchRequest) -> BatchOutcome:
        """Run an extract batch — items submitted concurrently so BatchFormer can batch them."""
        import asyncio  # noqa: PLC0415

        model_id = req.model_id

        record_ipc_batch_shape(
            model=model_id,
            endpoint="extract",
            total_items=len(req.items),
            sub_group_sizes=[1] * len(req.items),
        )

        tasks = [self._process_single_extract(model_id, bi) for bi in req.items]
        outcomes = await asyncio.gather(*tasks)
        return BatchOutcome(outcomes=list(outcomes))

    async def _process_single_extract(self, model_id: str, bi: ExtractBatchItem) -> ItemOutcome:
        try:
            worker = await self._registry.start_worker(model_id)
        except (KeyError, RuntimeError) as e:
            logger.info("Model %s not available for extract: %s — NAKing", model_id, e)
            return _nak_outcome(bi)

        server_item = _dict_to_item(bi.item)

        try:
            timing = RequestTiming()
            timing.start_tokenization()
            char_count = len(server_item.text) if server_item.text else 0
            prepared_items = [ExtractPreparedItem(cost=char_count, original_index=0)]
            timing.end_tokenization()

            future = await worker.submit_extract(
                prepared_items=prepared_items,
                items=[server_item],
                labels=bi.labels,
                output_schema=bi.output_schema,
                instruction=bi.instruction,
                options=bi.options or {},
                timing=timing,
            )
            worker_result = await future

            extraction_results = ExtractHandler.format_output(cast("Any", worker_result.output))
            if not extraction_results:
                # Adapter returned no results for a single-item request —
                # surface as an error instead of silently publishing an
                # empty object (which the client would parse as "success
                # with no fields").
                return _error_outcome(bi, "inference_error", "adapter returned no extraction results")
            result_msgpack = msgpack.packb(extraction_results[0], use_bin_type=True)

            return ItemOutcome(
                work_item_id=bi.work_item_id,
                request_id=bi.request_id,
                item_index=bi.item_index,
                disposition="publish_and_ack",
                result_msgpack=result_msgpack,
                inference_ms=timing.inference_ms,
                tokenization_ms=timing.tokenization_ms if timing.tokenization_ms > 0 else None,
                postprocessing_ms=timing.postprocessing_ms if timing.postprocessing_ms > 0 else None,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Extract failed for %s: %s", bi.work_item_id, e)
            return _inference_exception_outcome(bi, e)


# ---------------------------------------------------------------------------
# Outcome helpers
# ---------------------------------------------------------------------------


def _nak_outcome(bi: EncodeBatchItem | ScoreBatchItem | ExtractBatchItem) -> ItemOutcome:
    return ItemOutcome(
        work_item_id=bi.work_item_id,
        request_id=bi.request_id,
        item_index=bi.item_index,
        disposition="nak_retry",
        nak_delay_ms=int(_default_nak_delay_s() * 1000),
    )


def _oom_nak_outcome(bi: EncodeBatchItem | ScoreBatchItem | ExtractBatchItem) -> ItemOutcome:
    return ItemOutcome(
        work_item_id=bi.work_item_id,
        request_id=bi.request_id,
        item_index=bi.item_index,
        disposition="nak_retry",
        nak_delay_ms=int(_oom_nak_delay_s() * 1000),
    )


def _inference_exception_outcome(
    bi: EncodeBatchItem | ScoreBatchItem | ExtractBatchItem,
    exc: BaseException,
) -> ItemOutcome:
    if is_oom_error(exc):
        return _oom_nak_outcome(bi)
    if isinstance(exc, InvalidMediaError):
        return _error_outcome(bi, ErrorCode.INVALID_INPUT.value, str(exc))
    return _error_outcome(bi, "inference_error", str(exc))


def _error_outcome(bi: EncodeBatchItem | ScoreBatchItem | ExtractBatchItem, code: str, message: str) -> ItemOutcome:
    return ItemOutcome(
        work_item_id=bi.work_item_id,
        request_id=bi.request_id,
        item_index=bi.item_index,
        disposition="publish_error_and_ack",
        error=message,
        error_code=code,
    )


def _default_nak_delay_s() -> float:
    """NAK delay (seconds) used when a model is evicted / not yet ready.

    Must agree with the worker-sidecar's ``SIE_NAK_DELAY_S`` default so
    redelivery behaviour is consistent across either side choosing the
    fallback.
    """
    import os  # noqa: PLC0415

    return float(os.environ.get("SIE_NAK_DELAY_S", "5.0"))


def _oom_nak_delay_s() -> float:
    """NAK delay (seconds) for retryable OOM / RESOURCE_EXHAUSTED failures.

    This mirrors the deleted Python NATS worker contract: do not publish an
    error and ACK for OOM, because JetStream redelivery may land on a sibling
    worker after memory pressure clears.
    """
    return float(os.environ.get("SIE_OOM_NAK_DELAY_S", "10.0"))
