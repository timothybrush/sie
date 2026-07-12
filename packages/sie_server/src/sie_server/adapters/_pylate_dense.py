"""Shared pylate Dense-chain loading for ColBERT-style adapters.

pylate ColBERT checkpoints describe their projection head in ``modules.json``:
module 0 is the sentence-transformers ``Transformer`` backbone, and every later
module is a ``pylate.models.Dense.Dense`` whose weights live in
``<path>/model.safetensors`` under the single key ``linear.weight`` (shape
``[out_features, in_features]``). pylate's ``Dense.forward`` is literally
``self.linear(x)`` — it never applies the checkpoint's ``activation_function``
— so with ``bias: false`` / ``use_residual: false`` the whole head is the
sequential matmul ``hidden @ W0.T @ W1.T ...`` on the backbone's final-normed
``last_hidden_state``. This composition is proven bit-exact against official
pylate output in ``_poc/pylate-dense-chain/FINDINGS.md``. See #1680.

Any checkpoint whose modules.json is absent, has no Dense modules (e.g.
mxbai-colbert-large-v1 ships a Transformer-only modules.json), or declares math
this loader does not reproduce (bias, residual, non-Identity activation, dim
discontinuities) yields ``None`` so callers keep their existing behavior
(legacy projection probes / backbone truncation) — degrade, don't fail.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def _repo_file(model_name_or_path: str, rel: str, *, revision: str | None = None) -> str | None:
    """Resolve a repo-relative file locally or via the HF hub; None if absent."""
    local = Path(model_name_or_path) / rel
    if local.is_file():
        return str(local)
    try:
        return hf_hub_download(model_name_or_path, rel, revision=revision)
    except Exception:  # noqa: BLE001 - missing file/offline -> caller decides severity
        return None


def load_pylate_dense_chain(
    model_name_or_path: str,
    *,
    hidden_size: int,
    token_dim: int,
    device: str | torch.device | None,
    dtype: torch.dtype,
    revision: str | None = None,
) -> list[torch.Tensor] | None:
    """Load the trained pylate Dense chain for a checkpoint.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        hidden_size: Backbone hidden size; the first weight's in_features must
            match it.
        token_dim: Served token dimension; the last weight's out_features must
            match it (a head that does not land on token_dim degrades to
            truncation, preserving the old single-head semantics).
        device: Device to move the weights to.
        dtype: Dtype to cast the weights to.
        revision: Optional HuggingFace revision/branch/commit SHA to pin when
            downloading the checkpoint's Dense-chain artifacts (``modules.json``,
            per-module config/weights) from the Hub. The head lives in the same
            repo as the backbone, so callers pass their pinned model revision so
            the whole snapshot is version-consistent.

    Returns:
        The chain weights in modules.json order, or None to signal the caller
        to fall back to its existing behavior (backbone truncation / legacy
        projection probes).
    """
    modules_path = _repo_file(model_name_or_path, "modules.json", revision=revision)
    if modules_path is None:
        # Normal for stanford-format ColBERT repos (no modules.json shipped).
        logger.debug("No modules.json for %s; no pylate Dense chain to load", model_name_or_path)
        return None

    try:
        parsed = json.loads(Path(modules_path).read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Unparseable modules.json for %s; using backbone truncation (%s)", model_name_or_path, exc)
        return None
    modules: list[dict[str, Any]] = []
    if not isinstance(parsed, list) or not parsed:
        logger.warning("Malformed modules.json for %s; using backbone truncation", model_name_or_path)
        return None
    for entry in parsed:
        if not isinstance(entry, dict):
            logger.warning("Malformed modules.json for %s; using backbone truncation", model_name_or_path)
            return None
        modules.append(entry)

    modules.sort(key=lambda m: m.get("idx", 0))
    first_type = str(modules[0].get("type", ""))
    if not first_type.endswith("Transformer"):
        logger.warning(
            "modules.json for %s does not start with a Transformer module (got %s); using backbone truncation",
            model_name_or_path,
            first_type,
        )
        return None

    dense_modules: list[dict[str, Any]] = []
    for module in modules[1:]:
        module_type = str(module.get("type", ""))
        if module_type.rsplit(".", 1)[-1] != "Dense":
            logger.warning(
                "Unsupported module type %s in modules.json for %s; using backbone truncation",
                module_type,
                model_name_or_path,
            )
            return None
        dense_modules.append(module)

    if not dense_modules:
        # A Transformer-only modules.json is a healthy checkpoint layout
        # (e.g. mxbai-colbert-large-v1); the caller's existing probes apply.
        logger.debug("No Dense modules in modules.json for %s", model_name_or_path)
        return None

    weights: list[torch.Tensor] = []
    for module in dense_modules:
        rel_dir = str(module.get("path", ""))
        config_path = _repo_file(model_name_or_path, f"{rel_dir}/config.json", revision=revision)
        if config_path is None:
            logger.warning("Missing %s/config.json for %s; using backbone truncation", rel_dir, model_name_or_path)
            return None
        try:
            config = json.loads(Path(config_path).read_text())
        except (OSError, ValueError) as exc:
            logger.warning(
                "Unparseable %s/config.json for %s; using backbone truncation (%s)",
                rel_dir,
                model_name_or_path,
                exc,
            )
            return None
        if config.get("bias"):
            logger.warning(
                "Dense module %s has bias=true for %s; using backbone truncation", rel_dir, model_name_or_path
            )
            return None
        # Missing key == false (GTE-ModernColBERT's config has no use_residual key).
        if config.get("use_residual"):
            logger.warning(
                "Dense module %s has use_residual=true for %s; using backbone truncation",
                rel_dir,
                model_name_or_path,
            )
            return None
        # pylate's Dense.forward never applies activation_function, but
        # sentence-transformers' Dense.forward does — a non-Identity value means
        # the checkpoint was trained with math we would not reproduce: degrade.
        activation = config.get("activation_function")
        if activation is not None and str(activation).rsplit(".", 1)[-1] != "Identity":
            logger.warning(
                "Dense module %s has non-Identity activation_function=%s for %s; using backbone truncation",
                rel_dir,
                activation,
                model_name_or_path,
            )
            return None

        weights_path = _repo_file(model_name_or_path, f"{rel_dir}/model.safetensors", revision=revision)
        if weights_path is None:
            logger.warning(
                "Missing %s/model.safetensors for %s; using backbone truncation. "
                "If this checkpoint ships Dense modules (e.g. partial weight cache), embeddings are degraded.",
                rel_dir,
                model_name_or_path,
            )
            return None
        try:
            state = load_file(weights_path)
        except Exception as exc:  # noqa: BLE001 - corrupt/partial Dense -> degrade, don't fail the load
            logger.warning(
                "Failed to read %s/model.safetensors for %s (%s); using backbone truncation",
                rel_dir,
                model_name_or_path,
                exc,
            )
            return None
        weight = state.get("linear.weight")
        if weight is None or weight.ndim != 2:
            logger.warning(
                "Dense module %s for %s has no 2D linear.weight (keys=%s); using backbone truncation",
                rel_dir,
                model_name_or_path,
                sorted(state.keys()),
            )
            return None
        if "linear.bias" in state:
            logger.warning(
                "Dense module %s for %s ships a linear.bias tensor; using backbone truncation",
                rel_dir,
                model_name_or_path,
            )
            return None
        expected_shape = (config.get("out_features"), config.get("in_features"))
        if tuple(weight.shape) != expected_shape:
            logger.warning(
                "Dense module %s weight shape %s != config (out, in)=%s for %s; using backbone truncation",
                rel_dir,
                tuple(weight.shape),
                expected_shape,
                model_name_or_path,
            )
            return None
        weights.append(weight)

    # Shape-guard the chain ends plus adjacency: the first hop must consume the
    # backbone hidden size, every hop must consume the previous hop's output,
    # and the last hop must land exactly on token_dim (preserving the old
    # single-head semantics: a head not landing on token_dim degrades to
    # truncation, as today).
    if weights[0].shape[1] != hidden_size:
        logger.warning(
            "Dense chain in_features=%d != model hidden_size=%d for %s; using backbone truncation",
            weights[0].shape[1],
            hidden_size,
            model_name_or_path,
        )
        return None
    for prev, nxt in itertools.pairwise(weights):
        if nxt.shape[1] != prev.shape[0]:
            logger.warning(
                "Dense chain dimension discontinuity (%d -> %d) for %s; using backbone truncation",
                prev.shape[0],
                nxt.shape[1],
                model_name_or_path,
            )
            return None
    if weights[-1].shape[0] != token_dim:
        logger.warning(
            "Dense chain out_features=%d != token_dim=%d for %s; using backbone truncation",
            weights[-1].shape[0],
            token_dim,
            model_name_or_path,
        )
        return None

    logger.info(
        "Loaded pylate Dense chain for %s: %s",
        model_name_or_path,
        [tuple(w.shape) for w in weights],
    )
    return [w.to(device=device, dtype=dtype) for w in weights]


def apply_dense_chain(hidden: torch.Tensor, chain: Sequence[torch.Tensor]) -> torch.Tensor:
    """Apply the Dense chain: ``hidden @ W0.T @ W1.T ...`` in order.

    Works for 2D packed ``[tokens, hidden]`` and 3D padded
    ``[batch, seq, hidden]`` inputs (plain matmul broadcasting).
    """
    for weight in chain:
        hidden = hidden @ weight.T
    return hidden
