from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from sie_server.adapters import _pylate_dense
from sie_server.adapters._pylate_dense import apply_dense_chain, load_pylate_dense_chain

_MXBAI_DENSES = [(384, 768), (768, 768), (768, 64)]  # (in_features, out_features)
_TRANSFORMER_MODULE = {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"}


def _write_checkpoint(
    root: Path,
    denses: list[tuple[int, int]],
    *,
    modules: list[dict[str, Any]] | None = None,
    config_overrides: dict[int, dict[str, Any]] | None = None,
    state_overrides: dict[int, dict[str, torch.Tensor]] | None = None,
) -> str:
    """Write a real pylate-style checkpoint dir: modules.json + N_Dense modules.

    ``denses`` lists (in_features, out_features) per Dense module. ``modules``
    replaces modules.json wholesale; ``config_overrides``/``state_overrides``
    patch a single Dense module (keyed by its 1-based position).
    """
    root.mkdir(parents=True, exist_ok=True)
    if modules is None:
        modules = [_TRANSFORMER_MODULE] + [
            {"idx": i, "name": str(i), "path": f"{i}_Dense", "type": "pylate.models.Dense.Dense"}
            for i in range(1, len(denses) + 1)
        ]
    (root / "modules.json").write_text(json.dumps(modules))

    for i, (in_features, out_features) in enumerate(denses, start=1):
        dense_dir = root / f"{i}_Dense"
        dense_dir.mkdir(exist_ok=True)
        config: dict[str, Any] = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": False,
            "activation_function": "torch.nn.modules.linear.Identity",
        }
        config.update((config_overrides or {}).get(i, {}))
        (dense_dir / "config.json").write_text(json.dumps(config))
        state = {"linear.weight": torch.randn(out_features, in_features)}
        if state_overrides and i in state_overrides:
            state = state_overrides[i]
        save_file(state, str(dense_dir / "model.safetensors"))
    return str(root)


def test_chain_loads_mxbai_shape(tmp_path: Path) -> None:
    """The mxbai-edge 3-Dense chain (384->768->768->64) loads in order, dtype honored."""
    path = _write_checkpoint(tmp_path / "ckpt", _MXBAI_DENSES)

    chain = load_pylate_dense_chain(path, hidden_size=384, token_dim=64, device="cpu", dtype=torch.bfloat16)

    assert chain is not None
    assert [tuple(w.shape) for w in chain] == [(768, 384), (768, 768), (64, 768)]
    assert all(w.dtype == torch.bfloat16 for w in chain)


def test_chain_length_one_gte_shape(tmp_path: Path) -> None:
    """The GTE-style single-Dense chain (768->128) loads as one tensor."""
    path = _write_checkpoint(tmp_path / "ckpt", [(768, 128)])

    chain = load_pylate_dense_chain(path, hidden_size=768, token_dim=128, device="cpu", dtype=torch.float32)

    assert chain is not None
    assert [tuple(w.shape) for w in chain] == [(128, 768)]


def test_missing_modules_json_returns_none(tmp_path: Path, monkeypatch) -> None:
    """No modules.json (stanford-format repos) -> None, no exception."""

    def fail_download(*_args, **_kwargs):
        raise OSError("repo not found")

    monkeypatch.setattr(_pylate_dense, "hf_hub_download", fail_download)

    chain = load_pylate_dense_chain(str(tmp_path), hidden_size=768, token_dim=128, device="cpu", dtype=torch.float32)

    assert chain is None


def test_transformer_only_modules_json_returns_none(tmp_path: Path) -> None:
    """A Transformer-only modules.json (the real mxbai-colbert-large-v1 shape) -> None.

    LOAD-BEARING for the ColBERTAdapter fallback: this checkpoint's head is a
    root-safetensors linear.weight found by the legacy probes, which must run.
    """
    path = _write_checkpoint(tmp_path / "ckpt", [], modules=[_TRANSFORMER_MODULE])

    chain = load_pylate_dense_chain(path, hidden_size=1024, token_dim=128, device="cpu", dtype=torch.float32)

    assert chain is None


def test_unknown_module_type_returns_none(tmp_path: Path) -> None:
    """Any non-Dense module after the Transformer (e.g. Normalize) -> None."""
    modules = [
        _TRANSFORMER_MODULE,
        {"idx": 1, "name": "1", "path": "1_Dense", "type": "pylate.models.Dense.Dense"},
        {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
    ]
    path = _write_checkpoint(tmp_path / "ckpt", [(768, 128)], modules=modules)

    chain = load_pylate_dense_chain(path, hidden_size=768, token_dim=128, device="cpu", dtype=torch.float32)

    assert chain is None


def test_bias_true_rejected(tmp_path: Path) -> None:
    path = _write_checkpoint(tmp_path / "ckpt", [(768, 128)], config_overrides={1: {"bias": True}})

    chain = load_pylate_dense_chain(path, hidden_size=768, token_dim=128, device="cpu", dtype=torch.float32)

    assert chain is None


def test_use_residual_true_rejected(tmp_path: Path) -> None:
    path = _write_checkpoint(tmp_path / "ckpt", [(768, 128)], config_overrides={1: {"use_residual": True}})

    chain = load_pylate_dense_chain(path, hidden_size=768, token_dim=128, device="cpu", dtype=torch.float32)

    assert chain is None


def test_missing_use_residual_key_ok(tmp_path: Path) -> None:
    """GTE's Dense config has no use_residual key at all; missing == false."""
    path = _write_checkpoint(tmp_path / "ckpt", [(768, 128)])
    config = json.loads((tmp_path / "ckpt" / "1_Dense" / "config.json").read_text())
    assert "use_residual" not in config  # the regression being guarded

    chain = load_pylate_dense_chain(path, hidden_size=768, token_dim=128, device="cpu", dtype=torch.float32)

    assert chain is not None


def test_non_identity_activation_rejected(tmp_path: Path) -> None:
    """Reject a non-Identity activation_function: pylate never applies it, but its
    presence means the checkpoint was trained with math we would not reproduce.
    """
    path = _write_checkpoint(
        tmp_path / "ckpt",
        [(768, 128)],
        config_overrides={1: {"activation_function": "torch.nn.modules.activation.ReLU"}},
    )

    chain = load_pylate_dense_chain(path, hidden_size=768, token_dim=128, device="cpu", dtype=torch.float32)

    assert chain is None


def test_dim_discontinuity_rejected(tmp_path: Path) -> None:
    """Adjacent Dense modules must chain: 384->768 then 512->64 is invalid."""
    path = _write_checkpoint(tmp_path / "ckpt", [(384, 768), (512, 64)])

    chain = load_pylate_dense_chain(path, hidden_size=384, token_dim=64, device="cpu", dtype=torch.float32)

    assert chain is None


def test_first_in_features_mismatch_rejected(tmp_path: Path) -> None:
    """The first weight must consume the backbone hidden size."""
    path = _write_checkpoint(tmp_path / "ckpt", _MXBAI_DENSES)

    chain = load_pylate_dense_chain(path, hidden_size=999, token_dim=64, device="cpu", dtype=torch.float32)

    assert chain is None


def test_last_out_features_mismatch_rejected(tmp_path: Path) -> None:
    """A chain not landing on token_dim degrades to truncation — exactly the old
    single-head semantics (the pre-#1680 mxbai situation: only 1_Dense 384->768
    with token_dim 64).
    """
    path = _write_checkpoint(tmp_path / "ckpt", [(384, 768)])

    chain = load_pylate_dense_chain(path, hidden_size=384, token_dim=64, device="cpu", dtype=torch.float32)

    assert chain is None


def test_linear_bias_tensor_present_rejected(tmp_path: Path) -> None:
    """A linear.bias tensor in the safetensors means math we would not reproduce."""
    path = _write_checkpoint(
        tmp_path / "ckpt",
        [(768, 128)],
        state_overrides={1: {"linear.weight": torch.randn(128, 768), "linear.bias": torch.randn(128)}},
    )

    chain = load_pylate_dense_chain(path, hidden_size=768, token_dim=128, device="cpu", dtype=torch.float32)

    assert chain is None


def test_weight_shape_config_mismatch_rejected(tmp_path: Path) -> None:
    """The weight tensor must match the config's (out_features, in_features)."""
    path = _write_checkpoint(
        tmp_path / "ckpt",
        [(768, 128)],
        state_overrides={1: {"linear.weight": torch.randn(64, 768)}},
    )

    chain = load_pylate_dense_chain(path, hidden_size=768, token_dim=128, device="cpu", dtype=torch.float32)

    assert chain is None


def test_corrupt_safetensors_returns_none(tmp_path: Path) -> None:
    """A corrupt/truncated Dense weights file degrades instead of crashing load()."""
    path = _write_checkpoint(tmp_path / "ckpt", [(768, 128)])
    (tmp_path / "ckpt" / "1_Dense" / "model.safetensors").write_bytes(b"not a safetensors file")

    chain = load_pylate_dense_chain(path, hidden_size=768, token_dim=128, device="cpu", dtype=torch.float32)

    assert chain is None


def test_apply_dense_chain_matches_hand_matmul() -> None:
    """apply_dense_chain == explicit sequential matmuls for 2D and 3D inputs."""
    w0 = torch.randn(6, 8)
    w1 = torch.randn(5, 6)
    w2 = torch.randn(4, 5)
    chain = [w0, w1, w2]

    hidden_2d = torch.randn(7, 8)  # [tokens, hidden]
    out_2d = apply_dense_chain(hidden_2d, chain)
    assert out_2d.shape == (7, 4)
    assert torch.equal(out_2d, ((hidden_2d @ w0.T) @ w1.T) @ w2.T)

    hidden_3d = torch.randn(2, 3, 8)  # [batch, seq, hidden]
    out_3d = apply_dense_chain(hidden_3d, chain)
    assert out_3d.shape == (2, 3, 4)
    assert torch.equal(out_3d, ((hidden_3d @ w0.T) @ w1.T) @ w2.T)
