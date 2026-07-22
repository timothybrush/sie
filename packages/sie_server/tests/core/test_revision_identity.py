"""Served-model version-identity validation (design §6.6).

A promoted/served HF-backed model must pin ``hf_revision`` to an immutable
commit SHA so its ``sie_id`` maps to identical weights forever; a weights
change becomes a NEW versioned id. Package-backed live downloads are also
rejected; package locks are not model revisions.
"""

from pathlib import Path

import pytest
from sie_server.config.model import (
    AdapterOptions,
    EmbeddingDim,
    EncodeTask,
    ExtractTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.loader import (
    is_immutable_revision,
    load_model_configs,
    require_pinned_revisions,
    validate_pinned_revision,
)

_SHA = "5617a9f61b028005a4858fdac845db406aefb181"  # a real 40-hex commit SHA shape


def _encode_config(
    *,
    hf_id: str | None = "org/model",
    hf_revision: str | None = None,
    weights_path: Path | None = None,
    package_backed: bool = False,
    package_artifact_loadtime: dict[str, object] | None = None,
) -> ModelConfig:
    kwargs: dict[str, object] = {
        "sie_id": "org/model",
        "package_backed": package_backed,
        "tasks": Tasks(extract=ExtractTask())
        if package_backed
        else Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
        "profiles": {
            "default": ProfileConfig(
                adapter_path="test:Adapter",
                max_batch_tokens=8192,
                adapter_options=AdapterOptions(loadtime=package_artifact_loadtime or {}),
            )
        },
    }
    if not package_backed:
        kwargs["hf_id"] = hf_id
        kwargs["hf_revision"] = hf_revision
        kwargs["weights_path"] = weights_path
    return ModelConfig(**kwargs)


class TestIsImmutableRevision:
    @pytest.mark.parametrize("rev", [_SHA, "0" * 40, "abcdef0123456789abcdef0123456789abcdef01"])
    def test_accepts_40_hex_sha(self, rev: str) -> None:
        assert is_immutable_revision(rev) is True

    @pytest.mark.parametrize("rev", [None, "main", "v1.0", _SHA[:12], _SHA + "aa", _SHA.upper()])
    def test_rejects_non_immutable(self, rev: str | None) -> None:
        assert is_immutable_revision(rev) is False


class TestValidatePinnedRevision:
    def test_rejects_unpinned_hf_model(self) -> None:
        with pytest.raises(ValueError, match="has no 'hf_revision'"):
            validate_pinned_revision(_encode_config(hf_revision=None))

    def test_rejects_branch_pin(self) -> None:
        with pytest.raises(ValueError, match="immutable 40-char commit SHA"):
            validate_pinned_revision(_encode_config(hf_revision="main"))

    def test_accepts_immutable_sha(self) -> None:
        validate_pinned_revision(_encode_config(hf_revision=_SHA))  # no raise

    def test_weights_path_model_is_exempt(self) -> None:
        validate_pinned_revision(_encode_config(hf_id=None, weights_path=Path("/w/model")))

    def test_bundled_package_backed_model_is_not_promotable(self) -> None:
        with pytest.raises(ValueError, match="does not declare a staged artifact manifest"):
            validate_pinned_revision(_encode_config(package_backed=True))

    def test_live_package_artifacts_are_not_promotable(self) -> None:
        with pytest.raises(ValueError, match="does not declare a staged artifact manifest"):
            validate_pinned_revision(
                _encode_config(
                    package_backed=True,
                    package_artifact_loadtime={"package_artifact_mode": "live"},
                )
            )

    def test_staged_package_artifact_manifest_is_promotable(self) -> None:
        validate_pinned_revision(
            _encode_config(
                package_backed=True,
                package_artifact_loadtime={
                    "package_artifact_mode": "staged",
                    "package_artifact_manifest_path": "/models/package/model/manifest.json",
                    "package_artifact_manifest_sha256": "a" * 64,
                },
            )
        )


_MODEL_YAML = """\
sie_id: {sie_id}
hf_id: {hf_id}
{rev_line}tasks:
  encode:
    dense:
      dim: 768
profiles:
  default:
    adapter_path: test:Adapter
    max_batch_tokens: 8192
"""


def _write_yaml(models_dir: Path, name: str, *, revision: str | None) -> None:
    rev_line = f"hf_revision: {revision}\n" if revision is not None else ""
    (models_dir / f"{name}.yaml").write_text(
        _MODEL_YAML.format(sie_id=f"org/{name}", hf_id=f"org/{name}", rev_line=rev_line)
    )


class TestRequirePinnedRevisionsAtLoad:
    def test_strict_load_rejects_unpinned(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path, "pinned", revision=_SHA)
        _write_yaml(tmp_path, "unpinned", revision=None)
        with pytest.raises(ValueError, match="hf_revision"):
            load_model_configs(tmp_path, require_pinned_revision=True)

    def test_non_strict_load_accepts_unpinned(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path, "unpinned", revision=None)
        configs = load_model_configs(tmp_path)  # default: not strict
        assert "org/unpinned" in configs

    def test_strict_load_accepts_all_pinned(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path, "a", revision=_SHA)
        _write_yaml(tmp_path, "b", revision="cdbee75f17c01a7cc42f958dc650907174af0554")
        configs = load_model_configs(tmp_path, require_pinned_revision=True)
        assert {"org/a", "org/b"} <= set(configs)
        require_pinned_revisions(configs)  # idempotent, no raise
