from __future__ import annotations

from pathlib import Path

from sie_sdk.bundle_utils import match_bundle_models


def _write_model(models_dir: Path, name: str, *, pool: str | None = None) -> None:
    pool_line = f"pool: {pool}\n" if pool else ""
    (models_dir / f"{name.replace('/', '__')}.yaml").write_text(
        f"""
sie_id: {name}
{pool_line}profiles:
  default:
    adapter_path: pkg.adapters.sglang:Adapter
""".lstrip()
    )


def test_match_bundle_models_filters_by_pool(tmp_path: Path) -> None:
    bundle_path = tmp_path / "sglang.yaml"
    bundle_path.write_text("adapters:\n  - pkg.adapters.sglang\n")
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    _write_model(models_dir, "org/generation")
    _write_model(models_dir, "org/embedding", pool="SGLang-Embedding")

    assert set(match_bundle_models(bundle_path, models_dir)) == {"org/generation", "org/embedding"}
    assert match_bundle_models(bundle_path, models_dir, pool_name="default") == ["org/generation"]
    assert match_bundle_models(bundle_path, models_dir, pool_name="sglang-embedding") == ["org/embedding"]
    assert match_bundle_models(bundle_path, models_dir, pool_name="SGLANG-EMBEDDING") == ["org/embedding"]
