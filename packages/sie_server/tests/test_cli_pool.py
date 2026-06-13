from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from sie_server import cli
from sie_server.app.app_state_config import AppStateConfig
from sie_server.core.deps import collect_bundle_deps


def test_serve_passes_sie_pool_to_app_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_run_server(*, config: AppStateConfig, **kwargs: Any) -> None:
        captured["config"] = config
        captured["kwargs"] = kwargs

    monkeypatch.setenv("SIE_POOL", "customer-a")
    monkeypatch.setattr(cli, "run_server", fake_run_server)

    try:
        cli.serve(
            port=8080,
            host="127.0.0.1",
            device="cpu",
            models_dir=str(tmp_path),
            bundle=None,
            models=None,
            local_cache=None,
            cluster_cache=None,
            hf_fallback=True,
            reload=False,
            tracing=False,
            instrumentation=False,
            verbose=False,
            log_level="info",
            preload=None,
            json_logs=False,
        )
    finally:
        monkeypatch.delenv("SIE_BUNDLE", raising=False)
        monkeypatch.delenv("SIE_HF_FALLBACK", raising=False)
        monkeypatch.delenv("SIE_INSTRUMENTATION", raising=False)

    assert captured["config"].pool_name == "customer-a"


def _write_bundle(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "name": "shared",
                "adapters": ["sie_server.adapters.fake"],
            }
        )
    )


def _write_model(path: Path, *, sie_id: str, pool: str | None = None) -> None:
    data: dict[str, Any] = {
        "sie_id": sie_id,
        "profiles": {
            "default": {
                "adapter_path": "sie_server.adapters.fake:FakeAdapter",
            }
        },
    }
    if pool is not None:
        data["pool"] = pool
    path.write_text(yaml.safe_dump(data))


def test_load_bundle_filters_models_by_sie_pool(tmp_path: Path) -> None:
    bundles_dir = tmp_path / "bundles"
    models_dir = tmp_path / "models"
    bundles_dir.mkdir()
    models_dir.mkdir()
    _write_bundle(bundles_dir / "shared.yaml")
    _write_model(models_dir / "default.yaml", sie_id="org/default")
    _write_model(models_dir / "tenant.yaml", sie_id="org/tenant", pool="customer-a")

    assert cli.load_bundle("shared", bundles_dir, str(models_dir), pool_name="default") == ["org/default"]
    assert cli.load_bundle("shared", bundles_dir, str(models_dir), pool_name="customer-a") == ["org/tenant"]

    deps = collect_bundle_deps("shared", bundles_dir, models_dir, pool_name="customer-a")
    assert deps.models == ["org/tenant"]
