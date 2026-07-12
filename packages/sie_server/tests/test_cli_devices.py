from __future__ import annotations

from typing import Any

from sie_server import cli
from sie_server.app.app_state_config import AppStateConfig
from typer.testing import CliRunner


def test_serve_threads_sie_devices_env_to_app_config(monkeypatch, tmp_path) -> None:
    """Helm-rendered SIE_DEVICES reaches AppStateConfig through the CLI path."""
    captured: dict[str, AppStateConfig] = {}

    def fake_run_server(**kwargs: Any) -> None:
        captured["config"] = kwargs["config"]

    monkeypatch.setenv("SIE_DEVICES", "cuda:0, cuda:1,,")
    monkeypatch.setattr(cli, "run_server", fake_run_server)

    result = CliRunner().invoke(
        cli.app,
        [
            "serve",
            "--device",
            "cuda",
            "--models-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].device == "cuda"
    assert captured["config"].devices == ["cuda:0", "cuda:1"]
