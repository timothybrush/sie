"""Regression tests for model YAML filenames in packages/sie_server/models/.

sie_bench's `EvalRunner._get_local_model_info(model_name)` looks up a YAML by
converting `model_name` to `model_name.replace("/", "__").replace(":", "__") + ".yaml"`.
On a case-sensitive filesystem (Linux CI), any case mismatch between the filename
and `sie_id` makes the lookup return `{}`, which silently downgrades the
dispatched MTEB wrapper to text-only and trips the modality precheck. See
issue #1058.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _expected_filename(sie_id: str) -> str:
    return sie_id.replace("/", "__").replace(":", "__") + ".yaml"


@pytest.mark.parametrize("yaml_path", sorted(MODELS_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_yaml_filename_matches_sie_id(yaml_path: Path) -> None:
    with yaml_path.open() as f:
        config = yaml.safe_load(f) or {}
    sie_id = config.get("sie_id")
    assert sie_id, f"{yaml_path.name}: missing sie_id"
    assert yaml_path.name == _expected_filename(sie_id), (
        f"YAML filename {yaml_path.name!r} does not match sie_id {sie_id!r}; "
        f"expected {_expected_filename(sie_id)!r}. "
        f"Case-sensitive filesystems (Linux CI) will fail to find this config."
    )
