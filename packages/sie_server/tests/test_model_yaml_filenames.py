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


@pytest.mark.parametrize("yaml_path", sorted(MODELS_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_instruction_template_has_placeholder(yaml_path: Path) -> None:
    """An ``Instruct:``-prefixed ``query_template`` must contain the ``{instruction}``
    placeholder.

    Otherwise the eval harness's instruction gating
    (``EvalRunner._model_uses_instruction``, added in #1432) treats the model as
    non-instruction-following and silently drops the per-task MTEB prompt, so the
    model is measured with a hardcoded generic instruction instead of each task's
    instruction. This is the stella_en_*_v5 regression (#1340).
    """
    with yaml_path.open() as f:
        config = yaml.safe_load(f) or {}
    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        return
    offenders: list[str] = []
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        adapter_options = profile.get("adapter_options")
        runtime = adapter_options.get("runtime") if isinstance(adapter_options, dict) else None
        query_template = runtime.get("query_template") if isinstance(runtime, dict) else None
        if isinstance(query_template, str) and "Instruct:" in query_template and "{instruction}" not in query_template:
            offenders.append(profile_name)
    assert not offenders, (
        f"{yaml_path.name}: profile(s) {offenders} hardcode an 'Instruct:' instruction "
        f"without a '{{instruction}}' placeholder; the eval harness drops the per-task "
        f"MTEB instruction (#1340). Use 'Instruct: {{instruction}}' + 'default_instruction'."
    )
