# Catalog guard: ``extends`` profiles must not silently drop the parent's
# text templates. Profile resolution replaces ``adapter_options.runtime``
# wholesale when the child block is non-empty, so a child that re-states
# ``pooling``/``normalize`` but forgets ``query_template``/``doc_template``
# serves the model without its required prefixes — a silent quality
# regression (issue #1595: multilingual-e5-large lost its ``query: `` /
# ``passage: `` prefixes on the sentence_transformer profile, −12 to −16%
# on retrieval). To intentionally serve a profile without a parent
# template, set the key explicitly to ``null`` in the child runtime block.

from __future__ import annotations

from pathlib import Path

import yaml

SIE_SERVER_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = SIE_SERVER_ROOT / "models"

TEMPLATE_KEYS = ("query_template", "doc_template", "default_instruction")


def _runtime_block(profile: dict) -> dict:
    adapter_options = profile.get("adapter_options") or {}
    if not isinstance(adapter_options, dict):
        return {}
    runtime = adapter_options.get("runtime") or {}
    return runtime if isinstance(runtime, dict) else {}


def test_extends_profiles_do_not_silently_drop_templates() -> None:
    """Every ``extends`` child with a non-empty runtime block keeps the
    parent's template keys (an explicit ``null`` counts as a deliberate
    opt-out; a missing key is the silent-drop trap).
    """
    model_files = sorted(MODELS_DIR.glob("*.yaml"))
    assert model_files, f"No model configs found in {MODELS_DIR}"

    violations: list[str] = []
    for path in model_files:
        data = yaml.safe_load(path.read_text()) or {}
        profiles = data.get("profiles") or {}
        if not isinstance(profiles, dict):
            continue
        for name, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            parent_name = profile.get("extends")
            if parent_name is None:
                continue
            parent = profiles.get(parent_name)
            if not isinstance(parent, dict):
                continue
            child_runtime = _runtime_block(profile)
            if not child_runtime:
                # Empty child runtime inherits the parent's block wholesale.
                continue
            parent_runtime = _runtime_block(parent)
            dropped = [key for key in TEMPLATE_KEYS if key in parent_runtime and key not in child_runtime]
            if dropped:
                violations.append(f"{path.name} profile '{name}' (extends '{parent_name}') drops {dropped}")

    assert not violations, (
        "Profile runtime blocks are full replacements, not merges — these "
        "profiles silently lose the parent's text templates (see #1595). "
        "Re-state the template (or set it explicitly to null to opt out):\n  " + "\n  ".join(violations)
    )
