"""Conformance: sie_config's bundle_config_hash adapter_options canonicalization
matches the shared cross-language vectors (issue #1542).

The same `conformance/bundle_config_hash/canonical_profile_falsy_vectors.json`
fixture is asserted by the gateway Rust suite
(``packages/sie_gateway/src/types/model.rs``) and the sie_server suite, so the
config service's ``_canonical_adapter_options`` cannot drift from the gateway's
Rust ``canonicalize_adapter_options`` — a divergence would leave workers stuck in
``pending_workers`` with a ``bundle_config_hash`` that never matches the gateway's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sie_config.model_registry import _canonical_adapter_options


def _load_vectors() -> dict[str, Any]:
    for parent in Path(__file__).resolve().parents:
        fixture = parent / "conformance" / "bundle_config_hash" / "canonical_profile_falsy_vectors.json"
        if fixture.exists():
            return json.loads(fixture.read_text())
    msg = "canonical_profile_falsy_vectors.json not found in any parent directory"
    raise FileNotFoundError(msg)


_VECTORS = _load_vectors()


@pytest.mark.parametrize("case", _VECTORS["strip_to_none"], ids=lambda c: c["name"])
def test_falsy_only_options_strip_to_none(case: dict[str, Any]) -> None:
    assert _canonical_adapter_options(case["adapter_options"]) is None


@pytest.mark.parametrize("case", _VECTORS["keep_unchanged"], ids=lambda c: c["name"])
def test_meaningful_options_kept_unchanged(case: dict[str, Any]) -> None:
    assert _canonical_adapter_options(case["adapter_options"]) == case["adapter_options"]


@pytest.mark.parametrize("case", _VECTORS["transform"], ids=lambda c: c["name"])
def test_loadtime_runtime_compaction(case: dict[str, Any]) -> None:
    assert _canonical_adapter_options(case["adapter_options"]) == case["expected"]
