"""Conformance: the sie_server worker's bundle_config_hash adapter_options
canonicalization matches the shared cross-language vectors (issue #1542).

The worker advertises a ``bundle_config_hash`` that the gateway must reproduce or
the worker sits in ``pending_workers`` forever. Unlike the gateway (Rust
``canonicalize_adapter_options``) and the config service
(``sie_config._canonical_adapter_options``) — which are GENERIC flat canonicalizers
exercised by the ``strip_to_none`` / ``keep_unchanged`` vectors — the worker's
canonicalizer (``api.ws._compact_adapter_options_for_hash``) is structured: real
profiles always shape ``adapter_options`` as ``{loadtime, runtime}`` and the worker
compacts those two sub-maps. The shared ``transform`` vectors pin that
loadtime/runtime compaction, which all three implementations must agree on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sie_server.api.ws import _compact_adapter_options_for_hash


def _load_vectors() -> dict[str, Any]:
    for parent in Path(__file__).resolve().parents:
        fixture = parent / "conformance" / "bundle_config_hash" / "canonical_profile_falsy_vectors.json"
        if fixture.exists():
            return json.loads(fixture.read_text())
    msg = "canonical_profile_falsy_vectors.json not found in any parent directory"
    raise FileNotFoundError(msg)


_VECTORS = _load_vectors()


@pytest.mark.parametrize("case", _VECTORS["transform"], ids=lambda c: c["name"])
def test_loadtime_runtime_compaction(case: dict[str, Any]) -> None:
    options = case["adapter_options"]
    got = _compact_adapter_options_for_hash(options.get("loadtime", {}), options.get("runtime", {}))
    assert got == case["expected"]
