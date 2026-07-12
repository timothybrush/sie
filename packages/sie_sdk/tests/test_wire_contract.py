"""The Python SDK's wire enums must match the shared golden fixtures.

Round-trips ``packages/wire-fixtures/model_state.json`` against the SDK's typed
``ModelState`` so drift (for example a state added on one side only) fails in CI
rather than shipping. See ``packages/wire-fixtures/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

from sie_sdk.types import ModelState

_FIXTURES = Path(__file__).parents[2] / "wire-fixtures"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


def test_model_state_matches_golden_fixture() -> None:
    fixture = _load("model_state.json")
    assert set(get_args(ModelState)) == set(fixture["model_states"])
