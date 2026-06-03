# Release-gate guard: production Qwen3.5 profiles ship with
# ``grammar_backend: outlines`` (dottxt partnership; the codebase default).
# This test keeps the YAML aligned with that backend-of-record.

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SIE_SERVER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SIE_SERVER_ROOT.parents[1]
QWEN35_PROFILE = SIE_SERVER_ROOT / "models" / "Qwen__Qwen3.5-4B.yaml"
STATUS_DOC = REPO_ROOT / "product" / "research" / "generation-primitive-status.md"

EXPECTED_BACKEND = "outlines"


def _iter_profile_backends(profile_path: Path) -> list[tuple[str, str]]:
    """Return ``[(profile_name, grammar_backend), ...]`` for every profile in the file.

    The Qwen3.5 model YAML is a single top-level mapping with a ``profiles``
    dict. Each profile carries ``adapter_options.loadtime.grammar_backend``.
    """
    data = yaml.safe_load(profile_path.read_text()) or {}
    if not isinstance(data, dict):
        raise AssertionError(f"Expected {profile_path.name} to be a YAML mapping, got {type(data).__name__}")
    profiles = data.get("profiles") or {}
    if not isinstance(profiles, dict):
        raise AssertionError(
            f"Expected 'profiles' in {profile_path.name} to be a mapping, got {type(profiles).__name__}"
        )
    out: list[tuple[str, str]] = []
    for name, entry in profiles.items():
        if not isinstance(entry, dict):
            continue
        loadtime = (entry.get("adapter_options") or {}).get("loadtime") or {}
        backend = loadtime.get("grammar_backend")
        if backend is not None:
            out.append((str(name), str(backend)))
    return out


def test_qwen35_profiles_all_pin_outlines() -> None:
    """Every Qwen3.5 profile must set ``grammar_backend: outlines``.

    Outlines is the backend of record (dottxt partnership) for Qwen3.5. If a
    profile drifts to ``xgrammar`` or ``llguidance`` this test fires so the
    status doc gets updated alongside.
    """
    backends = _iter_profile_backends(QWEN35_PROFILE)
    assert backends, f"No profiles with grammar_backend found in {QWEN35_PROFILE}"
    mismatches = [(name, value) for name, value in backends if value != EXPECTED_BACKEND]
    assert not mismatches, (
        f"Qwen3.5 profile(s) drifted off the backend of record "
        f"(expected '{EXPECTED_BACKEND}'): {mismatches}. "
        f"If this is intentional, update both the YAML and the §4.2 Decision "
        f"record in {STATUS_DOC.relative_to(REPO_ROOT)}."
    )


def test_status_doc_names_outlines_as_backend_of_record() -> None:
    """The status doc must explicitly call Outlines the backend of record.

    The decision record in §4.2 is the single source of truth; this test
    guards against silent regressions where someone edits the YAML but not
    the doc (or vice versa).
    """
    if not STATUS_DOC.exists():
        pytest.skip(f"Status doc not present at {STATUS_DOC}")
    text = STATUS_DOC.read_text()
    # The decision record uses the exact phrase "backend of record" alongside
    # "Outlines". Both must appear; the doc must not say profiles are "pinned"
    # to xgrammar (the contradictory framing that finding H10 reconciled).
    assert "backend of record" in text.lower(), (
        "Expected the status doc to declare a 'backend of record' for structured outputs. Did §4.2 get rewritten?"
    )
    assert "outlines" in text.lower(), "Expected the status doc to name 'outlines' explicitly."
    forbidden_phrases = [
        "pin xgrammar",
        "pinned xgrammar",
        "profiles pin xgrammar",
        "profiles currently pin",
        "xgrammar as the validated fallback",
        "until the outlines a100 smoke",
        "until the outlines smoke",
    ]
    lowered = text.lower()
    hits = [p for p in forbidden_phrases if p in lowered]
    assert not hits, (
        f"Status doc contains contradictory 'pin xgrammar' phrasing that H10 "
        f"reconciled: {hits}. Production profiles ship grammar_backend: "
        f"outlines; the doc must match the code."
    )
