"""Regression guards for ``prewarm_grammars`` in the real model registry.

The Outlines grammar backend rejects regex **anchors** (``^`` / ``$``): its
FSM engine (``interegular``) raises ``interegular.patterns.Unsupported`` on
them, and inside SGLang's scheduler thread that uncaught exception SIGQUITs
the whole server (XGrammar tolerated anchors; Outlines does not). A single
anchored ``prewarm_grammars`` regex on an Outlines-backed model is therefore
a latent server-crasher — and exactly the bug a GPU smoke run surfaced for
Qwen3.5-4B / Qwen3-4B-Instruct-2507.

This module loads every shipped ``models/*.yaml`` and asserts that no
Outlines-backed model prewarms an anchored regex. ``interegular`` is a
worker-bundle-only dependency (not installed in the dev env), so the anchor
check is a string scan rather than a real parse; the scan is itself unit
tested below so the guard can be trusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sie_server.config.model import ModelConfig

# ``packages/sie_server/models`` relative to this test file
# (tests/config/ -> tests/ -> sie_server/).
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_MODEL_FILES = sorted(_MODELS_DIR.glob("*.yaml"))


def _unescaped_anchor(pattern: str) -> str | None:
    r"""Return the first unescaped, non-char-class ``^``/``$`` anchor, or None.

    Mirrors interegular's restriction closely enough for a config guard:
    escaped anchors (``\\^``, ``\\$``) and anchors inside a character class
    (``[$]``, the leading ``[^`` negation) are treated as literals and
    ignored; a bare ``^`` or ``$`` anywhere else is flagged.
    """
    i = 0
    in_class = False
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            i += 2  # skip the escaped char
            continue
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif c in "^$" and not in_class:
            return c
        i += 1
    return None


def _effective_grammar_backend(cfg: ModelConfig) -> str | None:
    """Resolve a generation model's grammar backend the way the worker does.

    Mirrors ``StreamingProcessor._grammar_backend_for_model_config``: the
    ``default`` profile's ``loadtime.grammar_backend``, defaulting to
    ``"outlines"`` when unset (the adapter's own default).
    """
    resolved = cfg.resolve_profile("default")
    loadtime = getattr(resolved, "loadtime", {}) or {}
    if not hasattr(loadtime, "get"):
        return "outlines"
    backend = loadtime.get("grammar_backend", "outlines")
    return str(backend) if backend is not None else None


def _load(path: Path) -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(path.read_text()))


@pytest.mark.parametrize("path", _MODEL_FILES, ids=lambda p: p.name)
def test_outlines_models_have_no_anchored_prewarm_regex(path: Path) -> None:
    """No Outlines-backed model may ship an anchored regex prewarm.

    Skips models without a ``generate`` task, without ``prewarm_grammars``,
    or that explicitly opt into a non-Outlines backend (anchors are fine
    there).
    """
    cfg = _load(path)
    gen = cfg.tasks.generate if cfg.tasks else None
    if gen is None or not gen.prewarm_grammars:
        pytest.skip("no generate task / no prewarm_grammars")
    if _effective_grammar_backend(cfg) != "outlines":
        pytest.skip("non-outlines grammar backend tolerates anchors")

    offenders: list[str] = []
    for entry in gen.prewarm_grammars:
        if entry.kind != "regex":
            continue
        anchor = _unescaped_anchor(str(entry.value))
        if anchor is not None:
            offenders.append(f"{entry.name}={entry.value!r} (anchor {anchor!r})")

    assert not offenders, (
        f"{path.name}: Outlines backend rejects regex anchors (^/$) and crashes "
        f"SGLang on them. De-anchor these prewarm regexes: {offenders}"
    )


def test_at_least_one_outlines_prewarm_model_is_covered() -> None:
    """Guard against the parametrized test silently skipping everything.

    If the registry stops shipping any Outlines model with regex prewarms
    (e.g. a refactor moves them), this canary fails so the guard above is
    not quietly a no-op.
    """
    covered = 0
    for path in _MODEL_FILES:
        cfg = _load(path)
        gen = cfg.tasks.generate if cfg.tasks else None
        if gen is None or not gen.prewarm_grammars:
            continue
        if _effective_grammar_backend(cfg) != "outlines":
            continue
        if any(e.kind == "regex" for e in gen.prewarm_grammars):
            covered += 1
    assert covered >= 1, "no Outlines model with a regex prewarm found — anchor guard is a no-op"


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("^(yes|no)$", "^"),  # the exact GPU-crash pattern
        ("(yes|no)", None),  # the de-anchored fix
        ("yes$", "$"),
        (r"\d{3}-\d{2}-\d{4}", None),
        (r"\^\d+", None),  # escaped caret is a literal
        (r"price:\$\d+", None),  # escaped dollar is a literal
        ("[$^]+", None),  # anchors inside a char class are literals
        ("[^abc]+", None),  # leading ^ is class negation, not an anchor
        ("a^b", "^"),  # bare anchor mid-pattern is still flagged
    ],
)
def test_unescaped_anchor_detector(pattern: str, expected: str | None) -> None:
    """The anchor scanner itself — the guard is only as good as this."""
    assert _unescaped_anchor(pattern) == expected
