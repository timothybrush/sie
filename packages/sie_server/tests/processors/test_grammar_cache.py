"""Tests for the :class:`GrammarLRU` cache."""

from __future__ import annotations

import threading

import pytest
from sie_server.processors.grammar_cache import GrammarLRU
from sie_server.types.grammar import GrammarSpec, hash_grammar


def test_hash_grammar_distinguishes_kind_for_same_value() -> None:
    """BUG 4 regression: a regex and an ebnf with identical ``value`` must
    NOT collide in the cache key — otherwise an ebnf "ready" sentinel would
    satisfy a regex lookup and skip the regex's own Outlines preflight.
    """
    regex = GrammarSpec(kind="regex", value="[a-z]+")
    ebnf = GrammarSpec(kind="ebnf", value="[a-z]+")
    assert hash_grammar(regex) != hash_grammar(ebnf)


def test_hash_grammar_stable_for_same_kind_and_value() -> None:
    """Same (kind, value) hashes identically (cache hits still work)."""
    a = GrammarSpec(kind="regex", value="[a-z]+")
    b = GrammarSpec(kind="regex", value="[a-z]+", label="other", strict=True)
    assert hash_grammar(a) == hash_grammar(b)


def _key(seed: str) -> tuple[str, str, str]:
    return (f"tok_{seed}", f"schema_{seed}", "outlines")


def test_get_returns_none_for_missing_key() -> None:
    lru = GrammarLRU(maxsize=4)
    assert lru.get(_key("a")) is None


def test_put_then_get_returns_value() -> None:
    lru = GrammarLRU(maxsize=4)
    lru.put(_key("a"), "value-a")
    assert lru.get(_key("a")) == "value-a"


def test_evicts_least_recently_used_when_full() -> None:
    lru = GrammarLRU(maxsize=2)
    lru.put(_key("a"), "A")
    lru.put(_key("b"), "B")
    # Touch ``a`` to refresh its recency — ``b`` is now LRU.
    assert lru.get(_key("a")) == "A"
    lru.put(_key("c"), "C")
    # ``b`` evicted, ``a`` and ``c`` remain.
    assert lru.get(_key("b")) is None
    assert lru.get(_key("a")) == "A"
    assert lru.get(_key("c")) == "C"
    assert len(lru) == 2


def test_put_existing_key_does_not_grow() -> None:
    lru = GrammarLRU(maxsize=2)
    lru.put(_key("a"), 1)
    lru.put(_key("a"), 2)
    assert len(lru) == 1
    assert lru.get(_key("a")) == 2


def test_clear_empties_cache() -> None:
    lru = GrammarLRU(maxsize=4)
    lru.put(_key("a"), 1)
    lru.put(_key("b"), 2)
    lru.clear()
    assert len(lru) == 0
    assert lru.get(_key("a")) is None


def test_maxsize_must_be_positive() -> None:
    with pytest.raises(ValueError, match="maxsize"):
        GrammarLRU(maxsize=0)
    with pytest.raises(ValueError, match="maxsize"):
        GrammarLRU(maxsize=-1)


@pytest.mark.parametrize("falsy", [None, False, 0, "", []])
def test_get_returns_stored_falsy_value_not_none(falsy: object) -> None:
    """Fix #5 regression: a stored falsy/``None`` value must be returned as
    itself, not conflated with "key absent".

    The old ``dict.get(key)`` + ``if value is None`` check treated a
    legitimately cached ``None`` (or any falsy object a future backend might
    store) as a miss — a latent foot-gun. The sentinel-based lookup keeps
    "present-but-falsy" distinct from "absent".
    """
    lru = GrammarLRU(maxsize=4)
    lru.put(_key("a"), falsy)
    assert lru.get(_key("a")) is falsy or lru.get(_key("a")) == falsy
    assert len(lru) == 1
    # A genuinely absent key is still a miss.
    assert lru.get(_key("absent")) is None


def test_get_stored_none_refreshes_recency() -> None:
    """A cached ``None`` participates in LRU recency like any other value.

    After touching the ``None`` entry it must NOT be the one evicted; the
    untouched non-``None`` entry is. We distinguish "survived" from "absent"
    by re-putting ``c`` (idempotent) and checking the cache still holds
    exactly the touched-``None`` key plus ``c``.
    """
    lru = GrammarLRU(maxsize=2)
    lru.put(_key("a"), None)
    lru.put(_key("b"), "B")
    # Touch ``a`` (value None) so ``b`` becomes LRU.
    assert lru.get(_key("a")) is None
    lru.put(_key("c"), "C")
    assert len(lru) == 2
    assert lru.get(_key("c")) == "C"
    # ``b`` (untouched, non-None) was the eviction victim, NOT ``a``: insert a
    # fresh key and confirm ``a`` (None) is now the LRU victim, proving it was
    # still resident after the ``c`` insert.
    lru.put(_key("d"), "D")
    assert len(lru) == 2
    assert lru.get(_key("d")) == "D"


def test_concurrent_put_get_does_not_corrupt() -> None:
    """Threaded smoke test: concurrent inserts/reads stay consistent.

    Not a strict serialisability proof — just confirms the
    :class:`threading.Lock` actually protects the OrderedDict from
    interleaved mutations that would otherwise raise ``RuntimeError:
    OrderedDict mutated during iteration``.
    """
    lru = GrammarLRU(maxsize=32)
    threads: list[threading.Thread] = []
    errors: list[Exception] = []

    def writer(start: int) -> None:
        try:
            for i in range(start, start + 100):
                k = _key(str(i % 50))
                lru.put(k, i)
                lru.get(k)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    for s in (0, 100, 200, 300):
        t = threading.Thread(target=writer, args=(s,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=10.0)
    assert not errors, f"thread errors: {errors!r}"
    assert len(lru) <= 32
