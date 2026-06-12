"""Smoke tests for sie_server package."""

import sie_server


def test_import() -> None:
    """Verify package can be imported and exposes a sensible __version__.

    We don't pin an exact version string here — the version is sourced from
    installed package metadata — but it must be non-empty so downstream code
    that logs or reports it doesn't print an empty string.
    """
    assert isinstance(sie_server.__version__, str)
    assert sie_server.__version__
