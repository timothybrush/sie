"""Single source of truth for turning a request's raw options into the
effective options an adapter sees.

A model profile declares ``adapter_options.runtime`` defaults (e.g. an
embedding model's ``query_template`` / ``default_instruction`` / ``pooling`` /
``normalize``). Those defaults must be merged under the per-request options so
the adapter receives them at encode/score time.

There are two ingress paths and they MUST agree:

* Single-server HTTP (``api.options.resolve_runtime_options`` →
  ``api.encode``) — used by ``mise run serve`` and local SDK calls.
* Cluster queue worker (``queue_executor.process_encode_batch``) — the Rust
  gateway publishes only the raw SDK options to the queue, so the worker has to
  perform the same merge itself.

Historically only the HTTP path merged profile runtime defaults; the cluster
path forwarded the raw SDK options verbatim, silently dropping
``query_template`` / ``default_instruction`` / ``pooling`` / ``normalize`` for
every queued request (issue #1489). Routing both paths through this helper
keeps them in lockstep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sie_server.config.model import ModelConfig


def merge_runtime_options(
    config: ModelConfig,
    request_options: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the selected profile and overlay request options on its runtime defaults.

    The ``"profile"`` key in ``request_options`` selects the profile and is
    consumed (not forwarded). Request-supplied values win over profile runtime
    defaults so per-request overrides still work.

    Args:
        config: The model configuration (carries the profiles).
        request_options: Raw options from the request (may be ``None`` or carry
            a ``"profile"`` key).

    Returns:
        The merged options dict ready to hand to the adapter.

    Raises:
        ValueError: If ``request_options`` names a profile that does not exist
            (propagated from ``config.resolve_profile``).
    """
    profile_name = request_options.get("profile") if request_options else None
    resolved = config.resolve_profile(profile_name or "default")

    merged: dict[str, Any] = dict(resolved.runtime)
    if request_options:
        merged |= {k: v for k, v in request_options.items() if k != "profile"}
    return merged
