from urllib.parse import urlparse

import pytest

# sie_mcp.server pulls in FastMCP (`mcp`), which is intentionally outside the public
# dependency closure — skip these in environments (integration/docker) without it.
pytest.importorskip("mcp")

from sie_mcp.config import (
    DEFAULT_MAX_DOCUMENT_BYTES,
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_VLOCR_MODEL,
    MCPConfig,
)
from sie_mcp.server import _transport_security


def _cfg(**overrides: object) -> MCPConfig:
    base: dict[str, object] = {
        "sie_base_url": "http://localhost:8080",
        "sie_api_key": None,
        "connector_secrets": {},
        "allow_anonymous": False,
        "allowed_hosts": [],
        "max_document_bytes": DEFAULT_MAX_DOCUMENT_BYTES,
        "max_image_bytes": DEFAULT_MAX_IMAGE_BYTES,
        "docling_model": "docling",
        "generate_model": "Qwen/Qwen3.5-4B",
        "max_output_tokens": 4096,
        "vlocr_model": DEFAULT_VLOCR_MODEL,
        "encode_model": "BAAI/bge-m3",
        "rerank_model": "BAAI/bge-reranker-v2-m3",
        "caption_model": "microsoft/Florence-2-base-ft",
        "embed_model": "openai/clip-vit-base-patch32",
        "image_labels": [],
        "image_top_k": 5,
        "gpu": None,
        "timeout_s": 300.0,
        "qa_top_k": 5,
        "qa_rerank_candidates": 20,
        "qa_chunk_chars": 1200,
        "qa_chunk_overlap_chars": 200,
        "qa_max_tokens": 512,
        "qa_max_document_chars": 2_000_000,
        "qa_max_questions": 50,
        "qa_max_chunks": 2000,
        "oauth_enabled": True,
        "public_base_url": None,
        "oauth_redirect_uris": ("https://claude.ai/api/mcp/auth_callback",),
    }
    base.update(overrides)
    return MCPConfig(**base)  # type: ignore[arg-type]


def test_transport_security_disabled_without_allowed_hosts() -> None:
    # No allow-list: FastMCP's localhost-only default would 421 the remote /mcp endpoint,
    # so Host validation is off (the connector-secret auth is the access control).
    settings = _transport_security(_cfg(allowed_hosts=[]))
    assert settings.enable_dns_rebinding_protection is False


def test_transport_security_scopes_to_allowed_hosts() -> None:
    settings = _transport_security(_cfg(allowed_hosts=["mcp.example.com"]))
    assert settings.enable_dns_rebinding_protection is True
    assert settings.allowed_hosts == ["mcp.example.com"]
    # Compare parsed (scheme, host) pairs rather than substring-matching the origin
    # strings, so the assertion is exact (and not flagged as URL substring checks).
    parsed_origins = {
        (parsed.scheme, parsed.hostname) for parsed in (urlparse(origin) for origin in settings.allowed_origins)
    }
    assert ("https", "mcp.example.com") in parsed_origins
    assert ("http", "mcp.example.com") in parsed_origins
