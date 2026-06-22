import pytest
from sie_mcp.config import (
    _DEFAULT_IMAGE_TOP_K,
    DEFAULT_MAX_DOCUMENT_BYTES,
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_VLOCR_MODEL,
    MCPConfig,
)


def test_vlocr_model_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIE_MCP_VLOCR_MODEL", raising=False)
    assert MCPConfig.from_env().vlocr_model == DEFAULT_VLOCR_MODEL


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_vlocr_model_falls_back_to_default(monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    monkeypatch.setenv("SIE_MCP_VLOCR_MODEL", blank)
    assert MCPConfig.from_env().vlocr_model == DEFAULT_VLOCR_MODEL


def test_explicit_vlocr_model_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_MCP_VLOCR_MODEL", "some/other-model")
    assert MCPConfig.from_env().vlocr_model == "some/other-model"


def test_anonymous_closed_by_default_without_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    # A missing/misnamed secret env var must fail closed, not open anonymous access.
    monkeypatch.delenv("SIE_MCP_CONNECTOR_SECRETS", raising=False)
    monkeypatch.delenv("SIE_MCP_ALLOW_ANONYMOUS", raising=False)
    assert MCPConfig.from_env().allow_anonymous is False


def test_anonymous_is_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIE_MCP_CONNECTOR_SECRETS", raising=False)
    monkeypatch.setenv("SIE_MCP_ALLOW_ANONYMOUS", "1")
    assert MCPConfig.from_env().allow_anonymous is True


def test_allowed_hosts_parsed_as_trimmed_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_MCP_ALLOWED_HOSTS", " mcp.example.com , mcp.example.com:* ,")
    assert MCPConfig.from_env().allowed_hosts == ["mcp.example.com", "mcp.example.com:*"]


def test_allowed_hosts_default_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIE_MCP_ALLOWED_HOSTS", raising=False)
    assert MCPConfig.from_env().allowed_hosts == []


@pytest.mark.parametrize("bad", ["", "   ", "not-a-number", "0", "-5"])
def test_max_document_bytes_falls_back_on_bad_value(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("SIE_MCP_MAX_DOCUMENT_BYTES", bad)
    assert MCPConfig.from_env().max_document_bytes == DEFAULT_MAX_DOCUMENT_BYTES


def test_max_document_bytes_honors_explicit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_MCP_MAX_DOCUMENT_BYTES", "1024")
    assert MCPConfig.from_env().max_document_bytes == 1024


@pytest.mark.parametrize("bad", ["", "   ", "not-a-number", "0", "-5"])
def test_max_image_bytes_falls_back_on_bad_value(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("SIE_MCP_MAX_IMAGE_BYTES", bad)
    assert MCPConfig.from_env().max_image_bytes == DEFAULT_MAX_IMAGE_BYTES


def test_max_image_bytes_honors_explicit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_MCP_MAX_IMAGE_BYTES", "1024")
    assert MCPConfig.from_env().max_image_bytes == 1024


def test_image_top_k_zero_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    # 0 is a valid caption-only request, not a bad value — it must pass through.
    monkeypatch.setenv("SIE_MCP_IMAGE_TOP_K", "0")
    assert MCPConfig.from_env().image_top_k == 0


@pytest.mark.parametrize("bad", ["", "   ", "not-a-number", "-3"])
def test_image_top_k_falls_back_on_bad_value(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("SIE_MCP_IMAGE_TOP_K", bad)
    assert MCPConfig.from_env().image_top_k == _DEFAULT_IMAGE_TOP_K
