import io
import zipfile
from pathlib import Path

import pytest
from sie_mcp.plugin_pack import build_plugin_pack, normalize_mcp_url

_SKILL_MD = """---
name: superlinked-docs
description: >-
  Offload document work to the Superlinked inference cluster: convert documents
  to clean markdown instead of ingesting the file directly.
---

# Body
"""

_PARSE_DOCUMENT_SKILL_MD = """---
name: parse-document
description: Parse documents through the Superlinked MCP edge.
---

# Parse
"""

_SUMMARIZE_DOCUMENT_SKILL_MD = """---
name: summarize-document
description: Summarize documents through the Superlinked MCP edge.
---

# Summarize
"""

_EXTRACT_ENTITIES_SKILL_MD = """---
name: extract-entities
description: Extract entities through the Superlinked MCP edge.
---

# Extract
"""

_REDACT_PII_SKILL_MD = """---
name: redact-pii
description: Redact PII through the Superlinked MCP edge.
---

# Redact
"""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://mcp.sie-test.example", "https://mcp.sie-test.example/mcp"),
        ("https://mcp.sie-test.example/", "https://mcp.sie-test.example/mcp"),
        ("https://mcp.sie-test.example/mcp/", "https://mcp.sie-test.example/mcp"),
        ("https://mcp.sie-test.example/api/mcp", "https://mcp.sie-test.example/api/mcp"),
    ],
)
def test_normalize_mcp_url(raw: str, expected: str) -> None:
    assert normalize_mcp_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "mcp.sie-test.example/mcp",
        "ftp://mcp.sie-test.example/mcp",
        "https://mcp.sie-test.example/not-mcp",
        "https://mcp.sie-test.example/mcp?secret=nope",
        "https://mcp.sie-test.example/mcp#frag",
    ],
)
def test_normalize_mcp_url_rejects_bad_values(raw: str) -> None:
    with pytest.raises(ValueError, match="MCP URL"):
        normalize_mcp_url(raw)


def test_build_plugin_pack_writes_install_artifacts(tmp_path: Path) -> None:
    connector_secret = "demo-connector-secret"  # noqa: S105 - inert test fixture
    report = build_plugin_pack(
        _SKILL_MD,
        mcp_url="https://mcp.sie-test.example",
        connector_secret=connector_secret,
        cluster_label="sie-test",
        claude_code_skill_mds=[
            _EXTRACT_ENTITIES_SKILL_MD,
            _PARSE_DOCUMENT_SKILL_MD,
            _REDACT_PII_SKILL_MD,
            _SUMMARIZE_DOCUMENT_SKILL_MD,
        ],
        cowork_guide_md="# Cowork install\n",
        out_dir=tmp_path,
    )

    assert report.mcp_url == "https://mcp.sie-test.example/mcp"
    assert report.skill_name == "superlinked-docs"
    assert report.skill_zip == tmp_path / "superlinked-docs-skill.zip"
    assert report.claude_code_skill == tmp_path / "claude-code" / "superlinked-docs" / "SKILL.md"
    assert report.claude_code_skills == (
        tmp_path / "claude-code" / "superlinked-docs" / "SKILL.md",
        tmp_path / "claude-code" / "extract-entities" / "SKILL.md",
        tmp_path / "claude-code" / "parse-document" / "SKILL.md",
        tmp_path / "claude-code" / "redact-pii" / "SKILL.md",
        tmp_path / "claude-code" / "summarize-document" / "SKILL.md",
    )
    assert report.cowork_guide == tmp_path / "cowork" / "superlinked.md"
    assert report.install_guide == tmp_path / "INSTALL.md"

    assert report.claude_code_skill.read_text(encoding="utf-8") == _SKILL_MD
    parse_skill = tmp_path / "claude-code" / "parse-document" / "SKILL.md"
    assert parse_skill.read_text(encoding="utf-8") == _PARSE_DOCUMENT_SKILL_MD
    assert report.cowork_guide is not None
    assert report.cowork_guide.read_text(encoding="utf-8") == "# Cowork install\n"

    guide = report.install_guide.read_text(encoding="utf-8")
    assert "https://mcp.sie-test.example/mcp" in guide
    assert "demo-connector-secret" in guide
    assert "claude mcp add --scope user --transport http superlinked-docs" in guide
    assert "cp -R claude-code/* ~/.claude/skills/" in guide
    assert "parse-document" in guide
    assert "summarize-document" in guide
    assert "extract-entities" in guide
    assert "redact-pii" in guide
    assert "PR #1336" in guide
    assert "superlinked-docs-skill.zip" in guide


def test_build_plugin_pack_keeps_connector_secret_out_of_skill_zip(tmp_path: Path) -> None:
    connector_secret = "secret-that-must-not-enter-the-skill"  # noqa: S105 - inert test fixture
    report = build_plugin_pack(
        _SKILL_MD,
        mcp_url="https://mcp.sie-test.example/mcp",
        connector_secret=connector_secret,
        out_dir=tmp_path,
    )

    with zipfile.ZipFile(io.BytesIO(report.skill_zip.read_bytes())) as archive:
        names = archive.namelist()
        skill_text = archive.read("superlinked-docs/SKILL.md").decode()

    assert names == ["superlinked-docs/SKILL.md"]
    assert "secret-that-must-not-enter-the-skill" not in skill_text


def test_build_plugin_pack_uses_placeholder_when_secret_omitted(tmp_path: Path) -> None:
    report = build_plugin_pack(_SKILL_MD, mcp_url="https://mcp.sie-test.example/mcp", out_dir=tmp_path)

    guide = report.install_guide.read_text(encoding="utf-8")
    assert "Authorization: Bearer <connector-secret>" in guide


def test_build_plugin_pack_refreshes_claude_code_skills(tmp_path: Path) -> None:
    stale = tmp_path / "claude-code" / "stale" / "SKILL.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")

    build_plugin_pack(_SKILL_MD, mcp_url="https://mcp.sie-test.example/mcp", out_dir=tmp_path)

    assert not stale.exists()
    assert (tmp_path / "claude-code" / "superlinked-docs" / "SKILL.md").exists()
