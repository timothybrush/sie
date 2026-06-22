import io
import zipfile
from pathlib import Path

import pytest
from sie_mcp.skill_zip import (
    SkillPackagingError,
    build_skill_zip,
    parse_frontmatter,
    skill_name,
)

_SKILL_MD = Path(__file__).resolve().parents[1] / "plugin" / "SKILL.md"

_VALID = """---
name: superlinked-docs
description: >-
  Offload document work to the Superlinked inference cluster: convert documents
  to clean markdown instead of ingesting the file directly.
---

# Body
"""


def test_parses_folded_description() -> None:
    fields = parse_frontmatter(_VALID)
    assert fields["name"] == "superlinked-docs"
    assert fields["description"].startswith("Offload document work")
    assert "\n" not in fields["description"]


def test_zip_nests_skill_md_under_name_folder() -> None:
    data = build_skill_zip(_VALID)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
    assert names == ["superlinked-docs/SKILL.md"]
    # SKILL.md must NOT sit at the archive root — claude.ai rejects that layout.
    assert "SKILL.md" not in names


def test_zip_is_reproducible() -> None:
    assert build_skill_zip(_VALID) == build_skill_zip(_VALID)


def test_extra_files_are_nested_too() -> None:
    data = build_skill_zip(_VALID, extra={"reference.md": "x"})
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert set(archive.namelist()) == {"superlinked-docs/SKILL.md", "superlinked-docs/reference.md"}


def test_rejects_uppercase_name() -> None:
    bad = _VALID.replace("name: superlinked-docs", "name: Superlinked-Docs")
    with pytest.raises(SkillPackagingError):
        skill_name(bad)


def test_rejects_reserved_word_in_name() -> None:
    bad = _VALID.replace("name: superlinked-docs", "name: claude-docs")
    with pytest.raises(SkillPackagingError):
        skill_name(bad)


def test_rejects_missing_description() -> None:
    bad = "---\nname: superlinked-docs\n---\n\n# Body\n"
    with pytest.raises(SkillPackagingError):
        skill_name(bad)


def test_rejects_overlong_description() -> None:
    bad = _VALID.replace("to clean markdown instead of ingesting the file directly.", "x" * 1100)
    with pytest.raises(SkillPackagingError):
        skill_name(bad)


def test_requires_frontmatter() -> None:
    with pytest.raises(SkillPackagingError):
        parse_frontmatter("# no frontmatter here\n")


def test_strips_trailing_comment_on_inline_name() -> None:
    text = _VALID.replace("name: superlinked-docs", "name: superlinked-docs  # prod build")
    assert skill_name(text) == "superlinked-docs"


def test_preserves_quoted_inline_values() -> None:
    fields = parse_frontmatter('---\nname: "superlinked-docs"\ndescription: "hello there"\n---\n')
    assert fields["name"] == "superlinked-docs"
    assert fields["description"] == "hello there"


def test_shipped_claude_ai_skill_is_valid_and_packs() -> None:
    text = _SKILL_MD.read_text(encoding="utf-8")
    assert skill_name(text) == "superlinked-docs"
    data = build_skill_zip(text)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.namelist() == ["superlinked-docs/SKILL.md"]
