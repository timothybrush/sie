"""CLI entry point for the SIE MCP edge service."""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from sie_mcp.skill_zip import build_skill_zip, skill_name

app = typer.Typer(name="sie-mcp", help="SIE MCP edge service (Req 12).", no_args_is_help=True)

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SKILL_MD = _PACKAGE_ROOT / "plugin" / "SKILL.md"


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind to.")] = "0.0.0.0",  # noqa: S104 — intentional bind to all interfaces for the edge service
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on.")] = 8088,
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", envvar="SIE_LOG_LEVEL", help="Log level."),
    ] = "info",
    reload: Annotated[
        bool,
        typer.Option("--reload", "-r", help="Auto-reload for development."),
    ] = False,
) -> None:
    """Start the MCP edge over remote streamable-HTTP."""
    _setup_logging(log_level)
    uvicorn.run(
        "sie_mcp.app:build_app",
        host=host,
        port=port,
        factory=True,
        reload=reload,
        log_level=log_level.lower(),
    )


@app.command("skill-zip")
def skill_zip(
    skill: Annotated[
        Path,
        typer.Option(
            "--skill", "-s", help="Path to the SKILL.md to package (default: the surface-agnostic plugin/SKILL.md)."
        ),
    ] = _DEFAULT_SKILL_MD,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Output .zip path (default: dist/<name>-skill.zip)."),
    ] = None,
) -> None:
    """Package the surface-agnostic Superlinked Agent Skill as a claude.ai-uploadable ZIP (#1312)."""
    if not skill.is_file():
        raise typer.BadParameter(f"SKILL.md not found at {skill}")
    text = skill.read_text(encoding="utf-8")
    data = build_skill_zip(text)
    target = out or (_PACKAGE_ROOT / "dist" / f"{skill_name(text)}-skill.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    typer.echo(f"wrote {target} ({len(data)} bytes)")


@app.command()
def version() -> None:
    """Show version information."""
    try:
        typer.echo(f"sie-mcp {_pkg_version('sie-mcp')}")
    except PackageNotFoundError:
        typer.echo("sie-mcp (version unknown)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
