"""Run the contract-review agent over one contract and show the model fan-out."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from agents import InputGuardrailTripwireTriggered
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sie_sdk import SIEAsyncClient

from .app import ContractReview, build_investigator, build_reasoning_agent, build_synthesizer, run_review
from .config import load_config
from .data import CUAD_DIR, GENERATED_DIR, MANIFEST_PATH, make_sample
from .runtime import AppContext, Ledger, chat_once, configure_runtime, make_openai_client

console = Console()

# (config role, human job, SIE function) — drives the catalog table.
ROLE_INFO = [
    ("orchestrator", "Plan, call tools, assemble the review", "chat + tools"),
    ("triage", "Classify the document type (fast)", "chat"),
    ("vision", "Read the scanned signature page", "chat + image"),
    ("reasoning", "Clause-risk specialist (sub-agent)", "chat"),
    ("sql", "Text-to-SQL over the obligations DB", "chat"),
    ("guard", "Safety / prompt-injection guardrail", "chat"),
    ("ocr", "Scanned page → markdown", "extract"),
    ("embed", "Clause search (embeddings)", "encode"),
    ("rerank", "Rerank retrieved clauses", "score"),
    ("entities", "Entity extraction (parties, dates, $)", "extract"),
]


def _print_catalog(cfg: dict) -> None:
    table = Table(title="One SIE cluster · the right model for each job", title_style="bold")
    table.add_column("Role", style="cyan")
    table.add_column("SIE catalog model", style="green")
    table.add_column("SIE function", style="magenta")
    table.add_column("Job")
    for role, job, fn in ROLE_INFO:
        if role == "sql":  # the SQL tool's wire path follows sql.mode (chat | completions)
            fn = (cfg.get("sql") or {}).get("mode", "chat")
        table.add_row(role, cfg["models"][role], fn, job)
    console.print(table)


def _print_ledger(ledger: Ledger) -> None:
    table = Table(title="Per-model observability (in call order)", title_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Step")
    table.add_column("Model", style="green")
    table.add_column("SIE fn", style="magenta")
    table.add_column("Warm-up", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Sent", justify="right")
    table.add_column("Got", justify="right")
    table.add_column("Throughput", justify="right")
    warmup_total = 0.0
    call_total = 0.0
    for i, e in enumerate(ledger.entries, 1):
        warmup_total += e.warmup_s
        call_total += e.latency_s
        table.add_row(
            str(i), e.step, e.model.split("/")[-1], e.sie_fn,
            f"{e.warmup_s:.1f}s" if e.warmup_s else "—",
            f"{e.latency_s:.2f}s" if e.latency_s else "—",
            e.sent or "—", e.got or "—", e.throughput or "—",
        )
    console.print(table)
    used = {e.model for e in ledger.entries}
    console.print(f"[bold]{len(used)} distinct SIE models[/] handled this request — "
                  f"{warmup_total:.0f}s cold-start (model warm-up) + {call_total:.0f}s warm calls.")


def _print_summary(cfg: dict, usage, wall_s: float) -> None:
    parts = [f"end-to-end wall time [bold]{wall_s:.1f}s[/]"]
    reqs = getattr(usage, "requests", None)
    if reqs is not None:
        it = getattr(usage, "input_tokens", 0) or 0
        ot = getattr(usage, "output_tokens", 0) or 0
        orch = cfg["models"]["orchestrator"].split("/")[-1]
        parts.append(f"investigator {orch}: {reqs} LLM calls, {it:,} in / {ot:,} out tok")
    console.print("Run summary — " + " · ".join(parts))


def _print_review(review: ContractReview) -> None:
    lines = [
        f"[bold]Document type[/]: {review.document_type}",
        f"[bold]Parties[/]: {', '.join(review.parties) or '—'}",
        f"[bold]Effective date[/]: {review.effective_date or '—'}",
        f"[bold]Governing law[/]: {review.governing_law or '—'}",
        f"[bold]Executed[/]: {review.executed}",
        f"[bold]Renewal terms[/]: {review.renewal_terms}",
        "",
        "[bold]Key obligations[/]:",
        *[f"  • {o}" for o in review.key_obligations],
        "",
        "[bold]Recommendation[/]:",
        f"  {review.recommendation}",
    ]
    console.print(Panel("\n".join(lines), title="Contract review", border_style="blue"))

    if review.risk_flags:
        risks = Table(title="Risk flags", title_style="bold red")
        risks.add_column("Severity", style="bold")
        risks.add_column("Clause")
        risks.add_column("Issue")
        risks.add_column("Suggested redline")
        sev_color = {"high": "red", "medium": "yellow", "low": "green"}
        for f in review.risk_flags:
            color = sev_color.get(f.severity.lower(), "white")
            risks.add_row(f"[{color}]{f.severity}[/]", f.clause, f.issue, f.suggested_redline)
        console.print(risks)


def _resolve_corpus(args) -> tuple[str, str, str, str]:
    """Return (contract_text, scan_path, db_path, label)."""
    # Explicit file path wins.
    if args.contract and Path(args.contract).is_file():
        p = Path(args.contract)
        scan = args.scan or str(GENERATED_DIR / "acme-msa-signature.png")
        return p.read_text(), scan, str(GENERATED_DIR / "obligations.db"), p.name

    if MANIFEST_PATH.exists():  # real CUAD corpus
        manifest = json.loads(MANIFEST_PATH.read_text())
        slug = args.contract or manifest["primary"]
        text = (CUAD_DIR / f"{slug}.txt").read_text()
        scan = args.scan or str(GENERATED_DIR / manifest["scan_path"])
        return text, scan, str(GENERATED_DIR / manifest["db_path"]), f"CUAD · {slug}"

    # Offline fallback: synthetic corpus (generate it if missing).
    if not (GENERATED_DIR / "acme-msa.md").exists():
        console.print("[yellow]No corpus found — generating the synthetic one. "
                      "Run `uv run fetch-contracts` for real CUAD contracts.[/]")
        make_sample.main()
    name = args.contract or "acme-msa"
    text = (GENERATED_DIR / f"{name}.md").read_text()
    scan = args.scan or str(GENERATED_DIR / "acme-msa-signature.png")
    return text, scan, str(GENERATED_DIR / "obligations.db"), f"synthetic · {name}"


def _list_contracts() -> None:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        console.print(f"[bold]CUAD corpus[/] ({manifest['license']}):")
        for c in manifest["contracts"]:
            console.print(f"  {c['slug']}  [dim]{c['type']} · {c['char_len']:,} chars[/]")
    elif (GENERATED_DIR / "acme-msa.md").exists():
        console.print("[bold]Synthetic corpus[/]: acme-msa, mutual-nda, acme-sow")
    else:
        console.print("No corpus yet. Run `uv run fetch-contracts` or `uv run make-sample`.")


async def _warm(app: AppContext) -> None:
    """Provision the generative models before the run.

    The orchestrator and reasoning sub-agent call models through the Agents SDK,
    which has no cold-start retry — so on a scale-from-zero cluster the first
    call would fail. Touch each model once (our helpers retry while it loads).
    """
    m = app.cfg["models"]
    # Only the orchestrator and reasoning sub-agent run through the Agents SDK,
    # which has no cold-start retry — so only these must be warm before the run.
    # Triage, vision, guard, SQL, and the encode/score/extract tools all retry
    # while their model provisions, so they load lazily on first use.
    for model in dict.fromkeys([m["orchestrator"], m["reasoning"]]):
        with console.status(f"Warming {model} (first call provisions it on a cold cluster)..."):
            try:
                await chat_once(app, model, [{"role": "user", "content": "ok"}], max_tokens=1)
            except Exception as exc:  # warm-up is best-effort; the run will retry
                console.print(f"[yellow]warm-up: {model} not ready ({type(exc).__name__}); will retry during the run.[/]")
    console.print("[green]Warm-up done.[/]\n")


async def _run(args) -> None:
    cfg = load_config()
    text, scan_path, db_path, label = _resolve_corpus(args)

    # Tool calls use our own provisioning-retry, so their client shouldn't also retry
    # (max_retries=0). Agents-SDK calls can't be wrapped, so that client retries hard
    # to survive a model being evicted/reloaded mid-run on a busy cluster.
    tool_client = make_openai_client(cfg["cluster"]["url"], cfg["cluster"]["api_key"], max_retries=0)
    agent_client = make_openai_client(cfg["cluster"]["url"], cfg["cluster"]["api_key"], max_retries=12, timeout_s=180)
    configure_runtime(agent_client)

    _print_catalog(cfg)
    console.print(f"Reviewing [bold]{label}[/] against SIE at "
                  f"[bold]{cfg['cluster']['url']}[/]\n")

    async with SIEAsyncClient(cfg["cluster"]["url"], api_key=cfg["cluster"]["api_key"] or None) as sie:
        ledger = Ledger()
        app = AppContext(
            sie=sie,
            oai=tool_client,
            cfg=cfg,
            ledger=ledger,
            contract_text=text,
            scan_path=scan_path,
            db_path=db_path,
            reasoning_agent=build_reasoning_agent(cfg, agent_client),
        )
        investigator = build_investigator(cfg, agent_client)
        synthesizer = build_synthesizer(cfg, agent_client)
        if not args.no_warm:
            await _warm(app)
        t0 = time.monotonic()
        try:
            gather, result = await run_review(app, investigator, synthesizer, args.instruction)
        except InputGuardrailTripwireTriggered:
            console.print(Panel("Request blocked by the granite-guardian safety guardrail.",
                                border_style="red", title="Guardrail tripped"))
            _print_ledger(ledger)
            await agent_client.close()
            await tool_client.close()
            return
        except Exception as exc:  # a model the SDK calls (investigator/sub-agent) was unreachable
            console.print(Panel(f"{type(exc).__name__}: {exc}", border_style="red",
                                title="Run failed (model unavailable)"))
            _print_ledger(ledger)
            await agent_client.close()
            await tool_client.close()
            return
        wall = time.monotonic() - t0
        await agent_client.close()
        await tool_client.close()

        try:
            review = result.final_output_as(ContractReview)
        except Exception:
            review = result.final_output

        console.print()
        if isinstance(review, ContractReview):
            _print_review(review)
        else:
            console.print(Panel(str(review), title="Agent output (unstructured)"))
        console.print()
        _print_ledger(ledger)
        usage = getattr(getattr(gather, "context_wrapper", None), "usage", None)
        _print_summary(cfg, usage, wall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a contract with a multi-model SIE agent.")
    parser.add_argument("--contract", default=None,
                        help="contract slug (CUAD), synthetic name, or a path to a .txt/.md file")
    parser.add_argument("--scan", default=None, help="path to a signature-page image (png/jpg)")
    parser.add_argument(
        "--instruction",
        default="Review this contract. Identify the parties and key terms, flag the "
        "biggest risks to the Customer with severity and redlines, confirm it is "
        "executed, and surface upcoming obligations and deadlines.",
        help="what to ask the agent to do",
    )
    parser.add_argument("--list", action="store_true", help="list available contracts and exit")
    parser.add_argument("--no-warm", action="store_true",
                        help="skip pre-warming models (faster when the cluster is already warm)")
    args = parser.parse_args()

    if args.list:
        _list_contracts()
        return
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
