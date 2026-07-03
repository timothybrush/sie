"""Wire the OpenAI Agents SDK to a SIE cluster.

The one idea that makes this whole example work: the Agents SDK speaks the
OpenAI wire protocol, and SIE serves an OpenAI-compatible ``/v1`` endpoint. So
we hand every agent an ``AsyncOpenAI`` client whose ``base_url`` points at SIE,
force the *chat completions* API (SIE doesn't implement the newer Responses
API), and disable tracing (so nothing is shipped to api.openai.com). After that,
each agent just names the SIE catalog model it should run on.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

from agents import (
    OpenAIChatCompletionsModel,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from sie_sdk import SIEAsyncClient

T = TypeVar("T")


def make_openai_client(
    cluster_url: str, api_key: str, *, max_retries: int = 2, timeout_s: float | None = None
) -> AsyncOpenAI:
    """An OpenAI client pointed at SIE's OpenAI-compatible endpoint.

    ``max_retries`` matters because Agents-SDK-driven calls can't be wrapped in our
    own provisioning retry — a client with generous retries survives a model being
    evicted and reloaded mid-run on a busy cluster.
    """
    base_url = cluster_url.rstrip("/") + "/v1"
    kwargs: dict[str, Any] = {"base_url": base_url, "api_key": api_key or "not-needed", "max_retries": max_retries}
    if timeout_s is not None:
        kwargs["timeout"] = timeout_s
    # SIE ignores the key locally; a managed cluster reads it as a Bearer token.
    return AsyncOpenAI(**kwargs)


def configure_runtime(client: AsyncOpenAI) -> None:
    """Point the whole Agents SDK at SIE instead of api.openai.com."""
    set_default_openai_client(client)  # every agent talks to SIE...
    set_default_openai_api("chat_completions")  # ...over chat completions, not Responses...
    set_tracing_disabled(True)  # ...and we never phone home with traces.


def model_for(model_id: str, client: AsyncOpenAI) -> OpenAIChatCompletionsModel:
    """Bind one SIE catalog model id to an Agents-SDK model an Agent can use."""
    return OpenAIChatCompletionsModel(model=model_id, openai_client=client)


@dataclass
class GenResult:
    """One generation's text plus the metrics we log for observability.

    ``provision_s`` is time spent waiting for the model to be ready — the cold
    start, measured as the failed 503/504 retries before the request that
    actually ran. ``gen_s`` is the duration of that successful (warm) call, so
    throughput is measured warm instead of being blended with the cold start.
    """

    text: str
    provision_s: float
    gen_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def latency_s(self) -> float:
        return self.provision_s + self.gen_s

    @property
    def tokens_per_s(self) -> float | None:
        if self.completion_tokens and self.gen_s > 0:
            return self.completion_tokens / self.gen_s
        return None


@dataclass
class LedgerEntry:
    step: str
    model: str
    sie_fn: str
    warmup_s: float = 0.0  # cold-start / provisioning wait (0 if warm or N/A)
    latency_s: float = 0.0  # the call itself (warm), excluding warm-up
    sent: str = ""
    got: str = ""
    throughput: str = ""


@dataclass
class Ledger:
    """Per-call observability for one agent run.

    Every tool, guardrail, and sub-agent records the model it used plus latency,
    how much data it sent, and throughput — so a normal run prints not just
    *which* models the cluster fanned the request across, but how each performed.
    """

    entries: list[LedgerEntry] = field(default_factory=list)

    def record(
        self,
        step: str,
        model: str,
        sie_fn: str,
        *,
        warmup_s: float = 0.0,
        latency_s: float = 0.0,
        sent: str = "",
        got: str = "",
        throughput: str = "",
    ) -> None:
        self.entries.append(LedgerEntry(step, model, sie_fn, warmup_s, latency_s, sent, got, throughput))


@dataclass
class AppContext:
    """Shared dependencies handed to every tool via ``RunContextWrapper``."""

    sie: SIEAsyncClient
    oai: AsyncOpenAI
    cfg: dict[str, Any]
    ledger: Ledger
    contract_text: str  # the contract body we have on file (template/clauses)
    scan_path: str  # the executed signature page, delivered as a scan image
    db_path: str  # the SQLite obligations database the SQL tool queries
    reasoning_agent: Any = None  # the risk-analyst sub-agent (set during build)
    # Cache the clause embeddings in a shared mutable dict, not a reassigned
    # attribute: the Agents SDK hands each tool call a shallow copy of the context,
    # so mutating a shared object persists but reassigning `app.x = ...` does not.
    clause_cache: dict[str, Any] = field(default_factory=dict)

    @property
    def provision_timeout_s(self) -> float:
        return float(self.cfg["cluster"].get("provision_timeout_s", 900))


async def with_provisioning_retry(
    make_call: Callable[[], Awaitable[T]], deadline: float
) -> tuple[T, float, float]:
    """Retry an OpenAI-client call while SIE scales a model from zero.

    A cold cluster answers 503/504/202 until the model is resident; we retry until
    ``deadline`` (monotonic seconds) before giving up. Returns
    ``(result, provision_s, call_s)``: ``provision_s`` is the time spent in failed
    retries before the attempt that succeeded (the cold start), ``call_s`` is the
    duration of that successful call — so callers can report warm-up and warm
    throughput separately.
    """
    start = time.monotonic()
    while True:
        attempt_start = time.monotonic()
        try:
            result = await make_call()
            return result, attempt_start - start, time.monotonic() - attempt_start
        except APIConnectionError:
            if time.monotonic() < deadline:
                await asyncio.sleep(5)
                continue
            raise
        except APIStatusError as exc:
            # 202 accepted (warming), 502/503 unavailable, 504 first_chunk_timeout —
            # all transient while SIE scales a model from zero.
            if exc.status_code in (202, 502, 503, 504) and time.monotonic() < deadline:
                await asyncio.sleep(5)
                continue
            raise


async def chat_once(
    app: AppContext,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout_s: float | None = None,
    **extra: Any,
) -> GenResult:
    """One OpenAI-compatible chat completion against SIE, with cold-start retry.

    ``timeout_s`` overrides how long to keep retrying a provisioning model (the
    guardrail uses a short budget so it fails open fast rather than blocking).
    """
    deadline = time.monotonic() + (timeout_s if timeout_s is not None else app.provision_timeout_s)
    resp, provision_s, gen_s = await with_provisioning_retry(
        lambda: app.oai.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, **extra
        ),
        deadline,
    )
    usage = resp.usage
    return GenResult(
        text=resp.choices[0].message.content or "",
        provision_s=provision_s,
        gen_s=gen_s,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
    )


async def complete_once(
    app: AppContext,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    stop: list[str] | None = None,
) -> GenResult:
    """One OpenAI-compatible *text completion* against SIE (for completion-only
    models like sqlcoder that expect a raw prompt, not a chat transcript)."""
    deadline = time.monotonic() + app.provision_timeout_s
    resp, provision_s, gen_s = await with_provisioning_retry(
        lambda: app.oai.completions.create(
            model=model, prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop
        ),
        deadline,
    )
    usage = resp.usage
    return GenResult(
        text=resp.choices[0].text or "",
        provision_s=provision_s,
        gen_s=gen_s,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
    )
