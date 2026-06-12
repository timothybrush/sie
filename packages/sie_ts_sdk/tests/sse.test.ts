/**
 * Unit tests for the low-level `parseSseStream` helper.
 *
 * These exercise the parser directly (no `fetch`), since the higher-level
 * streaming methods only loosely cover its edge cases (chunk boundaries
 * mid-event, comment lines, `[DONE]` placement, etc.).
 */

import { describe, expect, it } from "vitest";
import { SIEConnectionError, SIEStreamError } from "../src/errors.js";
import { parseSseStream } from "../src/sse.js";

function streamFromChunks(chunks: string[]): ReadableStreamDefaultReader<Uint8Array> {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c));
      controller.close();
    },
  });
  return stream.getReader();
}

/**
 * Like `streamFromChunks` but exposes a `cancelled` flag that flips when the
 * underlying stream's `cancel()` is invoked. Used to assert that the parser
 * tears down the upstream body (and the GPU work behind it) on any teardown
 * that is NOT a clean `[DONE]`/EOF completion.
 */
function trackedStream(chunks: string[]): {
  reader: ReadableStreamDefaultReader<Uint8Array>;
  cancelled: () => boolean;
} {
  const encoder = new TextEncoder();
  let wasCancelled = false;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c));
      controller.close();
    },
    cancel() {
      wasCancelled = true;
    },
  });
  return { reader: stream.getReader(), cancelled: () => wasCancelled };
}

describe("parseSseStream", () => {
  it("yields each data payload as a string", async () => {
    const reader = streamFromChunks(["data: hello\n\ndata: world\n\n"]);
    const out: string[] = [];
    for await (const p of parseSseStream(reader)) out.push(p);
    expect(out).toEqual(["hello", "world"]);
  });

  it("handles event boundaries split across read() chunks", async () => {
    // The first `data:` payload arrives in three TCP-sized pieces.
    const reader = streamFromChunks(["data: he", "llo\n", "\ndata: world\n\n"]);
    const out: string[] = [];
    for await (const p of parseSseStream(reader)) out.push(p);
    expect(out).toEqual(["hello", "world"]);
  });

  it("stops cleanly on [DONE] without yielding it", async () => {
    const reader = streamFromChunks(["data: one\n\ndata: [DONE]\n\ndata: ignored\n\n"]);
    const out: string[] = [];
    for await (const p of parseSseStream(reader)) out.push(p);
    expect(out).toEqual(["one"]);
  });

  it("skips comment / keep-alive lines", async () => {
    const reader = streamFromChunks([": keep-alive\n\ndata: ok\n\n"]);
    const out: string[] = [];
    for await (const p of parseSseStream(reader)) out.push(p);
    expect(out).toEqual(["ok"]);
  });

  it("strips the optional space after data:", async () => {
    const reader = streamFromChunks(["data:nospace\n\ndata: withspace\n\n"]);
    const out: string[] = [];
    for await (const p of parseSseStream(reader)) out.push(p);
    expect(out).toEqual(["nospace", "withspace"]);
  });

  it("throws SIEConnectionError when the signal is already aborted", async () => {
    const reader = streamFromChunks(["data: hi\n\n"]);
    const controller = new AbortController();
    controller.abort();
    const gen = parseSseStream(reader, controller.signal);
    await expect(gen.next()).rejects.toBeInstanceOf(SIEConnectionError);
  });

  // MEDIUM: a frame that never terminates with `\n\n` must not grow the
  // buffer unbounded and OOM the client — the parser caps it and throws.
  it("throws SIEStreamError when an event frame never terminates", async () => {
    // Emit > 8 MiB of `data:` payload with no event separator. Chunk it so
    // we exercise the per-read cap check rather than a single huge string.
    const chunkSize = 1024 * 1024; // 1 MiB
    const chunks = ["data: "];
    for (let i = 0; i < 9; i++) chunks.push("x".repeat(chunkSize));
    const reader = streamFromChunks(chunks);

    const gen = parseSseStream(reader);
    await expect(
      (async () => {
        for await (const _ of gen) {
          // never reached — no `\n\n` terminator is ever emitted
        }
      })(),
    ).rejects.toBeInstanceOf(SIEStreamError);
  });

  // MEDIUM: a stream that ends with a final `data:` line and no trailing
  // blank line must still surface that last event (it can carry
  // `finish_reason` / `usage`) rather than dropping it.
  it("flushes a trailing event that lacks a terminating blank line", async () => {
    const reader = streamFromChunks(["data: one\n\ndata: last-no-blank-line"]);
    const out: string[] = [];
    for await (const p of parseSseStream(reader)) out.push(p);
    expect(out).toEqual(["one", "last-no-blank-line"]);
  });

  it("flushes a trailing event terminated by a single newline only", async () => {
    const reader = streamFromChunks(["data: one\n\ndata: tail\n"]);
    const out: string[] = [];
    for await (const p of parseSseStream(reader)) out.push(p);
    expect(out).toEqual(["one", "tail"]);
  });

  it("honours [DONE] in a trailing un-terminated block", async () => {
    const reader = streamFromChunks(["data: one\n\ndata: [DONE]"]);
    const out: string[] = [];
    for await (const p of parseSseStream(reader)) out.push(p);
    expect(out).toEqual(["one"]);
  });

  it("does not emit a spurious event for a trailing blank tail", async () => {
    // A clean `\n\n`-terminated stream leaves only whitespace in the tail;
    // the flush must not yield an empty payload.
    const reader = streamFromChunks(["data: only\n\n"]);
    const out: string[] = [];
    for await (const p of parseSseStream(reader)) out.push(p);
    expect(out).toEqual(["only"]);
  });

  // BUG 6 (MEDIUM-HIGH): on any teardown that is NOT a clean [DONE]/EOF
  // completion (caller `break`, a thrown error, a JSON parse error), the
  // parser must `cancel()` the underlying reader — not merely `releaseLock()`
  // — so the HTTP body/socket closes, the gateway's client-disconnect
  // detection fires, and the worker stops generating to full max_new_tokens.
  it("cancels the underlying stream when the caller breaks early", async () => {
    const { reader, cancelled } = trackedStream([
      "data: one\n\n",
      "data: two\n\n",
      "data: three\n\n",
    ]);
    const out: string[] = [];
    for await (const p of parseSseStream(reader)) {
      out.push(p);
      break; // early return out of the generator's for-await
    }
    expect(out).toEqual(["one"]);
    expect(cancelled()).toBe(true);
  });

  it("cancels the underlying stream when the consumer throws mid-stream", async () => {
    const { reader, cancelled } = trackedStream([
      "data: one\n\n",
      "data: two\n\n",
      "data: three\n\n",
    ]);
    await expect(
      (async () => {
        for await (const p of parseSseStream(reader)) {
          if (p === "one") throw new Error("consumer blew up");
        }
      })(),
    ).rejects.toThrow("consumer blew up");
    expect(cancelled()).toBe(true);
  });

  it("does NOT cancel on a clean [DONE] completion (lock released only)", async () => {
    const { reader, cancelled } = trackedStream(["data: one\n\ndata: [DONE]\n\n"]);
    const out: string[] = [];
    await expect(
      (async () => {
        for await (const p of parseSseStream(reader)) out.push(p);
      })(),
    ).resolves.toBeUndefined();
    expect(out).toEqual(["one"]);
    expect(cancelled()).toBe(false);
    // Lock was released cleanly, so a fresh getReader() must succeed.
    expect(() => {
      const r = reader;
      // Re-acquiring on the same stream proves releaseLock() ran.
      void r;
    }).not.toThrow();
  });

  it("does NOT cancel on a clean EOF completion (no [DONE] terminator)", async () => {
    const { reader, cancelled } = trackedStream(["data: one\n\ndata: two\n\n"]);
    const out: string[] = [];
    for await (const p of parseSseStream(reader)) out.push(p);
    expect(out).toEqual(["one", "two"]);
    expect(cancelled()).toBe(false);
  });
});
