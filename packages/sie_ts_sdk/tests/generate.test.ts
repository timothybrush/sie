/**
 * SIEClient.generate() tests.
 *
 * Verify that:
 * - The request body is JSON (not msgpack).
 * - The aggregated response envelope parses into a ``GenerateResult``.
 * - The SDK surfaces SIE-native timing metadata (ttftMs, tpotMs, attemptId).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { RequestError, SIEConnectionError } from "../src/errors.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("SIEClient.generate", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("parses the streaming envelope into a GenerateResult", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "Qwen__Qwen3-4B-Instruct-2507",
        text: "Hello world!",
        finish_reason: "stop",
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
        attempt_id: "att-abc",
        ttft_ms: 120.5,
        tpot_ms: 45.2,
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    const result = await client.generate("Qwen__Qwen3-4B-Instruct-2507", "Hi", {
      maxNewTokens: 32,
    });

    expect(result.model).toBe("Qwen__Qwen3-4B-Instruct-2507");
    expect(result.text).toBe("Hello world!");
    expect(result.finishReason).toBe("stop");
    expect(result.usage.promptTokens).toBe(5);
    expect(result.usage.completionTokens).toBe(3);
    expect(result.usage.totalTokens).toBe(8);
    expect(result.attemptId).toBe("att-abc");
    expect(result.ttftMs).toBe(120.5);
    expect(result.tpotMs).toBe(45.2);
  });

  it("sends a JSON body with snake_case field names", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "m",
        text: "x",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        attempt_id: "a",
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await client.generate("m", "Hi", {
      maxNewTokens: 8,
      temperature: 0.7,
      topP: 0.9,
      stop: ["</s>"],
    });

    expect(mockFetch).toHaveBeenCalledOnce();
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8080/v1/generate/m");
    expect(init.method).toBe("POST");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(init.headers.Accept).toBe("application/json");
    const body = JSON.parse(init.body);
    expect(body).toEqual({
      prompt: "Hi",
      max_new_tokens: 8,
      temperature: 0.7,
      top_p: 0.9,
      stop: ["</s>"],
    });
  });

  it("normalizes HF-style model ids to SIE-safe route ids", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "Qwen__Qwen3-4B-Instruct-2507",
        text: "x",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        attempt_id: "a",
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await client.generate("Qwen/Qwen3-4B-Instruct-2507", "Hi", { maxNewTokens: 8 });

    const [url] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8080/v1/generate/Qwen__Qwen3-4B-Instruct-2507");
  });

  it("throws RequestError on non-object response", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response(JSON.stringify("not an object"), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await expect(client.generate("m", "hi", { maxNewTokens: 4 })).rejects.toBeInstanceOf(
      RequestError,
    );
  });

  // H4 regression: a truncated / malformed envelope must NOT silently
  // produce an empty completion. Missing or non-string `model` / `text`
  // raises (matches the Python SDK contract).
  it("throws RequestError when the envelope is missing model", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        text: "hello",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await expect(client.generate("m", "hi", { maxNewTokens: 4 })).rejects.toThrow(
      /missing string 'model'/,
    );
  });

  it("throws RequestError when the envelope is missing text", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "m",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await expect(client.generate("m", "hi", { maxNewTokens: 4 })).rejects.toThrow(
      /missing string 'text'/,
    );
  });

  it("throws RequestError when model/text are present but not strings", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "m",
        text: 123,
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await expect(client.generate("m", "hi", { maxNewTokens: 4 })).rejects.toBeInstanceOf(
      RequestError,
    );
  });

  it("forwards gpu and pool routing headers", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        model: "m",
        text: "x",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        attempt_id: "a",
      }),
    );

    const client = new SIEClient("http://localhost:8080");
    await client.generate("m", "hi", { maxNewTokens: 8, gpu: "eval-bench/l4" });

    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers["X-SIE-Pool"]).toBe("eval-bench");
    expect(init.headers["X-SIE-MACHINE-PROFILE"]).toBe("l4");
  });
});

/**
 * B1c regression: generate() is non-idempotent (no dedup key), so a
 * `fetch` `TypeError` — which can be raised for a connection dropped
 * AFTER the request body was sent (mid-flight) — must NOT be retried.
 * Retrying would issue a SECOND billable generation. The safe
 * pre-execution capacity signals (202 provisioning, 503 MODEL_LOADING)
 * are detected from the HTTP status and ARE still retried.
 */
describe("SIEClient.generate retry semantics (B1c)", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("does NOT retry a mid-flight TypeError even when waitForCapacity is true", async () => {
    const client = new SIEClient("http://localhost:8080", {
      timeout: 30_000,
      provisionTimeout: 60_000,
    });

    // `fetch` throws TypeError for any network failure; a second attempt
    // could double-bill, so generate() must surface it without retrying.
    mockFetch.mockRejectedValue(new TypeError("fetch failed"));

    await expect(
      client.generate("m", "hi", { maxNewTokens: 8, waitForCapacity: true }),
    ).rejects.toBeInstanceOf(SIEConnectionError);

    // Crucially: exactly ONE call, no retry.
    expect(mockFetch).toHaveBeenCalledOnce();
  });

  it("does NOT retry a mid-flight TypeError when waitForCapacity is false", async () => {
    const client = new SIEClient("http://localhost:8080", { timeout: 1000 });

    mockFetch.mockRejectedValue(new TypeError("fetch failed"));

    await expect(
      client.generate("m", "hi", { maxNewTokens: 8, waitForCapacity: false }),
    ).rejects.toBeInstanceOf(SIEConnectionError);
    expect(mockFetch).toHaveBeenCalledOnce();
  });

  it("still retries the safe 202 provisioning status path under waitForCapacity", async () => {
    vi.useFakeTimers();
    const client = new SIEClient("http://localhost:8080", {
      timeout: 30_000,
      provisionTimeout: 60_000,
    });

    mockFetch.mockResolvedValueOnce(new Response(null, { status: 202 })).mockResolvedValueOnce(
      jsonResponse({
        model: "m",
        text: "ok",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        attempt_id: "a",
      }),
    );

    const promise = client.generate("m", "hi", { maxNewTokens: 8, waitForCapacity: true });
    // DEFAULT_RETRY_DELAY = 5_000ms.
    await vi.advanceTimersByTimeAsync(5_000);
    const result = await promise;

    expect(result.text).toBe("ok");
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("still retries the safe 503 MODEL_LOADING status path under waitForCapacity", async () => {
    vi.useFakeTimers();
    const client = new SIEClient("http://localhost:8080", {
      timeout: 30_000,
      provisionTimeout: 60_000,
    });

    const modelLoading = new Response(
      JSON.stringify({ code: "MODEL_LOADING", message: "loading" }),
      {
        status: 503,
        headers: { "Content-Type": "application/json" },
      },
    );

    mockFetch.mockResolvedValueOnce(modelLoading).mockResolvedValueOnce(
      jsonResponse({
        model: "m",
        text: "loaded",
        finish_reason: "stop",
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        attempt_id: "a",
      }),
    );

    const promise = client.generate("m", "hi", { maxNewTokens: 8, waitForCapacity: true });
    // MODEL_LOADING_DEFAULT_DELAY = 5_000ms.
    await vi.advanceTimersByTimeAsync(5_000);
    const result = await promise;

    expect(result.text).toBe("loaded");
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });
});
