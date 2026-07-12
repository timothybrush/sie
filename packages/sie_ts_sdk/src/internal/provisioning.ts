/**
 * Shared provisioning / retry loop for non-streaming POST endpoints.
 *
 * Both {@link SIEClient.generate} and {@link SIEClient.chatCompletions}
 * receive identical pre-execution capacity signals from the gateway —
 * `503` with a known error code (`PROVISIONING`, `MODEL_LOADING` or
 * `RESOURCE_EXHAUSTED`). They both need to retry those SAFE
 * pre-execution signals while honouring a caller-supplied
 * `waitForCapacity` flag plus a `provisionTimeout` budget. A `504` is
 * post-publish and therefore terminal here (non-idempotent generation).
 *
 * This helper centralises that loop. Callers supply a `performFetch`
 * callback that issues a fresh `fetch` per attempt (the request must be
 * re-buildable, which the JSON chat path satisfies trivially since the
 * body is a plain object). The loop returns the first successful
 * response or throws a typed error.
 *
 * The streaming path keeps its own inline copy because it needs
 * abortable sleeps composed with the caller's `AbortSignal` (see
 * `consumeSseStream` in `client.ts`).
 */

import {
  ModelLoadingError,
  ProvisioningError,
  RequestError,
  ResourceExhaustedError,
  ServerError,
} from "../errors.js";
import {
  DEFAULT_RETRY_DELAY,
  HTTP_GATEWAY_TIMEOUT,
  MODEL_LOADING_DEFAULT_DELAY,
  MODEL_LOADING_ERROR_CODE,
  PROVISIONING_ERROR_CODE,
  RESOURCE_EXHAUSTED_ERROR_CODE,
  RESOURCE_EXHAUSTED_MAX_RETRIES,
} from "./constants.js";
import { getErrorCode, getRetryAfter, handleError, throwIfModelLoadFailed } from "./parsing.js";
import { applyRetryJitter, computeOomBackoff } from "./retry.js";

/** Options controlling the provisioning retry loop. */
export interface ProvisioningOptions {
  /** Model name (used to populate `ModelLoadingError.model`). */
  model: string;
  /** GPU label passed through to `ProvisioningError`. May be `undefined`. */
  gpu: string | undefined;
  /**
   * When `true`, the loop retries `503 PROVISIONING` / `503 MODEL_LOADING`
   * until the provision budget is exhausted. When `false`, the first
   * provisioning signal throws (the call-site opted out of waiting).
   */
  waitForCapacity: boolean;
  /**
   * Total cumulative wall-clock budget (ms) for retries. Defaults to
   * `DEFAULT_PROVISION_TIMEOUT` if omitted.
   */
  provisionTimeoutMs: number;
}

/** Sleep for `ms` milliseconds. Non-abortable; the non-streaming surface
 * does not expose an AbortSignal to the caller. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Budget check + backoff for one `503 RESOURCE_EXHAUSTED` retry.
 *
 * Mirrors the Python SDK's `_handle_oom_retry`: throws
 * {@link ResourceExhaustedError} when the retry budget is exhausted
 * (`oomRetries >= maxOomRetries`), the provision budget has elapsed, or
 * the next backoff would consume the remaining budget without leaving
 * room for the retried request to run (surfacing the *root cause* now
 * instead of a later "provisioning timeout"). Otherwise returns the
 * delay (ms) to sleep before the next attempt.
 *
 * Distinct from MODEL_LOADING: the model is already resident, the
 * request just lost the race for compute resources. This is a SAFE
 * pre-execution signal (the worker rejected the request before running
 * it), so on the buffered paths it is retried regardless of
 * `waitForCapacity`, matching the Python SDK.
 *
 * @internal
 */
export function nextOomRetryDelay(opts: {
  retryAfter: number | undefined;
  oomRetries: number;
  maxOomRetries: number;
  elapsedMs: number;
  provisionTimeoutMs: number;
  model: string;
}): number {
  const { retryAfter, oomRetries, maxOomRetries, elapsedMs, provisionTimeoutMs, model } = opts;
  const message = `Server resource exhausted after ${oomRetries} retry attempt(s) for model '${model}'`;
  if (oomRetries >= maxOomRetries || elapsedMs >= provisionTimeoutMs) {
    throw new ResourceExhaustedError(message, { model, retries: oomRetries });
  }
  const delay = computeOomBackoff(retryAfter, oomRetries);
  if (delay >= provisionTimeoutMs - elapsedMs) {
    throw new ResourceExhaustedError(message, { model, retries: oomRetries });
  }
  return delay;
}

/**
 * Wrap a non-streaming POST attempt in the shared provisioning retry loop.
 *
 * The `performFetch` callback MUST re-issue the request from scratch on
 * each invocation — never reuse a consumed `Response`. It is responsible
 * for its own per-attempt timeout and for translating low-level
 * `TypeError` / `AbortError` into `SIEConnectionError`.
 *
 * The loop returns the first non-retryable success (`status === 200`).
 * Any other terminal status is handed to {@link handleError}, which
 * always throws.
 *
 * @internal
 */
export async function withProvisioningRetry(
  performFetch: () => Promise<Response>,
  opts: ProvisioningOptions,
): Promise<Response> {
  const startTime = Date.now();
  let oomRetries = 0;

  while (true) {
    const response = await performFetch();

    // 502 MODEL_LOAD_FAILED is terminal — surface immediately.
    await throwIfModelLoadFailed(response, opts.model);

    if (response.status === 503) {
      const errorCode = await getErrorCode(response.clone());
      if (errorCode === PROVISIONING_ERROR_CODE) {
        if (!opts.waitForCapacity) {
          throw new ProvisioningError(
            "No capacity available. Server is provisioning.",
            opts.gpu,
            getRetryAfter(response),
          );
        }
        const elapsed = Date.now() - startTime;
        if (elapsed >= opts.provisionTimeoutMs) {
          throw new ProvisioningError(
            `Provisioning timeout after ${elapsed}ms`,
            opts.gpu,
            getRetryAfter(response),
          );
        }
        const retryAfter = getRetryAfter(response);
        const delay = retryAfter ?? applyRetryJitter(DEFAULT_RETRY_DELAY);
        await sleep(Math.min(delay, opts.provisionTimeoutMs - elapsed));
        continue;
      }
      if (errorCode === MODEL_LOADING_ERROR_CODE) {
        const elapsed = Date.now() - startTime;
        if (elapsed >= opts.provisionTimeoutMs) {
          throw new ModelLoadingError(`Model loading timeout for '${opts.model}'`, opts.model);
        }
        const delay = getRetryAfter(response) ?? MODEL_LOADING_DEFAULT_DELAY;
        await sleep(Math.min(delay, opts.provisionTimeoutMs - elapsed));
        continue;
      }
      if (errorCode === RESOURCE_EXHAUSTED_ERROR_CODE) {
        // Retried regardless of `waitForCapacity` (bounded budget), matching
        // the Python SDK: the signal fires before any generation starts.
        const delay = nextOomRetryDelay({
          retryAfter: getRetryAfter(response),
          oomRetries,
          maxOomRetries: RESOURCE_EXHAUSTED_MAX_RETRIES,
          elapsedMs: Date.now() - startTime,
          provisionTimeoutMs: opts.provisionTimeoutMs,
          model: opts.model,
        });
        oomRetries += 1;
        await sleep(delay);
        continue;
      }
    }

    // Do NOT retry 504. A 504 GATEWAY_TIMEOUT is a *post-publish* timeout:
    // the work item is already on the queue and a worker may be — or have
    // finished — generating. Generation is non-idempotent and carries no
    // dedup key, so retrying could issue a SECOND billable generation.
    // The pre-execution 503 signals above remain retryable because those
    // fire before any generation can have started (Python SDK parity).
    if (response.status === HTTP_GATEWAY_TIMEOUT) {
      throw new ServerError(
        "Gateway timed out (504) after the request was published to the queue; " +
          "a worker may already be generating. Not retried because generation is " +
          "non-idempotent (retrying could double-bill). Re-issue manually if needed.",
        await getErrorCode(response.clone()),
        HTTP_GATEWAY_TIMEOUT,
      );
    }

    if (!response.ok) {
      await handleError(response);
    }

    // Defensive: handleError always throws on !ok, but if a future caller
    // adds a non-200 success status we still want to surface it cleanly.
    if (response.status !== 200) {
      throw new RequestError(`Unexpected response status ${response.status}`);
    }
    return response;
  }
}
