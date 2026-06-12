/**
 * Shared provisioning / retry loop for non-streaming POST endpoints.
 *
 * Both {@link SIEClient.generate} and {@link SIEClient.chatCompletions}
 * receive identical pre-execution capacity signals from the gateway —
 * `202 Accepted` (provisioning) and `503` with a known error code
 * (`MODEL_LOADING`) or a generic 503 (scale-from-zero). They both need
 * to retry those SAFE pre-execution signals while honouring a caller-
 * supplied `waitForCapacity` flag plus a `provisionTimeout` budget.
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

import { ModelLoadingError, ProvisioningError, RequestError } from "../errors.js";
import {
  DEFAULT_RETRY_DELAY,
  HTTP_ACCEPTED,
  MODEL_LOADING_DEFAULT_DELAY,
  MODEL_LOADING_ERROR_CODE,
} from "./constants.js";
import { getErrorCode, getRetryAfter, handleError, throwIfModelLoadFailed } from "./parsing.js";

/** Options controlling the provisioning retry loop. */
export interface ProvisioningOptions {
  /** Model name (used to populate `ModelLoadingError.model`). */
  model: string;
  /** GPU label passed through to `ProvisioningError`. May be `undefined`. */
  gpu: string | undefined;
  /**
   * When `true`, the loop retries 202 / `503 MODEL_LOADING` / generic 503
   * until the provision budget is exhausted. When `false`, the first such
   * signal throws (the call-site opted out of waiting).
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

  while (true) {
    const response = await performFetch();

    if (response.status === HTTP_ACCEPTED) {
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
      const delay = getRetryAfter(response) ?? DEFAULT_RETRY_DELAY;
      await sleep(Math.min(delay, opts.provisionTimeoutMs - elapsed));
      continue;
    }

    // 502 MODEL_LOAD_FAILED is terminal — surface immediately.
    await throwIfModelLoadFailed(response, opts.model);

    if (response.status === 503) {
      const errorCode = await getErrorCode(response.clone());
      if (errorCode === MODEL_LOADING_ERROR_CODE) {
        const elapsed = Date.now() - startTime;
        if (elapsed >= opts.provisionTimeoutMs) {
          throw new ModelLoadingError(`Model loading timeout for '${opts.model}'`, opts.model);
        }
        const delay = getRetryAfter(response) ?? MODEL_LOADING_DEFAULT_DELAY;
        await sleep(Math.min(delay, opts.provisionTimeoutMs - elapsed));
        continue;
      }
      if (opts.waitForCapacity) {
        const elapsed = Date.now() - startTime;
        if (elapsed < opts.provisionTimeoutMs) {
          const delay = getRetryAfter(response) ?? DEFAULT_RETRY_DELAY;
          await sleep(Math.min(delay, opts.provisionTimeoutMs - elapsed));
          continue;
        }
      }
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
