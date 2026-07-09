/**
 * Retry logic with exponential backoff and jitter
 */

import {
  DEFAULT_MAX_RETRY_DELAY,
  DEFAULT_RETRY_DELAY,
  RESOURCE_EXHAUSTED_DEFAULT_DELAY,
  RESOURCE_EXHAUSTED_MAX_DELAY,
} from "./constants.js";

const RETRY_JITTER_FRACTION = 0.25;

/**
 * Compute backoff with decorrelated jitter
 * @param attempt - The current attempt number (0-indexed)
 * @param baseDelay - Base delay in milliseconds
 * @param maxDelay - Maximum delay in milliseconds
 */
export function computeBackoffWithJitter(
  attempt: number,
  baseDelay: number = DEFAULT_RETRY_DELAY,
  maxDelay: number = DEFAULT_MAX_RETRY_DELAY,
): number {
  const exponentialDelay = baseDelay * 2 ** attempt;
  const cappedDelay = Math.min(exponentialDelay, maxDelay);
  // Decorrelated jitter: random value between 0 and cappedDelay
  return Math.random() * cappedDelay;
}

/**
 * Apply bounded downward jitter to a fixed retry delay.
 *
 * The jittered value is drawn from [delay * 0.75, delay], so it never
 * exceeds the caller's existing timeout cap. A zero Retry-After hint
 * remains an immediate retry.
 */
export function applyRetryJitter(delay: number): number {
  if (delay <= 0) return Math.max(delay, 0);
  const low = delay * (1 - RETRY_JITTER_FRACTION);
  return Math.max(0, low + Math.random() * (delay - low));
}

/**
 * Compute the next sleep interval (ms) for a `503 RESOURCE_EXHAUSTED` retry.
 *
 * Mirrors the Python SDK's `compute_oom_backoff`: a first-retry
 * `Retry-After` hint is honoured verbatim (capped at `maxDelay`, no
 * jitter — the server gave an explicit instruction); subsequent retries
 * use bounded exponential backoff, `max(baseDelay, retryAfter) * 2**attempt`
 * capped at `maxDelay`, with bounded downward jitter so a fleet of clients
 * evicted by the same OOM event does not retry in lockstep. Always returns
 * a value in `[0, maxDelay]`.
 *
 * @param retryAfter - Parsed `Retry-After` hint in milliseconds, if any
 * @param attempt - 0-indexed retry number (0 = first retry)
 */
export function computeOomBackoff(
  retryAfter: number | undefined,
  attempt: number,
  baseDelay: number = RESOURCE_EXHAUSTED_DEFAULT_DELAY,
  maxDelay: number = RESOURCE_EXHAUSTED_MAX_DELAY,
): number {
  // Defensive floor: a negative Retry-After (malformed upstream) must not
  // produce a negative sleep.
  const safeRetryAfter = retryAfter !== undefined ? Math.max(retryAfter, 0) : undefined;
  if (safeRetryAfter !== undefined && attempt === 0) {
    return Math.min(safeRetryAfter, maxDelay);
  }
  // `max(...)`: a zero hint falls back to `baseDelay`, and a hint above
  // `baseDelay` keeps the schedule non-decreasing (see the Python SDK's
  // `compute_oom_backoff` for the full rationale).
  const base = safeRetryAfter !== undefined ? Math.max(baseDelay, safeRetryAfter) : baseDelay;
  const capped = Math.max(0, Math.min(base * 2 ** attempt, maxDelay));
  return applyRetryJitter(capped);
}

/**
 * Parse Retry-After header value
 * @param header - The Retry-After header value
 * @returns Delay in milliseconds, or undefined if invalid
 */
export function getRetryAfter(header: string | null): number | undefined {
  if (!header) return undefined;

  // Try parsing as seconds (integer). `Retry-After: 0` means "retry
  // immediately" and must be honored (>= 0), not treated as invalid and
  // replaced by the default delay.
  const seconds = Number.parseInt(header, 10);
  if (!Number.isNaN(seconds) && seconds >= 0) {
    return seconds * 1000;
  }

  // Try parsing as HTTP-date
  const date = new Date(header);
  if (!Number.isNaN(date.getTime())) {
    const delay = date.getTime() - Date.now();
    return delay > 0 ? delay : undefined;
  }

  return undefined;
}
