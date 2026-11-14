/**
 * Client display freshness follows the backend's configured state TTL. The
 * fallback mirrors the validated backend default for offline/config failure.
 */
export const DEFAULT_MAX_STATE_OBSERVATION_AGE_SECONDS = 0.5;

export function positiveFreshnessLimit(
  value: unknown,
  fallback = DEFAULT_MAX_STATE_OBSERVATION_AGE_SECONDS,
): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
    ? value
    : fallback;
}

/** Add browser-monotonic time since the last unique source observation. */
export function currentObservationAgeSeconds(
  reportedAgeSeconds: unknown,
  ageAtLastUniqueObservationSeconds: number | null,
  receivedAtMonotonicMs: number,
  nowMonotonicMs: number,
): number | null {
  if (
    typeof reportedAgeSeconds !== 'number'
    || !Number.isFinite(reportedAgeSeconds)
    || reportedAgeSeconds < 0
  ) {
    return null;
  }

  const ageAtReceipt = ageAtLastUniqueObservationSeconds ?? reportedAgeSeconds;
  const elapsedSeconds = Math.max(0, nowMonotonicMs - receivedAtMonotonicMs) / 1000;
  // A repeated publication may carry a newer producer-side age. Never let it
  // make the displayed state appear younger than the age already observed.
  return Math.max(reportedAgeSeconds, ageAtReceipt + elapsedSeconds);
}

/** Advance one independently identified evidence age without rejuvenating repeats. */
export function advanceEvidenceAgeMs(
  reportedAgeMs: unknown,
  ageAtLastUniqueEvidenceMs: number | null,
  receivedAtMonotonicMs: number,
  nowMonotonicMs: number,
): number | null {
  if (
    typeof reportedAgeMs !== "number"
    || !Number.isFinite(reportedAgeMs)
    || reportedAgeMs < 0
  ) return null;
  const ageAtReceipt = ageAtLastUniqueEvidenceMs ?? reportedAgeMs;
  const elapsedMs = Math.max(0, nowMonotonicMs - receivedAtMonotonicMs);
  return Math.max(reportedAgeMs, ageAtReceipt + elapsedMs);
}

export function observationIsStale(
  ageSeconds: number | null,
  maximumAgeSeconds: number,
  status?: {
    freshness_status?: string;
    ball_positions_status?: string;
    ball_detection_status?: string;
  } | null,
): boolean {
  if (ageSeconds !== null && ageSeconds >= maximumAgeSeconds) return true;
  return status?.freshness_status === 'source_silent'
    || status?.ball_positions_status === 'cached'
    || status?.ball_positions_status === 'unavailable'
    || status?.ball_detection_status === 'failed';
}
