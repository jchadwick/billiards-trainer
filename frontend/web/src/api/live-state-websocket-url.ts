const RETAIN_LAST_KNOWN_PARAMETER = 'retain_last_known_ms';

/**
 * Read the optional live-state retention window from the browser URL.
 * Invalid, zero, negative, fractional, and unsafe values all mean disabled.
 */
export function readRetainLastKnownMs(search: string): number | null {
  const rawValue = new URLSearchParams(search).get(RETAIN_LAST_KNOWN_PARAMETER);
  if (rawValue === null || !/^\d+$/.test(rawValue)) return null;

  const durationMs = Number(rawValue);
  return Number.isSafeInteger(durationMs) && durationMs > 0 ? durationMs : null;
}

/** Add the opt-in retention window to the state WebSocket URL. */
export function buildLiveStateWebSocketUrl(
  baseUrl: string,
  search: string,
): string {
  const durationMs = readRetainLastKnownMs(search);
  if (durationMs === null) return baseUrl;

  const url = new URL(baseUrl);
  url.searchParams.set(RETAIN_LAST_KNOWN_PARAMETER, String(durationMs));
  return url.toString();
}
