/** Validate the runtime-configured, browser-reachable camera stream URL. */
export function getConfiguredLiveStreamUrl(
  configuredUrl: unknown,
  apiOrigin?: string,
): string | undefined {
  const candidate = typeof configuredUrl === 'string' ? configuredUrl.trim() : '';
  if (!candidate) return undefined;

  try {
    const url = new URL(candidate);
    if (
      (url.protocol !== 'http:' && url.protocol !== 'https:') ||
      url.username ||
      url.password ||
      url.hash ||
      url.pathname.startsWith("/api/") ||
      (apiOrigin && url.origin === new URL(apiOrigin).origin)
    ) {
      return undefined;
    }
    return url.href;
  } catch {
    return undefined;
  }
}
