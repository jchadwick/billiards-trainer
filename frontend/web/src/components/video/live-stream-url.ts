/** Validate the runtime-configured, browser-reachable camera stream URL. */
export function getConfiguredLiveStreamUrl(
  configuredUrl: unknown,
): string | undefined {
  const candidate = typeof configuredUrl === 'string' ? configuredUrl.trim() : '';
  if (!candidate) return undefined;

  try {
    const url = new URL(candidate);
    if (
      (url.protocol !== 'http:' && url.protocol !== 'https:') ||
      url.username ||
      url.password ||
      url.hash
    ) {
      return undefined;
    }
    return url.href;
  } catch {
    return undefined;
  }
}
