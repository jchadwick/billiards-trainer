import { describe, expect, it } from 'vitest';
import { buildLiveStateWebSocketUrl, readRetainLastKnownMs } from '../../api/live-state-websocket-url';

describe('live state WebSocket retention option', () => {
  const endpoint = 'ws://localhost:8000/api/v1/websocket/ws';

  it('propagates the positive retention duration from the live page URL', () => {
    expect(readRetainLastKnownMs('?retain_last_known_ms=500')).toBe(500);
    expect(buildLiveStateWebSocketUrl(endpoint, '?retain_last_known_ms=500')).toBe(
      `${endpoint}?retain_last_known_ms=500`,
    );
  });

  it.each([
    '',
    '?retain_last_known_ms=0',
    '?retain_last_known_ms=-1',
    '?retain_last_known_ms=1.5',
    '?retain_last_known_ms=not-a-number',
    '?retain_last_known_ms=9007199254740992',
  ])('leaves retention disabled for %s', (search) => {
    expect(readRetainLastKnownMs(search)).toBeNull();
    expect(buildLiveStateWebSocketUrl(endpoint, search)).toBe(endpoint);
  });

  it('preserves unrelated WebSocket query settings', () => {
    expect(buildLiveStateWebSocketUrl(`${endpoint}?stream=state`, '?retain_last_known_ms=250'))
      .toBe(`${endpoint}?stream=state&retain_last_known_ms=250`);
  });
});
