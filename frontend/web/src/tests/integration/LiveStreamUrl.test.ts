import { describe, expect, it } from 'vitest';
import { getConfiguredLiveStreamUrl } from '../../components/video/live-stream-url';

describe('direct browser camera stream URL', () => {
  it('accepts browser-reachable HTTP streams and normalizes their URL', () => {
    expect(getConfiguredLiveStreamUrl(' http://camera.local:8001/stream.mjpg '))
      .toBe('http://camera.local:8001/stream.mjpg');
    expect(getConfiguredLiveStreamUrl('https://camera.local/live'))
      .toBe('https://camera.local/live');
  });

  it('rejects unsafe or malformed URLs so the API never proxies video', () => {
    expect(getConfiguredLiveStreamUrl('https://user:secret@camera.local/live')).toBeUndefined();
    expect(getConfiguredLiveStreamUrl('http://camera.local/live#frame')).toBeUndefined();
    expect(getConfiguredLiveStreamUrl('javascript:alert(1)')).toBeUndefined();
    expect(getConfiguredLiveStreamUrl('')).toBeUndefined();
  });
});
