import { describe, expect, it } from 'vitest';
import { AxiosClient } from '../../api/axios-client';

describe('unauthenticated API transport', () => {
  it('does not expose token helpers or add credentials by default', () => {
    const client = new AxiosClient({
      baseURL: 'http://localhost:8000',
      wsBaseURL: 'ws://localhost:8000',
    });

    expect(client).not.toHaveProperty('setAuthToken');
    expect(client).not.toHaveProperty('createAuthenticatedWebSocket');
    expect(client.getInstance().defaults.withCredentials).toBe(false);
    expect(client.getInstance().defaults.headers.common).not.toHaveProperty(
      'Authorization'
    );
  });

  it('creates a websocket URL without appending credential query parameters', () => {
    const client = new AxiosClient({
      baseURL: 'http://localhost:8000',
      wsBaseURL: 'ws://localhost:8000',
    });

    expect(client.getWebSocketUrl('/api/v1/websocket/ws')).toBe(
      'ws://localhost:8000/api/v1/websocket/ws'
    );
  });
});
