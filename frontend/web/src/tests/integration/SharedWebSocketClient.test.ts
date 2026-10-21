import { afterEach, describe, expect, it, vi } from 'vitest';
import { WebSocketClient } from '../../services/websocket-client';
import { SystemStore } from '../../stores/SystemStore';
import { VideoStore } from '../../stores/VideoStore';
import { apiClient } from '../../api/client';

class MockWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: MockWebSocket[] = [];

  readyState = MockWebSocket.CONNECTING;
  sent: string[] = [];
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(readonly url: string) {
    MockWebSocket.instances.push(this);
  }

  send(message: string): void {
    this.sent.push(message);
  }

  open(): void {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.(new Event('open'));
  }

  close(code = 1000, reason = ''): void {
    this.readyState = MockWebSocket.CLOSED;
    queueMicrotask(() => {
      this.onclose?.({ code, reason, wasClean: code === 1000 } as CloseEvent);
    });
  }
}

describe('shared WebSocket client subscriptions', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    MockWebSocket.instances = [];
  });

  it('shares an in-flight connection and restores desired streams exactly once after reconnect', async () => {
    vi.stubGlobal('WebSocket', MockWebSocket);
    const client = new WebSocketClient({
      url: 'ws://localhost:8000/api/v1/websocket/ws',
      reconnectDelay: 1,
      maxReconnectDelay: 1,
      heartbeatInterval: 60_000,
    });

    client.subscribe(['state']);
    const connection = client.connect();
    expect(client.connect()).toBe(connection);
    expect(MockWebSocket.instances).toHaveLength(1);

    const firstSocket = MockWebSocket.instances[0];
    firstSocket.open();
    await connection;
    expect(firstSocket.sent.map(message => JSON.parse(message))).toEqual([
      expect.objectContaining({ type: 'subscribe', data: { streams: ['state'] } }),
    ]);

    // Repeating the same intent does not create duplicate subscription traffic.
    client.subscribe(['state']);
    expect(firstSocket.sent).toHaveLength(1);

    firstSocket.close(1006, 'network interruption');
    await new Promise(resolve => setTimeout(resolve, 10));
    expect(MockWebSocket.instances).toHaveLength(2);

    const secondSocket = MockWebSocket.instances[1];
    secondSocket.open();
    expect(secondSocket.sent.map(message => JSON.parse(message))).toEqual([
      expect.objectContaining({ type: 'subscribe', data: { streams: ['state'] } }),
    ]);

    client.destroy();
  });

  it('delivers live state to SystemStore and VideoStore through one injected socket', async () => {
    vi.stubGlobal('WebSocket', MockWebSocket);
    vi.spyOn(apiClient, 'stopVideoCapture').mockResolvedValue({
      success: true,
      timestamp: new Date().toISOString(),
    });
    const url = 'ws://localhost:8000/api/v1/websocket/ws';
    const client = new WebSocketClient({ url, heartbeatInterval: 60_000 });
    const systemStore = new SystemStore(client);
    const videoStore = new VideoStore(client);

    try {
      const connecting = systemStore.connect(url);
      expect(MockWebSocket.instances).toHaveLength(1);
      const socket = MockWebSocket.instances[0];
      socket.open();
      expect((await connecting).success).toBe(true);
      expect(systemStore.status.isConnected).toBe(true);
      expect(videoStore.isRealtimeConnected).toBe(true);

      const state = {
        balls: [],
        ball_count: 0,
        frame_number: 7,
        game_type: '8ball',
      };
      socket.onmessage?.({
        data: JSON.stringify({ type: 'state', timestamp: new Date().toISOString(), data: state }),
      } as MessageEvent);

      expect(videoStore.latestGameState).toEqual(state);
      expect(MockWebSocket.instances).toHaveLength(1);
      expect(socket.sent.map(message => JSON.parse(message))).toEqual([
        expect.objectContaining({ type: 'subscribe', data: { streams: ['state'] } }),
      ]);
    } finally {
      videoStore.dispose();
      systemStore.destroy();
      client.destroy();
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  });

  it('keeps a standalone VideoStore self-contained and subscribed to state', async () => {
    vi.stubGlobal('WebSocket', MockWebSocket);
    const timestamp = new Date().toISOString();
    vi.spyOn(apiClient, 'getStreamStatus').mockResolvedValue({
      success: true,
      data: { camera: { connected: true } },
      timestamp,
    });
    vi.spyOn(apiClient, 'startVideoCapture').mockResolvedValue({ success: true, timestamp });
    vi.spyOn(apiClient, 'getCalibrationData').mockResolvedValue({ success: false, timestamp });
    vi.spyOn(apiClient, 'stopVideoCapture').mockResolvedValue({ success: true, timestamp });

    const videoStore = new VideoStore();
    try {
      const connecting = videoStore.connect();
      for (let attempt = 0; attempt < 10 && MockWebSocket.instances.length === 0; attempt++) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      expect(MockWebSocket.instances).toHaveLength(1);
      const socket = MockWebSocket.instances[0];
      socket.open();
      await connecting;

      const subscriptions = socket.sent.map(message => JSON.parse(message));
      expect(subscriptions).toEqual([
        expect.objectContaining({
          type: 'subscribe',
          data: { streams: ['state', 'frame', 'trajectory', 'metrics', 'alert'], filters: { quality: 'medium', frame_rate: 30 } },
        }),
      ]);
      expect(videoStore.isRealtimeConnected).toBe(true);
    } finally {
      videoStore.dispose();
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  });
});
