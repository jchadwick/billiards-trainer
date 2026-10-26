import { afterEach, describe, expect, it, vi } from 'vitest';
import { WebSocketClient } from '../../services/websocket-client';
import { RootStore } from '../../stores/RootStore';

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

describe('live WebSocket connectivity', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    MockWebSocket.instances = [];
  });

  it('shares an in-flight connection and restores the state subscription after reconnect', async () => {
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

  it('publishes only the newest accepted state and resets ordering after producer restart', async () => {
    vi.stubGlobal('WebSocket', MockWebSocket);
    const client = new WebSocketClient({
      url: 'ws://localhost:8000/api/v1/websocket/ws',
      heartbeatInterval: 60_000,
    });
    const store = new RootStore(client);

    try {
      expect(MockWebSocket.instances).toHaveLength(1);
      const socket = MockWebSocket.instances[0];
      socket.open();
      await new Promise(resolve => setTimeout(resolve, 0));
      expect(store.system.status.isConnected).toBe(true);
      expect(socket.sent.map(message => JSON.parse(message))).toEqual([
        expect.objectContaining({ type: 'subscribe', data: { streams: ['state'] } }),
      ]);

      const publish = (frameNumber: number, sessionId: string, sequence: number) => {
        socket.onmessage?.({
          data: JSON.stringify({
            type: 'state',
            timestamp: new Date().toISOString(),
            data: {
              balls: [],
              ball_count: 0,
              frame_number: frameNumber,
              observation_status: {
                observation_identity: {
                  engine_instance_id: 'engine',
                  session_id: sessionId,
                },
                transport_sequence: sequence,
              },
            },
          }),
        } as MessageEvent);
      };

      publish(12, 'session-a', 12);
      publish(11, 'session-a', 11);
      expect(store.videoStore.latestGameState?.frame_number).toBe(12);

      publish(1, 'session-b', 1);
      expect(store.videoStore.latestGameState?.frame_number).toBe(1);
    } finally {
      await store.shutdown();
    }
  });

  it('orders phase deliveries independently from raw scenes and preserves the newest value of each', async () => {
    vi.stubGlobal('WebSocket', MockWebSocket);
    const client = new WebSocketClient({
      url: 'ws://localhost:8000/api/v1/websocket/ws',
      heartbeatInterval: 60_000,
    });
    const store = new RootStore(client);

    try {
      const socket = MockWebSocket.instances[0];
      socket.open();
      await new Promise(resolve => setTimeout(resolve, 0));

      const publish = (
        frameNumber: number,
        transportSequence: number,
        deliverySequence: number,
      ) => {
        socket.onmessage?.({
          data: JSON.stringify({
            type: 'state',
            timestamp: new Date().toISOString(),
            data: {
              balls: [{ id: `frame-${frameNumber}` }],
              frame_number: frameNumber,
              observation_status: {
                observation_identity: {
                  engine_instance_id: 'engine',
                  session_id: 'session-a',
                },
                transport_sequence: transportSequence,
                perception: {
                  event_sequence: deliverySequence,
                  phase: 'aiming',
                  layout_trusted: true,
                  guidance_valid: true,
                },
              },
            },
          }),
        } as MessageEvent);
      };

      publish(10, 10, 4);
      publish(10, 10, 5); // phase-only delivery, same raw observation
      expect(store.videoStore.latestGameState?.frame_number).toBe(10);
      expect(store.videoStore.latestGameState?.observation_status?.perception?.event_sequence)
        .toBe(5);

      publish(9, 10, 4); // delayed phase event is rejected
      expect(store.videoStore.latestGameState?.observation_status?.perception?.event_sequence)
        .toBe(5);

      publish(11, 11, 4); // new raw scene with a stale embedded phase payload
      expect(store.videoStore.latestGameState?.frame_number).toBe(11);
      expect(store.videoStore.latestGameState?.balls[0].id).toBe('frame-11');
      expect(store.videoStore.latestGameState?.observation_status?.perception?.event_sequence)
        .toBe(5);

      publish(10, 10, 6); // newer phase event cannot roll raw state back
      expect(store.videoStore.latestGameState?.frame_number).toBe(11);
      expect(store.videoStore.latestGameState?.observation_status?.perception?.event_sequence)
        .toBe(6);
    } finally {
      await store.shutdown();
    }
  });
});
