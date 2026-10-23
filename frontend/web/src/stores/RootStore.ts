import { makeAutoObservable, runInAction } from 'mobx';
import axiosClient from '../api/axios-client';
import { createWebSocketClient, type WebSocketClient } from '../services/websocket-client';
import { isGameStateMessage, type GameStateData, type WebSocketMessage } from '../types/api';

/**
 * The browser runtime only owns the live table surface: one state subscription,
 * its connection status, and the latest accepted source state. Camera pixels
 * remain a direct browser MJPEG request owned by LiveView.
 */
export class RootStore {
  readonly system = { status: { isConnected: false } };
  readonly videoStore: { latestGameState: GameStateData | null } = {
    latestGameState: null,
  };

  private readonly websocketClient: WebSocketClient;
  private latestProducerEpoch: string | null = null;
  private latestTransportSequence: number | null = null;
  private latestSnapshotRevision: number | null = null;
  private latestSourceFrameNumber: number | null = null;

  constructor(websocketClient?: WebSocketClient) {
    this.websocketClient = websocketClient ?? createWebSocketClient({
      url: axiosClient.getWebSocketUrl('/api/v1/websocket/ws'),
      autoReconnect: true,
    });

    makeAutoObservable(this, {
      websocketClient: false,
      latestProducerEpoch: false,
      latestTransportSequence: false,
      latestSnapshotRevision: false,
      latestSourceFrameNumber: false,
    } as never);

    this.websocketClient.onConnectionState((state) => {
      runInAction(() => {
        this.system.status.isConnected = state === 'connected';
      });
    });
    this.websocketClient.on('state', this.handleStateMessage);
    this.websocketClient.subscribe(['state']);

    // Connection failure is visible through `system.status.isConnected`; the
    // client owns reconnect timing and replays this bounded subscription.
    void this.websocketClient.connect().catch((error: unknown) => {
      console.error('Live state WebSocket connection failed:', error);
    });
  }

  private handleStateMessage = (message: WebSocketMessage): void => {
    if (!isGameStateMessage(message)) return;
    if (!this.acceptObservation(message.data)) return;
    this.videoStore.latestGameState = message.data;
  };

  /** Reject delayed state within an explicitly identified producer session. */
  private acceptObservation(state: GameStateData): boolean {
    const status = state.observation_status;
    const identity = status?.observation_identity;
    const engineId = identity?.engine_instance_id
      ?? status?.engine_instance_id
      ?? status?.vision_instance_id;
    const reportedSessionId: unknown = identity?.session_id
      ?? status?.source_session_id
      ?? status?.source_process_id;
    if (
      typeof engineId !== 'string'
      || (typeof reportedSessionId !== 'string' && typeof reportedSessionId !== 'number')
    ) return true;

    const producerEpoch = `${engineId}\u0000${reportedSessionId}`;
    if (producerEpoch !== this.latestProducerEpoch) {
      this.latestProducerEpoch = producerEpoch;
      this.latestTransportSequence = null;
      this.latestSnapshotRevision = null;
      this.latestSourceFrameNumber = null;
    }

    const transportSequence = status?.transport_sequence;
    if (typeof transportSequence === 'number' && Number.isSafeInteger(transportSequence)) {
      if (
        this.latestTransportSequence !== null
        && transportSequence <= this.latestTransportSequence
      ) return false;
      this.latestTransportSequence = transportSequence as number;
      return true;
    }

    const snapshotRevision = identity?.snapshot_revision ?? status?.snapshot_revision;
    if (typeof snapshotRevision === 'number' && Number.isSafeInteger(snapshotRevision)) {
      if (
        this.latestSnapshotRevision !== null
        && snapshotRevision <= this.latestSnapshotRevision
      ) return false;
      this.latestSnapshotRevision = snapshotRevision as number;
      return true;
    }

    const sourceFrameNumber = status?.source_frame_number ?? state.frame_number;
    if (typeof sourceFrameNumber === 'number' && Number.isSafeInteger(sourceFrameNumber)) {
      if (
        this.latestSourceFrameNumber !== null
        && sourceFrameNumber < this.latestSourceFrameNumber
      ) return false;
      this.latestSourceFrameNumber = sourceFrameNumber as number;
    }
    return true;
  }

  async shutdown(): Promise<void> {
    this.websocketClient.destroy();
  }
}

let rootStoreInstance: RootStore | null = null;

export function getRootStore(): RootStore {
  rootStoreInstance ??= new RootStore();
  return rootStoreInstance;
}
