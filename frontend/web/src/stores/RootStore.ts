import { makeAutoObservable, runInAction } from 'mobx';
import axiosClient from '../api/axios-client';
import { buildLiveStateWebSocketUrl } from '../api/live-state-websocket-url';
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
  private latestDeliverySequence: number | null = null;
  private latestTransportSequence: number | null = null;
  private latestSnapshotRevision: number | null = null;
  private latestSourceFrameNumber: number | null = null;

  constructor(websocketClient?: WebSocketClient) {
    this.websocketClient = websocketClient ?? createWebSocketClient({
      url: buildLiveStateWebSocketUrl(
        axiosClient.getWebSocketUrl('/api/v1/websocket/ws'),
        typeof window === 'undefined' ? '' : window.location.search,
      ),
      autoReconnect: true,
    });

    makeAutoObservable(this, {
      websocketClient: false,
      latestProducerEpoch: false,
      latestDeliverySequence: false,
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
    const acceptedState = this.acceptObservation(message.data);
    if (!acceptedState) return;
    this.videoStore.latestGameState = acceptedState;
  };

  /**
   * Track phase-delivery and raw-scene sequence spaces independently. A phase
   * event may repeat the current scene transport sequence, while a later raw
   * scene may carry the last phase event again; neither stream may block or
   * roll back the other.
   */
  private acceptObservation(state: GameStateData): GameStateData | null {
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
    ) return state;

    const producerEpoch = `${engineId}\u0000${reportedSessionId}`;
    const producerEpochChanged = producerEpoch !== this.latestProducerEpoch;
    if (producerEpochChanged) {
      this.latestProducerEpoch = producerEpoch;
      this.latestDeliverySequence = null;
      this.latestTransportSequence = null;
      this.latestSnapshotRevision = null;
      this.latestSourceFrameNumber = null;
    }

    // Phase-only V2 events have a delivery sequence independent of the raw
    // scene transport sequence. Accept them even when ball measurements are
    // unchanged, while still rejecting a delayed/coalesced older event.
    const deliverySequence = status?.perception?.event_sequence;
    const hasDeliverySequence = typeof deliverySequence === 'number'
      && Number.isSafeInteger(deliverySequence);
    let deliveryIsNew = false;
    if (hasDeliverySequence) {
      deliveryIsNew = this.latestDeliverySequence === null
        || deliverySequence > this.latestDeliverySequence;
      if (deliveryIsNew) this.latestDeliverySequence = deliverySequence;
    }

    let observationIsNew = false;
    const transportSequence = status?.transport_sequence;
    if (typeof transportSequence === 'number' && Number.isSafeInteger(transportSequence)) {
      observationIsNew = this.latestTransportSequence === null
        || transportSequence > this.latestTransportSequence;
      if (observationIsNew) this.latestTransportSequence = transportSequence;
    } else {
      const snapshotRevision = identity?.snapshot_revision ?? status?.snapshot_revision;
      if (typeof snapshotRevision === 'number' && Number.isSafeInteger(snapshotRevision)) {
        observationIsNew = this.latestSnapshotRevision === null
          || snapshotRevision > this.latestSnapshotRevision;
        if (observationIsNew) this.latestSnapshotRevision = snapshotRevision;
      } else {
        const sourceFrameNumber = status?.source_frame_number ?? state.frame_number;
        if (typeof sourceFrameNumber === 'number' && Number.isSafeInteger(sourceFrameNumber)) {
          observationIsNew = this.latestSourceFrameNumber === null
            || sourceFrameNumber >= this.latestSourceFrameNumber;
          if (observationIsNew) this.latestSourceFrameNumber = sourceFrameNumber;
        } else {
          // Preserve the legacy acceptance behavior when the producer does
          // not supply any ordering field.
          observationIsNew = !hasDeliverySequence;
        }
      }
    }

    if (!deliveryIsNew && !observationIsNew) return null;

    const currentState = producerEpochChanged ? null : this.videoStore.latestGameState;
    if (deliveryIsNew && !observationIsNew && currentState) {
      // A phase-only event is self-contained for perception status but must
      // not replace the currently accepted ball layout with an older copy.
      return {
        ...currentState,
        observation_status: {
          ...currentState.observation_status,
          perception: state.observation_status?.perception,
        },
      };
    }

    if (observationIsNew && !deliveryIsNew && currentState?.observation_status?.perception) {
      // Raw scenes can be delivered after a newer coalesced phase event while
      // still carrying an older perception payload. Keep the newer evidence.
      return {
        ...state,
        observation_status: {
          ...status,
          perception: currentState.observation_status.perception,
        },
      };
    }
    return state;
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
