import { makeAutoObservable, observable, reaction, runInAction } from "mobx";
import {
  eventHistoryLimit,
  redactEvidence,
  type BrowserEvent,
} from "../components/events/event-model";
import {
  BrowserStateChangeObserver,
  type BrowserEventDraft,
} from "../components/events/state-change-observer";
import axiosClient from "../api/axios-client";
import { buildLiveStateWebSocketUrl } from "../api/live-state-websocket-url";
import {
  createWebSocketClient,
  type WebSocketClient,
} from "../services/websocket-client";
import {
  isGameStateMessage,
  type GameStateData,
  type WebSocketMessage,
} from "../types/api";
import type { DisplayMjpegFrame } from "../components/video/use-frame-aware-mjpeg";
import {
  BrowserSessionMetrics,
  extractObservationAges,
  getBrowserSessionMetricsPolicy,
  type BrowserSessionSample,
  type SourceFrameIdentity,
} from "../services/browser-session-metrics";
import {
  BrowserDiagnosticsWebSocket,
  isBrowserDiagnosticsSnapshot,
  type BrowserDiagnosticsSnapshot,
  type DiagnosticsConnectionState,
  type DiagnosticsTransport,
} from "../services/browser-diagnostics";

export type VideoDeliveryState =
  | "unconfigured"
  | "connecting"
  | "receiving"
  | "loaded"
  | "stalled"
  | "error"
  | "stopped";

export type VideoDeliveryMode = "identified" | "native" | null;

export interface VideoDeliveryStatus {
  state: VideoDeliveryState;
  mode: VideoDeliveryMode;
  configured: boolean;
  configurationSynchronized: boolean;
  /** Increments when the configured stream source is replaced. */
  sourceGeneration: number;
  /** Configuration that selected this source; survives route changes. */
  sourceConfigurationRevision?: number;
  startedAt: string | null;
  lastFrameReceiptAt: string | null;
  lastFrameIdentity: Omit<DisplayMjpegFrame, "imageUrl"> | null;
  /** Dimensions of the most recently displayed/loaded frame, if known. */
  dimensions: { width: number; height: number } | null;
  error: string | null;
}

interface AcceptedObservation {
  state: GameStateData;
  /** Phase-only deliveries refresh state but are not new source frames. */
  isNewSourceObservation: boolean;
  producerEpoch: string | null;
  producerEpochChanged: boolean;
}

export interface BrowserMetricsClock {
  monotonicNowMs(): number;
  wallTime(): Date;
}

export interface BrowserDiagnosticsStoreState {
  connection: DiagnosticsConnectionState;
  snapshot: BrowserDiagnosticsSnapshot | null;
  receivedAt: string | null;
  isStale: boolean;
  protocolError: string | null;
}

const SYSTEM_BROWSER_CLOCK: BrowserMetricsClock = {
  monotonicNowMs: () => typeof performance !== "undefined" ? performance.now() : Date.now(),
  wallTime: () => new Date(),
};

function stateSourceFrameIdentity(state: GameStateData): SourceFrameIdentity | null {
  const status = state.observation_status;
  const identity = status?.observation_identity;
  const engineInstanceId =
    identity?.engine_instance_id
    ?? status?.engine_instance_id
    ?? status?.vision_instance_id;
  const sessionId =
    identity?.session_id
    ?? status?.source_session_id
    ?? status?.source_process_id;
  const frameId =
    identity?.frame_id
    ?? status?.source_frame_id
    ?? status?.source_frame_number
    ?? state.frame_number;
  if (
    typeof engineInstanceId !== "string"
    || (typeof sessionId !== "string" && typeof sessionId !== "number")
    || (typeof frameId !== "string" && typeof frameId !== "number")
  ) return null;
  return { engineInstanceId, sessionId, frameId };
}

/**
 * The browser runtime owns one shared state subscription, browser transport evidence,
 * its connection status, the shared browser video receiver and latest accepted
 * source state. LiveView consumes this receiver; route changes do not create
 * or tear down another camera stream.
 */
export class RootStore {
  readonly system = { status: { isConnected: false } };
  readonly browserMetrics: {
    samples: BrowserSessionSample[];
    latest: BrowserSessionSample | null;
  } = { samples: [], latest: null };
  readonly diagnosticsStore: BrowserDiagnosticsStoreState = {
    connection: "connecting",
    snapshot: null,
    receivedAt: null,
    isStale: false,
    protocolError: null,
  };
  readonly videoStore: {
    latestGameState: GameStateData | null;
    latestFrame: DisplayMjpegFrame | null;
    matchedFrame: DisplayMjpegFrame | null;
    maximumObservationAgeSeconds: number;
    framePolicy: { maximumFrameBytes: number; maximumPartHeaderBytes: number; bufferedFrameCount: number };
    delivery: VideoDeliveryStatus;
  } = {
    latestGameState: null,
    latestFrame: null,
    matchedFrame: null,
    maximumObservationAgeSeconds: 3,
    framePolicy: { maximumFrameBytes: 16 * 1024 * 1024, maximumPartHeaderBytes: 16 * 1024, bufferedFrameCount: 64 },
    delivery: {
      state: "unconfigured",
      mode: null,
      configured: false,
      configurationSynchronized: false,
      sourceGeneration: 0,
      startedAt: null,
      lastFrameReceiptAt: null,
      lastFrameIdentity: null,
      dimensions: null,
      error: null,
    },
  };
  nativeVideoElement: HTMLImageElement | null = null;

  readonly eventSessionStartedAt = new Date().toISOString();
  readonly eventHistoryLimit = eventHistoryLimit;
  events: BrowserEvent[] = [];
  private nextEventId = 0;

  private readonly websocketClient: WebSocketClient;
  private readonly diagnosticsTransport: DiagnosticsTransport;
  private readonly metricsClock: BrowserMetricsClock;
  private readonly stateChangeObserver = new BrowserStateChangeObserver();
  private readonly sessionMetrics = new BrowserSessionMetrics(
    getBrowserSessionMetricsPolicy(),
  );
  private sessionMetricsSampler: number | null = null;
  private latestProducerEpoch: string | null = null;
  private latestDeliverySequence: number | null = null;
  private latestTransportSequence: number | null = null;
  private latestSnapshotRevision: number | null = null;
  private latestSourceFrameNumber: number | null = null;
  private readonly videoReceiverShutdownHandlers = new Set<() => void>();
  private readonly videoConfigurationReloadHandlers = new Set<() => void>();
  private lastConnectionState: string | null = null;
  private videoOutageGeneration: number | null = null;
  private eventFreshnessTimeout: number | null = null;
  private diagnosticsExpiryTimeout: number | null = null;
  private lastDiagnosticsConnectionState: DiagnosticsConnectionState | null = null;
  private readonly videoDeliveryReactionDisposer: () => void;
  private readonly freshnessPolicyReactionDisposer: () => void;
  private hasShutdown = false;

  constructor(
    websocketClient?: WebSocketClient,
    metricsClock: BrowserMetricsClock = SYSTEM_BROWSER_CLOCK,
    diagnosticsTransport?: DiagnosticsTransport,
  ) {
    this.metricsClock = metricsClock;
    this.websocketClient =
      websocketClient ??
      createWebSocketClient({
        url: buildLiveStateWebSocketUrl(
          axiosClient.getWebSocketUrl("/api/v1/websocket/ws"),
          typeof window === "undefined" ? "" : window.location.search,
        ),
        autoReconnect: true,
      });
    this.diagnosticsTransport = diagnosticsTransport ?? new BrowserDiagnosticsWebSocket(
      axiosClient.getWebSocketUrl("/api/v1/diagnostics/ws"),
    );

    makeAutoObservable(this, {
      websocketClient: false,
      diagnosticsTransport: false,
      metricsClock: false,
      stateChangeObserver: false,
      sessionMetrics: false,
      sessionMetricsSampler: false,
      nativeVideoElement: observable.ref,
      latestProducerEpoch: false,
      latestDeliverySequence: false,
      latestTransportSequence: false,
      latestSnapshotRevision: false,
      latestSourceFrameNumber: false,
      videoReceiverShutdownHandlers: false,
      videoConfigurationReloadHandlers: false,
      lastConnectionState: false,
      videoOutageGeneration: false,
      eventFreshnessTimeout: false,
      diagnosticsExpiryTimeout: false,
      lastDiagnosticsConnectionState: false,
      videoDeliveryReactionDisposer: false,
      freshnessPolicyReactionDisposer: false,
      hasShutdown: false,
    } as never);

    this.videoDeliveryReactionDisposer = reaction(
      () => {
        const delivery = this.videoStore.delivery;
        return JSON.stringify([
          delivery.state,
          delivery.mode,
          delivery.configured,
          delivery.sourceGeneration,
        ]);
      },
      (current, previous) => {
        this.recordVideoDeliveryTransition(previous, current);
      },
    );
    this.freshnessPolicyReactionDisposer = reaction(
      () => this.videoStore.maximumObservationAgeSeconds,
      () => this.refreshFreshnessPolicy(),
    );

    this.diagnosticsTransport.connect({
      onConnectionState: (connection) => {
        runInAction(() => {
          if (connection === this.lastDiagnosticsConnectionState) return;
          this.lastDiagnosticsConnectionState = connection;
          this.diagnosticsStore.connection = connection;
          if (connection !== "connected") {
            this.diagnosticsStore.isStale = this.diagnosticsStore.snapshot !== null;
          }
          this.recordEvent({
            severity: connection === "error" ? "Warning" : "Info",
            component: "Diagnostics stream",
            type: "Connection",
            origin: "Browser observed",
            message: `Pushed diagnostics stream ${connection}`,
            payload: {
              connection_state: connection,
              gateway_instance_id: this.diagnosticsStore.snapshot?.gateway.instance_id ?? null,
            },
          });
        });
      },
      onPayload: (payload) => this.handleDiagnosticsPayload(payload),
    });

    this.websocketClient.onConnectionState((state, error) => {
      runInAction(() => {
        if (state === this.lastConnectionState) return;
        this.lastConnectionState = state;
        this.system.status.isConnected = state === "connected";
        this.sessionMetrics.setConnected(
          state === "connected",
          this.metricsClock.monotonicNowMs(),
        );
        if (state === "disconnected" || state === "reconnecting" || state === "error") {
          this.stateChangeObserver.markDeliveryGap();
        }
        this.recordEvent({
          severity:
            state === "error"
              ? "Error"
              : state === "disconnected" || state === "reconnecting"
                ? "Warning"
                : "Info",
          component: "State delivery",
          type: "Connection",
          origin: "Browser observed",
          message: `State stream ${state}`,
          payload: {
            connection_state: state,
            ...(error ? { error: error.name || "WebSocket connection error" } : {}),
          },
        });
      });
    });
    // Record only transport/control evidence, not routine state frames or invented backend events.
    for (const type of [
      "status",
      "error",
      "subscribed",
      "unsubscribed",
    ] as const) {
      this.websocketClient.on(type, (message) =>
        this.recordEvent({
          severity: type === "error" ? "Error" : "Info",
          component: "State delivery",
          type: "Control",
          origin: "WebSocket control response",
          message: `State stream ${type} response`,
          payload: redactEvidence(message),
        }),
      );
    }
    this.websocketClient.on("state", this.handleStateMessage);
    this.websocketClient.subscribe(["state"]);
    if (typeof window !== "undefined") {
      this.sessionMetricsSampler = window.setInterval(
        () => this.sampleBrowserMetrics(),
        this.sessionMetrics.policy.sampleIntervalMs,
      );
    }

    // Connection failure is visible through `system.status.isConnected`; the
    // client owns reconnect timing and replays this bounded subscription.
    void this.websocketClient.connect().catch((error: unknown) => {
      console.error("Live state WebSocket connection failed:", error);
    });
  }

  recordEvent(event: Omit<BrowserEvent, "id" | "timestamp">): void {
    this.events.unshift({
      ...event,
      id: `browser-${++this.nextEventId}`,
      timestamp: this.metricsClock.wallTime().toISOString(),
    });
    this.events.length = Math.min(this.events.length, this.eventHistoryLimit);
  }

  /** Record a configuration apply only after its PUT response confirms success. */
  recordConfigurationApplied(input: {
    changedFieldPaths: string[];
    restartRequired: boolean;
    previousRevision?: string | number;
    appliedRevision?: string | number;
  }): void {
    const changedFieldPaths = input.changedFieldPaths
      .filter((path) => typeof path === "string" && path.length > 0)
      .slice(0, 500);
    this.recordEvent({
      severity: "Info",
      component: "Configuration",
      type: "Configuration",
      origin: "Interface action",
      message: "Configuration apply confirmed by the server",
      payload: {
        result: "applied",
        changed_field_paths: changedFieldPaths,
        restart_required: input.restartRequired,
        ...(input.previousRevision !== undefined
          ? { previous_revision: input.previousRevision }
          : {}),
        ...(input.appliedRevision !== undefined
          ? { applied_revision: input.appliedRevision }
          : { applied_revision: "unknown until canonical reload" }),
      },
    });
  }

  setVideoDelivery(delivery: VideoDeliveryStatus): void {
    this.videoStore.delivery = delivery;
  }

  setReceivedVideoFrame(frame: DisplayMjpegFrame): void {
    this.videoStore.latestFrame = frame;
    this.videoStore.delivery = {
      ...this.videoStore.delivery,
      state: "receiving",
      mode: "identified",
      configured: true,
      configurationSynchronized: true,
      lastFrameReceiptAt: new Date().toISOString(),
      lastFrameIdentity: {
        frameId: frame.frameId,
        engineInstanceId: frame.engineInstanceId,
        sourceSessionId: frame.sourceSessionId,
        configurationRevision: frame.configurationRevision,
        calibrationRevision: frame.calibrationRevision,
        rectificationRevision: frame.rectificationRevision,
      },
      error: null,
    };
  }

  setMatchedVideoFrame(frame: DisplayMjpegFrame | null): void {
    this.videoStore.matchedFrame = frame;
  }

  setVideoDimensions(width: number, height: number): void {
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;
    this.videoStore.delivery = {
      ...this.videoStore.delivery,
      dimensions: { width, height },
    };
  }

  registerVideoReceiverShutdown(handler: () => void): () => void {
    if (this.hasShutdown) {
      handler();
      return () => undefined;
    }
    this.videoReceiverShutdownHandlers.add(handler);
    return () => this.videoReceiverShutdownHandlers.delete(handler);
  }

  registerVideoConfigurationReload(handler: () => void): () => void {
    if (this.hasShutdown) return () => undefined;
    this.videoConfigurationReloadHandlers.add(handler);
    return () => this.videoConfigurationReloadHandlers.delete(handler);
  }

  /** Notify app-scoped consumers only after the server confirms configuration apply. */
  notifyVideoConfigurationApplied(): void {
    if (this.hasShutdown) return;
    for (const handler of this.videoConfigurationReloadHandlers) handler();
  }

  private handleStateMessage = (message: WebSocketMessage): void => {
    if (!isGameStateMessage(message)) return;
    const receivedAtMs = this.metricsClock.monotonicNowMs();
    // Count syntactically valid state messages before ordering filters, so a
    // producer that sends delayed/coalesced states remains visible in the rate.
    this.sessionMetrics.recordStateMessageReceipt(receivedAtMs);
    const accepted = this.acceptObservation(message.data);
    if (!accepted) return;
    if (accepted.producerEpochChanged && accepted.producerEpoch) {
      this.sessionMetrics.recordProducerEpoch(accepted.producerEpoch, receivedAtMs);
    }
    if (accepted.isNewSourceObservation) {
      this.sessionMetrics.recordAcceptedSourceFrame(
        stateSourceFrameIdentity(accepted.state),
        receivedAtMs,
      );
    }
    const eventDrafts = this.stateChangeObserver.observeAcceptedState(
      accepted.state,
      receivedAtMs,
      this.videoStore.maximumObservationAgeSeconds,
    );
    for (const event of eventDrafts) this.recordEvent(event);
    this.scheduleEventFreshnessDeadline();
    this.sessionMetrics.recordObservationAges(
      extractObservationAges(accepted.state),
      receivedAtMs,
    );
    this.videoStore.latestGameState = accepted.state;
  };

  private handleDiagnosticsPayload(payload: unknown): void {
    if (!isBrowserDiagnosticsSnapshot(payload)) {
      runInAction(() => {
        this.diagnosticsStore.protocolError = "Unsupported or invalid diagnostics snapshot";
        this.diagnosticsStore.isStale = this.diagnosticsStore.snapshot !== null;
      });
      return;
    }
    const current = this.diagnosticsStore.snapshot;
    if (
      current?.gateway.instance_id === payload.gateway.instance_id
      && payload.gateway.revision <= current.gateway.revision
    ) return;

    if (this.diagnosticsExpiryTimeout !== null && typeof window !== "undefined") {
      window.clearTimeout(this.diagnosticsExpiryTimeout);
      this.diagnosticsExpiryTimeout = null;
    }
    runInAction(() => {
      this.diagnosticsStore.snapshot = payload;
      this.diagnosticsStore.receivedAt = this.metricsClock.wallTime().toISOString();
      this.diagnosticsStore.isStale = false;
      this.diagnosticsStore.protocolError = null;
    });
    if (typeof window !== "undefined") {
      const { instance_id: gatewayId, revision } = payload.gateway;
      this.diagnosticsExpiryTimeout = window.setTimeout(() => {
        if (
          this.diagnosticsStore.snapshot?.gateway.instance_id !== gatewayId
          || this.diagnosticsStore.snapshot.gateway.revision !== revision
        ) return;
        this.diagnosticsExpiryTimeout = null;
        runInAction(() => {
          this.diagnosticsStore.isStale = true;
        });
      }, payload.stream.stale_after_ms);
    }
  }

  private recordVideoDeliveryTransition(previous: string, current: string): void {
    const before = JSON.parse(previous) as [VideoDeliveryState, VideoDeliveryMode, boolean, number];
    const after = JSON.parse(current) as [VideoDeliveryState, VideoDeliveryMode, boolean, number];
    const [previousState, previousMode, previousConfigured, previousGeneration] = before;
    const [currentState, currentMode, currentConfigured, currentGeneration] = after;
    if (currentGeneration !== previousGeneration) {
      this.videoOutageGeneration = null;
      this.recordEvent({
        severity: "Info",
        component: "Video",
        type: "Source",
        origin: "Browser observed",
        message: previousConfigured
          ? "Video source replaced"
          : currentConfigured
            ? "Video source configured"
            : "Video source removed",
        payload: {
          previous_source_generation: previousGeneration,
          source_generation: currentGeneration,
          previous_mode: previousMode,
          mode: currentMode,
        },
      });
    }
    if (currentState === previousState) return;
    if (currentMode !== "identified") this.videoOutageGeneration = null;
    if (currentState === "stalled" && currentMode === "identified") {
      this.videoOutageGeneration = currentGeneration;
      this.recordEvent({
        severity: "Warning",
        component: "Video",
        type: "Connection",
        origin: "Browser observed",
        message: "Identified video delivery stalled",
        payload: { mode: currentMode, source_generation: currentGeneration },
      });
      return;
    }
    if (currentState === "error") {
      if (currentMode === "identified") this.videoOutageGeneration = currentGeneration;
      this.recordEvent({
        severity: "Error",
        component: "Video",
        type: "Connection",
        origin: "Browser observed",
        message: "Video receiver failed to load the configured source",
        payload: { mode: currentMode, source_generation: currentGeneration },
      });
      return;
    }
    if (
      this.videoOutageGeneration === currentGeneration
      && currentState === "receiving"
      && currentMode === "identified"
    ) {
      this.videoOutageGeneration = null;
      this.recordEvent({
        severity: "Info",
        component: "Video",
        type: "Connection",
        origin: "Browser observed",
        message: "Identified video delivery recovered",
        payload: { mode: currentMode, source_generation: currentGeneration },
      });
    }
    // A native image load cannot prove ongoing frame freshness, so `loaded`
    // does not emit an ongoing-stream recovery event.
  }

  private scheduleEventFreshnessDeadline(): void {
    if (this.eventFreshnessTimeout !== null && typeof window !== "undefined") {
      window.clearTimeout(this.eventFreshnessTimeout);
      this.eventFreshnessTimeout = null;
    }
    if (typeof window === "undefined" || this.hasShutdown) return;
    const nowMs = this.metricsClock.monotonicNowMs();
    const deadline = this.stateChangeObserver.nextFreshnessDeadline(nowMs);
    if (deadline === null) return;
    this.eventFreshnessTimeout = window.setTimeout(() => {
      this.eventFreshnessTimeout = null;
      for (const event of this.stateChangeObserver.evaluateFreshness(this.metricsClock.monotonicNowMs())) {
        this.recordEvent(event);
      }
      this.scheduleEventFreshnessDeadline();
    }, Math.max(0, deadline - nowMs));
  }

  /** Recompute a pending stale deadline if app configuration changes the fallback limit. */
  private refreshFreshnessPolicy(): void {
    const state = this.videoStore.latestGameState;
    if (state) {
      const events = this.stateChangeObserver.observeAcceptedState(
        state,
        this.metricsClock.monotonicNowMs(),
        this.videoStore.maximumObservationAgeSeconds,
      );
      for (const event of events) this.recordEvent(event);
    }
    this.scheduleEventFreshnessDeadline();
  }

  private sampleBrowserMetrics(): void {
    const sample = this.sessionMetrics.sample(
      this.metricsClock.monotonicNowMs(),
      this.metricsClock.wallTime(),
    );
    runInAction(() => {
      this.browserMetrics.latest = sample;
      this.browserMetrics.samples.push(sample);
      const overflow = this.browserMetrics.samples.length
        - this.sessionMetrics.policy.retentionBuckets;
      if (overflow > 0) this.browserMetrics.samples.splice(0, overflow);
    });
  }

  /**
   * Track phase-delivery and raw-scene sequence spaces independently. A phase
   * event may repeat the current scene transport sequence, while a later raw
   * scene may carry the last phase event again; neither stream may block or
   * roll back the other.
   */
  private acceptObservation(state: GameStateData): AcceptedObservation | null {
    const status = state.observation_status;
    const identity = status?.observation_identity;
    const engineId =
      identity?.engine_instance_id ??
      status?.engine_instance_id ??
      status?.vision_instance_id;
    const reportedSessionId: unknown =
      identity?.session_id ??
      status?.source_session_id ??
      status?.source_process_id;
    if (
      typeof engineId !== "string" ||
      (typeof reportedSessionId !== "string" &&
        typeof reportedSessionId !== "number")
    ) return {
      state,
      // Legacy state snapshots have no ordering identity. They remain
      // accepted for display, while the metrics store will label frame rate
      // unknown until an engine/session/frame identity is observed.
      isNewSourceObservation: true,
      producerEpoch: null,
      producerEpochChanged: false,
    };

    const producerEpoch = `${engineId}\u0000${reportedSessionId}`;
    const producerEpochChanged = producerEpoch !== this.latestProducerEpoch;
    const hadPreviousProducerEpoch = this.latestProducerEpoch !== null;
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
    const hasDeliverySequence =
      typeof deliverySequence === "number" &&
      Number.isSafeInteger(deliverySequence);
    let deliveryIsNew = false;
    if (hasDeliverySequence) {
      deliveryIsNew =
        this.latestDeliverySequence === null ||
        deliverySequence > this.latestDeliverySequence;
      if (deliveryIsNew) this.latestDeliverySequence = deliverySequence;
    }

    let observationIsNew = false;
    const transportSequence = status?.transport_sequence;
    if (
      typeof transportSequence === "number" &&
      Number.isSafeInteger(transportSequence)
    ) {
      observationIsNew =
        this.latestTransportSequence === null ||
        transportSequence > this.latestTransportSequence;
      if (observationIsNew) this.latestTransportSequence = transportSequence;
    } else {
      const snapshotRevision =
        identity?.snapshot_revision ?? status?.snapshot_revision;
      if (
        typeof snapshotRevision === "number" &&
        Number.isSafeInteger(snapshotRevision)
      ) {
        observationIsNew =
          this.latestSnapshotRevision === null ||
          snapshotRevision > this.latestSnapshotRevision;
        if (observationIsNew) this.latestSnapshotRevision = snapshotRevision;
      } else {
        const sourceFrameNumber =
          status?.source_frame_number ?? state.frame_number;
        if (
          typeof sourceFrameNumber === "number" &&
          Number.isSafeInteger(sourceFrameNumber)
        ) {
          observationIsNew =
            this.latestSourceFrameNumber === null ||
            sourceFrameNumber >= this.latestSourceFrameNumber;
          if (observationIsNew)
            this.latestSourceFrameNumber = sourceFrameNumber;
        } else {
          // Preserve the legacy acceptance behavior when the producer does
          // not supply any ordering field.
          observationIsNew = !hasDeliverySequence;
        }
      }
    }

    if (!deliveryIsNew && !observationIsNew) return null;

    const currentState = producerEpochChanged
      ? null
      : this.videoStore.latestGameState;
    if (deliveryIsNew && !observationIsNew && currentState) {
      // A phase-only event is self-contained for perception status but must
      // not replace the currently accepted ball layout with an older copy.
      return {
        state: {
          ...currentState,
          observation_status: {
            ...currentState.observation_status,
            perception: state.observation_status?.perception,
          },
        },
        isNewSourceObservation: false,
        producerEpoch,
        producerEpochChanged: producerEpochChanged && hadPreviousProducerEpoch,
      };
    }

    if (
      observationIsNew &&
      !deliveryIsNew &&
      currentState?.observation_status?.perception
    ) {
      // Raw scenes can be delivered after a newer coalesced phase event while
      // still carrying an older perception payload. Keep the newer evidence.
      return {
        state: {
          ...state,
          observation_status: {
            ...status,
            perception: currentState.observation_status.perception,
          },
        },
        isNewSourceObservation: true,
        producerEpoch,
        producerEpochChanged: producerEpochChanged && hadPreviousProducerEpoch,
      };
    }
    return {
      state,
      isNewSourceObservation: observationIsNew,
      producerEpoch,
      producerEpochChanged: producerEpochChanged && hadPreviousProducerEpoch,
    };
  }

  async shutdown(): Promise<void> {
    if (this.hasShutdown) return;
    this.hasShutdown = true;
    if (this.sessionMetricsSampler !== null && typeof window !== "undefined") {
      window.clearInterval(this.sessionMetricsSampler);
      this.sessionMetricsSampler = null;
    }
    if (this.eventFreshnessTimeout !== null && typeof window !== "undefined") {
      window.clearTimeout(this.eventFreshnessTimeout);
      this.eventFreshnessTimeout = null;
    }
    if (this.diagnosticsExpiryTimeout !== null && typeof window !== "undefined") {
      window.clearTimeout(this.diagnosticsExpiryTimeout);
      this.diagnosticsExpiryTimeout = null;
    }
    this.videoDeliveryReactionDisposer();
    this.freshnessPolicyReactionDisposer();
    for (const handler of this.videoReceiverShutdownHandlers) handler();
    this.videoReceiverShutdownHandlers.clear();
    this.videoConfigurationReloadHandlers.clear();
    this.websocketClient.destroy();
    this.diagnosticsTransport.destroy();
    this.videoStore.delivery = {
      ...this.videoStore.delivery,
      state: "stopped",
    };
  }
}

let rootStoreInstance: RootStore | null = null;

export function getRootStore(): RootStore {
  rootStoreInstance ??= new RootStore();
  return rootStoreInstance;
}
