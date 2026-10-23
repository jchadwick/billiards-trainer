/** Shared state transport with reconnect and replay of the desired subscription. */

import type {
  WebSocketMessage,
  MessageType,
  SubscribeRequest,
  UnsubscribeRequest,
} from '../types/api';
import { isWebSocketMessage } from '../types/api';

export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error' | 'reconnecting';

export interface WebSocketConfig {
  url: string;
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectDelay?: number;
  maxReconnectDelay?: number;
  heartbeatInterval?: number;
  connectionTimeout?: number;
}

export type MessageHandler = (message: WebSocketMessage) => void;
export type ConnectionStateHandler = (state: ConnectionState, error?: Error) => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private state: ConnectionState = 'disconnected';
  private connectPromise: Promise<void> | null = null;
  private rejectPendingConnect: ((error: Error) => void) | null = null;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private connectionTimer: NodeJS.Timeout | null = null;
  private messageQueue: string[] = [];

  // Event handlers
  private messageHandlers = new Map<MessageType, Set<MessageHandler>>();
  private stateHandlers = new Set<ConnectionStateHandler>();
  /** Desired subscriptions survive reconnects and are replayed after every open. */
  private subscriptions = new Map<string, Record<string, any> | undefined>();

  constructor(config: WebSocketConfig) {
    this.config = {
      autoReconnect: true,
      maxReconnectAttempts: 10,
      reconnectDelay: 1000,
      maxReconnectDelay: 30000,
      heartbeatInterval: 30000,
      connectionTimeout: 10000,
      ...config,
    };

    // MessageType is erased at runtime, so initialize only the live stream and
    // the control responses that callers can observe.
    const messageTypes: MessageType[] = [
      'state', 'pong', 'subscribed', 'unsubscribed', 'status', 'error',
    ];

    messageTypes.forEach(type => {
      this.messageHandlers.set(type, new Set());
    });
  }

  // =============================================================================
  // Connection Management
  // =============================================================================

  connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) return Promise.resolve();
    if (this.connectPromise) return this.connectPromise;

    this.config.autoReconnect = true;
    this.clearTimers();
    this.setState('connecting');

    let resolveAttempt!: () => void;
    let rejectAttempt!: (error: Error) => void;
    const attempt = new Promise<void>((resolve, reject) => {
      resolveAttempt = resolve;
      rejectAttempt = reject;
    });
    this.connectPromise = attempt;
    let settled = false;
    const resolveOnce = () => {
      if (settled) return;
      settled = true;
      if (this.connectPromise === attempt) this.connectPromise = null;
      this.rejectPendingConnect = null;
      resolveAttempt();
    };
    const rejectOnce = (error: Error) => {
      if (settled) return;
      settled = true;
      if (this.connectPromise === attempt) this.connectPromise = null;
      this.rejectPendingConnect = null;
      rejectAttempt(error);
    };
    this.rejectPendingConnect = rejectOnce;

    try {
      this.ws = new WebSocket(this.config.url);
      const socket = this.ws;

      // A failed attempt may be followed by an automatic reconnect. Settle
      // every caller waiting for this attempt before scheduling that retry.
      this.connectionTimer = setTimeout(() => {
        if (socket.readyState === WebSocket.CONNECTING) {
          socket.close();
          rejectOnce(new Error('Connection timeout'));
        }
      }, this.config.connectionTimeout);

      socket.onopen = () => {
        if (this.ws !== socket) return;
        this.clearTimers();
        this.onOpen();
        resolveOnce();
      };

      socket.onmessage = (event) => {
        if (this.ws === socket) this.onMessage(event);
      };

      socket.onclose = (event) => {
        if (this.ws !== socket) return;
        this.onClose(event);
        if (!settled) rejectOnce(new Error('Connection closed before opening'));
      };

      socket.onerror = (event) => {
        if (this.ws !== socket) return;
        this.onError(event);
        if (!settled) rejectOnce(new Error('Connection failed'));
      };
    } catch (error) {
      this.setState('error', error instanceof Error ? error : new Error(String(error)));
      rejectOnce(error instanceof Error ? error : new Error(String(error)));
    }

    return attempt;
  }

  disconnect(): void {
    this.config.autoReconnect = false;
    this.clearTimers();
    this.rejectPendingConnect?.(new Error('Connection cancelled'));
    this.connectPromise = null;

    if (this.ws) {
      this.ws.close(1000, 'Client initiated disconnect');
      this.ws = null;
    }

    this.setState('disconnected');
  }

  // =============================================================================
  // Message Handling
  // =============================================================================

  send(message: Partial<WebSocketMessage>): boolean {
    const fullMessage: WebSocketMessage = {
      type: 'ping' as MessageType,
      timestamp: new Date().toISOString(),
      data: {},
      ...message,
    };

    const messageStr = JSON.stringify(fullMessage);

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(messageStr);
        return true;
      } catch (error) {
        console.error('Failed to send message:', error);
        return false;
      }
    } else {
      // Queue message for later sending
      if (this.config.autoReconnect) {
        this.messageQueue.push(messageStr);
      }
      return false;
    }
  }

  subscribe(streams: string[], filters?: Record<string, any>): boolean {
    const filterSnapshot = filters === undefined
      ? undefined
      : JSON.parse(this.serialize(filters)) as Record<string, any>;
    const changedStreams = streams.filter((stream) => {
      if (!stream) return false;
      const previous = this.subscriptions.get(stream);
      const wasSubscribed = this.subscriptions.has(stream);
      const changed = !wasSubscribed || this.serialize(previous) !== this.serialize(filterSnapshot);
      this.subscriptions.set(stream, filterSnapshot);
      return changed;
    });
    if (changedStreams.length === 0) return this.isConnected;
    return this.sendSubscription(changedStreams, filterSnapshot);
  }

  unsubscribe(streams: string[]): boolean {
    const removedStreams = streams.filter((stream) => this.subscriptions.delete(stream));
    if (removedStreams.length === 0) return this.isConnected;
    if (!this.isConnected) return false;
    const request: UnsubscribeRequest = { streams: removedStreams };
    return this.send({ type: 'unsubscribe', data: request });
  }

  ping(): boolean {
    return this.send({
      type: 'ping',
      data: { timestamp: new Date().toISOString() },
    });
  }

  // =============================================================================
  // Event Handler Management
  // =============================================================================

  on(messageType: MessageType, handler: MessageHandler): void {
    const handlers = this.messageHandlers.get(messageType);
    if (handlers) {
      handlers.add(handler);
    }
  }

  off(messageType: MessageType, handler: MessageHandler): void {
    const handlers = this.messageHandlers.get(messageType);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  onConnectionState(handler: ConnectionStateHandler): void {
    this.stateHandlers.add(handler);
  }

  offConnectionState(handler: ConnectionStateHandler): void {
    this.stateHandlers.delete(handler);
  }

  // =============================================================================
  // Private Methods
  // =============================================================================

  private onOpen(): void {
    console.log('WebSocket connected');
    this.setState('connected');
    this.reconnectAttempts = 0;

    // Restore stream subscriptions before flushing application messages. A
    // subscriber therefore never has to coordinate its own reconnect logic.
    this.replaySubscriptions();

    // Send queued messages
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message && this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(message);
      }
    }

    // Start heartbeat
    this.startHeartbeat();
  }

  private onMessage(event: MessageEvent): void {
    try {
      const data = JSON.parse(event.data);

      if (!isWebSocketMessage(data)) {
        console.warn('Received invalid message format:', data);
        return;
      }

      // Handle pong messages
      if (data.type === 'pong') {
        this.handlePong(data);
      }

      // Dispatch to registered handlers
      const handlers = this.messageHandlers.get(data.type as MessageType);
      if (handlers) {
        handlers.forEach(handler => {
          try {
            handler(data);
          } catch (error) {
            console.error(`Error in message handler for ${data.type}:`, error);
          }
        });
      }

    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  private onClose(event: CloseEvent): void {
    console.log('WebSocket closed:', event.code, event.reason);
    this.clearTimers();

    if (event.code !== 1000 && this.config.autoReconnect) {
      this.setState('reconnecting');
      this.scheduleReconnect();
    } else {
      this.setState('disconnected');
    }
  }

  private onError(event: Event): void {
    console.error('WebSocket error:', event);
    const error = new Error('WebSocket connection error');
    this.setState('error', error);
  }

  private setState(newState: ConnectionState, error?: Error): void {
    if (this.state !== newState) {
      this.state = newState;
      this.stateHandlers.forEach(handler => {
        try {
          handler(newState, error);
        } catch (err) {
          console.error('Error in connection state handler:', err);
        }
      });
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.setState('error', new Error('Max reconnection attempts exceeded'));
      return;
    }

    const delay = Math.min(
      this.config.reconnectDelay * Math.pow(2, this.reconnectAttempts),
      this.config.maxReconnectDelay
    );

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1})`);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++;
      this.connect().catch(error => {
        console.error('Reconnection failed:', error);
        this.scheduleReconnect();
      });
    }, delay);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ping();
      }
    }, this.config.heartbeatInterval);
  }

  private handlePong(message: WebSocketMessage): void {
    // Calculate latency if timestamp is available
    const clientTimestamp = message.data.timestamp;
    if (clientTimestamp) {
      const latency = Date.now() - new Date(clientTimestamp).getTime();
      console.debug(`WebSocket ping latency: ${latency}ms`);
    }
  }

  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer);
      this.connectionTimer = null;
    }
  }

  private sendSubscription(streams: string[], filters?: Record<string, any>): boolean {
    // Subscription intent is already retained in `subscriptions`; queueing this
    // command as well would send it twice when onOpen replays that intent.
    if (!this.isConnected) return false;
    const request: SubscribeRequest = { streams, filters };
    return this.send({ type: 'subscribe', data: request });
  }

  private replaySubscriptions(): void {
    const grouped = new Map<string, { filters: Record<string, any> | undefined; streams: string[] }>();
    for (const [stream, filters] of this.subscriptions) {
      const key = this.serialize(filters);
      const group = grouped.get(key) ?? { filters, streams: [] };
      group.streams.push(stream);
      grouped.set(key, group);
    }
    for (const { streams, filters } of grouped.values()) {
      this.sendSubscription(streams, filters);
    }
  }

  private serialize(value: unknown): string {
    if (value === undefined) return 'undefined';
    if (value === null || typeof value !== 'object') return JSON.stringify(value) ?? String(value);
    if (Array.isArray(value)) return `[${value.map(item => this.serialize(item)).join(',')}]`;
    const entries = Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right));
    return `{${entries.map(([key, entry]) => `${JSON.stringify(key)}:${this.serialize(entry)}`).join(',')}}`;
  }

  // =============================================================================
  // Public Accessors
  // =============================================================================

  get isConnected(): boolean {
    return this.state === 'connected';
  }

  // =============================================================================
  // Configuration Updates
  // =============================================================================

  updateConfig(newConfig: Partial<WebSocketConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  // =============================================================================
  // Cleanup
  // =============================================================================

  destroy(): void {
    this.disconnect();
    this.messageHandlers.clear();
    this.stateHandlers.clear();
    this.messageQueue.length = 0;
    this.subscriptions.clear();
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createWebSocketClient(config: WebSocketConfig): WebSocketClient {
  return new WebSocketClient(config);
}
