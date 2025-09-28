/**
 * WebSocket client with auto-reconnection and comprehensive error handling
 */

import {
  WebSocketMessage,
  MessageType,
  SubscribeRequest,
  UnsubscribeRequest,
  isWebSocketMessage,
} from '../types/api';

export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error' | 'reconnecting';

export interface WebSocketConfig {
  url: string;
  token?: string;
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectDelay?: number;
  maxReconnectDelay?: number;
  heartbeatInterval?: number;
  connectionTimeout?: number;
}

export interface ConnectionStats {
  connected_at?: Date;
  last_ping?: Date;
  uptime: number;
  messages_sent: number;
  messages_received: number;
  bytes_sent: number;
  bytes_received: number;
  reconnect_count: number;
  last_error?: string;
}

export type MessageHandler = (message: WebSocketMessage) => void;
export type ConnectionStateHandler = (state: ConnectionState, error?: Error) => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private state: ConnectionState = 'disconnected';
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private connectionTimer: NodeJS.Timeout | null = null;
  private messageQueue: string[] = [];

  // Event handlers
  private messageHandlers = new Map<MessageType, Set<MessageHandler>>();
  private stateHandlers = new Set<ConnectionStateHandler>();

  // Statistics
  private stats: ConnectionStats = {
    uptime: 0,
    messages_sent: 0,
    messages_received: 0,
    bytes_sent: 0,
    bytes_received: 0,
    reconnect_count: 0,
  };

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

    // Initialize message handlers map
    Object.values(MessageType).forEach(type => {
      this.messageHandlers.set(type as MessageType, new Set());
    });
  }

  // =============================================================================
  // Connection Management
  // =============================================================================

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
        resolve();
        return;
      }

      this.setState('connecting');
      this.clearTimers();

      try {
        // Build WebSocket URL with authentication
        const url = new URL(this.config.url);
        if (this.config.token) {
          url.searchParams.set('token', this.config.token);
        }

        this.ws = new WebSocket(url.toString());

        // Set connection timeout
        this.connectionTimer = setTimeout(() => {
          if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
            this.ws.close();
            reject(new Error('Connection timeout'));
          }
        }, this.config.connectionTimeout);

        this.ws.onopen = () => {
          this.clearTimers();
          this.onOpen();
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.onMessage(event);
        };

        this.ws.onclose = (event) => {
          this.onClose(event);
        };

        this.ws.onerror = (event) => {
          this.onError(event);
          if (this.state === 'connecting') {
            reject(new Error('Connection failed'));
          }
        };

      } catch (error) {
        this.setState('error', error as Error);
        reject(error);
      }
    });
  }

  disconnect(): void {
    this.config.autoReconnect = false;
    this.clearTimers();

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
        this.stats.messages_sent++;
        this.stats.bytes_sent += messageStr.length;
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
    const request: SubscribeRequest = { streams, filters };
    return this.send({
      type: 'subscribe',
      data: request,
    });
  }

  unsubscribe(streams: string[]): boolean {
    const request: UnsubscribeRequest = { streams };
    return this.send({
      type: 'unsubscribe',
      data: request,
    });
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
    this.stats.connected_at = new Date();
    this.reconnectAttempts = 0;

    // Send queued messages
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message && this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(message);
        this.stats.messages_sent++;
        this.stats.bytes_sent += message.length;
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

      this.stats.messages_received++;
      this.stats.bytes_received += event.data.length;
      this.stats.last_ping = new Date();

      // Handle pong messages
      if (data.type === 'pong') {
        this.handlePong(data);
        return;
      }

      // Dispatch to registered handlers
      const handlers = this.messageHandlers.get(data.type);
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
    this.stats.last_error = error.message;
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
      this.stats.reconnect_count++;
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

  // =============================================================================
  // Public Accessors
  // =============================================================================

  get connectionState(): ConnectionState {
    return this.state;
  }

  get isConnected(): boolean {
    return this.state === 'connected';
  }

  get connectionStats(): ConnectionStats {
    return {
      ...this.stats,
      uptime: this.stats.connected_at
        ? Date.now() - this.stats.connected_at.getTime()
        : 0,
    };
  }

  // =============================================================================
  // Configuration Updates
  // =============================================================================

  updateToken(token: string): void {
    this.config.token = token;

    // Reconnect if currently connected to use new token
    if (this.isConnected) {
      this.disconnect();
      setTimeout(() => this.connect(), 100);
    }
  }

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
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createWebSocketClient(config: WebSocketConfig): WebSocketClient {
  return new WebSocketClient(config);
}

// =============================================================================
// Connection Quality Monitor
// =============================================================================

export class ConnectionQualityMonitor {
  private client: WebSocketClient;
  private metrics = {
    latencyHistory: [] as number[],
    messageHistory: [] as number[],
    errorCount: 0,
    lastQualityCheck: Date.now(),
  };

  constructor(client: WebSocketClient) {
    this.client = client;
    this.startMonitoring();
  }

  private startMonitoring(): void {
    // Monitor connection quality every 30 seconds
    setInterval(() => {
      this.updateQualityMetrics();
    }, 30000);
  }

  private updateQualityMetrics(): void {
    const stats = this.client.connectionStats;
    const now = Date.now();
    const timeDiff = now - this.metrics.lastQualityCheck;

    // Calculate message rate
    const messageRate = (stats.messages_received * 1000) / timeDiff;
    this.metrics.messageHistory.push(messageRate);

    // Keep only last 10 measurements
    if (this.metrics.messageHistory.length > 10) {
      this.metrics.messageHistory.shift();
    }

    this.metrics.lastQualityCheck = now;
  }

  getQualityScore(): number {
    const stats = this.client.connectionStats();
    let score = 1.0;

    // Factor in reconnection count
    if (stats.reconnect_count > 0) {
      score -= Math.min(stats.reconnect_count * 0.1, 0.5);
    }

    // Factor in error count
    if (this.metrics.errorCount > 0) {
      score -= Math.min(this.metrics.errorCount * 0.05, 0.3);
    }

    // Factor in message consistency
    if (this.metrics.messageHistory.length > 0) {
      const avgRate = this.metrics.messageHistory.reduce((a, b) => a + b, 0) / this.metrics.messageHistory.length;
      const variance = this.metrics.messageHistory.reduce((acc, rate) => acc + Math.pow(rate - avgRate, 2), 0) / this.metrics.messageHistory.length;
      if (variance > 10) {
        score -= 0.1;
      }
    }

    return Math.max(0, Math.min(1, score));
  }

  getHealthStatus(): {
    score: number;
    status: 'excellent' | 'good' | 'fair' | 'poor';
    issues: string[];
  } {
    const score = this.getQualityScore();
    const stats = this.client.connectionStats();
    const issues: string[] = [];

    if (stats.reconnect_count > 3) {
      issues.push('Frequent reconnections detected');
    }

    if (this.metrics.errorCount > 5) {
      issues.push('Multiple connection errors');
    }

    if (score < 0.3) {
      issues.push('Poor connection stability');
    }

    let status: 'excellent' | 'good' | 'fair' | 'poor';
    if (score >= 0.9) status = 'excellent';
    else if (score >= 0.7) status = 'good';
    else if (score >= 0.5) status = 'fair';
    else status = 'poor';

    return { score, status, issues };
  }
}
