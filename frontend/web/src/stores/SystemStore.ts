import { makeAutoObservable, runInAction } from 'mobx';
import type {
  SystemStatus,
  SystemError,
  WebSocketMessage,
  SystemUpdateMessage,
  ActionResult
} from './types';

export class SystemStore {
  // Observable state
  status: SystemStatus = {
    isConnected: false,
    lastHeartbeat: null,
    backendVersion: null,
    websocketStatus: 'disconnected',
    errors: []
  };

  private websocket: WebSocket | null = null;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // Start with 1 second

  constructor() {
    makeAutoObservable(this, {}, { autoBind: true });
  }

  // Computed values
  get isHealthy(): boolean {
    return this.status.isConnected &&
           this.status.websocketStatus === 'connected' &&
           this.status.errors.filter(e => e.level === 'critical').length === 0;
  }

  get hasErrors(): boolean {
    return this.status.errors.length > 0;
  }

  get criticalErrors(): SystemError[] {
    return this.status.errors.filter(error => error.level === 'critical');
  }

  get recentErrors(): SystemError[] {
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
    return this.status.errors.filter(error => error.timestamp > oneHourAgo);
  }

  get connectionUptime(): number {
    if (!this.status.lastHeartbeat) return 0;
    return Date.now() - this.status.lastHeartbeat.getTime();
  }

  // Actions
  async connect(websocketUrl: string): Promise<ActionResult> {
    try {
      this.updateWebSocketStatus('connecting');

      this.websocket = new WebSocket(websocketUrl);

      this.websocket.onopen = () => {
        runInAction(() => {
          this.status.isConnected = true;
          this.status.websocketStatus = 'connected';
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          this.addInfo('System', 'Connected to backend successfully');
        });
      };

      this.websocket.onmessage = (event) => {
        this.handleWebSocketMessage(event);
      };

      this.websocket.onclose = (event) => {
        runInAction(() => {
          this.status.isConnected = false;
          this.status.websocketStatus = 'disconnected';
          this.stopHeartbeat();

          if (event.wasClean) {
            this.addInfo('System', 'Connection closed cleanly');
          } else {
            this.addError('System', `Connection lost: ${event.reason || 'Unknown reason'}`);
            this.scheduleReconnect();
          }
        });
      };

      this.websocket.onerror = (error) => {
        runInAction(() => {
          this.status.websocketStatus = 'error';
          this.addError('System', 'WebSocket connection error');
        });
      };

      return {
        success: true,
        timestamp: new Date()
      };
    } catch (error) {
      this.addError('System', `Failed to connect: ${error instanceof Error ? error.message : 'Unknown error'}`);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date()
      };
    }
  }

  disconnect(): void {
    if (this.websocket) {
      this.websocket.close(1000, 'User requested disconnect');
      this.websocket = null;
    }
    this.stopHeartbeat();
    this.clearReconnectTimeout();

    runInAction(() => {
      this.status.isConnected = false;
      this.status.websocketStatus = 'disconnected';
      this.status.lastHeartbeat = null;
    });
  }

  async sendMessage(message: WebSocketMessage): Promise<ActionResult> {
    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
      return {
        success: false,
        error: 'WebSocket is not connected',
        timestamp: new Date()
      };
    }

    try {
      this.websocket.send(JSON.stringify(message));
      return {
        success: true,
        timestamp: new Date()
      };
    } catch (error) {
      this.addError('System', `Failed to send message: ${error instanceof Error ? error.message : 'Unknown error'}`);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date()
      };
    }
  }

  clearErrors(): void {
    runInAction(() => {
      this.status.errors = [];
    });
  }

  clearError(errorId: string): void {
    runInAction(() => {
      this.status.errors = this.status.errors.filter(error => error.id !== errorId);
    });
  }

  addError(component: string, message: string, details?: any): void {
    this.addSystemError('error', component, message, details);
  }

  addWarning(component: string, message: string, details?: any): void {
    this.addSystemError('warning', component, message, details);
  }

  addInfo(component: string, message: string, details?: any): void {
    this.addSystemError('info', component, message, details);
  }

  addCritical(component: string, message: string, details?: any): void {
    this.addSystemError('critical', component, message, details);
  }

  // Private methods
  private updateWebSocketStatus(status: SystemStatus['websocketStatus']): void {
    runInAction(() => {
      this.status.websocketStatus = status;
    });
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatInterval = setInterval(() => {
      this.sendHeartbeat();
    }, 30000); // Send heartbeat every 30 seconds
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private sendHeartbeat(): void {
    const heartbeatMessage: WebSocketMessage = {
      type: 'heartbeat',
      timestamp: new Date(),
      data: { clientTime: Date.now() }
    };

    this.sendMessage(heartbeatMessage);
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.addCritical('System', 'Max reconnection attempts reached');
      return;
    }

    this.clearReconnectTimeout();

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts); // Exponential backoff
    this.reconnectAttempts++;

    this.reconnectTimeout = setTimeout(() => {
      if (this.status.websocketStatus === 'disconnected') {
        this.addInfo('System', `Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        // Note: In a real implementation, you'd need to store the websocket URL
        // For now, this is a placeholder that would need the URL to be passed
      }
    }, delay);
  }

  private clearReconnectTimeout(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }

  private handleWebSocketMessage(event: MessageEvent): void {
    try {
      const message = JSON.parse(event.data) as WebSocketMessage;

      switch (message.type) {
        case 'heartbeat_response':
          runInAction(() => {
            this.status.lastHeartbeat = new Date();
          });
          break;

        case 'system_update':
          this.handleSystemUpdate(message as SystemUpdateMessage);
          break;

        case 'version_info':
          runInAction(() => {
            this.status.backendVersion = message.data.version;
          });
          break;

        default:
          // Other message types will be handled by other stores
          break;
      }
    } catch (error) {
      this.addError('System', `Failed to parse WebSocket message: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private handleSystemUpdate(message: SystemUpdateMessage): void {
    runInAction(() => {
      if (message.data.status) {
        Object.assign(this.status, message.data.status);
      }

      if (message.data.errors) {
        this.status.errors.push(...message.data.errors);

        // Keep only the most recent 100 errors
        if (this.status.errors.length > 100) {
          this.status.errors = this.status.errors.slice(-100);
        }
      }
    });
  }

  private addSystemError(
    level: SystemError['level'],
    component: string,
    message: string,
    details?: any
  ): void {
    const error: SystemError = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      level,
      component,
      message,
      details
    };

    runInAction(() => {
      this.status.errors.push(error);

      // Keep only the most recent 100 errors
      if (this.status.errors.length > 100) {
        this.status.errors = this.status.errors.slice(-100);
      }
    });

    // Log to console for development
    if (process.env.NODE_ENV === 'development') {
      const logMethod = level === 'error' || level === 'critical' ? 'error' :
                       level === 'warning' ? 'warn' : 'log';
      console[logMethod](`[${component}] ${message}`, details);
    }
  }

  // Cleanup method
  destroy(): void {
    this.disconnect();
    this.clearReconnectTimeout();
  }
}
