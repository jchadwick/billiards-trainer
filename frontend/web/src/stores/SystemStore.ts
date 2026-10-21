import { makeAutoObservable, runInAction } from 'mobx';
import type { MessageType, WebSocketMessage as ApiWebSocketMessage } from '../types/api';
import type {
  SystemStatus,
  SystemError,
  WebSocketMessage,
  SystemUpdateMessage,
  ActionResult
} from './types';
import { createWebSocketClient, WebSocketClient } from '../services/websocket-client';

export class SystemStore {
  // Observable state
  status: SystemStatus = {
    isConnected: false,
    lastHeartbeat: null,
    backendVersion: null,
    websocketStatus: 'disconnected',
    errors: []
  };

  // Additional state for real API integration
  lastMetrics: any = null;
  lastMetricsUpdate: Date | null = null;
  lastHealthCheck: any = null;
  lastHealthUpdate: Date | null = null;

  private websocketClient: WebSocketClient | null = null;
  private readonly injectedWebSocketClient: WebSocketClient | null;
  private ownsWebSocketClient = false;
  private registeredWebSocketClient: WebSocketClient | null = null;
  private messageRouter: ((message: any) => void) | null = null;

  constructor(websocketClient?: WebSocketClient) {
    this.injectedWebSocketClient = websocketClient ?? null;
    this.websocketClient = websocketClient ?? null;
    makeAutoObservable(this, {
      injectedWebSocketClient: false,
      websocketClient: false,
      ownsWebSocketClient: false,
      registeredWebSocketClient: false,
      messageRouter: false,
    } as any, { autoBind: true });
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
      this.websocketClient ??= this.injectedWebSocketClient ?? createWebSocketClient({ url: websocketUrl });
      this.ownsWebSocketClient = this.injectedWebSocketClient === null;
      this.websocketClient.updateConfig({ url: websocketUrl, autoReconnect: true });
      this.registerWebSocketHandlers(this.websocketClient);

      // System and live state share this transport. The client retains this
      // subscription and restores it after reconnect without a second socket.
      this.websocketClient.subscribe(['state']);
      await this.websocketClient.connect();
      if (this.websocketClient.isConnected && !this.status.isConnected) {
        this.handleConnectionStateChange('connected');
      }

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
    this.websocketClient?.disconnect();

    runInAction(() => {
      this.status.isConnected = false;
      this.status.websocketStatus = 'disconnected';
      this.status.lastHeartbeat = null;
    });
  }

  async sendMessage(message: WebSocketMessage): Promise<ActionResult> {
    if (!this.websocketClient?.isConnected) {
      return {
        success: false,
        error: 'WebSocket is not connected',
        timestamp: new Date()
      };
    }

    try {
      const transportMessage = {
        ...message,
        type: message.type as MessageType,
        timestamp: message.timestamp.toISOString(),
      };
      if (!this.websocketClient.send(transportMessage)) {
        throw new Error('WebSocket message could not be sent');
      }
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

  // Public method to set message router
  addMessageRouter = (router: (message: any) => void): void => {
    this.messageRouter = router;
  }

  private registerWebSocketHandlers(client: WebSocketClient): void {
    if (this.registeredWebSocketClient === client) return;
    if (this.registeredWebSocketClient) {
      this.registeredWebSocketClient.offAny(this.handleWebSocketMessage);
      this.registeredWebSocketClient.offConnectionState(this.handleConnectionStateChange);
    }
    client.onAny(this.handleWebSocketMessage);
    client.onConnectionState(this.handleConnectionStateChange);
    this.registeredWebSocketClient = client;
  }

  private handleConnectionStateChange(state: string, error?: Error): void {
    runInAction(() => {
      this.status.isConnected = state === 'connected';
      this.status.websocketStatus = state === 'connected'
        ? 'connected'
        : state === 'error'
          ? 'error'
          : state === 'disconnected'
            ? 'disconnected'
            : 'connecting';
      if (state === 'connected') this.addInfo('System', 'Connected to backend successfully');
      if (error) this.addError('System', error.message);
    });
  }

  private handleWebSocketMessage(message: ApiWebSocketMessage): void {
    try {
      // Route every valid server message first so other stores can retain
      // existing payload contracts without owning the transport.
      this.messageRouter?.(message);

      switch (message.type as string) {
        case 'heartbeat_response':
        case 'pong':
          runInAction(() => {
            this.status.lastHeartbeat = new Date();
          });
          break;

        case 'system_update':
          this.handleSystemUpdate(message as unknown as SystemUpdateMessage);
          break;

        case 'version_info':
          runInAction(() => {
            this.status.backendVersion = message.data.version;
          });
          break;

        default:
          break;
      }
    } catch (error) {
      this.addError('System', `Failed to parse WebSocket message: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /** Detach store callbacks and release the client only when this store owns it. */
  destroy(): void {
    if (this.registeredWebSocketClient) {
      this.registeredWebSocketClient.offAny(this.handleWebSocketMessage);
      this.registeredWebSocketClient.offConnectionState(this.handleConnectionStateChange);
      this.registeredWebSocketClient = null;
    }
    if (this.ownsWebSocketClient) this.websocketClient?.destroy();
    this.websocketClient = null;
    this.status.isConnected = false;
    this.status.websocketStatus = 'disconnected';
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

  // Monitoring methods for dashboard integration
  async refreshMetrics(): Promise<void> {
    try {
      const { apiClient } = await import('../api/client');
      const response = await apiClient.getMetrics();

      if (response.success && response.data) {
        const data = response.data;
        // Store metrics data for use by dashboard components
        runInAction(() => {
          this.lastMetrics = data;
          this.lastMetricsUpdate = new Date();
        });
        this.addInfo('System', 'Metrics refreshed successfully');
      } else {
        this.addError('System', `Failed to refresh metrics: ${response.error || 'Unknown error'}`);
      }
    } catch (error) {
      this.addError('System', `Failed to refresh metrics: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async refreshModuleStatus(): Promise<void> {
    try {
      const { apiClient } = await import('../api/client');
      const response = await apiClient.getHealth(true, false);

      if (response.success && response.data) {
        const data = response.data;
        runInAction(() => {
          this.status.backendVersion = data.version;
          this.lastHealthCheck = data;
          this.lastHealthUpdate = new Date();
        });
        this.addInfo('System', 'Module status refreshed successfully');
      } else {
        this.addError('System', `Failed to refresh module status: ${response.error || 'Unknown error'}`);
      }
    } catch (error) {
      this.addError('System', `Failed to refresh module status: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async getHealthStatus(): Promise<any> {
    try {
      const { apiClient } = await import('../api/client');
      const response = await apiClient.getHealth(true, true);

      if (response.success && response.data) {
        const data = response.data;
        // Update internal state
        runInAction(() => {
          this.status.backendVersion = data.version;
          this.lastHealthCheck = data;
          this.lastHealthUpdate = new Date();
        });

        return {
          status: data?.status || 'unknown',
          timestamp: data?.timestamp || new Date().toISOString(),
          uptime: data?.uptime || 0,
          version: data?.version || 'unknown',
          components: data?.components || {},
          metrics: data?.metrics
        };
      } else {
        // Return fallback data based on current connection state
        return {
          status: this.isHealthy ? 'healthy' : 'unhealthy',
          timestamp: new Date().toISOString(),
          uptime: this.connectionUptime / 1000,
          version: this.status.backendVersion || 'unknown',
          components: {
            api: {
              name: 'API Server',
              status: this.status.isConnected ? 'healthy' : 'unhealthy',
              message: this.status.isConnected ? 'All endpoints responding' : 'Server unreachable',
              last_check: new Date().toISOString(),
              uptime: this.status.isConnected ? this.connectionUptime / 1000 : 0,
              errors: this.status.isConnected ? [] : ['Connection failed'],
            },
            websocket: {
              name: 'WebSocket Server',
              status: this.status.websocketStatus === 'connected' ? 'healthy' : 'unhealthy',
              message: this.status.websocketStatus === 'connected' ? 'WebSocket active' : 'WebSocket disconnected',
              last_check: new Date().toISOString(),
              uptime: this.status.websocketStatus === 'connected' ? this.connectionUptime / 1000 : 0,
              errors: this.status.websocketStatus !== 'connected' ? ['WebSocket disconnected'] : [],
            },
          },
        };
      }
    } catch (error) {
      this.addError('System', `Failed to get health status: ${error instanceof Error ? error.message : 'Unknown error'}`);

      // Return fallback data
      return {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        uptime: 0,
        version: 'unknown',
        components: {},
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  // Module management methods
  async startModule(moduleName: string): Promise<ActionResult> {
    try {
      const { apiClient } = await import('../api/client');
      let response: any;

      switch (moduleName.toLowerCase()) {
        case 'vision':
          // Start vision detection
          response = await apiClient.startDetection();
          break;

        case 'video':
        case 'stream':
          // Start video capture
          response = await apiClient.startVideoCapture();
          break;

        case 'api':
        case 'core':
        case 'config':
        case 'projector':
          // These modules don't have individual start/stop endpoints
          // They're managed as part of the overall system
          this.addInfo('System', `Module ${moduleName} is managed automatically by the system`);
          return {
            success: true,
            timestamp: new Date()
          };

        default:
          this.addWarning('System', `Unknown module: ${moduleName}`);
          return {
            success: false,
            error: `Unknown module: ${moduleName}`,
            timestamp: new Date()
          };
      }

      if (response.success) {
        this.addInfo('System', `Module ${moduleName} started successfully`);
        return {
          success: true,
          timestamp: new Date()
        };
      } else {
        this.addError('System', `Failed to start module ${moduleName}: ${response.error}`);
        return {
          success: false,
          error: response.error || `Failed to start module ${moduleName}`,
          timestamp: new Date()
        };
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.addError('System', `Error starting module ${moduleName}: ${errorMessage}`);
      return {
        success: false,
        error: errorMessage,
        timestamp: new Date()
      };
    }
  }

  async stopModule(moduleName: string): Promise<ActionResult> {
    try {
      const { apiClient } = await import('../api/client');
      let response: any;

      switch (moduleName.toLowerCase()) {
        case 'vision':
          // Stop vision detection
          response = await apiClient.stopDetection();
          break;

        case 'video':
        case 'stream':
          // Stop video capture
          response = await apiClient.stopVideoCapture();
          break;

        case 'api':
        case 'core':
        case 'config':
        case 'projector':
          // These modules don't have individual start/stop endpoints
          this.addWarning('System', `Module ${moduleName} cannot be stopped individually - it's part of the core system`);
          return {
            success: false,
            error: `Module ${moduleName} cannot be stopped individually`,
            timestamp: new Date()
          };

        default:
          this.addWarning('System', `Unknown module: ${moduleName}`);
          return {
            success: false,
            error: `Unknown module: ${moduleName}`,
            timestamp: new Date()
          };
      }

      if (response.success) {
        this.addInfo('System', `Module ${moduleName} stopped successfully`);
        return {
          success: true,
          timestamp: new Date()
        };
      } else {
        this.addError('System', `Failed to stop module ${moduleName}: ${response.error}`);
        return {
          success: false,
          error: response.error || `Failed to stop module ${moduleName}`,
          timestamp: new Date()
        };
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.addError('System', `Error stopping module ${moduleName}: ${errorMessage}`);
      return {
        success: false,
        error: errorMessage,
        timestamp: new Date()
      };
    }
  }

  async restartModule(moduleName: string): Promise<ActionResult> {
    try {
      // For restartable modules, stop then start
      if (['vision', 'video', 'stream'].includes(moduleName.toLowerCase())) {
        this.addInfo('System', `Restarting module ${moduleName}...`);

        // First stop the module
        const stopResult = await this.stopModule(moduleName);
        if (!stopResult.success) {
          return stopResult;
        }

        // Wait a brief moment
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Then start it again
        const startResult = await this.startModule(moduleName);
        if (startResult.success) {
          this.addInfo('System', `Module ${moduleName} restarted successfully`);
        }
        return startResult;
      } else {
        // For non-restartable modules
        this.addWarning('System', `Module ${moduleName} cannot be restarted individually`);
        return {
          success: false,
          error: `Module ${moduleName} cannot be restarted individually`,
          timestamp: new Date()
        };
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.addError('System', `Error restarting module ${moduleName}: ${errorMessage}`);
      return {
        success: false,
        error: errorMessage,
        timestamp: new Date()
      };
    }
  }

  // Get module status for management interface
  getModuleStatus(moduleName: string): {
    status: 'running' | 'stopped' | 'error' | 'starting' | 'stopping' | 'unknown';
    canStart: boolean;
    canStop: boolean;
    canRestart: boolean;
    lastAction?: Date;
  } {
    // This would ideally get real status from the backend
    // For now, provide reasonable defaults based on system connection
    const isSystemConnected = this.status.isConnected;

    switch (moduleName.toLowerCase()) {
      case 'vision':
      case 'video':
      case 'stream':
        return {
          status: isSystemConnected ? 'running' : 'stopped',
          canStart: !isSystemConnected,
          canStop: isSystemConnected,
          canRestart: true,
        };

      case 'api':
        return {
          status: isSystemConnected ? 'running' : 'stopped',
          canStart: false,
          canStop: false,
          canRestart: false,
        };

      case 'core':
      case 'config':
      case 'projector':
        return {
          status: isSystemConnected ? 'running' : 'unknown',
          canStart: false,
          canStop: false,
          canRestart: false,
        };

      default:
        return {
          status: 'unknown',
          canStart: false,
          canStop: false,
          canRestart: false,
        };
    }
  }

}
