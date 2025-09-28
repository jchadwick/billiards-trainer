/**
 * MobX store for system health and monitoring
 */

import { makeAutoObservable, runInAction, flow, observable } from 'mobx';
import { HealthResponse, ComponentHealth, SystemMetrics, ConnectionStatus } from '../types/api';
import type { RootStore } from './index';

export interface SystemAlert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  source: string;
  details?: Record<string, any>;
}

export interface PerformanceMetrics {
  cpu: number;
  memory: number;
  disk: number;
  network: { in: number; out: number };
  apiLatency: number;
  wsLatency: number;
  frameRate: number;
  timestamp: Date;
}

export class SystemStore {
  private rootStore: RootStore;

  // System health
  overallHealth: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
  components = observable.map<string, ComponentHealth>();
  metrics: SystemMetrics | null = null;

  // System information
  version = '';
  buildDate: Date | null = null;
  uptime = 0;

  // Connection status
  connectionStatus: ConnectionStatus = {
    api: 'disconnected',
    websocket: 'disconnected',
    lastApiCall: null,
    lastWebSocketMessage: null,
  };

  // Alerts and notifications
  alerts = observable.array<SystemAlert>([]);
  unacknowledgedAlerts = 0;

  // Performance monitoring
  performanceHistory = observable.array<PerformanceMetrics>([]);
  maxPerformanceHistory = 100;

  // Loading and error states
  isLoading = false;
  error: string | null = null;
  lastHealthCheck: Date | null = null;

  // Auto-refresh settings
  autoRefreshEnabled = true;
  refreshInterval = 30000; // 30 seconds
  private refreshTimer: NodeJS.Timeout | null = null;

  constructor(rootStore: RootStore) {
    makeAutoObservable(this, {}, { autoBind: true });
    this.rootStore = rootStore;
  }

  async initialize(): Promise<void> {
    // Subscribe to connection status updates
    this.rootStore.apiService.onConnectionStatus(this.handleConnectionStatusChange);

    // Subscribe to health check updates
    this.rootStore.apiService.onHealthCheck(this.handleHealthUpdate);

    // Subscribe to real-time alerts
    this.rootStore.apiService.onAlertData(this.handleAlert);

    // Load initial system info
    await this.loadSystemInfo();

    // Start auto-refresh if enabled
    if (this.autoRefreshEnabled) {
      this.startAutoRefresh();
    }
  }

  // =============================================================================
  // System Information Loading
  // =============================================================================

  loadSystemInfo = flow(function* (this: SystemStore) {
    this.isLoading = true;
    this.error = null;

    try {
      // Load system version
      const versionInfo = yield this.rootStore.apiService.getSystemVersion();

      // Load health status
      const healthInfo = yield this.rootStore.apiService.getHealth();

      runInAction(() => {
        this.version = versionInfo.version;
        this.buildDate = new Date(versionInfo.build_date);
        this.handleHealthUpdate(healthInfo);
        this.isLoading = false;
      });

    } catch (error) {
      runInAction(() => {
        this.isLoading = false;
        this.error = error instanceof Error ? error.message : 'Failed to load system information';
      });
    }
  });

  refreshHealth = flow(function* (this: SystemStore) {
    try {
      const healthInfo = yield this.rootStore.apiService.getHealth();
      runInAction(() => {
        this.handleHealthUpdate(healthInfo);
        this.error = null;
      });
    } catch (error) {
      runInAction(() => {
        this.error = error instanceof Error ? error.message : 'Health check failed';
      });
    }
  });

  // =============================================================================
  // Event Handlers
  // =============================================================================

  private handleHealthUpdate = (health: HealthResponse): void => {
    runInAction(() => {
      this.overallHealth = health.status;
      this.uptime = health.uptime;
      this.lastHealthCheck = new Date();

      // Update components
      this.components.clear();
      Object.entries(health.components).forEach(([name, component]) => {
        this.components.set(name, component);
      });

      // Update metrics
      this.metrics = health.metrics || null;

      // Add performance data to history
      if (health.metrics) {
        this.addPerformanceData({
          cpu: health.metrics.cpu_usage,
          memory: health.metrics.memory_usage,
          disk: health.metrics.disk_usage,
          network: {
            in: health.metrics.network_io.bytes_received || 0,
            out: health.metrics.network_io.bytes_sent || 0,
          },
          apiLatency: health.metrics.average_response_time,
          wsLatency: 0, // Will be updated from connection status
          frameRate: 0, // Will be updated from game store
          timestamp: new Date(),
        });
      }
    });
  };

  private handleConnectionStatusChange = (status: ConnectionStatus): void => {
    runInAction(() => {
      this.connectionStatus = { ...status };

      // Update WebSocket latency in performance data
      if (this.performanceHistory.length > 0) {
        const latest = this.performanceHistory[this.performanceHistory.length - 1];
        if (status.lastWebSocketMessage) {
          latest.wsLatency = Date.now() - status.lastWebSocketMessage.getTime();
        }
      }
    });
  };

  private handleAlert = (alert: any): void => {
    runInAction(() => {
      const systemAlert: SystemAlert = {
        id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: alert.level,
        title: this.getAlertTitle(alert.code),
        message: alert.message,
        timestamp: new Date(),
        acknowledged: false,
        source: 'system',
        details: alert.details,
      };

      this.alerts.push(systemAlert);
      this.unacknowledgedAlerts++;

      // Auto-acknowledge info alerts after 5 seconds
      if (alert.level === 'info') {
        setTimeout(() => {
          this.acknowledgeAlert(systemAlert.id);
        }, 5000);
      }
    });
  };

  // =============================================================================
  // Alert Management
  // =============================================================================

  acknowledgeAlert(alertId: string): void {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert && !alert.acknowledged) {
      alert.acknowledged = true;
      this.unacknowledgedAlerts = Math.max(0, this.unacknowledgedAlerts - 1);
    }
  }

  acknowledgeAllAlerts(): void {
    this.alerts.forEach(alert => {
      if (!alert.acknowledged) {
        alert.acknowledged = true;
      }
    });
    this.unacknowledgedAlerts = 0;
  }

  dismissAlert(alertId: string): void {
    const index = this.alerts.findIndex(a => a.id === alertId);
    if (index !== -1) {
      const alert = this.alerts[index];
      if (!alert.acknowledged) {
        this.unacknowledgedAlerts = Math.max(0, this.unacknowledgedAlerts - 1);
      }
      this.alerts.splice(index, 1);
    }
  }

  clearAllAlerts(): void {
    this.alerts.clear();
    this.unacknowledgedAlerts = 0;
  }

  // =============================================================================
  // Performance Monitoring
  // =============================================================================

  private addPerformanceData(data: PerformanceMetrics): void {
    this.performanceHistory.push(data);

    // Limit history size
    if (this.performanceHistory.length > this.maxPerformanceHistory) {
      this.performanceHistory.shift();
    }
  }

  getPerformanceData(minutes: number = 30): PerformanceMetrics[] {
    const cutoff = new Date(Date.now() - minutes * 60 * 1000);
    return this.performanceHistory.filter(data => data.timestamp >= cutoff);
  }

  // =============================================================================
  // Auto-refresh Management
  // =============================================================================

  setAutoRefresh(enabled: boolean, interval?: number): void {
    this.autoRefreshEnabled = enabled;
    if (interval) {
      this.refreshInterval = interval;
    }

    if (enabled) {
      this.startAutoRefresh();
    } else {
      this.stopAutoRefresh();
    }
  }

  private startAutoRefresh(): void {
    this.stopAutoRefresh();

    this.refreshTimer = setInterval(() => {
      this.refreshHealth();
    }, this.refreshInterval);
  }

  private stopAutoRefresh(): void {
    if (this.refreshTimer) {
      clearInterval(this.refreshTimer);
      this.refreshTimer = null;
    }
  }

  // =============================================================================
  // Computed Properties
  // =============================================================================

  get isConnected(): boolean {
    return this.connectionStatus.api === 'connected' || this.connectionStatus.websocket === 'connected';
  }

  get healthyComponents(): ComponentHealth[] {
    return Array.from(this.components.values()).filter(c => c.status === 'healthy');
  }

  get unhealthyComponents(): ComponentHealth[] {
    return Array.from(this.components.values()).filter(c => c.status !== 'healthy');
  }

  get criticalAlerts(): SystemAlert[] {
    return this.alerts.filter(a => a.type === 'critical' && !a.acknowledged);
  }

  get errorAlerts(): SystemAlert[] {
    return this.alerts.filter(a => a.type === 'error' && !a.acknowledged);
  }

  get warningAlerts(): SystemAlert[] {
    return this.alerts.filter(a => a.type === 'warning' && !a.acknowledged);
  }

  get recentAlerts(): SystemAlert[] {
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
    return this.alerts.filter(a => a.timestamp >= oneHourAgo);
  }

  get systemStatus(): {
    overall: 'excellent' | 'good' | 'fair' | 'poor';
    details: string;
    issues: string[];
  } {
    const issues: string[] = [];
    let overall: 'excellent' | 'good' | 'fair' | 'poor' = 'excellent';

    // Check overall health
    if (this.overallHealth === 'unhealthy') {
      overall = 'poor';
      issues.push('System health is unhealthy');
    } else if (this.overallHealth === 'degraded') {
      overall = 'fair';
      issues.push('System health is degraded');
    }

    // Check connection status
    if (!this.isConnected) {
      overall = 'poor';
      issues.push('Not connected to backend services');
    }

    // Check for critical alerts
    if (this.criticalAlerts.length > 0) {
      overall = 'poor';
      issues.push(`${this.criticalAlerts.length} critical alerts`);
    }

    // Check for error alerts
    if (this.errorAlerts.length > 0) {
      if (overall === 'excellent') overall = 'fair';
      issues.push(`${this.errorAlerts.length} error alerts`);
    }

    // Check performance metrics
    if (this.metrics) {
      if (this.metrics.cpu_usage > 90) {
        if (overall === 'excellent') overall = 'fair';
        issues.push('High CPU usage');
      }
      if (this.metrics.memory_usage > 90) {
        if (overall === 'excellent') overall = 'fair';
        issues.push('High memory usage');
      }
    }

    let details = '';
    switch (overall) {
      case 'excellent':
        details = 'All systems operating normally';
        break;
      case 'good':
        details = 'Minor issues detected';
        break;
      case 'fair':
        details = 'Some issues require attention';
        break;
      case 'poor':
        details = 'Critical issues detected';
        break;
    }

    return { overall, details, issues };
  }

  get uptimeFormatted(): string {
    const seconds = this.uptime;
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else {
      return `${minutes}m`;
    }
  }

  // =============================================================================
  // Helper Methods
  // =============================================================================

  private getAlertTitle(code: string): string {
    const titles: Record<string, string> = {
      'HW_CAMERA_UNAVAILABLE': 'Camera Unavailable',
      'HW_PROJECTOR_UNAVAILABLE': 'Projector Unavailable',
      'HW_CALIBRATION_FAILED': 'Calibration Failed',
      'PROC_VISION_FAILED': 'Vision Processing Failed',
      'PROC_TRACKING_LOST': 'Ball Tracking Lost',
      'SYS_INTERNAL_ERROR': 'System Error',
      'SYS_SERVICE_UNAVAILABLE': 'Service Unavailable',
      'WS_CONNECTION_FAILED': 'WebSocket Connection Failed',
      'WS_STREAM_UNAVAILABLE': 'Stream Unavailable',
    };

    return titles[code] || 'System Alert';
  }

  getComponentHealth(name: string): ComponentHealth | null {
    return this.components.get(name) || null;
  }

  isComponentHealthy(name: string): boolean {
    const component = this.getComponentHealth(name);
    return component ? component.status === 'healthy' : false;
  }

  clearError(): void {
    this.error = null;
  }

  // =============================================================================
  // Store Lifecycle
  // =============================================================================

  reset(): void {
    this.overallHealth = 'healthy';
    this.components.clear();
    this.metrics = null;
    this.version = '';
    this.buildDate = null;
    this.uptime = 0;
    this.connectionStatus = {
      api: 'disconnected',
      websocket: 'disconnected',
      lastApiCall: null,
      lastWebSocketMessage: null,
    };
    this.alerts.clear();
    this.unacknowledgedAlerts = 0;
    this.performanceHistory.clear();
    this.isLoading = false;
    this.error = null;
    this.lastHealthCheck = null;
  }

  destroy(): void {
    this.stopAutoRefresh();
    this.reset();
  }
}
