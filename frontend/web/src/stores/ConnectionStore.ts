import { makeAutoObservable } from 'mobx'
import type { ConnectionState } from '../types'

export class ConnectionStore {
  state: ConnectionState = {
    isConnected: false,
    isConnecting: false,
  }

  // Connection retry settings
  private retryCount = 0
  private maxRetries = 5
  private retryDelay = 1000 // Start with 1 second
  private retryTimer?: NodeJS.Timeout

  constructor() {
    makeAutoObservable(this)
    this.initializeConnection()
  }

  private initializeConnection() {
    // Start connection attempt
    this.connect()
  }

  async connect() {
    if (this.state.isConnecting || this.state.isConnected) {
      return
    }

    this.state.isConnecting = true
    this.state.error = undefined

    try {
      // Test actual connection to backend
      await this.testConnection()

      this.state.isConnected = true
      this.state.lastConnected = new Date()
      this.retryCount = 0

      console.log('Connected to billiards trainer backend')
    } catch (error) {
      this.state.isConnected = false
      this.state.error = error instanceof Error ? error.message : 'Connection failed'

      console.error('Connection failed:', error)
      this.scheduleRetry()
    } finally {
      this.state.isConnecting = false
    }
  }

  disconnect() {
    this.state.isConnected = false
    this.state.isConnecting = false
    this.state.error = undefined

    if (this.retryTimer) {
      clearTimeout(this.retryTimer)
      this.retryTimer = undefined
    }
  }

  private async testConnection(): Promise<void> {
    try {
      const { apiClient } = await import('../api/client');
      const response = await apiClient.getHealth();

      if (!response.success) {
        throw new Error(response.error || 'Health check failed');
      }
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Connection test failed');
    }
  }

  private scheduleRetry() {
    if (this.retryCount >= this.maxRetries) {
      console.error('Max retry attempts reached')
      return
    }

    const delay = this.retryDelay * Math.pow(2, this.retryCount) // Exponential backoff
    this.retryCount++

    console.log(`Scheduling reconnection attempt ${this.retryCount} in ${delay}ms`)

    this.retryTimer = setTimeout(() => {
      this.connect()
    }, delay)
  }

  // Getters for computed values
  get connectionStatus(): 'connected' | 'connecting' | 'disconnected' | 'error' {
    if (this.state.error) return 'error'
    if (this.state.isConnecting) return 'connecting'
    if (this.state.isConnected) return 'connected'
    return 'disconnected'
  }

  get connectionStatusText(): string {
    switch (this.connectionStatus) {
      case 'connected':
        return 'Connected'
      case 'connecting':
        return 'Connecting...'
      case 'error':
        return `Error: ${this.state.error}`
      case 'disconnected':
        return 'Disconnected'
      default:
        return 'Unknown'
    }
  }

  get connectionColor(): string {
    switch (this.connectionStatus) {
      case 'connected':
        return 'text-success-600'
      case 'connecting':
        return 'text-warning-600'
      case 'error':
        return 'text-error-600'
      case 'disconnected':
        return 'text-secondary-500'
      default:
        return 'text-secondary-500'
    }
  }

  // Monitoring dashboard methods
  async refreshStatus(): Promise<void> {
    // Refresh connection status by attempting to connect if disconnected
    if (!this.state.isConnected && !this.state.isConnecting) {
      await this.connect();
    }
  }

  getConnectionMetrics() {
    return {
      status: this.connectionStatus,
      isConnected: this.state.isConnected,
      isConnecting: this.state.isConnecting,
      lastConnected: this.state.lastConnected,
      error: this.state.error,
      retryCount: this.retryCount,
    };
  }

  // Get detailed connection info from backend
  async getDetailedConnectionInfo(): Promise<any> {
    try {
      const { apiClient } = await import('../api/client');
      const [healthResponse, metricsResponse] = await Promise.all([
        apiClient.getHealth(true, false),
        apiClient.getMetrics()
      ]);

      return {
        health: healthResponse.data,
        metrics: metricsResponse.data,
        connectionState: this.getConnectionMetrics(),
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : 'Failed to get connection info',
        connectionState: this.getConnectionMetrics(),
        timestamp: new Date().toISOString()
      };
    }
  }

  // Test connection without changing state
  async testConnectionHealth(): Promise<boolean> {
    try {
      await this.testConnection();
      return true;
    } catch (error) {
      return false;
    }
  }
}
