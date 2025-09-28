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
      // TODO: Replace with actual connection logic
      // For now, simulate connection attempt
      await this.simulateConnection()

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

  private async simulateConnection(): Promise<void> {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000))

    // Simulate occasional connection failures for testing
    if (Math.random() < 0.1) {
      throw new Error('Network timeout')
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
}
