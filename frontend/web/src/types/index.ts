// Common types used throughout the application

export interface ConnectionState {
  isConnected: boolean
  isConnecting: boolean
  lastConnected?: Date
  error?: string
}

export interface Notification {
  id: string
  type: 'info' | 'success' | 'warning' | 'error'
  title: string
  message: string
  timestamp: Date
  read: boolean
  autoHide?: boolean
  duration?: number
}

export interface NavItem {
  id: string
  label: string
  icon?: string
  path: string
  badge?: string | number
  disabled?: boolean
  children?: NavItem[]
}

export interface SystemInfo {
  version: string
  buildDate: string
  environment: string
  apiVersion?: string
}

// Re-export types from other modules
export * from './api'
export * from './video'
export * from './monitoring'
