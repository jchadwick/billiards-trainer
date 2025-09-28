import React from 'react'
import { observer } from 'mobx-react-lite'
import { useSystemStore } from '../../hooks/useStores'

export interface ConnectionStatusProps {
  className?: string
  showText?: boolean
  size?: 'sm' | 'md' | 'lg'
}

const iconSizes = {
  sm: 'w-3 h-3',
  md: 'w-4 h-4',
  lg: 'w-5 h-5',
}

const ConnectionIcon: React.FC<{ status: string; className: string }> = ({ status, className }) => {
  switch (status) {
    case 'connected':
      return (
        <svg className={className} fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
            clipRule="evenodd"
          />
        </svg>
      )
    case 'connecting':
      return (
        <svg className={`${className} animate-spin`} fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      )
    case 'error':
      return (
        <svg className={className} fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
            clipRule="evenodd"
          />
        </svg>
      )
    case 'disconnected':
    default:
      return (
        <svg className={className} fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
            clipRule="evenodd"
          />
        </svg>
      )
  }
}

export const ConnectionStatus = observer<ConnectionStatusProps>(({
  className = '',
  showText = true,
  size = 'md',
}) => {
  const systemStore = useSystemStore()
  const iconSize = iconSizes[size]

  // Map system status to our connection status
  const connectionStatus = systemStore.status.isConnected ? 'connected' : 'disconnected'
  const connectionStatusText = systemStore.status.isConnected ? 'Connected' : 'Disconnected'
  const connectionColor = systemStore.status.isConnected ? 'text-success-600' : 'text-error-600'

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <div className={`flex items-center ${connectionColor}`}>
        <ConnectionIcon status={connectionStatus} className={iconSize} />
        {showText && (
          <span className="ml-2 text-sm font-medium">
            {connectionStatusText}
          </span>
        )}
      </div>
      {systemStore.status.lastHeartbeat && systemStore.status.isConnected && (
        <span className="text-xs text-secondary-500">
          Connected at {systemStore.status.lastHeartbeat.toLocaleTimeString()}
        </span>
      )}
    </div>
  )
})
