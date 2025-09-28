import React, { useState, useRef, useEffect } from 'react'
import { observer } from 'mobx-react-lite'
import { useUIStore } from '../../hooks/useStores'
import { Button } from '../ui/Button'
import type { Notification } from '../../types'

export interface NotificationCenterProps {
  className?: string
}

const NotificationIcon: React.FC<{ type: Notification['type']; className?: string }> = ({
  type,
  className = 'w-5 h-5',
}) => {
  const iconClass = `${className} flex-shrink-0`

  switch (type) {
    case 'success':
      return (
        <svg className={`${iconClass} text-success-500`} fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
            clipRule="evenodd"
          />
        </svg>
      )
    case 'error':
      return (
        <svg className={`${iconClass} text-error-500`} fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
            clipRule="evenodd"
          />
        </svg>
      )
    case 'warning':
      return (
        <svg className={`${iconClass} text-warning-500`} fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
            clipRule="evenodd"
          />
        </svg>
      )
    case 'info':
    default:
      return (
        <svg className={`${iconClass} text-primary-500`} fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
            clipRule="evenodd"
          />
        </svg>
      )
  }
}

const NotificationItem: React.FC<{
  notification: Notification
  onRead: (id: string) => void
  onRemove: (id: string) => void
}> = observer(({ notification, onRead, onRemove }) => {
  const handleMarkRead = () => {
    if (!notification.read) {
      onRead(notification.id)
    }
  }

  const handleRemove = () => {
    onRemove(notification.id)
  }

  return (
    <div
      className={`p-4 border-b border-secondary-200 dark:border-secondary-700 hover:bg-secondary-50 dark:hover:bg-secondary-800/50 ${
        !notification.read ? 'bg-primary-50/50 dark:bg-primary-900/10' : ''
      }`}
    >
      <div className="flex items-start space-x-3">
        <NotificationIcon type={notification.type} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
              {notification.title}
            </h4>
            <button
              onClick={handleRemove}
              className="ml-2 text-secondary-400 hover:text-secondary-600 dark:hover:text-secondary-300"
              aria-label="Remove notification"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <p className="mt-1 text-sm text-secondary-600 dark:text-secondary-400">
            {notification.message}
          </p>
          <div className="mt-2 flex items-center justify-between">
            <span className="text-xs text-secondary-500">
              {notification.timestamp.toLocaleString()}
            </span>
            {!notification.read && (
              <button
                onClick={handleMarkRead}
                className="text-xs text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300"
              >
                Mark as read
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
})

export const NotificationCenter = observer<NotificationCenterProps>(({ className = '' }) => {
  const [isOpen, setIsOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const uiStore = useUIStore()

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleMarkRead = (id: string) => {
    uiStore.markNotificationRead(id)
  }

  const handleRemove = (id: string) => {
    uiStore.removeNotification(id)
  }

  const handleClearAll = () => {
    uiStore.clearAllNotifications()
    setIsOpen(false)
  }

  return (
    <div className={`relative ${className}`} ref={menuRef}>
      {/* Notification Bell */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative p-2 rounded-md text-secondary-700 hover:bg-secondary-100 hover:text-secondary-900 dark:text-secondary-300 dark:hover:bg-secondary-800 dark:hover:text-secondary-100 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
        aria-label="Open notifications"
      >
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
          />
        </svg>
        {uiStore.unreadNotificationCount > 0 && (
          <span className="absolute -top-1 -right-1 bg-error-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
            {uiStore.unreadNotificationCount > 9 ? '9+' : uiStore.unreadNotificationCount}
          </span>
        )}
      </button>

      {/* Dropdown Panel */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-secondary-800 rounded-md shadow-lg border border-secondary-200 dark:border-secondary-700 z-50 max-h-96">
          {/* Header */}
          <div className="px-4 py-3 border-b border-secondary-200 dark:border-secondary-700">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                Notifications
              </h3>
              {uiStore.notifications.length > 0 && (
                <Button variant="ghost" size="sm" onClick={handleClearAll}>
                  Clear all
                </Button>
              )}
            </div>
          </div>

          {/* Notifications List */}
          <div className="max-h-64 overflow-y-auto">
            {uiStore.notifications.length === 0 ? (
              <div className="p-8 text-center">
                <svg className="w-12 h-12 mx-auto text-secondary-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1}
                    d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
                  />
                </svg>
                <p className="text-sm text-secondary-500">No notifications</p>
              </div>
            ) : (
              uiStore.notifications.map((notification) => (
                <NotificationItem
                  key={notification.id}
                  notification={notification}
                  onRead={handleMarkRead}
                  onRemove={handleRemove}
                />
              ))
            )}
          </div>
        </div>
      )}
    </div>
  )
})
