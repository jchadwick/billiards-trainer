import React from 'react'
import { observer } from 'mobx-react-lite'
import { useUIStore } from '../../hooks/useStores'
import { ConnectionStatus, UserMenu, NotificationCenter } from '../navigation'
import { Button } from '../ui/Button'

export interface HeaderProps {
  className?: string
}

export const Header = observer<HeaderProps>(({ className = '' }) => {
  const uiStore = useUIStore()

  const handleToggleSidebar = () => {
    uiStore.toggleSidebar()
  }

  const handleToggleMobileSidebar = () => {
    uiStore.toggleMobileSidebar()
  }

  return (
    <header className={`bg-white dark:bg-secondary-800 border-b border-secondary-200 dark:border-secondary-700 ${className}`}>
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Left side */}
          <div className="flex items-center space-x-4">
            {/* Mobile menu button */}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleToggleMobileSidebar}
              className="lg:hidden"
              aria-label="Open mobile menu"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </Button>

            {/* Desktop sidebar toggle */}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleToggleSidebar}
              className="hidden lg:flex"
              aria-label="Toggle sidebar"
            >
              {uiStore.sidebarCollapsed ? (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                </svg>
              )}
            </Button>

            {/* Logo and title */}
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>
              </div>
              <div className="hidden sm:block">
                <h1 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">
                  Billiards Trainer
                </h1>
              </div>
            </div>
          </div>

          {/* Center - Breadcrumb or search could go here */}
          <div className="flex-1 max-w-lg mx-4">
            {/* Future: Search bar or breadcrumb navigation */}
          </div>

          {/* Right side */}
          <div className="flex items-center space-x-2">
            {/* Connection Status */}
            <ConnectionStatus
              className="hidden sm:flex"
              showText={false}
              size="md"
            />

            {/* Notifications */}
            <NotificationCenter />

            {/* User Menu */}
            <UserMenu />
          </div>
        </div>
      </div>

      {/* Mobile connection status */}
      <div className="sm:hidden px-4 pb-2">
        <ConnectionStatus showText={true} size="sm" />
      </div>
    </header>
  )
})
