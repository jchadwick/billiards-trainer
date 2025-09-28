import React from 'react'
import { observer } from 'mobx-react-lite'
import { useUIStore } from '../../hooks/useStores'
import { Header } from './Header'
import { Sidebar } from './Sidebar'
import { MainContent } from './MainContent'
import { Footer } from './Footer'
import { FullScreenLoading } from '../ui/LoadingSpinner'
import type { SystemInfo } from '../../types'

export interface AppLayoutProps {
  children: React.ReactNode
  className?: string
  systemInfo?: SystemInfo
  showFooter?: boolean
}

export const AppLayout = observer<AppLayoutProps>(({
  children,
  className = '',
  systemInfo,
  showFooter = true,
}) => {
  const uiStore = useUIStore()

  return (
    <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 ${className}`}>
      {/* Global Loading Overlay */}
      {uiStore.globalLoading && (
        <FullScreenLoading text={uiStore.loadingText} />
      )}

      {/* Header */}
      <Header />

      {/* Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex flex-col min-h-screen">
        {/* Main content */}
        <div className="flex-1">
          <MainContent>
            {children}
          </MainContent>
        </div>

        {/* Footer */}
        {showFooter && (
          <div className={`${uiStore.sidebarCollapsed ? 'lg:ml-16' : 'lg:ml-64'} transition-all duration-300`}>
            <Footer systemInfo={systemInfo} />
          </div>
        )}
      </div>
    </div>
  )
})
