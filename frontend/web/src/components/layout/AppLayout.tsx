import React from 'react'
import { observer } from 'mobx-react-lite'
import { useUIStore } from '../../hooks/useStores'
import { Header } from './Header'
import { Sidebar } from './Sidebar'
import { MainContent } from './MainContent'
import { Footer } from './Footer'
import { FullScreenLoading } from '../ui/LoadingSpinner'
import { SkipLinks, SkipLinkTarget } from '../accessibility/SkipLinks'
import { AnnouncementRegion } from '../accessibility/AnnouncementRegion'
import { useKeyboardDetection } from '../../hooks/useAccessibility'
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
  const isKeyboardUser = useKeyboardDetection()

  return (
    <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 ${className} ${isKeyboardUser ? 'keyboard-user' : ''}`}>
      {/* Skip Links for keyboard navigation */}
      <SkipLinks />

      {/* Accessibility Announcement Region */}
      <AnnouncementRegion />

      {/* Global Loading Overlay */}
      {uiStore.globalLoading && (
        <FullScreenLoading text={uiStore.loadingText} />
      )}

      {/* Header */}
      <SkipLinkTarget id="main-navigation" as="header">
        <Header />
      </SkipLinkTarget>

      {/* Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex flex-col min-h-screen">
        {/* Main content */}
        <SkipLinkTarget id="main-content" as="main" className="flex-1">
          <MainContent>
            {children}
          </MainContent>
        </SkipLinkTarget>

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
