import React from 'react'
import { observer } from 'mobx-react-lite'
import { useUIStore } from '../../hooks/useStores'
import { NavMenu } from '../navigation'
import type { NavItem } from '../../types'

export interface SidebarProps {
  className?: string
}

// Define navigation items for the billiards trainer
const navigationItems: NavItem[] = [
  {
    id: 'home',
    label: 'Dashboard',
    icon: 'home',
    path: '/',
  },
  {
    id: 'calibration',
    label: 'Calibration',
    icon: 'calibration',
    path: '/calibration',
  },
  {
    id: 'configuration',
    label: 'Configuration',
    icon: 'configuration',
    path: '/configuration',
  },
  {
    id: 'system-management',
    label: 'System Management',
    icon: 'system-management',
    path: '/system-management',
  },
  {
    id: 'diagnostics',
    label: 'Diagnostics',
    icon: 'diagnostics',
    path: '/diagnostics',
  },
]

export const Sidebar = observer<SidebarProps>(({ className = '' }) => {
  const uiStore = useUIStore()

  const handleNavItemClick = (item: NavItem) => {
    // Close mobile sidebar when item is clicked
    if (uiStore.sidebarMobileOpen) {
      uiStore.setMobileSidebarOpen(false)
    }
  }

  return (
    <>
      {/* Desktop Sidebar */}
      <aside
        className={`hidden lg:flex flex-col fixed inset-y-0 left-0 z-50 bg-white dark:bg-secondary-800 border-r border-secondary-200 dark:border-secondary-700 transition-all duration-300 ${
          uiStore.sidebarCollapsed ? 'w-16' : 'w-64'
        } ${className}`}
      >
        <div className="flex-1 flex flex-col pt-5 pb-4 overflow-y-auto">
          <div className="flex-1 px-3 space-y-1">
            <NavMenu
              items={navigationItems}
              collapsed={uiStore.sidebarCollapsed}
              onItemClick={handleNavItemClick}
            />
          </div>

          {/* Footer area */}
          {!uiStore.sidebarCollapsed && (
            <div className="px-3 py-4 border-t border-secondary-200 dark:border-secondary-700">
              <div className="text-xs text-secondary-500 text-center">
                <div>Billiards Trainer v1.0.0</div>
                <div className="mt-1">© 2025 Jess Chadwick</div>
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Mobile Sidebar */}
      {uiStore.sidebarMobileOpen && (
        <>
          {/* Mobile backdrop */}
          <div
            className="lg:hidden fixed inset-0 z-40 bg-black/50"
            onClick={() => uiStore.setMobileSidebarOpen(false)}
            aria-hidden="true"
          />

          {/* Mobile sidebar panel */}
          <aside className="lg:hidden fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-secondary-800 border-r border-secondary-200 dark:border-secondary-700 transform transition-transform duration-300 ease-in-out">
            <div className="flex-1 flex flex-col h-full pt-5 pb-4 overflow-y-auto">
              {/* Mobile header */}
              <div className="flex items-center justify-between px-4 mb-6">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                    </svg>
                  </div>
                  <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">
                    Billiards Trainer
                  </h2>
                </div>
                <button
                  onClick={() => uiStore.setMobileSidebarOpen(false)}
                  className="p-2 rounded-md text-secondary-400 hover:text-secondary-600 hover:bg-secondary-100 dark:hover:bg-secondary-700 focus:outline-none focus:ring-2 focus:ring-primary-500"
                  aria-label="Close sidebar"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Mobile navigation */}
              <div className="flex-1 px-3 space-y-1">
                <NavMenu
                  items={navigationItems}
                  collapsed={false}
                  onItemClick={handleNavItemClick}
                />
              </div>

              {/* Mobile footer */}
              <div className="px-3 py-4 border-t border-secondary-200 dark:border-secondary-700">
                <div className="text-xs text-secondary-500 text-center">
                  <div>Billiards Trainer v1.0.0</div>
                  <div className="mt-1">© 2025 Jess Chadwick</div>
                </div>
              </div>
            </div>
          </aside>
        </>
      )}
    </>
  )
})
