import React from 'react'
import { observer } from 'mobx-react-lite'
import { useUIStore } from '../../hooks/useStores'

export interface MainContentProps {
  children: React.ReactNode
  className?: string
  padding?: 'none' | 'sm' | 'md' | 'lg'
  maxWidth?: 'none' | 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full'
}

const paddingClasses = {
  none: '',
  sm: 'p-4',
  md: 'p-6',
  lg: 'p-8',
}

const maxWidthClasses = {
  none: '',
  sm: 'max-w-sm mx-auto',
  md: 'max-w-md mx-auto',
  lg: 'max-w-lg mx-auto',
  xl: 'max-w-xl mx-auto',
  '2xl': 'max-w-2xl mx-auto',
  full: 'max-w-full',
}

export const MainContent = observer<MainContentProps>(({
  children,
  className = '',
  padding = 'md',
  maxWidth = 'full',
}) => {
  const uiStore = useUIStore()

  const paddingClass = paddingClasses[padding]
  const maxWidthClass = maxWidthClasses[maxWidth]

  // Calculate left margin based on sidebar state
  const getMainStyles = () => {
    if (uiStore.sidebarCollapsed) {
      return 'lg:ml-16' // Collapsed sidebar width
    }
    return 'lg:ml-64' // Full sidebar width
  }

  return (
    <main
      className={`min-h-screen bg-gray-50 dark:bg-gray-900 transition-all duration-300 ${getMainStyles()} ${className}`}
    >
      <div className={`${maxWidthClass} ${paddingClass}`}>
        {children}
      </div>
    </main>
  )
})

export interface PageContainerProps {
  children: React.ReactNode
  title?: string
  description?: string
  actions?: React.ReactNode
  className?: string
}

export const PageContainer = observer<PageContainerProps>(({
  children,
  title,
  description,
  actions,
  className = '',
}) => {
  return (
    <div className={`${className}`}>
      {/* Page Header */}
      {(title || description || actions) && (
        <div className="mb-6">
          <div className="flex items-center justify-between">
            <div>
              {title && (
                <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {title}
                </h1>
              )}
              {description && (
                <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                  {description}
                </p>
              )}
            </div>
            {actions && (
              <div className="flex items-center space-x-3">
                {actions}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Page Content */}
      <div>
        {children}
      </div>
    </div>
  )
})

export interface SectionProps {
  children: React.ReactNode
  title?: string
  description?: string
  className?: string
  headerActions?: React.ReactNode
}

export const Section = observer<SectionProps>(({
  children,
  title,
  description,
  className = '',
  headerActions,
}) => {
  return (
    <section className={`${className}`}>
      {/* Section Header */}
      {(title || description || headerActions) && (
        <div className="mb-4">
          <div className="flex items-center justify-between">
            <div>
              {title && (
                <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {title}
                </h2>
              )}
              {description && (
                <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                  {description}
                </p>
              )}
            </div>
            {headerActions && (
              <div className="flex items-center space-x-2">
                {headerActions}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Section Content */}
      <div>
        {children}
      </div>
    </section>
  )
})
