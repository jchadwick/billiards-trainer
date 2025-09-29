import React, { useState } from 'react'
import { observer } from 'mobx-react-lite'
import { Card, CardContent, Button } from '../ui'

export interface ConfigSection {
  id: string
  label: string
  icon: React.ReactNode
  component: React.ComponentType
  description?: string
}

export interface ConfigurationLayoutProps {
  sections: ConfigSection[]
  activeSection?: string
  onSectionChange?: (sectionId: string) => void
  children?: React.ReactNode
}

export const ConfigurationLayout = observer<ConfigurationLayoutProps>(({
  sections,
  activeSection: controlledActiveSection,
  onSectionChange,
  children
}) => {
  const [localActiveSection, setLocalActiveSection] = useState(sections[0]?.id || '')

  const activeSection = controlledActiveSection !== undefined ? controlledActiveSection : localActiveSection
  const isControlled = controlledActiveSection !== undefined

  const handleSectionChange = (sectionId: string) => {
    if (!isControlled) {
      setLocalActiveSection(sectionId)
    }
    onSectionChange?.(sectionId)
  }

  const activeSectionData = sections.find(section => section.id === activeSection)
  const ActiveComponent = activeSectionData?.component

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            System Configuration
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Configure your billiards training system settings and preferences.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar Navigation */}
          <div className="lg:col-span-1">
            <Card className="sticky top-6">
              <CardContent padding="sm">
                <nav className="space-y-1">
                  {sections.map((section) => (
                    <button
                      key={section.id}
                      onClick={() => handleSectionChange(section.id)}
                      className={`
                        w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors
                        ${activeSection === section.id
                          ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200'
                          : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                        }
                      `}
                    >
                      <span className="mr-3 flex-shrink-0">
                        {section.icon}
                      </span>
                      <span className="truncate">{section.label}</span>
                    </button>
                  ))}
                </nav>
              </CardContent>
            </Card>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            <Card>
              <CardContent>
                {/* Section Header */}
                {activeSectionData && (
                  <div className="mb-6 pb-4 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex items-center mb-2">
                      <span className="mr-3 text-blue-600 dark:text-blue-400">
                        {activeSectionData.icon}
                      </span>
                      <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                        {activeSectionData.label}
                      </h2>
                    </div>
                    {activeSectionData.description && (
                      <p className="text-gray-600 dark:text-gray-400">
                        {activeSectionData.description}
                      </p>
                    )}
                  </div>
                )}

                {/* Section Content */}
                <div className="space-y-6">
                  {ActiveComponent ? (
                    <ActiveComponent />
                  ) : (
                    children
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
})

// Configuration section icons
export const ConfigIcons = {
  System: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
    </svg>
  ),
  Camera: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
    </svg>
  ),
  Table: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2 2z" />
    </svg>
  ),
  Physics: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
  Projector: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m3 0H4a1 1 0 00-1 1v10a1 1 0 001 1h16a1 1 0 001-1V5a1 1 0 00-1-1z" />
    </svg>
  ),
  Vision: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
    </svg>
  ),
  Calibration: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  Profiles: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
    </svg>
  ),
  ImportExport: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z" />
    </svg>
  )
}
