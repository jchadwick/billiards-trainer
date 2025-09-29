/**
 * Accessibility Settings Component
 * Provides user interface for managing accessibility preferences
 */

import React, { useState, useCallback } from 'react'
import { observer } from 'mobx-react-lite'
import { useAccessibility } from '../../hooks/useAccessibility'
import { createFormControlAria, createButtonAria, KEYBOARD_SHORTCUTS } from '../../utils/accessibility'
import { Modal } from '../ui/Modal'
import { Button } from '../ui/Button'
import { Card } from '../ui/Card'

export interface AccessibilitySettingsProps {
  isOpen: boolean
  onClose: () => void
  className?: string
}

export const AccessibilitySettings: React.FC<AccessibilitySettingsProps> = observer(({
  isOpen,
  onClose,
  className = '',
}) => {
  const {
    reducedMotion,
    highContrast,
    fontSize,
    screenReaderOptimized,
    keyboardNavigation,
    colorBlindnessMode,
    setReducedMotion,
    setHighContrast,
    setFontSize,
    setScreenReaderOptimized,
    setKeyboardNavigation,
    setColorBlindnessMode,
    resetToDefaults,
    announce,
  } = useAccessibility()

  const [showKeyboardShortcuts, setShowKeyboardShortcuts] = useState(false)

  const handleSave = useCallback(() => {
    announce('Accessibility settings saved')
    onClose()
  }, [announce, onClose])

  const handleReset = useCallback(() => {
    resetToDefaults()
    announce('Accessibility settings reset to defaults')
  }, [resetToDefaults, announce])

  const handleClose = useCallback(() => {
    onClose()
  }, [onClose])

  const fontSizeOptions = [
    { value: 'small', label: 'Small (14px)' },
    { value: 'medium', label: 'Medium (16px)' },
    { value: 'large', label: 'Large (18px)' },
    { value: 'extra-large', label: 'Extra Large (20px)' },
  ] as const

  const colorBlindnessOptions = [
    { value: 'none', label: 'None' },
    { value: 'protanopia', label: 'Protanopia (Red-blind)' },
    { value: 'deuteranopia', label: 'Deuteranopia (Green-blind)' },
    { value: 'tritanopia', label: 'Tritanopia (Blue-blind)' },
  ] as const

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      title="Accessibility Settings"
      className={className}
      size="lg"
    >
      <div className="space-y-6">
        {/* Visual Preferences */}
        <Card>
          <div className="p-4">
            <h3 className="text-lg font-semibold mb-4">Visual Preferences</h3>

            <div className="space-y-4">
              {/* High Contrast */}
              <div className="flex items-center justify-between">
                <div>
                  <label
                    htmlFor="high-contrast"
                    className="text-sm font-medium text-gray-700 dark:text-gray-300"
                  >
                    High Contrast Mode
                  </label>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Increases contrast for better visibility
                  </p>
                </div>
                <input
                  id="high-contrast"
                  type="checkbox"
                  checked={highContrast}
                  onChange={(e) => setHighContrast(e.target.checked)}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  {...createFormControlAria({
                    label: 'Enable high contrast mode for better visibility'
                  })}
                />
              </div>

              {/* Font Size */}
              <div>
                <label
                  htmlFor="font-size"
                  className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
                >
                  Font Size
                </label>
                <select
                  id="font-size"
                  value={fontSize}
                  onChange={(e) => setFontSize(e.target.value as any)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  {...createFormControlAria({
                    label: 'Select font size preference'
                  })}
                >
                  {fontSizeOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Color Blindness Support */}
              <div>
                <label
                  htmlFor="color-blindness"
                  className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
                >
                  Color Blindness Support
                </label>
                <select
                  id="color-blindness"
                  value={colorBlindnessMode}
                  onChange={(e) => setColorBlindnessMode(e.target.value as any)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  {...createFormControlAria({
                    label: 'Select color blindness support option'
                  })}
                >
                  {colorBlindnessOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </Card>

        {/* Motion Preferences */}
        <Card>
          <div className="p-4">
            <h3 className="text-lg font-semibold mb-4">Motion Preferences</h3>

            <div className="flex items-center justify-between">
              <div>
                <label
                  htmlFor="reduced-motion"
                  className="text-sm font-medium text-gray-700 dark:text-gray-300"
                >
                  Reduce Motion
                </label>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Minimizes animations and transitions
                </p>
              </div>
              <input
                id="reduced-motion"
                type="checkbox"
                checked={reducedMotion}
                onChange={(e) => setReducedMotion(e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                {...createFormControlAria({
                  label: 'Enable reduced motion to minimize animations'
                })}
              />
            </div>
          </div>
        </Card>

        {/* Navigation Preferences */}
        <Card>
          <div className="p-4">
            <h3 className="text-lg font-semibold mb-4">Navigation Preferences</h3>

            <div className="space-y-4">
              {/* Keyboard Navigation */}
              <div className="flex items-center justify-between">
                <div>
                  <label
                    htmlFor="keyboard-navigation"
                    className="text-sm font-medium text-gray-700 dark:text-gray-300"
                  >
                    Enhanced Keyboard Navigation
                  </label>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Enables additional keyboard shortcuts and focus indicators
                  </p>
                </div>
                <input
                  id="keyboard-navigation"
                  type="checkbox"
                  checked={keyboardNavigation}
                  onChange={(e) => setKeyboardNavigation(e.target.checked)}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  {...createFormControlAria({
                    label: 'Enable enhanced keyboard navigation features'
                  })}
                />
              </div>

              {/* Screen Reader Optimization */}
              <div className="flex items-center justify-between">
                <div>
                  <label
                    htmlFor="screen-reader"
                    className="text-sm font-medium text-gray-700 dark:text-gray-300"
                  >
                    Screen Reader Optimization
                  </label>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Optimizes interface for screen readers
                  </p>
                </div>
                <input
                  id="screen-reader"
                  type="checkbox"
                  checked={screenReaderOptimized}
                  onChange={(e) => setScreenReaderOptimized(e.target.checked)}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  {...createFormControlAria({
                    label: 'Enable screen reader optimization features'
                  })}
                />
              </div>
            </div>
          </div>
        </Card>

        {/* Keyboard Shortcuts */}
        <Card>
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Keyboard Shortcuts</h3>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowKeyboardShortcuts(!showKeyboardShortcuts)}
                {...createButtonAria({
                  label: showKeyboardShortcuts ? 'Hide keyboard shortcuts' : 'Show keyboard shortcuts',
                  expanded: showKeyboardShortcuts
                })}
              >
                {showKeyboardShortcuts ? 'Hide' : 'Show'} Shortcuts
              </Button>
            </div>

            {showKeyboardShortcuts && (
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {Object.entries(KEYBOARD_SHORTCUTS).map(([shortcut, description]) => (
                  <div key={shortcut} className="flex justify-between items-center py-1 border-b border-gray-200 dark:border-gray-700 last:border-b-0">
                    <code className="text-sm bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">
                      {shortcut}
                    </code>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {description}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Card>

        {/* Action Buttons */}
        <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button
            variant="outline"
            onClick={handleReset}
            {...createButtonAria({
              label: 'Reset accessibility settings to default values'
            })}
          >
            Reset to Defaults
          </Button>

          <Button
            variant="outline"
            onClick={handleClose}
            {...createButtonAria({
              label: 'Cancel changes and close accessibility settings'
            })}
          >
            Cancel
          </Button>

          <Button
            variant="primary"
            onClick={handleSave}
            {...createButtonAria({
              label: 'Save accessibility settings and close dialog'
            })}
          >
            Save Settings
          </Button>
        </div>
      </div>
    </Modal>
  )
})

/**
 * Accessibility Settings Toggle Button
 */
export interface AccessibilityToggleProps {
  className?: string
  onOpen: () => void
}

export const AccessibilityToggle: React.FC<AccessibilityToggleProps> = ({
  className = '',
  onOpen,
}) => {
  const handleClick = useCallback(() => {
    onOpen()
  }, [onOpen])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      onOpen()
    }
  }, [onOpen])

  return (
    <button
      type="button"
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      className={`inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${className}`}
      {...createButtonAria({
        label: 'Open accessibility settings dialog'
      })}
    >
      <svg
        className="w-4 h-4 mr-2"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 3c.552 0 1 .448 1 1v4c0 .552-.448 1-1 1s-1-.448-1-1V4c0-.552.448-1 1-1zm0 14c.552 0 1 .448 1 1v3c0 .552-.448 1-1 1s-1-.448-1-1v-3c0-.552.448-1 1-1zm9-5c0 .552-.448 1-1 1h-4c-.552 0-1-.448-1-1s.448-1 1-1h4c.552 0 1 .448 1 1zM8 12c0 .552-.448 1-1 1H4c-.552 0-1-.448-1-1s.448-1 1-1h3c.552 0 1 .448 1 1zm10.657-5.657c.39-.39 1.024-.39 1.414 0 .39.39.39 1.024 0 1.414l-2.828 2.828c-.39.39-1.024.39-1.414 0-.39-.39-.39-1.024 0-1.414l2.828-2.828zm-12.02 12.02c.39-.39 1.024-.39 1.414 0 .39.39.39 1.024 0 1.414l-2.829 2.829c-.39.39-1.024.39-1.414 0-.39-.39-.39-1.024 0-1.414l2.829-2.829zm12.02 0c.39.39.39 1.024 0 1.414l-2.828 2.828c-.39.39-1.024.39-1.414 0-.39-.39-.39-1.024 0-1.414l2.828-2.828zm-12.02-12.02c.39.39.39 1.024 0 1.414L5.808 8.606c-.39.39-1.024.39-1.414 0-.39-.39-.39-1.024 0-1.414L7.222 4.364c.39-.39 1.024-.39 1.414 0z"
        />
      </svg>
      Accessibility
    </button>
  )
}
