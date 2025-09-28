import React, { useState, useRef, useEffect } from 'react'
import { observer } from 'mobx-react-lite'
import { useUIStore } from '../../hooks/useStores'
import { Button } from '../ui/Button'

export interface UserMenuProps {
  className?: string
}

export const UserMenu = observer<UserMenuProps>(({ className = '' }) => {
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

  const handleThemeChange = (theme: 'light' | 'dark' | 'system') => {
    uiStore.setTheme(theme)
    setIsOpen(false)
  }

  const themeOptions = [
    { value: 'light', label: 'Light', icon: '☀️' },
    { value: 'dark', label: 'Dark', icon: '🌙' },
    { value: 'system', label: 'System', icon: '💻' },
  ] as const

  return (
    <div className={`relative ${className}`} ref={menuRef}>
      {/* Avatar/Menu Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 p-2 rounded-md text-secondary-700 hover:bg-secondary-100 hover:text-secondary-900 dark:text-secondary-300 dark:hover:bg-secondary-800 dark:hover:text-secondary-100 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
        aria-expanded={isOpen}
        aria-haspopup="true"
      >
        {/* User Avatar */}
        <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center text-white text-sm font-medium">
          U
        </div>
        <span className="hidden md:block text-sm font-medium">User</span>
        <svg
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-secondary-800 rounded-md shadow-lg border border-secondary-200 dark:border-secondary-700 z-50">
          <div className="py-1">
            {/* User Info */}
            <div className="px-4 py-2 text-sm text-secondary-500 border-b border-secondary-200 dark:border-secondary-700">
              <div className="font-medium text-secondary-900 dark:text-secondary-100">Guest User</div>
              <div className="text-xs">guest@example.com</div>
            </div>

            {/* Theme Selection */}
            <div className="px-4 py-2 border-b border-secondary-200 dark:border-secondary-700">
              <div className="text-xs font-medium text-secondary-500 uppercase tracking-wider mb-2">
                Theme
              </div>
              <div className="space-y-1">
                {themeOptions.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => handleThemeChange(option.value)}
                    className={`w-full flex items-center px-2 py-1 text-sm rounded transition-colors ${
                      uiStore.theme === option.value
                        ? 'bg-primary-100 text-primary-900 dark:bg-primary-900 dark:text-primary-100'
                        : 'text-secondary-700 hover:bg-secondary-100 dark:text-secondary-300 dark:hover:bg-secondary-700'
                    }`}
                  >
                    <span className="mr-2">{option.icon}</span>
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Menu Items */}
            <div className="py-1">
              <button
                className="w-full text-left px-4 py-2 text-sm text-secondary-700 hover:bg-secondary-100 dark:text-secondary-300 dark:hover:bg-secondary-700"
                onClick={() => setIsOpen(false)}
              >
                Profile Settings
              </button>
              <button
                className="w-full text-left px-4 py-2 text-sm text-secondary-700 hover:bg-secondary-100 dark:text-secondary-300 dark:hover:bg-secondary-700"
                onClick={() => setIsOpen(false)}
              >
                Preferences
              </button>
              <div className="border-t border-secondary-200 dark:border-secondary-700 my-1" />
              <button
                className="w-full text-left px-4 py-2 text-sm text-error-700 hover:bg-error-50 dark:text-error-400 dark:hover:bg-error-900/20"
                onClick={() => setIsOpen(false)}
              >
                Sign Out
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
})
