/**
 * Accessibility state management hook
 * Manages user accessibility preferences and provides utilities for screen readers
 */

import { useState, useEffect, useCallback, useRef } from 'react'

export interface AccessibilityState {
  // User preferences
  reducedMotion: boolean
  highContrast: boolean
  fontSize: 'small' | 'medium' | 'large' | 'extra-large'
  screenReaderOptimized: boolean
  keyboardNavigation: boolean

  // Screen reader support
  announcements: string[]
  liveRegionMode: 'polite' | 'assertive' | 'off'

  // Focus management
  focusVisible: boolean
  skipLinksVisible: boolean

  // Color and visual preferences
  colorBlindnessMode: 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia'
}

export interface AccessibilityActions {
  // Preferences
  setReducedMotion: (enabled: boolean) => void
  setHighContrast: (enabled: boolean) => void
  setFontSize: (size: AccessibilityState['fontSize']) => void
  setScreenReaderOptimized: (enabled: boolean) => void
  setKeyboardNavigation: (enabled: boolean) => void
  setColorBlindnessMode: (mode: AccessibilityState['colorBlindnessMode']) => void

  // Screen reader announcements
  announce: (message: string, priority?: 'polite' | 'assertive') => void
  clearAnnouncements: () => void

  // Focus management
  setFocusVisible: (visible: boolean) => void
  setSkipLinksVisible: (visible: boolean) => void

  // Utilities
  resetToDefaults: () => void
  savePreferences: () => void
  loadPreferences: () => void
}

const DEFAULT_STATE: AccessibilityState = {
  reducedMotion: false,
  highContrast: false,
  fontSize: 'medium',
  screenReaderOptimized: false,
  keyboardNavigation: false,
  announcements: [],
  liveRegionMode: 'polite',
  focusVisible: false,
  skipLinksVisible: false,
  colorBlindnessMode: 'none',
}

const STORAGE_KEY = 'billiards-trainer-accessibility-preferences'

/**
 * Custom hook for managing accessibility state and preferences
 */
export function useAccessibility(): AccessibilityState & AccessibilityActions {
  const [state, setState] = useState<AccessibilityState>(DEFAULT_STATE)
  const announcementTimeoutRef = useRef<NodeJS.Timeout>()

  // Load preferences from localStorage on mount
  useEffect(() => {
    loadPreferences()

    // Detect system preferences
    if (window.matchMedia) {
      const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
      const prefersHighContrast = window.matchMedia('(prefers-contrast: high)')

      if (prefersReducedMotion.matches) {
        setState(prev => ({ ...prev, reducedMotion: true }))
      }

      if (prefersHighContrast.matches) {
        setState(prev => ({ ...prev, highContrast: true }))
      }

      // Listen for changes
      const handleReducedMotionChange = (e: MediaQueryListEvent) => {
        setState(prev => ({ ...prev, reducedMotion: e.matches }))
      }

      const handleHighContrastChange = (e: MediaQueryListEvent) => {
        setState(prev => ({ ...prev, highContrast: e.matches }))
      }

      prefersReducedMotion.addEventListener('change', handleReducedMotionChange)
      prefersHighContrast.addEventListener('change', handleHighContrastChange)

      return () => {
        prefersReducedMotion.removeEventListener('change', handleReducedMotionChange)
        prefersHighContrast.removeEventListener('change', handleHighContrastChange)
      }
    }
  }, [])

  // Apply CSS custom properties when state changes
  useEffect(() => {
    const root = document.documentElement

    // Font size
    const fontSizeMap = {
      small: '14px',
      medium: '16px',
      large: '18px',
      'extra-large': '20px',
    }
    root.style.setProperty('--font-size-base', fontSizeMap[state.fontSize])

    // High contrast
    if (state.highContrast) {
      root.classList.add('high-contrast')
    } else {
      root.classList.remove('high-contrast')
    }

    // Reduced motion
    if (state.reducedMotion) {
      root.classList.add('reduced-motion')
    } else {
      root.classList.remove('reduced-motion')
    }

    // Color blindness mode
    root.setAttribute('data-color-mode', state.colorBlindnessMode)

    // Screen reader optimization
    if (state.screenReaderOptimized) {
      root.classList.add('screen-reader-optimized')
    } else {
      root.classList.remove('screen-reader-optimized')
    }

    // Keyboard navigation
    if (state.keyboardNavigation) {
      root.classList.add('keyboard-navigation')
    } else {
      root.classList.remove('keyboard-navigation')
    }
  }, [state])

  // Actions
  const setReducedMotion = useCallback((enabled: boolean) => {
    setState(prev => ({ ...prev, reducedMotion: enabled }))
  }, [])

  const setHighContrast = useCallback((enabled: boolean) => {
    setState(prev => ({ ...prev, highContrast: enabled }))
  }, [])

  const setFontSize = useCallback((size: AccessibilityState['fontSize']) => {
    setState(prev => ({ ...prev, fontSize: size }))
  }, [])

  const setScreenReaderOptimized = useCallback((enabled: boolean) => {
    setState(prev => ({ ...prev, screenReaderOptimized: enabled }))
  }, [])

  const setKeyboardNavigation = useCallback((enabled: boolean) => {
    setState(prev => ({ ...prev, keyboardNavigation: enabled }))
  }, [])

  const setColorBlindnessMode = useCallback((mode: AccessibilityState['colorBlindnessMode']) => {
    setState(prev => ({ ...prev, colorBlindnessMode: mode }))
  }, [])

  const announce = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    setState(prev => ({
      ...prev,
      announcements: [...prev.announcements, message],
      liveRegionMode: priority
    }))

    // Clear announcement after a delay
    if (announcementTimeoutRef.current) {
      clearTimeout(announcementTimeoutRef.current)
    }

    announcementTimeoutRef.current = setTimeout(() => {
      setState(prev => ({
        ...prev,
        announcements: prev.announcements.filter(a => a !== message)
      }))
    }, 5000)
  }, [])

  const clearAnnouncements = useCallback(() => {
    setState(prev => ({ ...prev, announcements: [] }))
    if (announcementTimeoutRef.current) {
      clearTimeout(announcementTimeoutRef.current)
    }
  }, [])

  const setFocusVisible = useCallback((visible: boolean) => {
    setState(prev => ({ ...prev, focusVisible: visible }))
  }, [])

  const setSkipLinksVisible = useCallback((visible: boolean) => {
    setState(prev => ({ ...prev, skipLinksVisible: visible }))
  }, [])

  const resetToDefaults = useCallback(() => {
    setState(DEFAULT_STATE)
    localStorage.removeItem(STORAGE_KEY)
  }, [])

  const savePreferences = useCallback(() => {
    const preferencesToSave = {
      reducedMotion: state.reducedMotion,
      highContrast: state.highContrast,
      fontSize: state.fontSize,
      screenReaderOptimized: state.screenReaderOptimized,
      keyboardNavigation: state.keyboardNavigation,
      colorBlindnessMode: state.colorBlindnessMode,
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(preferencesToSave))
  }, [state])

  const loadPreferences = useCallback(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        const preferences = JSON.parse(saved)
        setState(prev => ({ ...prev, ...preferences }))
      }
    } catch (error) {
      console.warn('Failed to load accessibility preferences:', error)
    }
  }, [])

  // Auto-save preferences when they change
  useEffect(() => {
    savePreferences()
  }, [state.reducedMotion, state.highContrast, state.fontSize, state.screenReaderOptimized, state.keyboardNavigation, state.colorBlindnessMode, savePreferences])

  return {
    ...state,
    setReducedMotion,
    setHighContrast,
    setFontSize,
    setScreenReaderOptimized,
    setKeyboardNavigation,
    setColorBlindnessMode,
    announce,
    clearAnnouncements,
    setFocusVisible,
    setSkipLinksVisible,
    resetToDefaults,
    savePreferences,
    loadPreferences,
  }
}

/**
 * Hook for detecting keyboard navigation
 */
export function useKeyboardDetection() {
  const [isKeyboardUser, setIsKeyboardUser] = useState(false)

  useEffect(() => {
    let keyboardThreshold = 0

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        keyboardThreshold += 1
        if (keyboardThreshold >= 3 && !isKeyboardUser) {
          setIsKeyboardUser(true)
          document.body.classList.add('user-is-tabbing')
        }
      }
    }

    const handleMouseDown = () => {
      keyboardThreshold = 0
      if (isKeyboardUser) {
        setIsKeyboardUser(false)
        document.body.classList.remove('user-is-tabbing')
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    document.addEventListener('mousedown', handleMouseDown)

    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.removeEventListener('mousedown', handleMouseDown)
    }
  }, [isKeyboardUser])

  return isKeyboardUser
}

/**
 * Hook for managing focus trapping in modals/overlays
 */
export function useFocusTrap(isActive: boolean = false) {
  const containerRef = useRef<HTMLElement>(null)

  useEffect(() => {
    if (!isActive || !containerRef.current) return

    const container = containerRef.current
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )

    if (focusableElements.length === 0) return

    const firstFocusable = focusableElements[0] as HTMLElement
    const lastFocusable = focusableElements[focusableElements.length - 1] as HTMLElement

    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return

      if (e.shiftKey) {
        if (document.activeElement === firstFocusable) {
          lastFocusable.focus()
          e.preventDefault()
        }
      } else {
        if (document.activeElement === lastFocusable) {
          firstFocusable.focus()
          e.preventDefault()
        }
      }
    }

    // Focus first element when trap activates
    firstFocusable.focus()

    document.addEventListener('keydown', handleTabKey)
    return () => document.removeEventListener('keydown', handleTabKey)
  }, [isActive])

  return containerRef
}
