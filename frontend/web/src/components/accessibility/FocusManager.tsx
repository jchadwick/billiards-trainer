/**
 * Focus Manager Component
 * Manages focus state and provides focus trapping for modals and overlays
 */

import React, { useRef, useEffect, useCallback, useState } from 'react'
import { useFocusTrap } from '../../hooks/useAccessibility'

export interface FocusManagerProps {
  children: React.ReactNode
  trapFocus?: boolean
  restoreOnUnmount?: boolean
  autoFocus?: boolean
  className?: string
  onEscape?: () => void
  onFocusLeave?: () => void
}

/**
 * Focus Manager component that handles focus trapping and restoration
 */
export const FocusManager: React.FC<FocusManagerProps> = ({
  children,
  trapFocus = false,
  restoreOnUnmount = true,
  autoFocus = false,
  className = '',
  onEscape,
  onFocusLeave,
}) => {
  const containerRef = useFocusTrap(trapFocus)
  const previousActiveElement = useRef<HTMLElement | null>(null)
  const [isMounted, setIsMounted] = useState(false)

  // Store the previously focused element
  useEffect(() => {
    if (trapFocus) {
      previousActiveElement.current = document.activeElement as HTMLElement
    }
    setIsMounted(true)

    return () => {
      // Restore focus on unmount if requested
      if (restoreOnUnmount && previousActiveElement.current) {
        previousActiveElement.current.focus()
      }
    }
  }, [trapFocus, restoreOnUnmount])

  // Auto focus first focusable element
  useEffect(() => {
    if (autoFocus && isMounted && containerRef.current) {
      const focusableElement = containerRef.current.querySelector(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      ) as HTMLElement

      if (focusableElement) {
        focusableElement.focus()
      }
    }
  }, [autoFocus, isMounted, containerRef])

  // Handle escape key
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.stopPropagation()
      onEscape?.()
    }
  }, [onEscape])

  // Handle focus leaving the container
  const handleFocusOut = useCallback((e: React.FocusEvent) => {
    if (!trapFocus) return

    const container = containerRef.current
    if (!container) return

    // Check if focus is moving outside the container
    setTimeout(() => {
      if (!container.contains(document.activeElement)) {
        onFocusLeave?.()
      }
    }, 0)
  }, [trapFocus, onFocusLeave, containerRef])

  return (
    <div
      ref={containerRef}
      className={className}
      onKeyDown={handleKeyDown}
      onBlur={handleFocusOut}
    >
      {children}
    </div>
  )
}

/**
 * Hook for managing focus restoration
 */
export function useFocusRestore() {
  const previousFocusRef = useRef<HTMLElement | null>(null)

  const storeFocus = useCallback(() => {
    previousFocusRef.current = document.activeElement as HTMLElement
  }, [])

  const restoreFocus = useCallback(() => {
    if (previousFocusRef.current) {
      previousFocusRef.current.focus()
      previousFocusRef.current = null
    }
  }, [])

  return { storeFocus, restoreFocus }
}

/**
 * Hook for managing focus within a specific container
 */
export function useFocusWithin(containerRef: React.RefObject<HTMLElement>) {
  const [hasFocusWithin, setHasFocusWithin] = useState(false)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const handleFocusIn = (e: FocusEvent) => {
      if (container.contains(e.target as Node)) {
        setHasFocusWithin(true)
      }
    }

    const handleFocusOut = (e: FocusEvent) => {
      if (!container.contains(e.relatedTarget as Node)) {
        setHasFocusWithin(false)
      }
    }

    document.addEventListener('focusin', handleFocusIn)
    document.addEventListener('focusout', handleFocusOut)

    return () => {
      document.removeEventListener('focusin', handleFocusIn)
      document.removeEventListener('focusout', handleFocusOut)
    }
  }, [containerRef])

  return hasFocusWithin
}

/**
 * Component for creating focus boundaries
 */
export interface FocusBoundaryProps {
  children: React.ReactNode
  onBoundaryFocus?: (direction: 'start' | 'end') => void
}

export const FocusBoundary: React.FC<FocusBoundaryProps> = ({
  children,
  onBoundaryFocus,
}) => {
  const handleStartBoundaryFocus = useCallback(() => {
    onBoundaryFocus?.('start')
  }, [onBoundaryFocus])

  const handleEndBoundaryFocus = useCallback(() => {
    onBoundaryFocus?.('end')
  }, [onBoundaryFocus])

  return (
    <>
      <div
        tabIndex={0}
        onFocus={handleStartBoundaryFocus}
        className="sr-only"
        aria-hidden="true"
      />
      {children}
      <div
        tabIndex={0}
        onFocus={handleEndBoundaryFocus}
        className="sr-only"
        aria-hidden="true"
      />
    </>
  )
}

/**
 * Component for managing focus indicators
 */
export interface FocusIndicatorProps {
  children: React.ReactNode
  className?: string
  showIndicator?: boolean
}

export const FocusIndicator: React.FC<FocusIndicatorProps> = ({
  children,
  className = '',
  showIndicator = true,
}) => {
  const [isFocused, setIsFocused] = useState(false)

  const handleFocus = useCallback(() => {
    setIsFocused(true)
  }, [])

  const handleBlur = useCallback(() => {
    setIsFocused(false)
  }, [])

  const focusClasses = showIndicator && isFocused
    ? 'ring-2 ring-blue-500 ring-offset-2'
    : ''

  return (
    <div
      className={`${className} ${focusClasses} transition-all`}
      onFocus={handleFocus}
      onBlur={handleBlur}
    >
      {children}
    </div>
  )
}

/**
 * Utility functions for focus management
 */
export const FocusUtils = {
  /**
   * Get all focusable elements within a container
   */
  getFocusableElements(container: HTMLElement): HTMLElement[] {
    const selector = [
      'button:not([disabled])',
      '[href]',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      '[tabindex]:not([tabindex="-1"])'
    ].join(', ')

    return Array.from(container.querySelectorAll(selector)) as HTMLElement[]
  },

  /**
   * Focus the first focusable element in a container
   */
  focusFirst(container: HTMLElement): boolean {
    const focusable = this.getFocusableElements(container)
    if (focusable.length > 0) {
      focusable[0].focus()
      return true
    }
    return false
  },

  /**
   * Focus the last focusable element in a container
   */
  focusLast(container: HTMLElement): boolean {
    const focusable = this.getFocusableElements(container)
    if (focusable.length > 0) {
      focusable[focusable.length - 1].focus()
      return true
    }
    return false
  },

  /**
   * Move focus to the next focusable element
   */
  focusNext(container: HTMLElement, currentElement: HTMLElement): boolean {
    const focusable = this.getFocusableElements(container)
    const currentIndex = focusable.indexOf(currentElement)

    if (currentIndex >= 0 && currentIndex < focusable.length - 1) {
      focusable[currentIndex + 1].focus()
      return true
    }

    return false
  },

  /**
   * Move focus to the previous focusable element
   */
  focusPrevious(container: HTMLElement, currentElement: HTMLElement): boolean {
    const focusable = this.getFocusableElements(container)
    const currentIndex = focusable.indexOf(currentElement)

    if (currentIndex > 0) {
      focusable[currentIndex - 1].focus()
      return true
    }

    return false
  },

  /**
   * Check if an element is focusable
   */
  isFocusable(element: HTMLElement): boolean {
    const focusable = this.getFocusableElements(element.ownerDocument.body)
    return focusable.includes(element)
  },

  /**
   * Trap focus within a container
   */
  trapFocus(container: HTMLElement, event: KeyboardEvent) {
    if (event.key !== 'Tab') return

    const focusable = this.getFocusableElements(container)
    if (focusable.length === 0) return

    const first = focusable[0]
    const last = focusable[focusable.length - 1]

    if (event.shiftKey) {
      if (document.activeElement === first) {
        last.focus()
        event.preventDefault()
      }
    } else {
      if (document.activeElement === last) {
        first.focus()
        event.preventDefault()
      }
    }
  }
}
