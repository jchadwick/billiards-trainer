/**
 * Accessibility components index
 * Exports all accessibility-related components and utilities
 */

// Components
export * from './SkipLinks'
export * from './ScreenReaderOnly'
export * from './FocusManager'
export * from './AnnouncementRegion'

// Re-export hooks and utilities for convenience
export { useAccessibility, useKeyboardDetection, useFocusTrap } from '../../hooks/useAccessibility'
export * from '../../utils/accessibility'
