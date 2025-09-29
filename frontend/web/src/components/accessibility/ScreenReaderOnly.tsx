/**
 * Screen Reader Only Component
 * Provides content that is only visible to screen readers
 */

import React from 'react'

export interface ScreenReaderOnlyProps {
  children: React.ReactNode
  className?: string
  as?: keyof JSX.IntrinsicElements
  live?: 'polite' | 'assertive' | 'off'
  atomic?: boolean
}

/**
 * Component that renders content only for screen readers
 * Uses sr-only class that visually hides content but keeps it accessible
 */
export const ScreenReaderOnly: React.FC<ScreenReaderOnlyProps> = ({
  children,
  className = '',
  as: Component = 'span',
  live,
  atomic,
}) => {
  const liveRegionProps = live ? {
    'aria-live': live,
    'aria-atomic': atomic,
  } : {}

  return React.createElement(
    Component,
    {
      className: `sr-only ${className}`,
      ...liveRegionProps,
    },
    children
  )
}

/**
 * Component for providing visual descriptions to screen readers
 */
export interface VisualDescriptionProps {
  description: string
  className?: string
  live?: 'polite' | 'assertive'
}

export const VisualDescription: React.FC<VisualDescriptionProps> = ({
  description,
  className = '',
  live = 'polite',
}) => {
  return (
    <ScreenReaderOnly className={className} live={live}>
      {description}
    </ScreenReaderOnly>
  )
}

/**
 * Component for providing instructions to screen readers
 */
export interface InstructionTextProps {
  instruction: string
  className?: string
}

export const InstructionText: React.FC<InstructionTextProps> = ({
  instruction,
  className = '',
}) => {
  return (
    <ScreenReaderOnly className={className}>
      <span role="note" aria-label="Instruction">
        {instruction}
      </span>
    </ScreenReaderOnly>
  )
}

/**
 * Component for providing status updates to screen readers
 */
export interface StatusUpdateProps {
  status: string
  className?: string
  priority?: 'polite' | 'assertive'
}

export const StatusUpdate: React.FC<StatusUpdateProps> = ({
  status,
  className = '',
  priority = 'polite',
}) => {
  return (
    <ScreenReaderOnly
      className={className}
      live={priority}
      atomic={true}
      role="status"
    >
      {status}
    </ScreenReaderOnly>
  )
}

/**
 * Component for providing keyboard shortcut information
 */
export interface KeyboardShortcutProps {
  shortcut: string
  description: string
  className?: string
}

export const KeyboardShortcutInfo: React.FC<KeyboardShortcutProps> = ({
  shortcut,
  description,
  className = '',
}) => {
  return (
    <ScreenReaderOnly className={className}>
      <span role="note">
        Press {shortcut} to {description}
      </span>
    </ScreenReaderOnly>
  )
}

/**
 * Component for providing context about complex visuals
 */
export interface ContextualHelpProps {
  context: string
  className?: string
}

export const ContextualHelp: React.FC<ContextualHelpProps> = ({
  context,
  className = '',
}) => {
  return (
    <ScreenReaderOnly className={className}>
      <span role="note" aria-label="Additional context">
        {context}
      </span>
    </ScreenReaderOnly>
  )
}

/**
 * Component for providing loading states to screen readers
 */
export interface LoadingAnnouncementProps {
  isLoading: boolean
  loadingText?: string
  completedText?: string
  className?: string
}

export const LoadingAnnouncement: React.FC<LoadingAnnouncementProps> = ({
  isLoading,
  loadingText = 'Loading...',
  completedText = 'Loading complete',
  className = '',
}) => {
  const text = isLoading ? loadingText : completedText

  return (
    <ScreenReaderOnly
      className={className}
      live="polite"
      atomic={true}
      role="status"
    >
      {text}
    </ScreenReaderOnly>
  )
}

/**
 * Component for providing error announcements to screen readers
 */
export interface ErrorAnnouncementProps {
  error: string | null
  className?: string
}

export const ErrorAnnouncement: React.FC<ErrorAnnouncementProps> = ({
  error,
  className = '',
}) => {
  if (!error) return null

  return (
    <ScreenReaderOnly
      className={className}
      live="assertive"
      atomic={true}
      role="alert"
    >
      Error: {error}
    </ScreenReaderOnly>
  )
}

/**
 * Component for providing progress updates to screen readers
 */
export interface ProgressAnnouncementProps {
  progress: number
  total: number
  action?: string
  className?: string
}

export const ProgressAnnouncement: React.FC<ProgressAnnouncementProps> = ({
  progress,
  total,
  action = 'Processing',
  className = '',
}) => {
  const percentage = Math.round((progress / total) * 100)

  return (
    <ScreenReaderOnly
      className={className}
      live="polite"
      atomic={true}
      role="status"
    >
      {action} {percentage}% complete. {progress} of {total} items processed.
    </ScreenReaderOnly>
  )
}
