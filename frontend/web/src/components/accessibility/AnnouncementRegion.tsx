/**
 * Announcement Region Component
 * Provides ARIA live regions for dynamic content announcements to screen readers
 */

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { useAccessibility } from '../../hooks/useAccessibility'
import { createLiveRegionAria } from '../../utils/accessibility'

export interface AnnouncementRegionProps {
  className?: string
  level?: 'polite' | 'assertive'
  atomic?: boolean
  relevant?: 'additions' | 'removals' | 'text' | 'all'
  clearDelay?: number
}

/**
 * Global announcement region that integrates with useAccessibility hook
 */
export const AnnouncementRegion: React.FC<AnnouncementRegionProps> = ({
  className = '',
  level = 'polite',
  atomic = true,
  relevant = 'additions text',
  clearDelay = 5000,
}) => {
  const { announcements, liveRegionMode, clearAnnouncements } = useAccessibility()
  const clearTimeoutRef = useRef<NodeJS.Timeout>()

  // Auto-clear announcements after delay
  useEffect(() => {
    if (announcements.length > 0) {
      if (clearTimeoutRef.current) {
        clearTimeout(clearTimeoutRef.current)
      }

      clearTimeoutRef.current = setTimeout(() => {
        clearAnnouncements()
      }, clearDelay)
    }

    return () => {
      if (clearTimeoutRef.current) {
        clearTimeout(clearTimeoutRef.current)
      }
    }
  }, [announcements, clearDelay, clearAnnouncements])

  const ariaProps = createLiveRegionAria({
    level: liveRegionMode === 'off' ? 'off' : liveRegionMode || level,
    atomic,
    relevant,
  })

  if (liveRegionMode === 'off') {
    return null
  }

  return (
    <div
      className={`sr-only ${className}`}
      {...ariaProps}
      role="log"
    >
      {announcements.map((announcement, index) => (
        <div key={`${announcement}-${index}`}>
          {announcement}
        </div>
      ))}
    </div>
  )
}

/**
 * Status region for general status updates
 */
export interface StatusRegionProps {
  status: string
  className?: string
  clearAfter?: number
}

export const StatusRegion: React.FC<StatusRegionProps> = ({
  status,
  className = '',
  clearAfter = 3000,
}) => {
  const [currentStatus, setCurrentStatus] = useState(status)
  const clearTimeoutRef = useRef<NodeJS.Timeout>()

  useEffect(() => {
    setCurrentStatus(status)

    if (status && clearAfter > 0) {
      if (clearTimeoutRef.current) {
        clearTimeout(clearTimeoutRef.current)
      }

      clearTimeoutRef.current = setTimeout(() => {
        setCurrentStatus('')
      }, clearAfter)
    }

    return () => {
      if (clearTimeoutRef.current) {
        clearTimeout(clearTimeoutRef.current)
      }
    }
  }, [status, clearAfter])

  if (!currentStatus) {
    return null
  }

  return (
    <div
      className={`sr-only ${className}`}
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      {currentStatus}
    </div>
  )
}

/**
 * Alert region for urgent announcements
 */
export interface AlertRegionProps {
  alert: string
  className?: string
  clearAfter?: number
}

export const AlertRegion: React.FC<AlertRegionProps> = ({
  alert,
  className = '',
  clearAfter = 5000,
}) => {
  const [currentAlert, setCurrentAlert] = useState(alert)
  const clearTimeoutRef = useRef<NodeJS.Timeout>()

  useEffect(() => {
    setCurrentAlert(alert)

    if (alert && clearAfter > 0) {
      if (clearTimeoutRef.current) {
        clearTimeout(clearTimeoutRef.current)
      }

      clearTimeoutRef.current = setTimeout(() => {
        setCurrentAlert('')
      }, clearAfter)
    }

    return () => {
      if (clearTimeoutRef.current) {
        clearTimeout(clearTimeoutRef.current)
      }
    }
  }, [alert, clearAfter])

  if (!currentAlert) {
    return null
  }

  return (
    <div
      className={`sr-only ${className}`}
      role="alert"
      aria-live="assertive"
      aria-atomic="true"
    >
      {currentAlert}
    </div>
  )
}

/**
 * Progress region for progress updates
 */
export interface ProgressRegionProps {
  progress: {
    current: number
    total: number
    label?: string
  }
  className?: string
  announceEvery?: number
}

export const ProgressRegion: React.FC<ProgressRegionProps> = ({
  progress,
  className = '',
  announceEvery = 25, // Announce every 25% by default
}) => {
  const [lastAnnouncedProgress, setLastAnnouncedProgress] = useState(0)
  const [announcement, setAnnouncement] = useState('')

  useEffect(() => {
    const percentage = Math.round((progress.current / progress.total) * 100)
    const shouldAnnounce = percentage >= lastAnnouncedProgress + announceEvery || percentage === 100

    if (shouldAnnounce && percentage !== lastAnnouncedProgress) {
      const label = progress.label || 'Progress'
      setAnnouncement(`${label}: ${percentage}% complete`)
      setLastAnnouncedProgress(percentage)

      // Clear announcement after 2 seconds
      setTimeout(() => {
        setAnnouncement('')
      }, 2000)
    }
  }, [progress, lastAnnouncedProgress, announceEvery])

  if (!announcement) {
    return null
  }

  return (
    <div
      className={`sr-only ${className}`}
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      {announcement}
    </div>
  )
}

/**
 * Video status region for video player announcements
 */
export interface VideoStatusRegionProps {
  isPlaying?: boolean
  isMuted?: boolean
  volume?: number
  duration?: number
  currentTime?: number
  className?: string
}

export const VideoStatusRegion: React.FC<VideoStatusRegionProps> = ({
  isPlaying,
  isMuted,
  volume,
  duration,
  currentTime,
  className = '',
}) => {
  const [announcement, setAnnouncement] = useState('')

  const announceStatus = useCallback((message: string) => {
    setAnnouncement(message)
    setTimeout(() => setAnnouncement(''), 2000)
  }, [])

  useEffect(() => {
    if (isPlaying !== undefined) {
      announceStatus(isPlaying ? 'Video playing' : 'Video paused')
    }
  }, [isPlaying, announceStatus])

  useEffect(() => {
    if (isMuted !== undefined) {
      announceStatus(isMuted ? 'Video muted' : 'Video unmuted')
    }
  }, [isMuted, announceStatus])

  useEffect(() => {
    if (volume !== undefined) {
      announceStatus(`Volume ${Math.round(volume * 100)}%`)
    }
  }, [volume, announceStatus])

  if (!announcement) {
    return null
  }

  return (
    <div
      className={`sr-only ${className}`}
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      {announcement}
    </div>
  )
}

/**
 * Game status region for billiards-specific announcements
 */
export interface GameStatusRegionProps {
  ballCount?: number
  detectedObjects?: {
    balls: number
    trajectories: number
    table: boolean
    cue: boolean
  }
  gameState?: string
  className?: string
}

export const GameStatusRegion: React.FC<GameStatusRegionProps> = ({
  ballCount,
  detectedObjects,
  gameState,
  className = '',
}) => {
  const [announcement, setAnnouncement] = useState('')
  const prevDetectedObjects = useRef(detectedObjects)

  const announceStatus = useCallback((message: string) => {
    setAnnouncement(message)
    setTimeout(() => setAnnouncement(''), 3000)
  }, [])

  // Announce ball count changes
  useEffect(() => {
    if (ballCount !== undefined) {
      announceStatus(`${ballCount} balls detected`)
    }
  }, [ballCount, announceStatus])

  // Announce detection changes
  useEffect(() => {
    if (detectedObjects && prevDetectedObjects.current) {
      const prev = prevDetectedObjects.current
      const curr = detectedObjects

      if (prev.table !== curr.table) {
        announceStatus(curr.table ? 'Table detected' : 'Table lost')
      }

      if (prev.cue !== curr.cue) {
        announceStatus(curr.cue ? 'Cue stick detected' : 'Cue stick lost')
      }

      if (prev.trajectories !== curr.trajectories && curr.trajectories > 0) {
        announceStatus(`${curr.trajectories} trajectory predictions available`)
      }
    }

    prevDetectedObjects.current = detectedObjects
  }, [detectedObjects, announceStatus])

  // Announce game state changes
  useEffect(() => {
    if (gameState) {
      announceStatus(`Game state: ${gameState}`)
    }
  }, [gameState, announceStatus])

  if (!announcement) {
    return null
  }

  return (
    <div
      className={`sr-only ${className}`}
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      {announcement}
    </div>
  )
}

/**
 * Hook for creating custom announcement regions
 */
export function useAnnouncementRegion(level: 'polite' | 'assertive' = 'polite') {
  const [announcement, setAnnouncement] = useState('')
  const timeoutRef = useRef<NodeJS.Timeout>()

  const announce = useCallback((message: string, clearAfter: number = 3000) => {
    setAnnouncement(message)

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    if (clearAfter > 0) {
      timeoutRef.current = setTimeout(() => {
        setAnnouncement('')
      }, clearAfter)
    }
  }, [])

  const clear = useCallback(() => {
    setAnnouncement('')
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }
  }, [])

  const region = announcement ? (
    <div
      className="sr-only"
      role={level === 'assertive' ? 'alert' : 'status'}
      aria-live={level}
      aria-atomic="true"
    >
      {announcement}
    </div>
  ) : null

  return { announce, clear, region }
}
