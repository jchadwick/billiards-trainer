/**
 * Skip Links Component
 * Provides skip navigation links for keyboard users and screen readers
 */

import React, { useCallback } from 'react'
import { useAccessibility } from '../../hooks/useAccessibility'

export interface SkipLink {
  id: string
  label: string
  target: string
  order: number
}

export interface SkipLinksProps {
  links?: SkipLink[]
  className?: string
  onLinkClick?: (target: string) => void
}

const DEFAULT_SKIP_LINKS: SkipLink[] = [
  { id: 'skip-to-main', label: 'Skip to main content', target: '#main-content', order: 1 },
  { id: 'skip-to-nav', label: 'Skip to navigation', target: '#main-navigation', order: 2 },
  { id: 'skip-to-video', label: 'Skip to video player', target: '#video-player', order: 3 },
  { id: 'skip-to-controls', label: 'Skip to video controls', target: '#video-controls', order: 4 },
  { id: 'skip-to-settings', label: 'Skip to accessibility settings', target: '#accessibility-settings', order: 5 },
]

export const SkipLinks: React.FC<SkipLinksProps> = ({
  links = DEFAULT_SKIP_LINKS,
  className = '',
  onLinkClick,
}) => {
  const { skipLinksVisible, setSkipLinksVisible } = useAccessibility()

  const handleLinkClick = useCallback((e: React.MouseEvent<HTMLAnchorElement>, target: string) => {
    e.preventDefault()

    // Hide skip links after use
    setSkipLinksVisible(false)

    // Find and focus target element
    const targetElement = document.querySelector(target) as HTMLElement
    if (targetElement) {
      targetElement.focus()
      targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }

    onLinkClick?.(target)
  }, [onLinkClick, setSkipLinksVisible])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Tab') {
      setSkipLinksVisible(true)
    } else if (e.key === 'Escape') {
      setSkipLinksVisible(false)
    }
  }, [setSkipLinksVisible])

  const sortedLinks = [...links].sort((a, b) => a.order - b.order)

  return (
    <nav
      className={`skip-links ${className}`}
      role="navigation"
      aria-label="Skip navigation links"
      onKeyDown={handleKeyDown}
    >
      <div className="sr-only focus-within:not-sr-only">
        <ul className="fixed top-0 left-0 z-[9999] bg-blue-600 text-white p-2 rounded-br-md shadow-lg">
          {sortedLinks.map((link) => (
            <li key={link.id} className="mb-1 last:mb-0">
              <a
                href={link.target}
                className="block px-3 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-blue-600 rounded transition-colors"
                onClick={(e) => handleLinkClick(e, link.target)}
                tabIndex={0}
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  )
}

/**
 * Hook to register skip link targets
 */
export function useSkipLinkTarget(id: string) {
  const targetRef = React.useRef<HTMLElement>(null)

  React.useEffect(() => {
    const element = targetRef.current
    if (element && !element.id) {
      element.id = id
    }

    // Ensure element is focusable
    if (element && !element.hasAttribute('tabindex')) {
      element.tabIndex = -1
    }
  }, [id])

  return targetRef
}

/**
 * Component to mark skip link targets
 */
export interface SkipLinkTargetProps {
  id: string
  children: React.ReactNode
  className?: string
  as?: keyof JSX.IntrinsicElements
}

export const SkipLinkTarget: React.FC<SkipLinkTargetProps> = ({
  id,
  children,
  className = '',
  as: Component = 'div',
}) => {
  const targetRef = useSkipLinkTarget(id)

  return React.createElement(
    Component,
    {
      ref: targetRef,
      id,
      tabIndex: -1,
      className: `skip-link-target ${className}`,
    },
    children
  )
}
