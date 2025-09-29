import React from 'react'
import { Link, useRouterState } from '@tanstack/react-router'
import { observer } from 'mobx-react-lite'
import { createNavigationAria, createListItemAria } from '../../utils/accessibility'
import { ScreenReaderOnly } from '../accessibility/ScreenReaderOnly'
import type { NavItem } from '../../types'

export interface NavMenuProps {
  items: NavItem[]
  collapsed?: boolean
  className?: string
  onItemClick?: (item: NavItem) => void
}

const NavIcon: React.FC<{ name?: string; className?: string; label?: string }> = ({ name, className = 'w-5 h-5', label }) => {
  if (!name) return null

  // Icon mapping - replace with your preferred icon library
  const icons: Record<string, React.ReactNode> = {
    home: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
      </svg>
    ),
    calibration: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
      </svg>
    ),
    configuration: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
    'system-management': (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
      </svg>
    ),
    diagnostics: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  }

  const icon = icons[name] || null

  if (!icon) return null

  return (
    <span role="img" aria-label={label || `${name} icon`} aria-hidden={!label}>
      {icon}
    </span>
  )
}

export const NavMenuItem: React.FC<{
  item: NavItem
  collapsed?: boolean
  isActive?: boolean
  onItemClick?: (item: NavItem) => void
}> = observer(({ item, collapsed = false, isActive = false, onItemClick }) => {
  const handleClick = () => {
    onItemClick?.(item)
  }

  const baseClasses = 'flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200'
  const activeClasses = isActive
    ? 'bg-primary-100 text-primary-900 dark:bg-primary-900 dark:text-primary-100'
    : 'text-secondary-700 hover:bg-secondary-100 hover:text-secondary-900 dark:text-secondary-300 dark:hover:bg-secondary-800 dark:hover:text-secondary-100'

  const disabledClasses = item.disabled
    ? 'opacity-50 cursor-not-allowed'
    : 'cursor-pointer'

  return (
    <Link
      to={item.path}
      className={`${baseClasses} ${activeClasses} ${disabledClasses}`}
      onClick={item.disabled ? (e) => e.preventDefault() : handleClick}
      aria-disabled={item.disabled}
      aria-current={isActive ? 'page' : undefined}
      role="menuitem"
      tabIndex={item.disabled ? -1 : 0}
    >
      <NavIcon
        name={item.icon}
        className="w-5 h-5 flex-shrink-0"
        label={collapsed ? item.label : undefined}
      />
      {!collapsed && (
        <>
          <span className="ml-3 flex-1">{item.label}</span>
          {item.badge && (
            <span
              className="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200"
              aria-label={`${item.badge} notifications`}
            >
              {item.badge}
            </span>
          )}
        </>
      )}
      {collapsed && (
        <ScreenReaderOnly>
          {item.label}
          {item.badge && ` (${item.badge} notifications)`}
        </ScreenReaderOnly>
      )}
      {isActive && (
        <ScreenReaderOnly>
          Current page
        </ScreenReaderOnly>
      )}
    </Link>
  )
})

export const NavMenu = observer<NavMenuProps>(({
  items,
  collapsed = false,
  className = '',
  onItemClick,
}) => {
  const router = useRouterState()
  const currentPath = router.location.pathname

  const navigationAria = createNavigationAria({
    label: collapsed ? 'Main navigation (collapsed)' : 'Main navigation'
  })

  return (
    <nav
      className={`space-y-1 ${className}`}
      {...navigationAria}
      role="menu"
    >
      <ScreenReaderOnly>
        Navigation menu with {items.length} items
        {collapsed && ' (collapsed view)'}
      </ScreenReaderOnly>

      {items.map((item, index) => {
        const isActive = currentPath === item.path
        const listItemAria = createListItemAria({
          position: index + 1,
          total: items.length,
          current: isActive
        })

        return (
          <div key={item.id} {...listItemAria}>
            <NavMenuItem
              item={item}
              collapsed={collapsed}
              isActive={isActive}
              onItemClick={onItemClick}
            />
          </div>
        )
      })}
    </nav>
  )
})
