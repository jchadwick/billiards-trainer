import React from 'react'

export interface CardProps {
  children: React.ReactNode
  className?: string
  padding?: 'none' | 'sm' | 'md' | 'lg'
  shadow?: 'none' | 'sm' | 'md' | 'lg'
  hover?: boolean
}

const paddingClasses = {
  none: '',
  sm: 'p-4',
  md: 'p-6',
  lg: 'p-8',
}

const shadowClasses = {
  none: '',
  sm: 'shadow-sm',
  md: 'shadow-md',
  lg: 'shadow-lg',
}

export const Card: React.FC<CardProps> = ({
  children,
  className = '',
  padding = 'md',
  shadow = 'md',
  hover = false,
}) => {
  const baseClasses = 'bg-white dark:bg-secondary-800 rounded-lg border border-secondary-200 dark:border-secondary-700'
  const paddingClass = paddingClasses[padding]
  const shadowClass = shadowClasses[shadow]
  const hoverClass = hover ? 'transition-shadow hover:shadow-lg' : ''

  return (
    <div className={`${baseClasses} ${paddingClass} ${shadowClass} ${hoverClass} ${className}`}>
      {children}
    </div>
  )
}

export interface CardHeaderProps {
  children: React.ReactNode
  className?: string
}

export const CardHeader: React.FC<CardHeaderProps> = ({ children, className = '' }) => (
  <div className={`mb-4 ${className}`}>
    {children}
  </div>
)

export interface CardTitleProps {
  children: React.ReactNode
  className?: string
  as?: 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6'
}

export const CardTitle: React.FC<CardTitleProps> = ({
  children,
  className = '',
  as: Component = 'h3'
}) => (
  <Component className={`text-lg font-semibold text-gray-900 dark:text-gray-100 ${className}`}>
    {children}
  </Component>
)

export interface CardContentProps {
  children: React.ReactNode
  className?: string
}

export const CardContent: React.FC<CardContentProps> = ({ children, className = '' }) => (
  <div className={`${className}`}>
    {children}
  </div>
)

export interface CardFooterProps {
  children: React.ReactNode
  className?: string
}

export const CardFooter: React.FC<CardFooterProps> = ({ children, className = '' }) => (
  <div className={`mt-4 pt-4 border-t border-secondary-200 dark:border-secondary-700 ${className}`}>
    {children}
  </div>
)
