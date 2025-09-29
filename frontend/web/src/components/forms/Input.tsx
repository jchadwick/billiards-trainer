import React, { forwardRef } from 'react'
import { observer } from 'mobx-react-lite'

export interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'> {
  label?: string
  helperText?: string
  error?: string
  variant?: 'default' | 'filled' | 'outlined'
  size?: 'sm' | 'md' | 'lg'
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
  fullWidth?: boolean
  required?: boolean
}

const inputVariants = {
  default: 'border border-secondary-300 bg-white dark:border-secondary-600 dark:bg-secondary-800',
  filled: 'border-0 bg-secondary-100 dark:bg-secondary-800',
  outlined: 'border-2 border-secondary-300 bg-transparent dark:border-secondary-600',
}

const inputSizes = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-sm',
  lg: 'px-4 py-3 text-base',
}

const inputFocusStates = {
  default: 'focus:ring-2 focus:ring-primary-500 focus:border-primary-500',
  error: 'focus:ring-2 focus:ring-error-500 focus:border-error-500 border-error-300 dark:border-error-600',
}

export const Input = observer(forwardRef<HTMLInputElement, InputProps>(({
  label,
  helperText,
  error,
  variant = 'default',
  size = 'md',
  leftIcon,
  rightIcon,
  fullWidth = false,
  required = false,
  className = '',
  disabled,
  id,
  ...props
}, ref) => {
  const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`
  const hasError = !!error

  const baseClasses = 'rounded-md transition-colors placeholder-secondary-400 dark:placeholder-secondary-500 disabled:opacity-50 disabled:cursor-not-allowed'
  const variantClasses = inputVariants[variant]
  const sizeClasses = inputSizes[size]
  const focusClasses = hasError ? inputFocusStates.error : inputFocusStates.default
  const widthClasses = fullWidth ? 'w-full' : ''
  const textClasses = 'text-secondary-900 dark:text-secondary-100'

  const iconClasses = size === 'sm' ? 'w-4 h-4' : size === 'lg' ? 'w-6 h-6' : 'w-5 h-5'
  const iconColor = 'text-secondary-400 dark:text-secondary-500'

  const inputElement = (
    <div className={`relative ${fullWidth ? 'w-full' : ''}`}>
      {leftIcon && (
        <div className={`absolute left-3 top-1/2 transform -translate-y-1/2 ${iconClasses} ${iconColor}`}>
          {leftIcon}
        </div>
      )}
      <input
        ref={ref}
        id={inputId}
        className={`
          ${baseClasses}
          ${variantClasses}
          ${sizeClasses}
          ${focusClasses}
          ${textClasses}
          ${widthClasses}
          ${leftIcon ? (size === 'sm' ? 'pl-9' : size === 'lg' ? 'pl-12' : 'pl-10') : ''}
          ${rightIcon ? (size === 'sm' ? 'pr-9' : size === 'lg' ? 'pr-12' : 'pr-10') : ''}
          ${className}
        `}
        disabled={disabled}
        aria-invalid={hasError}
        aria-describedby={
          helperText || error
            ? `${inputId}-description`
            : undefined
        }
        {...props}
      />
      {rightIcon && (
        <div className={`absolute right-3 top-1/2 transform -translate-y-1/2 ${iconClasses} ${iconColor}`}>
          {rightIcon}
        </div>
      )}
    </div>
  )

  if (!label && !helperText && !error) {
    return inputElement
  }

  return (
    <div className={fullWidth ? 'w-full' : ''}>
      {label && (
        <label
          htmlFor={inputId}
          className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2"
        >
          {label}
          {required && <span className="text-error-500 ml-1">*</span>}
        </label>
      )}
      {inputElement}
      {(helperText || error) && (
        <p
          id={`${inputId}-description`}
          className={`mt-2 text-sm ${
            error
              ? 'text-error-600 dark:text-error-400'
              : 'text-secondary-500 dark:text-secondary-400'
          }`}
        >
          {error || helperText}
        </p>
      )}
    </div>
  )
}))

Input.displayName = 'Input'
