import React, { forwardRef } from 'react'
import { observer } from 'mobx-react-lite'

export interface CheckboxProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'> {
  label?: string
  helperText?: string
  error?: string
  size?: 'sm' | 'md' | 'lg'
  variant?: 'checkbox' | 'switch'
  indeterminate?: boolean
  labelPosition?: 'left' | 'right'
}

const checkboxSizes = {
  sm: 'w-4 h-4',
  md: 'w-5 h-5',
  lg: 'w-6 h-6',
}

const labelSizes = {
  sm: 'text-sm',
  md: 'text-sm',
  lg: 'text-base',
}

export const Checkbox = observer(forwardRef<HTMLInputElement, CheckboxProps>(({
  label,
  helperText,
  error,
  size = 'md',
  variant = 'checkbox',
  indeterminate = false,
  labelPosition = 'right',
  className = '',
  disabled,
  id,
  checked,
  ...props
}, ref) => {
  const checkboxId = id || `checkbox-${Math.random().toString(36).substr(2, 9)}`
  const hasError = !!error

  const sizeClasses = checkboxSizes[size]
  const labelSizeClasses = labelSizes[size]

  const baseClasses = variant === 'switch'
    ? 'relative inline-flex items-center cursor-pointer'
    : 'cursor-pointer'

  const inputClasses = variant === 'switch'
    ? 'sr-only peer'
    : `
        ${sizeClasses}
        rounded border-2 transition-colors cursor-pointer
        ${hasError
          ? 'border-error-300 dark:border-error-600 focus:ring-error-500'
          : 'border-secondary-300 dark:border-secondary-600 focus:ring-primary-500'
        }
        ${checked || indeterminate
          ? 'bg-primary-600 border-primary-600 text-white'
          : 'bg-white dark:bg-secondary-800'
        }
        focus:ring-2 focus:ring-offset-2
        disabled:opacity-50 disabled:cursor-not-allowed
      `

  const switchClasses = variant === 'switch' ? `
    w-11 h-6 bg-secondary-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 dark:peer-focus:ring-primary-800 rounded-full peer dark:bg-secondary-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-secondary-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-secondary-600 peer-checked:bg-primary-600
    ${hasError ? 'peer-focus:ring-error-300 dark:peer-focus:ring-error-800' : ''}
    ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
  ` : ''

  const checkboxElement = (
    <div className={`relative ${baseClasses}`}>
      <input
        ref={ref}
        type="checkbox"
        id={checkboxId}
        className={inputClasses}
        disabled={disabled}
        checked={checked}
        aria-invalid={hasError}
        aria-describedby={
          helperText || error ? `${checkboxId}-description` : undefined
        }
        {...props}
      />

      {variant === 'switch' && (
        <div className={switchClasses}></div>
      )}

      {variant === 'checkbox' && (checked || indeterminate) && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          {indeterminate ? (
            <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M4 10a1 1 0 011-1h10a1 1 0 110 2H5a1 1 0 01-1-1z" clipRule="evenodd" />
            </svg>
          ) : (
            <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
          )}
        </div>
      )}
    </div>
  )

  const labelElement = label && (
    <label
      htmlFor={checkboxId}
      className={`
        ${labelSizeClasses}
        font-medium text-secondary-700 dark:text-secondary-300 cursor-pointer
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        ${labelPosition === 'left' ? 'mr-2' : 'ml-2'}
      `}
    >
      {label}
    </label>
  )

  const mainElement = (
    <div className={`flex items-center ${className}`}>
      {labelPosition === 'left' && labelElement}
      {checkboxElement}
      {labelPosition === 'right' && labelElement}
    </div>
  )

  if (!helperText && !error) {
    return mainElement
  }

  return (
    <div>
      {mainElement}
      {(helperText || error) && (
        <p
          id={`${checkboxId}-description`}
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

Checkbox.displayName = 'Checkbox'

// Convenience component for switch variant
export const Switch = observer(forwardRef<HTMLInputElement, Omit<CheckboxProps, 'variant'>>(
  (props, ref) => (
    <Checkbox {...props} variant="switch" ref={ref} />
  )
))

Switch.displayName = 'Switch'
