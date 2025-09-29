import React, { forwardRef } from 'react'
import { observer } from 'mobx-react-lite'

export interface CheckboxProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'> {
  label?: string
  error?: string
  hint?: string
  size?: 'sm' | 'md' | 'lg'
  indeterminate?: boolean
}

const checkboxSizes = {
  sm: 'w-4 h-4',
  md: 'w-5 h-5',
  lg: 'w-6 h-6',
}

export const Checkbox = observer(forwardRef<HTMLInputElement, CheckboxProps>(({
  label,
  error,
  hint,
  size = 'md',
  indeterminate = false,
  className = '',
  disabled,
  ...props
}, ref) => {
  const hasError = !!error
  const sizeClasses = checkboxSizes[size]

  const baseClasses = 'rounded border-2 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2'
  const stateClasses = hasError
    ? 'border-red-300 text-red-600 focus:ring-red-500'
    : 'border-gray-300 text-blue-600 focus:ring-blue-500'
  const disabledClasses = disabled
    ? 'opacity-50 cursor-not-allowed'
    : 'cursor-pointer'

  const checkboxElement = (
    <div className={`relative ${disabled ? 'opacity-50' : ''}`}>
      <input
        ref={ref}
        type="checkbox"
        className={`
          ${baseClasses} ${sizeClasses} ${stateClasses} ${disabledClasses} ${className}
        `}
        disabled={disabled}
        {...props}
      />
      {indeterminate && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-2.5 h-0.5 bg-blue-600 rounded-full"></div>
        </div>
      )}
    </div>
  )

  if (label || error || hint) {
    return (
      <div className="space-y-1">
        <div className="flex items-start space-x-3">
          {checkboxElement}
          <div className="flex-1 min-w-0">
            {label && (
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 cursor-pointer">
                {label}
              </label>
            )}
          </div>
        </div>
        {error && (
          <p className="text-sm text-red-600 dark:text-red-400 ml-8">{error}</p>
        )}
        {hint && !error && (
          <p className="text-sm text-gray-500 dark:text-gray-400 ml-8">{hint}</p>
        )}
      </div>
    )
  }

  return checkboxElement
}))

Checkbox.displayName = 'Checkbox'
