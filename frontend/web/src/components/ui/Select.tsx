import React, { forwardRef } from 'react'
import { observer } from 'mobx-react-lite'

export interface SelectOption {
  value: string | number
  label: string
  disabled?: boolean
}

export interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, 'size'> {
  label?: string
  error?: string
  hint?: string
  size?: 'sm' | 'md' | 'lg'
  options: SelectOption[]
  placeholder?: string
  loading?: boolean
  fullWidth?: boolean
}

const selectSizes = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-sm',
  lg: 'px-4 py-3 text-base',
}

export const Select = observer(forwardRef<HTMLSelectElement, SelectProps>(({
  label,
  error,
  hint,
  size = 'md',
  options,
  placeholder,
  loading = false,
  fullWidth = false,
  className = '',
  disabled,
  ...props
}, ref) => {
  const isDisabled = disabled || loading
  const hasError = !!error

  const baseClasses = 'border rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 appearance-none'
  const sizeClasses = selectSizes[size]
  const stateClasses = hasError
    ? 'border-red-300 focus:border-red-500 focus:ring-red-500'
    : 'border-gray-300 hover:border-gray-400 focus:border-blue-500 focus:ring-blue-500'
  const backgroundClasses = isDisabled
    ? 'bg-gray-50 text-gray-400 cursor-not-allowed'
    : 'bg-white text-gray-900 dark:bg-gray-800 dark:text-gray-100'
  const widthClasses = fullWidth ? 'w-full' : ''

  const selectElement = (
    <div className={`relative ${fullWidth ? 'w-full' : ''}`}>
      <select
        ref={ref}
        className={`
          ${baseClasses} ${sizeClasses} ${stateClasses} ${backgroundClasses} ${widthClasses}
          pr-10 ${className}
        `}
        disabled={isDisabled}
        {...props}
      >
        {placeholder && (
          <option value="" disabled>
            {placeholder}
          </option>
        )}
        {options.map((option) => (
          <option
            key={option.value}
            value={option.value}
            disabled={option.disabled}
          >
            {option.label}
          </option>
        ))}
      </select>

      <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
        {loading ? (
          <svg
            className="w-4 h-4 animate-spin text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        ) : (
          <svg
            className="w-5 h-5 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        )}
      </div>
    </div>
  )

  if (label || error || hint) {
    return (
      <div className={`space-y-1 ${fullWidth ? 'w-full' : ''}`}>
        {label && (
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            {label}
          </label>
        )}
        {selectElement}
        {error && (
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        )}
        {hint && !error && (
          <p className="text-sm text-gray-500 dark:text-gray-400">{hint}</p>
        )}
      </div>
    )
  }

  return selectElement
}))

Select.displayName = 'Select'
