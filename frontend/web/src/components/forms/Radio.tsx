import React, { forwardRef } from 'react'
import { observer } from 'mobx-react-lite'

export interface RadioOption {
  value: string | number
  label: string
  helperText?: string
  disabled?: boolean
}

export interface RadioProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'> {
  label?: string
  helperText?: string
  error?: string
  size?: 'sm' | 'md' | 'lg'
  orientation?: 'horizontal' | 'vertical'
  options?: RadioOption[]
  value?: string | number
  onValueChange?: (value: string | number) => void
}

export interface RadioGroupProps {
  label?: string
  helperText?: string
  error?: string
  required?: boolean
  size?: 'sm' | 'md' | 'lg'
  orientation?: 'horizontal' | 'vertical'
  options: RadioOption[]
  value?: string | number
  name: string
  onValueChange?: (value: string | number) => void
  className?: string
  disabled?: boolean
}

const radioSizes = {
  sm: 'w-4 h-4',
  md: 'w-5 h-5',
  lg: 'w-6 h-6',
}

const labelSizes = {
  sm: 'text-sm',
  md: 'text-sm',
  lg: 'text-base',
}

export const Radio = observer(forwardRef<HTMLInputElement, RadioProps>(({
  label,
  helperText,
  error,
  size = 'md',
  className = '',
  disabled,
  id,
  checked,
  value,
  ...props
}, ref) => {
  const radioId = id || `radio-${Math.random().toString(36).substr(2, 9)}`
  const hasError = !!error

  const sizeClasses = radioSizes[size]
  const labelSizeClasses = labelSizes[size]

  const inputClasses = `
    ${sizeClasses}
    border-2 transition-colors cursor-pointer
    ${hasError
      ? 'border-error-300 dark:border-error-600 focus:ring-error-500'
      : 'border-secondary-300 dark:border-secondary-600 focus:ring-primary-500'
    }
    ${checked
      ? 'bg-primary-600 border-primary-600 text-primary-600'
      : 'bg-white dark:bg-secondary-800'
    }
    focus:ring-2 focus:ring-offset-2
    disabled:opacity-50 disabled:cursor-not-allowed
  `

  const radioElement = (
    <div className="relative flex items-center">
      <input
        ref={ref}
        type="radio"
        id={radioId}
        className={inputClasses}
        disabled={disabled}
        checked={checked}
        value={value}
        aria-invalid={hasError}
        aria-describedby={
          helperText || error ? `${radioId}-description` : undefined
        }
        {...props}
      />
      {label && (
        <label
          htmlFor={radioId}
          className={`
            ml-2 ${labelSizeClasses}
            font-medium text-secondary-700 dark:text-secondary-300 cursor-pointer
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          {label}
        </label>
      )}
    </div>
  )

  if (!helperText && !error) {
    return <div className={className}>{radioElement}</div>
  }

  return (
    <div className={className}>
      {radioElement}
      {(helperText || error) && (
        <p
          id={`${radioId}-description`}
          className={`mt-1 ml-6 text-sm ${
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

Radio.displayName = 'Radio'

export const RadioGroup = observer<RadioGroupProps>(({
  label,
  helperText,
  error,
  required = false,
  size = 'md',
  orientation = 'vertical',
  options,
  value,
  name,
  onValueChange,
  className = '',
  disabled,
}) => {
  const groupId = `radio-group-${Math.random().toString(36).substr(2, 9)}`
  const hasError = !!error

  const handleChange = (optionValue: string | number) => {
    onValueChange?.(optionValue)
  }

  const containerClasses = orientation === 'horizontal'
    ? 'flex flex-wrap gap-6'
    : 'space-y-3'

  return (
    <fieldset className={className} aria-invalid={hasError}>
      {label && (
        <legend className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-3">
          {label}
          {required && <span className="text-error-500 ml-1">*</span>}
        </legend>
      )}

      <div
        className={containerClasses}
        role="radiogroup"
        aria-labelledby={label ? `${groupId}-label` : undefined}
        aria-describedby={
          helperText || error ? `${groupId}-description` : undefined
        }
      >
        {options.map((option, index) => {
          const isChecked = value === option.value
          const isDisabled = disabled || option.disabled

          return (
            <div key={option.value} className="relative">
              <Radio
                name={name}
                value={option.value}
                checked={isChecked}
                disabled={isDisabled}
                size={size}
                label={option.label}
                helperText={option.helperText}
                error={hasError && isChecked ? error : undefined}
                onChange={() => handleChange(option.value)}
              />
            </div>
          )
        })}
      </div>

      {(helperText || error) && (
        <p
          id={`${groupId}-description`}
          className={`mt-3 text-sm ${
            error
              ? 'text-error-600 dark:text-error-400'
              : 'text-secondary-500 dark:text-secondary-400'
          }`}
        >
          {error || helperText}
        </p>
      )}
    </fieldset>
  )
})

RadioGroup.displayName = 'RadioGroup'
