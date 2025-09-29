import React, { forwardRef, useState, useRef, useEffect } from 'react'
import { observer } from 'mobx-react-lite'

export interface SelectOption {
  value: string | number
  label: string
  disabled?: boolean
  group?: string
}

export interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, 'size'> {
  label?: string
  helperText?: string
  error?: string
  variant?: 'default' | 'filled' | 'outlined'
  size?: 'sm' | 'md' | 'lg'
  fullWidth?: boolean
  required?: boolean
  placeholder?: string
  options: SelectOption[]
  searchable?: boolean
  multiSelect?: boolean
  maxSelection?: number
  onSelectionChange?: (values: (string | number)[]) => void
}

const selectVariants = {
  default: 'border border-secondary-300 bg-white dark:border-secondary-600 dark:bg-secondary-800',
  filled: 'border-0 bg-secondary-100 dark:bg-secondary-800',
  outlined: 'border-2 border-secondary-300 bg-transparent dark:border-secondary-600',
}

const selectSizes = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-sm',
  lg: 'px-4 py-3 text-base',
}

export const Select = observer(forwardRef<HTMLSelectElement, SelectProps>(({
  label,
  helperText,
  error,
  variant = 'default',
  size = 'md',
  fullWidth = false,
  required = false,
  placeholder,
  options,
  searchable = false,
  multiSelect = false,
  maxSelection,
  onSelectionChange,
  className = '',
  disabled,
  id,
  value,
  onChange,
  ...props
}, ref) => {
  const selectId = id || `select-${Math.random().toString(36).substr(2, 9)}`
  const hasError = !!error
  const [isOpen, setIsOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedValues, setSelectedValues] = useState<(string | number)[]>(
    multiSelect
      ? Array.isArray(value) ? value : []
      : value ? [value] : []
  )
  const dropdownRef = useRef<HTMLDivElement>(null)

  const baseClasses = 'rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed'
  const variantClasses = selectVariants[variant]
  const sizeClasses = selectSizes[size]
  const focusClasses = hasError
    ? 'focus:ring-2 focus:ring-error-500 focus:border-error-500 border-error-300 dark:border-error-600'
    : 'focus:ring-2 focus:ring-primary-500 focus:border-primary-500'
  const widthClasses = fullWidth ? 'w-full' : ''
  const textClasses = 'text-secondary-900 dark:text-secondary-100'

  // Filter options based on search term
  const filteredOptions = searchable
    ? options.filter(option =>
        option.label.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : options

  // Group options if needed
  const groupedOptions = filteredOptions.reduce((acc, option) => {
    const group = option.group || 'default'
    if (!acc[group]) acc[group] = []
    acc[group].push(option)
    return acc
  }, {} as Record<string, SelectOption[]>)

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
        setSearchTerm('')
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleSelection = (optionValue: string | number) => {
    if (multiSelect) {
      let newValues: (string | number)[]
      if (selectedValues.includes(optionValue)) {
        newValues = selectedValues.filter(v => v !== optionValue)
      } else {
        if (maxSelection && selectedValues.length >= maxSelection) {
          return // Don't add if max selection reached
        }
        newValues = [...selectedValues, optionValue]
      }
      setSelectedValues(newValues)
      onSelectionChange?.(newValues)
    } else {
      setSelectedValues([optionValue])
      onSelectionChange?.([optionValue])
      setIsOpen(false)
      setSearchTerm('')
    }
  }

  const getDisplayValue = () => {
    if (selectedValues.length === 0) {
      return placeholder || 'Select...'
    }
    if (multiSelect) {
      if (selectedValues.length === 1) {
        const option = options.find(opt => opt.value === selectedValues[0])
        return option?.label || selectedValues[0]
      }
      return `${selectedValues.length} selected`
    }
    const option = options.find(opt => opt.value === selectedValues[0])
    return option?.label || selectedValues[0]
  }

  // For simple (non-searchable, non-multiselect) cases, use native select
  if (!searchable && !multiSelect) {
    const selectElement = (
      <select
        ref={ref}
        id={selectId}
        className={`
          ${baseClasses}
          ${variantClasses}
          ${sizeClasses}
          ${focusClasses}
          ${textClasses}
          ${widthClasses}
          ${className}
        `}
        disabled={disabled}
        aria-invalid={hasError}
        aria-describedby={
          helperText || error ? `${selectId}-description` : undefined
        }
        value={selectedValues[0] || ''}
        onChange={(e) => {
          const newValue = e.target.value
          setSelectedValues(newValue ? [newValue] : [])
          onSelectionChange?.(newValue ? [newValue] : [])
          onChange?.(e)
        }}
        {...props}
      >
        {placeholder && (
          <option value="" disabled>
            {placeholder}
          </option>
        )}
        {Object.entries(groupedOptions).map(([groupName, groupOptions]) => (
          groupName === 'default' ? (
            groupOptions.map((option) => (
              <option
                key={option.value}
                value={option.value}
                disabled={option.disabled}
              >
                {option.label}
              </option>
            ))
          ) : (
            <optgroup key={groupName} label={groupName}>
              {groupOptions.map((option) => (
                <option
                  key={option.value}
                  value={option.value}
                  disabled={option.disabled}
                >
                  {option.label}
                </option>
              ))}
            </optgroup>
          )
        ))}
      </select>
    )

    if (!label && !helperText && !error) {
      return selectElement
    }

    return (
      <div className={fullWidth ? 'w-full' : ''}>
        {label && (
          <label
            htmlFor={selectId}
            className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2"
          >
            {label}
            {required && <span className="text-error-500 ml-1">*</span>}
          </label>
        )}
        {selectElement}
        {(helperText || error) && (
          <p
            id={`${selectId}-description`}
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
  }

  // Custom dropdown for searchable/multiselect functionality
  const customSelectElement = (
    <div className={`relative ${fullWidth ? 'w-full' : ''}`} ref={dropdownRef}>
      <button
        type="button"
        className={`
          ${baseClasses}
          ${variantClasses}
          ${sizeClasses}
          ${focusClasses}
          ${textClasses}
          ${widthClasses}
          ${className}
          flex items-center justify-between
        `}
        disabled={disabled}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className={selectedValues.length === 0 ? 'text-secondary-400 dark:text-secondary-500' : ''}>
          {getDisplayValue()}
        </span>
        <svg
          className={`w-5 h-5 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-white dark:bg-secondary-800 border border-secondary-300 dark:border-secondary-600 rounded-md shadow-lg max-h-60 overflow-auto">
          {searchable && (
            <div className="p-2 border-b border-secondary-300 dark:border-secondary-600">
              <input
                type="text"
                placeholder="Search..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-3 py-1 text-sm border border-secondary-300 dark:border-secondary-600 rounded bg-white dark:bg-secondary-700 text-secondary-900 dark:text-secondary-100"
                autoFocus
              />
            </div>
          )}
          <div role="listbox" aria-multiselectable={multiSelect}>
            {Object.entries(groupedOptions).map(([groupName, groupOptions]) => (
              <div key={groupName}>
                {groupName !== 'default' && (
                  <div className="px-3 py-2 text-xs font-semibold text-secondary-500 dark:text-secondary-400 uppercase tracking-wider bg-secondary-50 dark:bg-secondary-900">
                    {groupName}
                  </div>
                )}
                {groupOptions.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    role="option"
                    aria-selected={selectedValues.includes(option.value)}
                    disabled={option.disabled}
                    className={`
                      w-full px-3 py-2 text-left text-sm transition-colors
                      ${selectedValues.includes(option.value)
                        ? 'bg-primary-100 dark:bg-primary-900 text-primary-900 dark:text-primary-100'
                        : 'hover:bg-secondary-100 dark:hover:bg-secondary-700'
                      }
                      ${option.disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                    `}
                    onClick={() => !option.disabled && handleSelection(option.value)}
                  >
                    <div className="flex items-center">
                      {multiSelect && (
                        <div className={`mr-2 w-4 h-4 border rounded ${
                          selectedValues.includes(option.value)
                            ? 'bg-primary-600 border-primary-600'
                            : 'border-secondary-300 dark:border-secondary-600'
                        }`}>
                          {selectedValues.includes(option.value) && (
                            <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                            </svg>
                          )}
                        </div>
                      )}
                      <span>{option.label}</span>
                    </div>
                  </button>
                ))}
              </div>
            ))}
            {filteredOptions.length === 0 && (
              <div className="px-3 py-2 text-sm text-secondary-500 dark:text-secondary-400">
                No options found
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )

  if (!label && !helperText && !error) {
    return customSelectElement
  }

  return (
    <div className={fullWidth ? 'w-full' : ''}>
      {label && (
        <label
          className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2"
        >
          {label}
          {required && <span className="text-error-500 ml-1">*</span>}
        </label>
      )}
      {customSelectElement}
      {(helperText || error) && (
        <p
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

Select.displayName = 'Select'
