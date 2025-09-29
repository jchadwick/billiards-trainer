import React, { forwardRef, useRef, useEffect, useState } from 'react'
import { observer } from 'mobx-react-lite'

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string
  helperText?: string
  error?: string
  variant?: 'default' | 'filled' | 'outlined'
  fullWidth?: boolean
  required?: boolean
  autoResize?: boolean
  minRows?: number
  maxRows?: number
  showCharCount?: boolean
  maxLength?: number
}

const textareaVariants = {
  default: 'border border-secondary-300 bg-white dark:border-secondary-600 dark:bg-secondary-800',
  filled: 'border-0 bg-secondary-100 dark:bg-secondary-800',
  outlined: 'border-2 border-secondary-300 bg-transparent dark:border-secondary-600',
}

const focusStates = {
  default: 'focus:ring-2 focus:ring-primary-500 focus:border-primary-500',
  error: 'focus:ring-2 focus:ring-error-500 focus:border-error-500 border-error-300 dark:border-error-600',
}

export const Textarea = observer(forwardRef<HTMLTextAreaElement, TextareaProps>(({
  label,
  helperText,
  error,
  variant = 'default',
  fullWidth = false,
  required = false,
  autoResize = false,
  minRows = 3,
  maxRows,
  showCharCount = false,
  maxLength,
  className = '',
  disabled,
  id,
  value,
  onChange,
  ...props
}, ref) => {
  const textareaId = id || `textarea-${Math.random().toString(36).substr(2, 9)}`
  const hasError = !!error
  const internalRef = useRef<HTMLTextAreaElement>(null)
  const [charCount, setCharCount] = useState(0)

  // Use the forwarded ref or internal ref
  const textareaRef = (ref as React.RefObject<HTMLTextAreaElement>) || internalRef

  const baseClasses = 'rounded-md transition-colors placeholder-secondary-400 dark:placeholder-secondary-500 disabled:opacity-50 disabled:cursor-not-allowed resize-none'
  const variantClasses = textareaVariants[variant]
  const focusClasses = hasError ? focusStates.error : focusStates.default
  const widthClasses = fullWidth ? 'w-full' : ''
  const textClasses = 'text-secondary-900 dark:text-secondary-100'
  const paddingClasses = 'px-4 py-3 text-sm'

  // Auto-resize functionality
  useEffect(() => {
    if (autoResize && textareaRef.current) {
      const textarea = textareaRef.current
      const adjustHeight = () => {
        textarea.style.height = 'auto'

        const lineHeight = parseInt(window.getComputedStyle(textarea).lineHeight, 10)
        const minHeight = minRows * lineHeight + 24 // 24px for padding
        const maxHeight = maxRows ? maxRows * lineHeight + 24 : Infinity

        const scrollHeight = textarea.scrollHeight
        const newHeight = Math.min(Math.max(scrollHeight, minHeight), maxHeight)

        textarea.style.height = `${newHeight}px`
        textarea.style.overflowY = scrollHeight > maxHeight ? 'auto' : 'hidden'
      }

      adjustHeight()

      // Adjust on content change
      const handleInput = () => adjustHeight()
      textarea.addEventListener('input', handleInput)

      return () => textarea.removeEventListener('input', handleInput)
    }
  }, [autoResize, minRows, maxRows, value])

  // Character count
  useEffect(() => {
    if (showCharCount && value !== undefined) {
      setCharCount(String(value).length)
    }
  }, [value, showCharCount])

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (showCharCount) {
      setCharCount(e.target.value.length)
    }
    onChange?.(e)
  }

  const getRowsProps = () => {
    if (autoResize) {
      return { rows: minRows }
    }
    return props.rows ? { rows: props.rows } : { rows: minRows }
  }

  const textareaElement = (
    <div className={`relative ${fullWidth ? 'w-full' : ''}`}>
      <textarea
        ref={textareaRef}
        id={textareaId}
        className={`
          ${baseClasses}
          ${variantClasses}
          ${focusClasses}
          ${textClasses}
          ${widthClasses}
          ${paddingClasses}
          ${className}
        `}
        disabled={disabled}
        aria-invalid={hasError}
        aria-describedby={
          helperText || error || showCharCount
            ? `${textareaId}-description`
            : undefined
        }
        value={value}
        onChange={handleChange}
        maxLength={maxLength}
        {...getRowsProps()}
        {...props}
      />

      {showCharCount && (
        <div className="absolute bottom-2 right-2 text-xs text-secondary-400 dark:text-secondary-500 bg-white dark:bg-secondary-800 px-1 rounded">
          {charCount}{maxLength && `/${maxLength}`}
        </div>
      )}
    </div>
  )

  if (!label && !helperText && !error && !showCharCount) {
    return textareaElement
  }

  const getDescription = () => {
    const parts = []
    if (error) return error
    if (helperText) parts.push(helperText)
    if (showCharCount && maxLength) {
      parts.push(`${charCount}/${maxLength} characters`)
    }
    return parts.join(' • ')
  }

  return (
    <div className={fullWidth ? 'w-full' : ''}>
      {label && (
        <label
          htmlFor={textareaId}
          className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2"
        >
          {label}
          {required && <span className="text-error-500 ml-1">*</span>}
        </label>
      )}
      {textareaElement}
      {(helperText || error || (showCharCount && maxLength)) && (
        <p
          id={`${textareaId}-description`}
          className={`mt-2 text-sm ${
            error
              ? 'text-error-600 dark:text-error-400'
              : 'text-secondary-500 dark:text-secondary-400'
          }`}
        >
          {getDescription()}
        </p>
      )}
    </div>
  )
}))

Textarea.displayName = 'Textarea'
