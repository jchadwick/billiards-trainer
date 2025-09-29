import React, { forwardRef, useState } from 'react'
import { observer } from 'mobx-react-lite'

export interface SliderProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string
  error?: string
  hint?: string
  showValue?: boolean
  formatValue?: (value: number) => string
  fullWidth?: boolean
}

export const Slider = observer(forwardRef<HTMLInputElement, SliderProps>(({
  label,
  error,
  hint,
  showValue = true,
  formatValue,
  fullWidth = false,
  className = '',
  min = 0,
  max = 100,
  step = 1,
  value,
  onChange,
  ...props
}, ref) => {
  const [localValue, setLocalValue] = useState(value || min)
  const currentValue = value !== undefined ? value : localValue
  const hasError = !!error

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = Number(e.target.value)
    if (value === undefined) {
      setLocalValue(newValue)
    }
    onChange?.(e)
  }

  const displayValue = formatValue ? formatValue(Number(currentValue)) : String(currentValue)
  const percentage = ((Number(currentValue) - Number(min)) / (Number(max) - Number(min))) * 100

  const sliderElement = (
    <div className={`relative ${fullWidth ? 'w-full' : ''}`}>
      <div className="flex items-center space-x-3">
        <div className="flex-1 relative">
          {/* Track */}
          <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
            {/* Progress */}
            <div
              className={`h-2 rounded-full transition-all ${
                hasError ? 'bg-red-500' : 'bg-blue-500'
              }`}
              style={{ width: `${percentage}%` }}
            />
          </div>

          {/* Input */}
          <input
            ref={ref}
            type="range"
            className={`
              absolute inset-0 w-full h-2 bg-transparent appearance-none cursor-pointer focus:outline-none
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5
              [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-blue-500
              [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:cursor-pointer
              [&::-webkit-slider-thumb]:transition-all [&::-webkit-slider-thumb]:hover:scale-110
              [&::-moz-range-thumb]:w-5 [&::-moz-range-thumb]:h-5 [&::-moz-range-thumb]:bg-white
              [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-blue-500 [&::-moz-range-thumb]:rounded-full
              [&::-moz-range-thumb]:shadow-md [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:transition-all
              [&::-moz-range-track]:bg-transparent
              ${hasError ? '[&::-webkit-slider-thumb]:border-red-500 [&::-moz-range-thumb]:border-red-500' : ''}
              ${className}
            `}
            min={min}
            max={max}
            step={step}
            value={currentValue}
            onChange={handleChange}
            {...props}
          />
        </div>

        {showValue && (
          <div className="text-sm font-medium text-gray-700 dark:text-gray-300 min-w-12 text-right">
            {displayValue}
          </div>
        )}
      </div>
    </div>
  )

  if (label || error || hint) {
    return (
      <div className={`space-y-2 ${fullWidth ? 'w-full' : ''}`}>
        {label && (
          <div className="flex items-center justify-between">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              {label}
            </label>
            {showValue && (
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {displayValue}
              </span>
            )}
          </div>
        )}
        {sliderElement}
        {error && (
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        )}
        {hint && !error && (
          <p className="text-sm text-gray-500 dark:text-gray-400">{hint}</p>
        )}
      </div>
    )
  }

  return sliderElement
}))

Slider.displayName = 'Slider'
