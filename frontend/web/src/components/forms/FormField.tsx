import React from 'react'
import { observer } from 'mobx-react-lite'
import { useFormContext } from './FormProvider'

export interface FormFieldProps {
  name: string
  label?: string
  helperText?: string
  required?: boolean
  children: React.ReactElement
  className?: string
  labelClassName?: string
  errorClassName?: string
  orientation?: 'vertical' | 'horizontal'
  showErrorIcon?: boolean
}

export const FormField = observer(({
  name,
  label,
  helperText,
  required = false,
  children,
  className = '',
  labelClassName = '',
  errorClassName = '',
  orientation = 'vertical',
  showErrorIcon = true,
}: FormFieldProps) => {
  const form = useFormContext()
  const fieldState = form.getFieldState(name)
  const fieldProps = form.getFieldProps(name)

  const fieldId = `field-${name}`
  const hasError = fieldState.hasError
  const displayError = hasError ? fieldState.error : null

  // Clone the child element and inject form props
  const childElement = React.cloneElement(children, {
    id: fieldId,
    ...fieldProps,
    error: displayError,
    'aria-invalid': hasError,
    'aria-describedby': displayError || helperText ? `${fieldId}-description` : undefined,
  })

  const labelElement = label && (
    <label
      htmlFor={fieldId}
      className={`
        block text-sm font-medium text-secondary-700 dark:text-secondary-300
        ${orientation === 'horizontal' ? 'mb-0' : 'mb-2'}
        ${labelClassName}
      `}
    >
      {label}
      {required && <span className="text-error-500 ml-1" aria-label="required">*</span>}
    </label>
  )

  const descriptionElement = (displayError || helperText) && (
    <div
      id={`${fieldId}-description`}
      className={`mt-2 text-sm flex items-start gap-2 ${
        displayError
          ? `text-error-600 dark:text-error-400 ${errorClassName}`
          : 'text-secondary-500 dark:text-secondary-400'
      }`}
    >
      {displayError && showErrorIcon && (
        <svg
          className="w-4 h-4 mt-0.5 flex-shrink-0"
          fill="currentColor"
          viewBox="0 0 20 20"
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
            clipRule="evenodd"
          />
        </svg>
      )}
      <span>{displayError || helperText}</span>
    </div>
  )

  if (orientation === 'horizontal') {
    return (
      <div className={`grid grid-cols-1 lg:grid-cols-3 gap-4 items-start ${className}`}>
        <div className="lg:pt-2">
          {labelElement}
        </div>
        <div className="lg:col-span-2">
          {childElement}
          {descriptionElement}
        </div>
      </div>
    )
  }

  return (
    <div className={className}>
      {labelElement}
      {childElement}
      {descriptionElement}
    </div>
  )
})

FormField.displayName = 'FormField'

// Specialized form field components for common patterns
export interface TextFieldProps extends Omit<FormFieldProps, 'children'> {
  type?: 'text' | 'email' | 'password' | 'url' | 'tel' | 'search'
  placeholder?: string
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
  autoComplete?: string
  size?: 'sm' | 'md' | 'lg'
  variant?: 'default' | 'filled' | 'outlined'
  fullWidth?: boolean
}

export const TextField = observer(({
  name,
  type = 'text',
  placeholder,
  leftIcon,
  rightIcon,
  autoComplete,
  size,
  variant,
  fullWidth = true,
  ...fieldProps
}: TextFieldProps) => {
  // Import Input component dynamically to avoid circular imports
  const InputComponent = React.lazy(() =>
    import('./Input').then(module => ({ default: module.Input }))
  )

  return (
    <FormField name={name} {...fieldProps}>
      <React.Suspense fallback={<div>Loading...</div>}>
        <InputComponent
          type={type}
          placeholder={placeholder}
          leftIcon={leftIcon}
          rightIcon={rightIcon}
          autoComplete={autoComplete}
          size={size}
          variant={variant}
          fullWidth={fullWidth}
        />
      </React.Suspense>
    </FormField>
  )
})

TextField.displayName = 'TextField'

export interface SelectFieldProps extends Omit<FormFieldProps, 'children'> {
  options: Array<{ value: string | number; label: string; disabled?: boolean; group?: string }>
  placeholder?: string
  searchable?: boolean
  multiSelect?: boolean
  maxSelection?: number
  size?: 'sm' | 'md' | 'lg'
  variant?: 'default' | 'filled' | 'outlined'
  fullWidth?: boolean
}

export const SelectField = observer(({
  name,
  options,
  placeholder,
  searchable,
  multiSelect,
  maxSelection,
  size,
  variant,
  fullWidth = true,
  ...fieldProps
}: SelectFieldProps) => {
  const SelectComponent = React.lazy(() =>
    import('./Select').then(module => ({ default: module.Select }))
  )

  return (
    <FormField name={name} {...fieldProps}>
      <React.Suspense fallback={<div>Loading...</div>}>
        <SelectComponent
          options={options}
          placeholder={placeholder}
          searchable={searchable}
          multiSelect={multiSelect}
          maxSelection={maxSelection}
          size={size}
          variant={variant}
          fullWidth={fullWidth}
        />
      </React.Suspense>
    </FormField>
  )
})

SelectField.displayName = 'SelectField'

export interface TextareaFieldProps extends Omit<FormFieldProps, 'children'> {
  placeholder?: string
  autoResize?: boolean
  minRows?: number
  maxRows?: number
  showCharCount?: boolean
  maxLength?: number
  variant?: 'default' | 'filled' | 'outlined'
  fullWidth?: boolean
}

export const TextareaField = observer(({
  name,
  placeholder,
  autoResize,
  minRows,
  maxRows,
  showCharCount,
  maxLength,
  variant,
  fullWidth = true,
  ...fieldProps
}: TextareaFieldProps) => {
  const TextareaComponent = React.lazy(() =>
    import('./Textarea').then(module => ({ default: module.Textarea }))
  )

  return (
    <FormField name={name} {...fieldProps}>
      <React.Suspense fallback={<div>Loading...</div>}>
        <TextareaComponent
          placeholder={placeholder}
          autoResize={autoResize}
          minRows={minRows}
          maxRows={maxRows}
          showCharCount={showCharCount}
          maxLength={maxLength}
          variant={variant}
          fullWidth={fullWidth}
        />
      </React.Suspense>
    </FormField>
  )
})

TextareaField.displayName = 'TextareaField'

export interface CheckboxFieldProps extends Omit<FormFieldProps, 'children' | 'orientation'> {
  size?: 'sm' | 'md' | 'lg'
  variant?: 'checkbox' | 'switch'
  labelPosition?: 'left' | 'right'
}

export const CheckboxField = observer(({
  name,
  size,
  variant,
  labelPosition,
  label,
  ...fieldProps
}: CheckboxFieldProps) => {
  const CheckboxComponent = React.lazy(() =>
    import('./Checkbox').then(module => ({ default: module.Checkbox }))
  )

  return (
    <FormField name={name} {...fieldProps}>
      <React.Suspense fallback={<div>Loading...</div>}>
        <CheckboxComponent
          label={label}
          size={size}
          variant={variant}
          labelPosition={labelPosition}
        />
      </React.Suspense>
    </FormField>
  )
})

CheckboxField.displayName = 'CheckboxField'

export interface RadioFieldProps extends Omit<FormFieldProps, 'children'> {
  options: Array<{ value: string | number; label: string; helperText?: string; disabled?: boolean }>
  size?: 'sm' | 'md' | 'lg'
  orientation?: 'horizontal' | 'vertical'
}

export const RadioField = observer(({
  name,
  options,
  size,
  orientation: radioOrientation = 'vertical',
  ...fieldProps
}: RadioFieldProps) => {
  const RadioGroupComponent = React.lazy(() =>
    import('./Radio').then(module => ({ default: module.RadioGroup }))
  )

  return (
    <FormField name={name} {...fieldProps}>
      <React.Suspense fallback={<div>Loading...</div>}>
        <RadioGroupComponent
          name={name}
          options={options}
          size={size}
          orientation={radioOrientation}
        />
      </React.Suspense>
    </FormField>
  )
})

RadioField.displayName = 'RadioField'
