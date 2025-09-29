import React, { createContext, useContext, ReactNode } from 'react'
import { observer } from 'mobx-react-lite'
import { UseFormReturn } from './useForm'

// Generic form context that can work with any form data structure
interface FormContextValue<T extends Record<string, any> = Record<string, any>> {
  form: UseFormReturn<T>
}

const FormContext = createContext<FormContextValue | null>(null)

export interface FormProviderProps<T extends Record<string, any>> {
  form: UseFormReturn<T>
  children: ReactNode
}

export const FormProvider = observer(<T extends Record<string, any>>({
  form,
  children,
}: FormProviderProps<T>) => {
  return (
    <FormContext.Provider value={{ form: form as UseFormReturn<Record<string, any>> }}>
      {children}
    </FormContext.Provider>
  )
})

FormProvider.displayName = 'FormProvider'

// Hook to access form context
export const useFormContext = <T extends Record<string, any> = Record<string, any>>(): UseFormReturn<T> => {
  const context = useContext(FormContext)
  if (!context) {
    throw new Error('useFormContext must be used within a FormProvider')
  }
  return context.form as UseFormReturn<T>
}

// Higher-order component to wrap form fields with context
export const withFormContext = <P extends object>(
  Component: React.ComponentType<P>
) => {
  const WrappedComponent = observer((props: P) => {
    const form = useFormContext()
    return <Component {...props} form={form} />
  })

  WrappedComponent.displayName = `withFormContext(${Component.displayName || Component.name})`
  return WrappedComponent
}

// Form component that automatically handles form submission
export interface FormProps<T extends Record<string, any>> extends React.FormHTMLAttributes<HTMLFormElement> {
  form: UseFormReturn<T>
  children: ReactNode
  className?: string
}

export const Form = observer(<T extends Record<string, any>>({
  form,
  children,
  className = '',
  onSubmit,
  ...props
}: FormProps<T>) => {
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    await form.handleSubmit(e)
    onSubmit?.(e)
  }

  return (
    <FormProvider form={form}>
      <form
        onSubmit={handleSubmit}
        className={className}
        noValidate // We handle validation ourselves
        {...props}
      >
        {children}
      </form>
    </FormProvider>
  )
})

Form.displayName = 'Form'

// Form section for organizing related fields
export interface FormSectionProps {
  title?: string
  description?: string
  children: ReactNode
  className?: string
  collapsible?: boolean
  defaultCollapsed?: boolean
}

export const FormSection = observer(({
  title,
  description,
  children,
  className = '',
  collapsible = false,
  defaultCollapsed = false,
}: FormSectionProps) => {
  const [isCollapsed, setIsCollapsed] = React.useState(defaultCollapsed)

  return (
    <div className={`space-y-4 ${className}`}>
      {(title || description) && (
        <div className="border-b border-secondary-200 dark:border-secondary-700 pb-4">
          {title && (
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-secondary-900 dark:text-secondary-100">
                {title}
              </h3>
              {collapsible && (
                <button
                  type="button"
                  onClick={() => setIsCollapsed(!isCollapsed)}
                  className="p-2 text-secondary-400 hover:text-secondary-600 dark:hover:text-secondary-300 transition-colors"
                  aria-expanded={!isCollapsed}
                  aria-label={isCollapsed ? 'Expand section' : 'Collapse section'}
                >
                  <svg
                    className={`w-5 h-5 transition-transform ${isCollapsed ? 'rotate-0' : 'rotate-90'}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              )}
            </div>
          )}
          {description && (
            <p className="mt-1 text-sm text-secondary-500 dark:text-secondary-400">
              {description}
            </p>
          )}
        </div>
      )}

      {(!collapsible || !isCollapsed) && (
        <div className="space-y-6">
          {children}
        </div>
      )}
    </div>
  )
})

FormSection.displayName = 'FormSection'

// Form actions container for submit/cancel buttons
export interface FormActionsProps {
  children: ReactNode
  className?: string
  align?: 'left' | 'center' | 'right'
  sticky?: boolean
}

export const FormActions = observer(({
  children,
  className = '',
  align = 'right',
  sticky = false,
}: FormActionsProps) => {
  const alignClasses = {
    left: 'justify-start',
    center: 'justify-center',
    right: 'justify-end',
  }

  const baseClasses = `flex items-center gap-3 ${alignClasses[align]}`
  const stickyClasses = sticky
    ? 'sticky bottom-0 bg-white dark:bg-secondary-900 border-t border-secondary-200 dark:border-secondary-700 p-4 -mx-4 -mb-4'
    : 'pt-6 border-t border-secondary-200 dark:border-secondary-700'

  return (
    <div className={`${baseClasses} ${stickyClasses} ${className}`}>
      {children}
    </div>
  )
})

FormActions.displayName = 'FormActions'

// Form error summary component
export interface FormErrorSummaryProps {
  className?: string
  title?: string
}

export const FormErrorSummary = observer(({
  className = '',
  title = 'Please correct the following errors:',
}: FormErrorSummaryProps) => {
  const form = useFormContext()

  const errors = Object.entries(form.errors)
    .filter(([_, error]) => error)
    .map(([field, error]) => ({ field, error: error! }))

  if (errors.length === 0 || form.submitCount === 0) {
    return null
  }

  return (
    <div className={`rounded-md bg-error-50 dark:bg-error-900/20 p-4 ${className}`}>
      <div className="flex">
        <div className="flex-shrink-0">
          <svg
            className="h-5 w-5 text-error-400"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clipRule="evenodd"
            />
          </svg>
        </div>
        <div className="ml-3">
          <h3 className="text-sm font-medium text-error-800 dark:text-error-200">
            {title}
          </h3>
          <div className="mt-2 text-sm text-error-700 dark:text-error-300">
            <ul className="list-disc pl-5 space-y-1">
              {errors.map(({ field, error }) => (
                <li key={field}>
                  <span className="font-medium capitalize">{field}:</span> {error}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
})

FormErrorSummary.displayName = 'FormErrorSummary'
