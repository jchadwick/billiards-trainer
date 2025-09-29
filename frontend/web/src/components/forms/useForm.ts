import { useState, useCallback, useRef, useEffect } from 'react'
import {
  FormValidationSchema,
  ValidationResult,
  validateField,
  validateForm,
  isFormValid,
  getFirstError
} from './validation'

export interface UseFormOptions<T extends Record<string, any>> {
  initialValues: T
  validationSchema?: FormValidationSchema
  validateOnChange?: boolean
  validateOnBlur?: boolean
  onSubmit?: (values: T) => void | Promise<void>
  onError?: (errors: Record<string, ValidationResult>) => void
}

export interface FormState<T extends Record<string, any>> {
  values: T
  errors: Record<string, string | null>
  touched: Record<string, boolean>
  isSubmitting: boolean
  isValidating: boolean
  submitCount: number
}

export interface FormActions<T extends Record<string, any>> {
  setValue: (name: keyof T, value: any) => void
  setValues: (values: Partial<T>) => void
  setError: (name: keyof T, error: string | null) => void
  setErrors: (errors: Record<string, string | null>) => void
  setTouched: (name: keyof T, touched?: boolean) => void
  setTouchedFields: (touched: Record<string, boolean>) => void
  resetForm: (newValues?: Partial<T>) => void
  validateField: (name: keyof T) => Promise<boolean>
  validateForm: () => Promise<boolean>
  handleSubmit: (e?: React.FormEvent) => Promise<void>
  getFieldProps: (name: keyof T) => {
    name: keyof T
    value: any
    onChange: (e: React.ChangeEvent<any>) => void
    onBlur: (e: React.FocusEvent<any>) => void
    error?: string
  }
  getFieldState: (name: keyof T) => {
    value: any
    error: string | null
    touched: boolean
    hasError: boolean
  }
}

export interface UseFormReturn<T extends Record<string, any>> extends FormState<T>, FormActions<T> {
  isValid: boolean
  isDirty: boolean
  canSubmit: boolean
}

export function useForm<T extends Record<string, any>>({
  initialValues,
  validationSchema,
  validateOnChange = false,
  validateOnBlur = true,
  onSubmit,
  onError,
}: UseFormOptions<T>): UseFormReturn<T> {
  const [values, setValuesState] = useState<T>(initialValues)
  const [errors, setErrorsState] = useState<Record<string, string | null>>({})
  const [touched, setTouchedState] = useState<Record<string, boolean>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isValidating, setIsValidating] = useState(false)
  const [submitCount, setSubmitCount] = useState(0)

  const initialValuesRef = useRef(initialValues)
  const validationSchemaRef = useRef(validationSchema)

  // Update refs when props change
  useEffect(() => {
    initialValuesRef.current = initialValues
    validationSchemaRef.current = validationSchema
  }, [initialValues, validationSchema])

  // Computed values
  const isValid = Object.values(errors).every(error => !error)
  const isDirty = JSON.stringify(values) !== JSON.stringify(initialValuesRef.current)
  const canSubmit = isValid && !isSubmitting && !isValidating

  // Actions
  const setValue = useCallback((name: keyof T, value: any) => {
    setValuesState(prev => ({ ...prev, [name]: value }))

    if (validateOnChange && validationSchemaRef.current?.[name as string]) {
      setIsValidating(true)
      const result = validateField(value, validationSchemaRef.current[name as string], { ...values, [name]: value })
      setErrorsState(prev => ({ ...prev, [name]: getFirstError(result) }))
      setIsValidating(false)
    }
  }, [values, validateOnChange])

  const setValues = useCallback((newValues: Partial<T>) => {
    setValuesState(prev => ({ ...prev, ...newValues }))
  }, [])

  const setError = useCallback((name: keyof T, error: string | null) => {
    setErrorsState(prev => ({ ...prev, [name]: error }))
  }, [])

  const setErrors = useCallback((newErrors: Record<string, string | null>) => {
    setErrorsState(prev => ({ ...prev, ...newErrors }))
  }, [])

  const setTouched = useCallback((name: keyof T, touchedValue = true) => {
    setTouchedState(prev => ({ ...prev, [name]: touchedValue }))
  }, [])

  const setTouchedFields = useCallback((newTouched: Record<string, boolean>) => {
    setTouchedState(prev => ({ ...prev, ...newTouched }))
  }, [])

  const resetForm = useCallback((newValues?: Partial<T>) => {
    const resetValues = { ...initialValuesRef.current, ...newValues }
    setValuesState(resetValues as T)
    setErrorsState({})
    setTouchedState({})
    setIsSubmitting(false)
    setIsValidating(false)
    setSubmitCount(0)
  }, [])

  const validateFieldAction = useCallback(async (name: keyof T): Promise<boolean> => {
    if (!validationSchemaRef.current?.[name as string]) {
      return true
    }

    setIsValidating(true)

    try {
      const result = validateField(
        values[name],
        validationSchemaRef.current[name as string],
        values
      )

      const error = getFirstError(result)
      setErrorsState(prev => ({ ...prev, [name]: error }))

      return result.isValid
    } finally {
      setIsValidating(false)
    }
  }, [values])

  const validateFormAction = useCallback(async (): Promise<boolean> => {
    if (!validationSchemaRef.current) {
      return true
    }

    setIsValidating(true)

    try {
      const results = validateForm(values, validationSchemaRef.current)
      const newErrors: Record<string, string | null> = {}

      for (const [fieldName, result] of Object.entries(results)) {
        newErrors[fieldName] = getFirstError(result)
      }

      setErrorsState(newErrors)

      const formIsValid = isFormValid(results)

      if (!formIsValid && onError) {
        onError(results)
      }

      return formIsValid
    } finally {
      setIsValidating(false)
    }
  }, [values, onError])

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault()

    setSubmitCount(prev => prev + 1)

    // Mark all fields as touched
    const allTouched: Record<string, boolean> = {}
    Object.keys(values).forEach(key => {
      allTouched[key] = true
    })
    setTouchedState(allTouched)

    // Validate form
    const formIsValid = await validateFormAction()

    if (!formIsValid) {
      return
    }

    if (onSubmit) {
      setIsSubmitting(true)
      try {
        await onSubmit(values)
      } catch (error) {
        console.error('Form submission error:', error)
        // You might want to set a general form error here
      } finally {
        setIsSubmitting(false)
      }
    }
  }, [values, validateFormAction, onSubmit])

  const getFieldProps = useCallback((name: keyof T) => {
    return {
      name,
      value: values[name] ?? '',
      onChange: (e: React.ChangeEvent<any>) => {
        const value = e.target.type === 'checkbox' ? e.target.checked : e.target.value
        setValue(name, value)
      },
      onBlur: (e: React.FocusEvent<any>) => {
        setTouched(name, true)
        if (validateOnBlur) {
          validateFieldAction(name)
        }
      },
      error: touched[name as string] ? errors[name as string] : undefined,
    }
  }, [values, errors, touched, setValue, setTouched, validateOnBlur, validateFieldAction])

  const getFieldState = useCallback((name: keyof T) => {
    return {
      value: values[name],
      error: errors[name as string] || null,
      touched: touched[name as string] || false,
      hasError: !!(touched[name as string] && errors[name as string]),
    }
  }, [values, errors, touched])

  return {
    // State
    values,
    errors,
    touched,
    isSubmitting,
    isValidating,
    submitCount,
    isValid,
    isDirty,
    canSubmit,

    // Actions
    setValue,
    setValues,
    setError,
    setErrors,
    setTouched,
    setTouchedFields,
    resetForm,
    validateField: validateFieldAction,
    validateForm: validateFormAction,
    handleSubmit,
    getFieldProps,
    getFieldState,
  }
}
