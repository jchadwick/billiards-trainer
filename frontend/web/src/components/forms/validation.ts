/**
 * Form validation utilities and common validation rules
 * Provides a flexible validation system that integrates with form components
 */

export type ValidationResult = {
  isValid: boolean
  errors: string[]
}

export type ValidationRule<T = any> = (value: T, allValues?: Record<string, any>) => string | null

export type FieldValidator<T = any> = {
  rules: ValidationRule<T>[]
  validateOnChange?: boolean
  validateOnBlur?: boolean
}

export type FormValidationSchema = Record<string, FieldValidator>

// Common validation rules
export const validationRules = {
  required: (message = 'This field is required'): ValidationRule =>
    (value) => {
      if (value === null || value === undefined || value === '' ||
          (Array.isArray(value) && value.length === 0)) {
        return message
      }
      return null
    },

  minLength: (min: number, message?: string): ValidationRule<string> =>
    (value) => {
      if (typeof value !== 'string') return null
      if (value.length < min) {
        return message || `Must be at least ${min} characters`
      }
      return null
    },

  maxLength: (max: number, message?: string): ValidationRule<string> =>
    (value) => {
      if (typeof value !== 'string') return null
      if (value.length > max) {
        return message || `Must be no more than ${max} characters`
      }
      return null
    },

  email: (message = 'Please enter a valid email address'): ValidationRule<string> =>
    (value) => {
      if (typeof value !== 'string' || !value) return null
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
      if (!emailRegex.test(value)) {
        return message
      }
      return null
    },

  url: (message = 'Please enter a valid URL'): ValidationRule<string> =>
    (value) => {
      if (typeof value !== 'string' || !value) return null
      try {
        new URL(value)
        return null
      } catch {
        return message
      }
    },

  pattern: (regex: RegExp, message = 'Invalid format'): ValidationRule<string> =>
    (value) => {
      if (typeof value !== 'string' || !value) return null
      if (!regex.test(value)) {
        return message
      }
      return null
    },

  min: (min: number, message?: string): ValidationRule<number> =>
    (value) => {
      if (typeof value !== 'number') return null
      if (value < min) {
        return message || `Must be at least ${min}`
      }
      return null
    },

  max: (max: number, message?: string): ValidationRule<number> =>
    (value) => {
      if (typeof value !== 'number') return null
      if (value > max) {
        return message || `Must be no more than ${max}`
      }
      return null
    },

  integer: (message = 'Must be a whole number'): ValidationRule<number> =>
    (value) => {
      if (typeof value !== 'number') return null
      if (!Number.isInteger(value)) {
        return message
      }
      return null
    },

  positive: (message = 'Must be a positive number'): ValidationRule<number> =>
    (value) => {
      if (typeof value !== 'number') return null
      if (value <= 0) {
        return message
      }
      return null
    },

  oneOf: (allowedValues: any[], message?: string): ValidationRule =>
    (value) => {
      if (!allowedValues.includes(value)) {
        return message || `Must be one of: ${allowedValues.join(', ')}`
      }
      return null
    },

  custom: (validator: (value: any, allValues?: Record<string, any>) => boolean, message: string): ValidationRule =>
    (value, allValues) => {
      if (!validator(value, allValues)) {
        return message
      }
      return null
    },

  // Conditional validation
  requiredIf: (condition: (allValues: Record<string, any>) => boolean, message = 'This field is required'): ValidationRule =>
    (value, allValues = {}) => {
      if (condition(allValues)) {
        return validationRules.required(message)(value, allValues)
      }
      return null
    },

  // Cross-field validation
  mustMatch: (fieldName: string, message?: string): ValidationRule =>
    (value, allValues = {}) => {
      if (value !== allValues[fieldName]) {
        return message || `Must match ${fieldName}`
      }
      return null
    },

  // Network validation rules
  port: (message = 'Must be a valid port number (1-65535)'): ValidationRule<number> =>
    (value) => {
      if (typeof value !== 'number') return null
      if (value < 1 || value > 65535 || !Number.isInteger(value)) {
        return message
      }
      return null
    },

  ipAddress: (message = 'Must be a valid IP address'): ValidationRule<string> =>
    (value) => {
      if (typeof value !== 'string' || !value) return null
      const ipRegex = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/
      if (!ipRegex.test(value)) {
        return message
      }
      return null
    },

  // File validation
  fileSize: (maxSizeBytes: number, message?: string): ValidationRule<File> =>
    (value) => {
      if (!(value instanceof File)) return null
      if (value.size > maxSizeBytes) {
        const maxSizeMB = (maxSizeBytes / (1024 * 1024)).toFixed(2)
        return message || `File size must be less than ${maxSizeMB}MB`
      }
      return null
    },

  fileType: (allowedTypes: string[], message?: string): ValidationRule<File> =>
    (value) => {
      if (!(value instanceof File)) return null
      if (!allowedTypes.includes(value.type)) {
        return message || `File type must be one of: ${allowedTypes.join(', ')}`
      }
      return null
    },
}

// Validation utilities
export const validateField = (
  value: any,
  validator: FieldValidator,
  allValues?: Record<string, any>
): ValidationResult => {
  const errors: string[] = []

  for (const rule of validator.rules) {
    const error = rule(value, allValues)
    if (error) {
      errors.push(error)
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
  }
}

export const validateForm = (
  values: Record<string, any>,
  schema: FormValidationSchema
): Record<string, ValidationResult> => {
  const results: Record<string, ValidationResult> = {}

  for (const [fieldName, validator] of Object.entries(schema)) {
    results[fieldName] = validateField(values[fieldName], validator, values)
  }

  return results
}

export const isFormValid = (validationResults: Record<string, ValidationResult>): boolean => {
  return Object.values(validationResults).every(result => result.isValid)
}

export const getFirstError = (validationResult: ValidationResult): string | null => {
  return validationResult.errors.length > 0 ? validationResult.errors[0] : null
}

export const getAllErrors = (validationResults: Record<string, ValidationResult>): string[] => {
  return Object.values(validationResults)
    .flatMap(result => result.errors)
    .filter(Boolean)
}

// Common validation schemas for the billiards trainer
export const commonSchemas = {
  server: {
    host: {
      rules: [
        validationRules.required(),
        validationRules.pattern(
          /^[a-zA-Z0-9.-]+$/,
          'Host must contain only letters, numbers, dots, and hyphens'
        ),
      ],
    },
    port: {
      rules: [
        validationRules.required(),
        validationRules.port(),
      ],
    },
  },

  user: {
    email: {
      rules: [
        validationRules.required(),
        validationRules.email(),
      ],
    },
    password: {
      rules: [
        validationRules.required(),
        validationRules.minLength(8),
        validationRules.pattern(
          /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
          'Password must contain at least one lowercase letter, one uppercase letter, and one number'
        ),
      ],
    },
    confirmPassword: {
      rules: [
        validationRules.required(),
        validationRules.mustMatch('password', 'Passwords must match'),
      ],
    },
  },

  camera: {
    deviceId: {
      rules: [validationRules.required()],
    },
    resolution: {
      rules: [
        validationRules.required(),
        validationRules.pattern(
          /^\d+x\d+$/,
          'Resolution must be in format "width x height" (e.g., 1920x1080)'
        ),
      ],
    },
    frameRate: {
      rules: [
        validationRules.required(),
        validationRules.min(1),
        validationRules.max(120),
        validationRules.integer(),
      ],
    },
  },

  calibration: {
    cornerPoints: {
      rules: [
        validationRules.custom(
          (value) => Array.isArray(value) && value.length === 4,
          'Must provide exactly 4 corner points'
        ),
      ],
    },
    tableLength: {
      rules: [
        validationRules.required(),
        validationRules.positive(),
        validationRules.min(6), // Minimum reasonable pool table length in feet
        validationRules.max(12), // Maximum reasonable pool table length in feet
      ],
    },
    tableWidth: {
      rules: [
        validationRules.required(),
        validationRules.positive(),
        validationRules.min(3), // Minimum reasonable pool table width in feet
        validationRules.max(6), // Maximum reasonable pool table width in feet
      ],
    },
  },
}
