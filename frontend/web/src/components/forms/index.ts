/**
 * Form component library for the billiards trainer frontend
 *
 * This module provides a comprehensive set of form components, validation utilities,
 * and form management tools that integrate with the MobX store architecture.
 *
 * Key features:
 * - Type-safe form components with accessibility support
 * - Flexible validation system with common validation rules
 * - Integration with MobX stores for reactive state management
 * - Consistent styling with Tailwind CSS
 * - Responsive design and dark mode support
 */

// Core form components
export { Input, type InputProps } from './Input'
export { Select, type SelectProps, type SelectOption } from './Select'
export { Checkbox, Switch, type CheckboxProps } from './Checkbox'
export { Radio, RadioGroup, type RadioProps, type RadioGroupProps, type RadioOption } from './Radio'
export { Textarea, type TextareaProps } from './Textarea'
export {
  FormButton,
  SubmitButton,
  ResetButton,
  CancelButton,
  DeleteButton,
  type FormButtonProps
} from './Button'

// Form validation system
export {
  validationRules,
  validateField,
  validateForm,
  isFormValid,
  getFirstError,
  getAllErrors,
  commonSchemas,
  type ValidationResult,
  type ValidationRule,
  type FieldValidator,
  type FormValidationSchema
} from './validation'

// Form management hooks and context
export {
  useForm,
  type UseFormOptions,
  type FormState,
  type FormActions,
  type UseFormReturn
} from './useForm'

export {
  FormProvider,
  useFormContext,
  withFormContext,
  Form,
  FormSection,
  FormErrorSummary,
  type FormProviderProps,
  type FormProps,
  type FormSectionProps,
  type FormActionsProps,
  type FormErrorSummaryProps
} from './FormProvider'

export {
  FormActions
} from './FormProvider'

// Composite form components
export {
  FormField,
  TextField,
  SelectField,
  TextareaField,
  CheckboxField,
  RadioField,
  type FormFieldProps,
  type TextFieldProps,
  type SelectFieldProps,
  type TextareaFieldProps,
  type CheckboxFieldProps,
  type RadioFieldProps
} from './FormField'

export {
  FormGroup,
  AddressFormGroup,
  NetworkFormGroup,
  CameraFormGroup,
  type FormGroupProps,
  type AddressFormGroupProps,
  type NetworkFormGroupProps,
  type CameraFormGroupProps
} from './FormGroup'

export {
  FormModal,
  ConfirmationModal,
  ConfigurationModal,
  CreateModal,
  EditModal,
  type FormModalProps,
  type ConfirmationModalProps,
  type ConfigurationModalProps,
  type CreateModalProps,
  type EditModalProps
} from './FormModal'

// Re-export commonly used types
export type FormValues = Record<string, any>
export type FormErrors = Record<string, string | null>
export type FormTouched = Record<string, boolean>

// Utility functions for common form operations
export const createFormSchema = (fields: Record<string, any>): any => fields

export const createInitialValues = <T extends Record<string, any>>(schema: T): T => {
  const initialValues: any = {}

  for (const [key, value] of Object.entries(schema)) {
    if (typeof value === 'string') {
      initialValues[key] = ''
    } else if (typeof value === 'number') {
      initialValues[key] = 0
    } else if (typeof value === 'boolean') {
      initialValues[key] = false
    } else if (Array.isArray(value)) {
      initialValues[key] = []
    } else if (value !== null && typeof value === 'object') {
      initialValues[key] = createInitialValues(value)
    } else {
      initialValues[key] = value
    }
  }

  return initialValues as T
}

// Common form patterns for the billiards trainer application
export const formPatterns = {
  /**
   * Creates a server configuration form with host and port fields
   */
  serverConfig: (initialHost = 'localhost', initialPort = 8080) => ({
    initialValues: {
      host: initialHost,
      port: initialPort,
    },
    validationSchema: {
      host: {
        rules: [validationRules.required()],
      },
      port: {
        rules: [validationRules.required(), validationRules.port()],
      },
    },
  }),

  /**
   * Creates a user authentication form with email and password
   */
  userAuth: () => ({
    initialValues: {
      email: '',
      password: '',
    },
    validationSchema: {
      email: {
        rules: [validationRules.required(), validationRules.email()],
      },
      password: {
        rules: [validationRules.required(), validationRules.minLength(8)],
      },
    },
  }),

  /**
   * Creates a user registration form with email, password, and confirmation
   */
  userRegistration: () => ({
    initialValues: {
      email: '',
      password: '',
      confirmPassword: '',
    },
    validationSchema: {
      email: {
        rules: [validationRules.required(), validationRules.email()],
      },
      password: {
        rules: [validationRules.required(), validationRules.minLength(8)],
      },
      confirmPassword: {
        rules: [validationRules.required(), validationRules.mustMatch('password')],
      },
    },
  }),

  /**
   * Creates a camera configuration form
   */
  cameraConfig: () => ({
    initialValues: {
      deviceId: '',
      resolution: '1920x1080',
      frameRate: 30,
    },
    validationSchema: {
      deviceId: {
        rules: [validationRules.required()],
      },
      resolution: {
        rules: [validationRules.required()],
      },
      frameRate: {
        rules: [validationRules.required(), validationRules.min(1), validationRules.max(120)],
      },
    },
  }),

  /**
   * Creates a table calibration form
   */
  tableCalibration: () => ({
    initialValues: {
      cornerPoints: [],
      tableLength: 9,
      tableWidth: 4.5,
    },
    validationSchema: {
      cornerPoints: {
        rules: [validationRules.required()],
      },
      tableLength: {
        rules: [validationRules.required(), validationRules.positive()],
      },
      tableWidth: {
        rules: [validationRules.required(), validationRules.positive()],
      },
    },
  }),
}

// CSS class utilities for consistent styling
export const formStyles = {
  container: 'space-y-6',
  section: 'space-y-4',
  group: 'space-y-3',
  field: 'space-y-2',
  actions: 'flex items-center justify-end gap-3 pt-6 border-t border-secondary-200 dark:border-secondary-700',
  error: 'text-error-600 dark:text-error-400',
  helper: 'text-secondary-500 dark:text-secondary-400',
  required: 'text-error-500',
}
