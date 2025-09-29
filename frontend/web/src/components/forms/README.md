# Form Component Library

A comprehensive form component library for the billiards trainer frontend, built with React, TypeScript, MobX, and Tailwind CSS.

## Features

- **Type-safe**: Full TypeScript support with proper type definitions
- **Accessible**: WCAG-compliant components with ARIA labels, keyboard navigation, and screen reader support
- **Reactive**: Integrates seamlessly with MobX stores for reactive state management
- **Flexible**: Supports both controlled and uncontrolled component patterns
- **Validation**: Comprehensive validation system with common validation rules
- **Responsive**: Mobile-first design with dark mode support
- **Composable**: Build complex forms using simple, reusable components

## Quick Start

```tsx
import { useForm, Form, TextField, SubmitButton, validationRules } from '@/components/forms'

function MyForm() {
  const form = useForm({
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
    onSubmit: async (values) => {
      console.log('Form submitted:', values)
    },
  })

  return (
    <Form form={form}>
      <TextField
        name="email"
        label="Email"
        type="email"
        required
      />

      <TextField
        name="password"
        label="Password"
        type="password"
        required
      />

      <SubmitButton>Sign In</SubmitButton>
    </Form>
  )
}
```

## Core Components

### Input Components

- **Input**: Text input with validation, error states, and various types
- **Select**: Dropdown with search, multi-select, and grouping support
- **Checkbox**: Checkbox and switch variants
- **Radio**: Radio buttons and radio groups
- **Textarea**: Multi-line text input with auto-resize
- **Button**: Form buttons with loading states and confirmation

### Form Management

- **useForm**: React hook for form state management and validation
- **FormProvider**: React context for sharing form state
- **Form**: Form wrapper component that handles submission

### Composite Components

- **FormField**: Wrapper component with label, error display, and help text
- **FormGroup**: Grouped form elements with consistent spacing
- **FormSection**: Section dividers with collapsible content
- **FormModal**: Modal wrapper for form dialogs

## Validation

The library includes a comprehensive validation system:

```tsx
import { validationRules } from '@/components/forms'

const schema = {
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
        'Must contain uppercase, lowercase, and number'
      ),
    ],
  },
  confirmPassword: {
    rules: [
      validationRules.required(),
      validationRules.mustMatch('password'),
    ],
  },
}
```

### Available Validation Rules

- `required(message?)`: Field is required
- `minLength(min, message?)`: Minimum string length
- `maxLength(max, message?)`: Maximum string length
- `email(message?)`: Valid email format
- `url(message?)`: Valid URL format
- `pattern(regex, message)`: Custom regex pattern
- `min(min, message?)`: Minimum number value
- `max(max, message?)`: Maximum number value
- `integer(message?)`: Must be an integer
- `positive(message?)`: Must be positive number
- `port(message?)`: Valid port number (1-65535)
- `ipAddress(message?)`: Valid IP address
- `mustMatch(fieldName, message?)`: Must match another field
- `custom(validator, message)`: Custom validation function

## MobX Integration

The form components integrate seamlessly with MobX stores:

```tsx
import { observer } from 'mobx-react-lite'
import { useStores } from '@/hooks/useStores'

const ConfigForm = observer(() => {
  const { configStore } = useStores()

  const form = useForm({
    initialValues: configStore.config,
    onSubmit: async (values) => {
      await configStore.updateConfiguration(values)
    },
  })

  return (
    <Form form={form}>
      {/* Form fields */}
    </Form>
  )
})
```

## Styling and Theming

The components use Tailwind CSS for styling and support:

- **Dark mode**: Automatic dark mode support
- **Responsive design**: Mobile-first responsive layouts
- **Custom colors**: Uses the project's color palette
- **Accessibility**: High contrast mode support
- **Reduced motion**: Respects user motion preferences

## Examples

### Server Configuration Form

```tsx
import { NetworkFormGroup, FormSection } from '@/components/forms'

function ServerConfigForm() {
  const form = useForm({
    initialValues: {
      host: 'localhost',
      port: 8080,
      protocol: 'http',
    },
    validationSchema: {
      host: { rules: [validationRules.required()] },
      port: { rules: [validationRules.required(), validationRules.port()] },
      protocol: { rules: [validationRules.required()] },
    },
  })

  return (
    <Form form={form}>
      <FormSection title="Server Configuration">
        <NetworkFormGroup
          hostField="host"
          portField="port"
          protocolField="protocol"
          includeProtocol
        />
      </FormSection>
    </Form>
  )
}
```

### Modal Form

```tsx
import { FormModal, ConfigurationModal } from '@/components/forms'

function ConfigModal({ isOpen, onClose }) {
  const form = useForm({
    initialValues: { /* ... */ },
    onSubmit: async (values) => {
      // Save configuration
      onClose()
    },
  })

  return (
    <ConfigurationModal
      form={form}
      configType="Camera"
      isOpen={isOpen}
      onClose={onClose}
    >
      {/* Form content */}
    </ConfigurationModal>
  )
}
```

## Accessibility Features

- **ARIA labels**: Proper labeling for screen readers
- **Keyboard navigation**: Full keyboard support
- **Focus management**: Logical focus order
- **Error announcements**: Screen reader error notifications
- **High contrast**: Support for high contrast mode
- **Reduced motion**: Respects motion preferences

## Best Practices

1. **Always use TypeScript**: Define proper types for form data
2. **Validate on blur**: Use `validateOnBlur: true` for better UX
3. **Group related fields**: Use FormGroup and FormSection for organization
4. **Provide helpful error messages**: Use clear, actionable error text
5. **Use semantic HTML**: Components generate proper form markup
6. **Test with screen readers**: Ensure accessibility compliance
7. **Handle loading states**: Show loading indicators during submission

## File Structure

```
src/components/forms/
├── index.ts              # Main exports
├── README.md            # This documentation
├── validation.ts        # Validation utilities
├── useForm.ts          # Form management hook
├── FormProvider.tsx    # Form context provider
├── Input.tsx           # Text input component
├── Select.tsx          # Select/dropdown component
├── Checkbox.tsx        # Checkbox and switch components
├── Radio.tsx           # Radio button components
├── Textarea.tsx        # Textarea component
├── Button.tsx          # Form button components
├── FormField.tsx       # Field wrapper components
├── FormGroup.tsx       # Form grouping components
├── FormModal.tsx       # Modal form components
└── examples/           # Example implementations
    └── ConfigurationFormExample.tsx
```

This form component library provides everything needed to build robust, accessible, and user-friendly forms for the billiards trainer application.
