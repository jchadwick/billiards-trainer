import React from 'react'
import { observer } from 'mobx-react-lite'
import {
  useForm,
  Form,
  FormSection,
  FormGroup,
  TextField,
  SelectField,
  CheckboxField,
  NetworkFormGroup,
  CameraFormGroup,
  FormActions,
  SubmitButton,
  ResetButton,
  FormErrorSummary,
  validationRules,
  FormModal,
  formPatterns,
} from '../index'
import { useStores } from '../../../hooks/useStores'

// Example configuration form data structure
interface ConfigurationFormData {
  server: {
    host: string
    port: number
    protocol: string
    enableSsl: boolean
  }
  camera: {
    deviceId: string
    resolution: string
    frameRate: number
    enableRecording: boolean
  }
  system: {
    debugMode: boolean
    logLevel: string
    autoStart: boolean
  }
}

const protocolOptions = [
  { value: 'http', label: 'HTTP' },
  { value: 'https', label: 'HTTPS' },
  { value: 'ws', label: 'WebSocket' },
  { value: 'wss', label: 'Secure WebSocket' },
]

const logLevelOptions = [
  { value: 'error', label: 'Error' },
  { value: 'warn', label: 'Warning' },
  { value: 'info', label: 'Info' },
  { value: 'debug', label: 'Debug' },
]

const resolutionOptions = [
  { value: '640x480', label: '640x480 (VGA)' },
  { value: '1280x720', label: '1280x720 (HD)' },
  { value: '1920x1080', label: '1920x1080 (Full HD)' },
  { value: '2560x1440', label: '2560x1440 (QHD)' },
  { value: '3840x2160', label: '3840x2160 (4K)' },
]

const frameRateOptions = [
  { value: '15', label: '15 fps' },
  { value: '24', label: '24 fps' },
  { value: '30', label: '30 fps' },
  { value: '60', label: '60 fps' },
  { value: '120', label: '120 fps' },
]

export const ConfigurationFormExample = observer(() => {
  const { configStore } = useStores()
  const [showModal, setShowModal] = React.useState(false)

  // Initialize form with validation schema
  const form = useForm<ConfigurationFormData>({
    initialValues: {
      server: {
        host: 'localhost',
        port: 8080,
        protocol: 'http',
        enableSsl: false,
      },
      camera: {
        deviceId: '',
        resolution: '1920x1080',
        frameRate: 30,
        enableRecording: false,
      },
      system: {
        debugMode: false,
        logLevel: 'info',
        autoStart: true,
      },
    },
    validationSchema: {
      'server.host': {
        rules: [
          validationRules.required(),
          validationRules.pattern(
            /^[a-zA-Z0-9.-]+$/,
            'Host must contain only letters, numbers, dots, and hyphens'
          ),
        ],
      },
      'server.port': {
        rules: [
          validationRules.required(),
          validationRules.port(),
        ],
      },
      'server.protocol': {
        rules: [validationRules.required()],
      },
      'camera.deviceId': {
        rules: [validationRules.required('Please select a camera device')],
      },
      'camera.resolution': {
        rules: [validationRules.required()],
      },
      'camera.frameRate': {
        rules: [
          validationRules.required(),
          validationRules.min(1),
          validationRules.max(120),
        ],
      },
      'system.logLevel': {
        rules: [validationRules.required()],
      },
    },
    validateOnBlur: true,
    onSubmit: async (values) => {
      console.log('Form submitted with values:', values)

      // Simulate saving to MobX store
      try {
        // Update config store with new values
        await configStore.updateConfiguration(values)

        // Show success message
        alert('Configuration saved successfully!')
      } catch (error) {
        console.error('Failed to save configuration:', error)
        alert('Failed to save configuration. Please try again.')
      }
    },
  })

  const handleReset = () => {
    form.resetForm()
  }

  const handleLoadDefaults = () => {
    form.setValues({
      server: {
        host: 'localhost',
        port: 8080,
        protocol: 'http',
        enableSsl: false,
      },
      camera: {
        deviceId: 'default',
        resolution: '1920x1080',
        frameRate: 30,
        enableRecording: false,
      },
      system: {
        debugMode: false,
        logLevel: 'info',
        autoStart: true,
      },
    })
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-secondary-900 dark:text-secondary-100">
          System Configuration
        </h1>
        <p className="mt-2 text-secondary-600 dark:text-secondary-400">
          Configure your billiards trainer system settings
        </p>
      </div>

      <Form form={form} className="space-y-8">
        <FormErrorSummary />

        <FormSection
          title="Server Configuration"
          description="Configure the connection to the billiards trainer server"
          collapsible
        >
          <FormGroup layout="grid" columns={3}>
            <SelectField
              name="server.protocol"
              label="Protocol"
              options={protocolOptions}
              required
            />

            <TextField
              name="server.host"
              label="Server Host"
              placeholder="localhost"
              helperText="IP address or hostname"
              required
            />

            <TextField
              name="server.port"
              label="Port"
              type="text"
              placeholder="8080"
              helperText="Port number (1-65535)"
              required
            />
          </FormGroup>

          <CheckboxField
            name="server.enableSsl"
            label="Enable SSL/TLS encryption"
            helperText="Use secure connection when available"
          />
        </FormSection>

        <FormSection
          title="Camera Settings"
          description="Configure video capture and processing settings"
          collapsible
        >
          <FormGroup layout="grid" columns={2}>
            <TextField
              name="camera.deviceId"
              label="Camera Device"
              placeholder="Select or enter device ID"
              helperText="Camera device identifier"
              required
            />

            <SelectField
              name="camera.resolution"
              label="Video Resolution"
              options={resolutionOptions}
              searchable
              required
            />

            <SelectField
              name="camera.frameRate"
              label="Frame Rate"
              options={frameRateOptions}
              required
            />

            <div className="space-y-3">
              <CheckboxField
                name="camera.enableRecording"
                label="Enable video recording"
                helperText="Save video sessions for analysis"
              />
            </div>
          </FormGroup>
        </FormSection>

        <FormSection
          title="System Settings"
          description="General system behavior and debugging options"
          collapsible
        >
          <FormGroup layout="grid" columns={2}>
            <SelectField
              name="system.logLevel"
              label="Log Level"
              options={logLevelOptions}
              helperText="Amount of detail in system logs"
              required
            />

            <div className="space-y-3">
              <CheckboxField
                name="system.debugMode"
                label="Enable debug mode"
                helperText="Show additional debugging information"
              />

              <CheckboxField
                name="system.autoStart"
                label="Auto-start system"
                helperText="Automatically start when application opens"
              />
            </div>
          </FormGroup>
        </FormSection>

        <FormActions>
          <button
            type="button"
            onClick={handleLoadDefaults}
            className="btn btn-secondary"
          >
            Load Defaults
          </button>

          <button
            type="button"
            onClick={() => setShowModal(true)}
            className="btn btn-secondary"
          >
            Preview in Modal
          </button>

          <ResetButton onClick={handleReset}>
            Reset Form
          </ResetButton>

          <SubmitButton loading={form.isSubmitting}>
            Save Configuration
          </SubmitButton>
        </FormActions>
      </Form>

      {/* Example of form in modal */}
      {showModal && (
        <FormModal
          form={form}
          title="Configuration Preview"
          description="Review your configuration settings"
          onClose={() => setShowModal(false)}
          onSubmit={async (values) => {
            console.log('Modal form submitted:', values)
          }}
          submitText="Apply Settings"
          size="lg"
        >
          <div className="space-y-4">
            <div className="bg-secondary-50 dark:bg-secondary-800 p-4 rounded-lg">
              <h3 className="font-medium text-secondary-900 dark:text-secondary-100 mb-2">
                Current Configuration
              </h3>
              <pre className="text-sm text-secondary-600 dark:text-secondary-400 overflow-auto">
                {JSON.stringify(form.values, null, 2)}
              </pre>
            </div>

            <div className="text-sm text-secondary-500 dark:text-secondary-400">
              <p>Form state:</p>
              <ul className="list-disc pl-5 space-y-1">
                <li>Valid: {form.isValid ? 'Yes' : 'No'}</li>
                <li>Dirty: {form.isDirty ? 'Yes' : 'No'}</li>
                <li>Submitting: {form.isSubmitting ? 'Yes' : 'No'}</li>
                <li>Submit count: {form.submitCount}</li>
              </ul>
            </div>
          </div>
        </FormModal>
      )}
    </div>
  )
})

ConfigurationFormExample.displayName = 'ConfigurationFormExample'

// Simple example demonstrating all form components
export const FormComponentsShowcase = observer(() => {
  const form = useForm({
    initialValues: {
      textField: '',
      emailField: '',
      passwordField: '',
      selectField: '',
      multiSelectField: [],
      textareaField: '',
      checkboxField: false,
      switchField: false,
      radioField: '',
      numberField: 0,
    },
    validationSchema: {
      textField: { rules: [validationRules.required(), validationRules.minLength(3)] },
      emailField: { rules: [validationRules.required(), validationRules.email()] },
      passwordField: { rules: [validationRules.required(), validationRules.minLength(8)] },
      selectField: { rules: [validationRules.required()] },
      radioField: { rules: [validationRules.required()] },
    },
    onSubmit: async (values) => {
      console.log('Showcase form submitted:', values)
      alert('Form submitted! Check console for values.')
    },
  })

  const selectOptions = [
    { value: 'option1', label: 'Option 1' },
    { value: 'option2', label: 'Option 2' },
    { value: 'option3', label: 'Option 3' },
  ]

  const radioOptions = [
    { value: 'choice1', label: 'Choice 1', helperText: 'First choice' },
    { value: 'choice2', label: 'Choice 2', helperText: 'Second choice' },
    { value: 'choice3', label: 'Choice 3', helperText: 'Third choice' },
  ]

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h2 className="text-xl font-bold mb-6">Form Components Showcase</h2>

      <Form form={form} className="space-y-6">
        <FormErrorSummary />

        <TextField
          name="textField"
          label="Text Field"
          placeholder="Enter some text"
          helperText="This is a required field with minimum 3 characters"
          required
        />

        <TextField
          name="emailField"
          label="Email Field"
          type="email"
          placeholder="user@example.com"
          required
        />

        <TextField
          name="passwordField"
          label="Password Field"
          type="password"
          placeholder="Enter password"
          required
        />

        <SelectField
          name="selectField"
          label="Select Field"
          options={selectOptions}
          placeholder="Choose an option"
          searchable
          required
        />

        <SelectField
          name="multiSelectField"
          label="Multi-Select Field"
          options={selectOptions}
          placeholder="Choose multiple options"
          multiSelect
          maxSelection={2}
        />

        <TextareaField
          name="textareaField"
          label="Textarea Field"
          placeholder="Enter multiple lines of text"
          autoResize
          minRows={3}
          maxRows={6}
          showCharCount
          maxLength={500}
        />

        <CheckboxField
          name="checkboxField"
          label="Checkbox Field"
          helperText="Check this box to enable the option"
        />

        <CheckboxField
          name="switchField"
          label="Switch Field"
          variant="switch"
          helperText="Toggle this switch on or off"
        />

        <RadioField
          name="radioField"
          label="Radio Field"
          options={radioOptions}
          orientation="vertical"
          required
        />

        <FormActions>
          <ResetButton>Reset</ResetButton>
          <SubmitButton>Submit Showcase</SubmitButton>
        </FormActions>
      </Form>
    </div>
  )
})

FormComponentsShowcase.displayName = 'FormComponentsShowcase'
