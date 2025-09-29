import React from 'react'
import { observer } from 'mobx-react-lite'

export interface FormGroupProps {
  title?: string
  description?: string
  children: React.ReactNode
  className?: string
  spacing?: 'tight' | 'normal' | 'loose'
  layout?: 'vertical' | 'horizontal' | 'grid'
  columns?: number
  required?: boolean
  disabled?: boolean
}

const spacingClasses = {
  tight: 'space-y-3',
  normal: 'space-y-4',
  loose: 'space-y-6',
}

const gridColumns = {
  1: 'grid-cols-1',
  2: 'grid-cols-1 md:grid-cols-2',
  3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
  4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4',
  5: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5',
  6: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6',
}

export const FormGroup = observer(({
  title,
  description,
  children,
  className = '',
  spacing = 'normal',
  layout = 'vertical',
  columns = 2,
  required = false,
  disabled = false,
}: FormGroupProps) => {
  const getLayoutClasses = () => {
    switch (layout) {
      case 'horizontal':
        return 'flex flex-wrap gap-4'
      case 'grid':
        const columnClass = gridColumns[Math.min(Math.max(columns, 1), 6) as keyof typeof gridColumns]
        return `grid ${columnClass} gap-4`
      default:
        return spacingClasses[spacing]
    }
  }

  const headerElement = (title || description) && (
    <div className="mb-6">
      {title && (
        <h3 className="text-base font-semibold text-secondary-900 dark:text-secondary-100 flex items-center">
          {title}
          {required && (
            <span className="text-error-500 ml-1" aria-label="required">*</span>
          )}
        </h3>
      )}
      {description && (
        <p className="mt-1 text-sm text-secondary-500 dark:text-secondary-400">
          {description}
        </p>
      )}
    </div>
  )

  const contentElement = (
    <div className={getLayoutClasses()}>
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child) && disabled) {
          // Pass disabled prop to form field children
          return React.cloneElement(child, { disabled: true } as any)
        }
        return child
      })}
    </div>
  )

  return (
    <fieldset
      disabled={disabled}
      className={`
        ${disabled ? 'opacity-60 pointer-events-none' : ''}
        ${className}
      `}
    >
      {title && <legend className="sr-only">{title}</legend>}
      {headerElement}
      {contentElement}
    </fieldset>
  )
})

FormGroup.displayName = 'FormGroup'

// Specialized form groups for common patterns
export interface AddressFormGroupProps extends Omit<FormGroupProps, 'children'> {
  streetField?: string
  cityField?: string
  stateField?: string
  zipField?: string
  countryField?: string
  includeCountry?: boolean
}

export const AddressFormGroup = observer(({
  streetField = 'address.street',
  cityField = 'address.city',
  stateField = 'address.state',
  zipField = 'address.zip',
  countryField = 'address.country',
  includeCountry = false,
  title = 'Address',
  layout = 'grid',
  columns = 2,
  ...props
}: AddressFormGroupProps) => {
  const { TextField, SelectField } = React.useMemo(() => {
    // Import components to avoid circular dependencies
    return {
      TextField: React.lazy(() =>
        import('./FormField').then(module => ({ default: module.TextField }))
      ),
      SelectField: React.lazy(() =>
        import('./FormField').then(module => ({ default: module.SelectField }))
      ),
    }
  }, [])

  const stateOptions = [
    { value: 'AL', label: 'Alabama' },
    { value: 'AK', label: 'Alaska' },
    { value: 'AZ', label: 'Arizona' },
    { value: 'AR', label: 'Arkansas' },
    { value: 'CA', label: 'California' },
    // Add more states as needed...
  ]

  const countryOptions = [
    { value: 'US', label: 'United States' },
    { value: 'CA', label: 'Canada' },
    { value: 'MX', label: 'Mexico' },
    // Add more countries as needed...
  ]

  return (
    <FormGroup
      title={title}
      layout={layout}
      columns={columns}
      {...props}
    >
      <React.Suspense fallback={<div>Loading...</div>}>
        <div className="col-span-full">
          <TextField
            name={streetField}
            label="Street Address"
            placeholder="1234 Main St"
            autoComplete="street-address"
          />
        </div>

        <TextField
          name={cityField}
          label="City"
          placeholder="San Francisco"
          autoComplete="address-level2"
        />

        <SelectField
          name={stateField}
          label="State"
          placeholder="Select state"
          options={stateOptions}
          searchable
        />

        <TextField
          name={zipField}
          label="ZIP / Postal Code"
          placeholder="12345"
          autoComplete="postal-code"
        />

        {includeCountry && (
          <SelectField
            name={countryField}
            label="Country"
            placeholder="Select country"
            options={countryOptions}
            searchable
          />
        )}
      </React.Suspense>
    </FormGroup>
  )
})

AddressFormGroup.displayName = 'AddressFormGroup'

export interface NetworkFormGroupProps extends Omit<FormGroupProps, 'children'> {
  hostField?: string
  portField?: string
  protocolField?: string
  includeProtocol?: boolean
}

export const NetworkFormGroup = observer(({
  hostField = 'host',
  portField = 'port',
  protocolField = 'protocol',
  includeProtocol = false,
  title = 'Network Configuration',
  layout = 'grid',
  columns = includeProtocol ? 3 : 2,
  ...props
}: NetworkFormGroupProps) => {
  const { TextField, SelectField } = React.useMemo(() => {
    return {
      TextField: React.lazy(() =>
        import('./FormField').then(module => ({ default: module.TextField }))
      ),
      SelectField: React.lazy(() =>
        import('./FormField').then(module => ({ default: module.SelectField }))
      ),
    }
  }, [])

  const protocolOptions = [
    { value: 'http', label: 'HTTP' },
    { value: 'https', label: 'HTTPS' },
    { value: 'ws', label: 'WebSocket' },
    { value: 'wss', label: 'Secure WebSocket' },
    { value: 'tcp', label: 'TCP' },
    { value: 'udp', label: 'UDP' },
  ]

  return (
    <FormGroup
      title={title}
      layout={layout}
      columns={columns}
      {...props}
    >
      <React.Suspense fallback={<div>Loading...</div>}>
        {includeProtocol && (
          <SelectField
            name={protocolField}
            label="Protocol"
            placeholder="Select protocol"
            options={protocolOptions}
          />
        )}

        <TextField
          name={hostField}
          label="Host"
          placeholder="localhost"
          helperText="IP address or hostname"
        />

        <TextField
          name={portField}
          label="Port"
          type="text"
          placeholder="8080"
          helperText="Port number (1-65535)"
        />
      </React.Suspense>
    </FormGroup>
  )
})

NetworkFormGroup.displayName = 'NetworkFormGroup'

export interface CameraFormGroupProps extends Omit<FormGroupProps, 'children'> {
  deviceField?: string
  resolutionField?: string
  frameRateField?: string
  includeAdvanced?: boolean
}

export const CameraFormGroup = observer(({
  deviceField = 'camera.deviceId',
  resolutionField = 'camera.resolution',
  frameRateField = 'camera.frameRate',
  includeAdvanced = false,
  title = 'Camera Settings',
  layout = 'grid',
  columns = 3,
  ...props
}: CameraFormGroupProps) => {
  const { TextField, SelectField } = React.useMemo(() => {
    return {
      TextField: React.lazy(() =>
        import('./FormField').then(module => ({ default: module.TextField }))
      ),
      SelectField: React.lazy(() =>
        import('./FormField').then(module => ({ default: module.SelectField }))
      ),
    }
  }, [])

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

  return (
    <FormGroup
      title={title}
      layout={layout}
      columns={columns}
      {...props}
    >
      <React.Suspense fallback={<div>Loading...</div>}>
        <TextField
          name={deviceField}
          label="Camera Device"
          placeholder="Default camera"
          helperText="Camera device ID or name"
        />

        <SelectField
          name={resolutionField}
          label="Resolution"
          placeholder="Select resolution"
          options={resolutionOptions}
        />

        <SelectField
          name={frameRateField}
          label="Frame Rate"
          placeholder="Select frame rate"
          options={frameRateOptions}
        />
      </React.Suspense>
    </FormGroup>
  )
})

CameraFormGroup.displayName = 'CameraFormGroup'
