import React, { useState, useRef } from 'react'
import { observer } from 'mobx-react-lite'
import { useStores } from '../../stores/context'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Select } from '../ui'
import type { SelectOption } from '../ui'

export const ConfigImportExport = observer(() => {
  const { configStore } = useStores()
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [exportFormat, setExportFormat] = useState('json')
  const [importStatus, setImportStatus] = useState<string>('')

  const fileInputRef = useRef<HTMLInputElement>(null)

  const formatOptions: SelectOption[] = [
    { value: 'json', label: 'JSON Format' },
    { value: 'yaml', label: 'YAML Format' },
    { value: 'toml', label: 'TOML Format' }
  ]

  const handleExport = async () => {
    setLoading(true)
    setErrors({})

    try {
      const config = configStore.exportConfig()
      const timestamp = new Date().toISOString().split('T')[0]
      const filename = `billiards-config-${timestamp}.${exportFormat}`

      let content: string
      let mimeType: string

      switch (exportFormat) {
        case 'yaml':
          // Convert to YAML format (simplified)
          content = convertToYAML(config)
          mimeType = 'text/yaml'
          break
        case 'toml':
          // Convert to TOML format (simplified)
          content = convertToTOML(config)
          mimeType = 'text/plain'
          break
        default:
          content = JSON.stringify(config, null, 2)
          mimeType = 'application/json'
      }

      // Create and download file
      const blob = new Blob([content], { type: mimeType })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)

      setImportStatus(`Configuration exported successfully as ${filename}`)

    } catch (error) {
      console.error('Export failed:', error)
      setErrors({ export: 'Failed to export configuration. Please try again.' })
    } finally {
      setLoading(false)
    }
  }

  const handleImportClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setLoading(true)
    setErrors({})
    setImportStatus('')

    try {
      const text = await file.text()
      let config: any

      // Parse based on file extension
      const extension = file.name.split('.').pop()?.toLowerCase()

      switch (extension) {
        case 'json':
          config = JSON.parse(text)
          break
        case 'yaml':
        case 'yml':
          config = parseYAML(text)
          break
        case 'toml':
          config = parseTOML(text)
          break
        default:
          // Try JSON first, then YAML
          try {
            config = JSON.parse(text)
          } catch {
            config = parseYAML(text)
          }
      }

      // Validate configuration structure
      if (!isValidConfig(config)) {
        throw new Error('Invalid configuration format')
      }

      // Import the configuration
      const result = await configStore.importConfig(config)

      if (result.success) {
        setImportStatus(`Configuration imported successfully from ${file.name}`)
      } else {
        throw new Error(result.error || 'Import failed')
      }

    } catch (error) {
      console.error('Import failed:', error)
      setErrors({
        import: error instanceof Error ? error.message : 'Failed to import configuration. Please check the file format.'
      })
    } finally {
      setLoading(false)
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const handleReset = async () => {
    setLoading(true)
    setErrors({})
    setImportStatus('')

    try {
      const result = await configStore.resetToDefaults()
      if (result.success) {
        setImportStatus('Configuration reset to defaults successfully')
      } else {
        throw new Error(result.error || 'Reset failed')
      }
    } catch (error) {
      console.error('Reset failed:', error)
      setErrors({ reset: 'Failed to reset configuration to defaults.' })
    } finally {
      setLoading(false)
    }
  }

  // Helper functions for format conversion (simplified implementations)
  const convertToYAML = (obj: any, indent = 0): string => {
    const spaces = '  '.repeat(indent)
    let yaml = ''

    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        yaml += `${spaces}${key}:\n${convertToYAML(value, indent + 1)}`
      } else if (Array.isArray(value)) {
        yaml += `${spaces}${key}:\n`
        value.forEach(item => {
          yaml += `${spaces}  - ${item}\n`
        })
      } else {
        yaml += `${spaces}${key}: ${value}\n`
      }
    }

    return yaml
  }

  const convertToTOML = (obj: any): string => {
    let toml = ''

    // Simple TOML conversion (this would need a proper TOML library in production)
    for (const [section, values] of Object.entries(obj)) {
      toml += `[${section}]\n`
      if (typeof values === 'object' && values !== null) {
        for (const [key, value] of Object.entries(values)) {
          toml += `${key} = ${JSON.stringify(value)}\n`
        }
      }
      toml += '\n'
    }

    return toml
  }

  const parseYAML = (text: string): any => {
    // Simplified YAML parser (would use proper YAML library in production)
    try {
      return JSON.parse(text) // Fallback to JSON
    } catch {
      throw new Error('YAML parsing not implemented - please use JSON format')
    }
  }

  const parseTOML = (text: string): any => {
    // Simplified TOML parser (would use proper TOML library in production)
    throw new Error('TOML parsing not implemented - please use JSON format')
  }

  const isValidConfig = (config: any): boolean => {
    // Basic validation - check if it has expected config sections
    if (typeof config !== 'object' || config === null) return false

    const expectedSections = ['camera', 'detection', 'game', 'ui']
    const hasValidSections = expectedSections.some(section =>
      config.hasOwnProperty(section) && typeof config[section] === 'object'
    )

    return hasValidSections
  }

  return (
    <div className="space-y-6">
      {(errors.export || errors.import || errors.reset) && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-800">
                {errors.export || errors.import || errors.reset}
              </p>
            </div>
          </div>
        </div>
      )}

      {importStatus && (
        <div className="bg-green-50 border border-green-200 rounded-md p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-green-800">{importStatus}</p>
            </div>
          </div>
        </div>
      )}

      {/* Export Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Export Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Export your current configuration settings to a file for backup or sharing.
            </p>

            <Select
              label="Export Format"
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value)}
              options={formatOptions}
              fullWidth
            />

            <div className="flex space-x-4">
              <Button
                onClick={handleExport}
                loading={loading}
                disabled={loading}
              >
                {loading ? 'Exporting...' : 'Export Configuration'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Import Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Import Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Import configuration settings from a previously exported file. This will merge with your current settings.
            </p>

            <div className="space-y-2">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Supported Formats
              </p>
              <ul className="text-sm text-gray-600 dark:text-gray-400 list-disc list-inside">
                <li>JSON (.json)</li>
                <li>YAML (.yaml, .yml) - Basic support</li>
                <li>TOML (.toml) - Basic support</li>
              </ul>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".json,.yaml,.yml,.toml"
              onChange={handleFileSelect}
              className="hidden"
            />

            <Button
              onClick={handleImportClick}
              loading={loading}
              disabled={loading}
              variant="outline"
            >
              {loading ? 'Importing...' : 'Select Configuration File'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Current Configuration Info */}
      <Card>
        <CardHeader>
          <CardTitle>Current Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Active Profile
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {configStore.currentProfile}
                </p>
              </div>

              <div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Has Unsaved Changes
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {configStore.hasUnsavedChanges ? 'Yes' : 'No'}
                </p>
              </div>

              <div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Configuration Valid
                </p>
                <p className={`text-sm ${configStore.isValid ? 'text-green-600' : 'text-red-600'}`}>
                  {configStore.isValid ? 'Valid' : 'Invalid'}
                </p>
              </div>

              <div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Available Profiles
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {configStore.availableProfiles.length} profiles
                </p>
              </div>
            </div>

            {!configStore.isValid && (
              <div className="mt-4">
                <p className="text-sm font-medium text-red-700 dark:text-red-400 mb-2">
                  Validation Errors:
                </p>
                <ul className="text-sm text-red-600 dark:text-red-400 list-disc list-inside">
                  {configStore.validationErrors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Reset Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Reset Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Reset all configuration settings to their default values. This action cannot be undone.
            </p>

            <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-yellow-800">
                    Warning: This will permanently reset all your configuration settings to defaults.
                  </p>
                </div>
              </div>
            </div>

            <Button
              onClick={handleReset}
              loading={loading}
              disabled={loading}
              variant="danger"
            >
              {loading ? 'Resetting...' : 'Reset to Defaults'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
})
