import React, { useState } from 'react'
import { observer } from 'mobx-react-lite'
import { useStores } from '../../stores/context'
import axiosClient from '../../api/axios-client'
import { Card, CardHeader, CardTitle, CardContent, Input, Select, Checkbox, Button, LoadingSpinner } from '../ui'
import type { SelectOption } from '../ui'

export const SystemSettings = observer(() => {
  const { configStore } = useStores()
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})

  // API Settings
  const [apiUrl, setApiUrl] = useState(axiosClient.getBaseURL())
  const [apiTimeout, setApiTimeout] = useState(10000)
  const [enableRetries, setEnableRetries] = useState(true)
  const [maxRetries, setMaxRetries] = useState(3)

  // Authentication Settings
  const [tokenExpiration, setTokenExpiration] = useState(3600)
  const [enableRefreshToken, setEnableRefreshToken] = useState(true)
  const [sessionTimeout, setSessionTimeout] = useState(1800)

  // Logging Settings
  const [logLevel, setLogLevel] = useState<string>('info')
  const [enableFileLogging, setEnableFileLogging] = useState(true)
  const [logRotation, setLogRotation] = useState(true)
  const [maxLogSize, setMaxLogSize] = useState(10)

  // Performance Settings
  const [enableCaching, setEnableCaching] = useState(true)
  const [cacheSize, setCacheSize] = useState(100)
  const [enableCompression, setEnableCompression] = useState(true)
  const [compressionLevel, setCompressionLevel] = useState(6)

  const logLevelOptions: SelectOption[] = [
    { value: 'debug', label: 'Debug' },
    { value: 'info', label: 'Information' },
    { value: 'warning', label: 'Warning' },
    { value: 'error', label: 'Error' },
    { value: 'critical', label: 'Critical' }
  ]

  const handleSave = async () => {
    setLoading(true)
    setErrors({})

    try {
      // Validate inputs
      const newErrors: Record<string, string> = {}

      if (!apiUrl.trim()) {
        newErrors.apiUrl = 'API URL is required'
      } else if (!apiUrl.match(/^https?:\/\/.+/)) {
        newErrors.apiUrl = 'API URL must be a valid HTTP/HTTPS URL'
      }

      if (apiTimeout < 1000 || apiTimeout > 60000) {
        newErrors.apiTimeout = 'API timeout must be between 1000ms and 60000ms'
      }

      if (maxRetries < 0 || maxRetries > 10) {
        newErrors.maxRetries = 'Max retries must be between 0 and 10'
      }

      if (tokenExpiration < 300 || tokenExpiration > 86400) {
        newErrors.tokenExpiration = 'Token expiration must be between 5 minutes and 24 hours'
      }

      if (sessionTimeout < 300 || sessionTimeout > 14400) {
        newErrors.sessionTimeout = 'Session timeout must be between 5 minutes and 4 hours'
      }

      if (maxLogSize < 1 || maxLogSize > 1000) {
        newErrors.maxLogSize = 'Max log size must be between 1MB and 1000MB'
      }

      if (cacheSize < 10 || cacheSize > 1000) {
        newErrors.cacheSize = 'Cache size must be between 10 and 1000 items'
      }

      if (compressionLevel < 1 || compressionLevel > 9) {
        newErrors.compressionLevel = 'Compression level must be between 1 and 9'
      }

      if (Object.keys(newErrors).length > 0) {
        setErrors(newErrors)
        return
      }

      // Save system configuration
      const systemConfig = {
        api: {
          baseUrl: apiUrl,
          timeout: apiTimeout,
          enableRetries,
          maxRetries
        },
        auth: {
          tokenExpiration,
          enableRefreshToken,
          sessionTimeout
        },
        logging: {
          level: logLevel,
          enableFileLogging,
          enableRotation: logRotation,
          maxFileSize: maxLogSize
        },
        performance: {
          enableCaching,
          cacheSize,
          enableCompression,
          compressionLevel
        }
      }

      // Here you would typically save to the backend API
      console.log('Saving system configuration:', systemConfig)

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))

    } catch (error) {
      console.error('Failed to save system settings:', error)
      setErrors({ general: 'Failed to save system settings. Please try again.' })
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setApiUrl(axiosClient.getBaseURL())
    setApiTimeout(10000)
    setEnableRetries(true)
    setMaxRetries(3)
    setTokenExpiration(3600)
    setEnableRefreshToken(true)
    setSessionTimeout(1800)
    setLogLevel('info')
    setEnableFileLogging(true)
    setLogRotation(true)
    setMaxLogSize(10)
    setEnableCaching(true)
    setCacheSize(100)
    setEnableCompression(true)
    setCompressionLevel(6)
    setErrors({})
  }

  return (
    <div className="space-y-6">
      {errors.general && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-800">{errors.general}</p>
            </div>
          </div>
        </div>
      )}

      {/* API Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>API Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="API Base URL"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              error={errors.apiUrl}
              placeholder={axiosClient.getBaseURL()}
              fullWidth
            />

            <Input
              label="Request Timeout (ms)"
              type="number"
              value={apiTimeout}
              onChange={(e) => setApiTimeout(Number(e.target.value))}
              error={errors.apiTimeout}
              min={1000}
              max={60000}
              fullWidth
            />

            <Checkbox
              label="Enable Request Retries"
              checked={enableRetries}
              onChange={(e) => setEnableRetries(e.target.checked)}
            />

            <Input
              label="Maximum Retries"
              type="number"
              value={maxRetries}
              onChange={(e) => setMaxRetries(Number(e.target.value))}
              error={errors.maxRetries}
              disabled={!enableRetries}
              min={0}
              max={10}
              fullWidth
            />
          </div>
        </CardContent>
      </Card>

      {/* Authentication Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Authentication Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Token Expiration (seconds)"
              type="number"
              value={tokenExpiration}
              onChange={(e) => setTokenExpiration(Number(e.target.value))}
              error={errors.tokenExpiration}
              min={300}
              max={86400}
              hint="300 seconds (5 minutes) to 86400 seconds (24 hours)"
              fullWidth
            />

            <Input
              label="Session Timeout (seconds)"
              type="number"
              value={sessionTimeout}
              onChange={(e) => setSessionTimeout(Number(e.target.value))}
              error={errors.sessionTimeout}
              min={300}
              max={14400}
              hint="300 seconds (5 minutes) to 14400 seconds (4 hours)"
              fullWidth
            />

            <div className="md:col-span-2">
              <Checkbox
                label="Enable Refresh Tokens"
                checked={enableRefreshToken}
                onChange={(e) => setEnableRefreshToken(e.target.checked)}
                hint="Allow automatic token renewal without re-authentication"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Logging Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Logging Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Select
              label="Log Level"
              value={logLevel}
              onChange={(e) => setLogLevel(e.target.value)}
              options={logLevelOptions}
              fullWidth
            />

            <Input
              label="Maximum Log File Size (MB)"
              type="number"
              value={maxLogSize}
              onChange={(e) => setMaxLogSize(Number(e.target.value))}
              error={errors.maxLogSize}
              disabled={!enableFileLogging}
              min={1}
              max={1000}
              fullWidth
            />

            <Checkbox
              label="Enable File Logging"
              checked={enableFileLogging}
              onChange={(e) => setEnableFileLogging(e.target.checked)}
              hint="Write logs to files in addition to console output"
            />

            <Checkbox
              label="Enable Log Rotation"
              checked={logRotation}
              onChange={(e) => setLogRotation(e.target.checked)}
              disabled={!enableFileLogging}
              hint="Automatically rotate log files when they reach maximum size"
            />
          </div>
        </CardContent>
      </Card>

      {/* Performance Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Checkbox
              label="Enable Caching"
              checked={enableCaching}
              onChange={(e) => setEnableCaching(e.target.checked)}
              hint="Cache API responses and computed results for better performance"
            />

            <Input
              label="Cache Size (items)"
              type="number"
              value={cacheSize}
              onChange={(e) => setCacheSize(Number(e.target.value))}
              error={errors.cacheSize}
              disabled={!enableCaching}
              min={10}
              max={1000}
              fullWidth
            />

            <Checkbox
              label="Enable Compression"
              checked={enableCompression}
              onChange={(e) => setEnableCompression(e.target.checked)}
              hint="Compress API requests and responses to reduce bandwidth usage"
            />

            <Input
              label="Compression Level"
              type="number"
              value={compressionLevel}
              onChange={(e) => setCompressionLevel(Number(e.target.value))}
              error={errors.compressionLevel}
              disabled={!enableCompression}
              min={1}
              max={9}
              hint="1 = fastest, 9 = best compression"
              fullWidth
            />
          </div>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <div className="flex justify-end space-x-4">
        <Button
          variant="outline"
          onClick={handleReset}
          disabled={loading}
        >
          Reset to Defaults
        </Button>
        <Button
          onClick={handleSave}
          loading={loading}
          disabled={loading}
        >
          {loading ? 'Saving...' : 'Save Settings'}
        </Button>
      </div>
    </div>
  )
})
