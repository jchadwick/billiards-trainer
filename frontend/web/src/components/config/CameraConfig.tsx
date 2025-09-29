import React, { useState, useEffect } from 'react'
import { observer } from 'mobx-react-lite'
import { useStores } from '../../stores/context'
import { Card, CardHeader, CardTitle, CardContent, Input, Select, Checkbox, Button, Slider } from '../ui'
import type { SelectOption } from '../ui'

interface CameraDevice {
  id: string
  label: string
  kind: string
}

export const CameraConfig = observer(() => {
  const { configStore } = useStores()
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [availableCameras, setAvailableCameras] = useState<CameraDevice[]>([])
  const [previewEnabled, setPreviewEnabled] = useState(false)

  // Camera settings from store
  const cameraConfig = configStore.cameraConfig

  // Local state for form values
  const [selectedCameraId, setSelectedCameraId] = useState(cameraConfig.selectedCameraId || '')
  const [resolution, setResolution] = useState(cameraConfig.resolution)
  const [fps, setFps] = useState(cameraConfig.fps)
  const [autoExposure, setAutoExposure] = useState(cameraConfig.autoExposure)
  const [exposure, setExposure] = useState(cameraConfig.exposure)
  const [brightness, setBrightness] = useState(cameraConfig.brightness)
  const [contrast, setContrast] = useState(cameraConfig.contrast)

  // Advanced settings
  const [whiteBalance, setWhiteBalance] = useState(5000)
  const [saturation, setSaturation] = useState(1.0)
  const [sharpness, setSharpness] = useState(0.5)
  const [enableHDR, setEnableHDR] = useState(false)
  const [denoise, setDenoise] = useState(true)

  const resolutionOptions: SelectOption[] = [
    { value: '640x480', label: '640 × 480 (VGA)' },
    { value: '1280x720', label: '1280 × 720 (HD)' },
    { value: '1920x1080', label: '1920 × 1080 (Full HD)' },
    { value: '2560x1440', label: '2560 × 1440 (QHD)' },
    { value: '3840x2160', label: '3840 × 2160 (4K)' }
  ]

  const fpsOptions: SelectOption[] = [
    { value: '15', label: '15 FPS' },
    { value: '24', label: '24 FPS' },
    { value: '30', label: '30 FPS' },
    { value: '60', label: '60 FPS' },
    { value: '120', label: '120 FPS' }
  ]

  useEffect(() => {
    // Enumerate available cameras
    const getCameras = async () => {
      try {
        if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
          const devices = await navigator.mediaDevices.enumerateDevices()
          const cameras = devices
            .filter(device => device.kind === 'videoinput')
            .map(device => ({
              id: device.deviceId,
              label: device.label || `Camera ${device.deviceId.slice(0, 8)}`,
              kind: device.kind
            }))
          setAvailableCameras(cameras)
        }
      } catch (error) {
        console.error('Failed to enumerate cameras:', error)
        setErrors({ camera: 'Failed to detect available cameras' })
      }
    }

    getCameras()
  }, [])

  const cameraOptions: SelectOption[] = [
    { value: '', label: 'No camera selected' },
    ...availableCameras.map(camera => ({
      value: camera.id,
      label: camera.label
    }))
  ]

  const handleResolutionChange = (value: string) => {
    const [width, height] = value.split('x').map(Number)
    setResolution({ width, height })
  }

  const getCurrentResolutionString = () => {
    return `${resolution.width}x${resolution.height}`
  }

  const handleTestCamera = async () => {
    if (!selectedCameraId) {
      setErrors({ camera: 'Please select a camera first' })
      return
    }

    setLoading(true)
    setErrors({})

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: selectedCameraId,
          width: resolution.width,
          height: resolution.height,
          frameRate: fps
        }
      })

      // Test successful - stop the stream
      stream.getTracks().forEach(track => track.stop())
      setPreviewEnabled(true)

      // Auto-hide preview after 5 seconds
      setTimeout(() => setPreviewEnabled(false), 5000)

    } catch (error) {
      console.error('Camera test failed:', error)
      setErrors({ camera: 'Failed to access camera with current settings' })
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    setLoading(true)
    setErrors({})

    try {
      // Validate settings
      const newErrors: Record<string, string> = {}

      if (!selectedCameraId) {
        newErrors.camera = 'Please select a camera'
      }

      if (fps < 1 || fps > 240) {
        newErrors.fps = 'FPS must be between 1 and 240'
      }

      if (exposure < 0 || exposure > 1) {
        newErrors.exposure = 'Exposure must be between 0 and 1'
      }

      if (brightness < 0 || brightness > 1) {
        newErrors.brightness = 'Brightness must be between 0 and 1'
      }

      if (contrast < 0 || contrast > 1) {
        newErrors.contrast = 'Contrast must be between 0 and 1'
      }

      if (Object.keys(newErrors).length > 0) {
        setErrors(newErrors)
        return
      }

      // Update config store
      configStore.updateCameraConfig({
        selectedCameraId,
        resolution,
        fps,
        autoExposure,
        exposure,
        brightness,
        contrast
      })

      // Save additional settings (would typically go to backend)
      const advancedConfig = {
        whiteBalance,
        saturation,
        sharpness,
        enableHDR,
        denoise
      }

      console.log('Saving camera configuration:', {
        ...configStore.cameraConfig,
        advanced: advancedConfig
      })

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))

    } catch (error) {
      console.error('Failed to save camera settings:', error)
      setErrors({ general: 'Failed to save camera settings. Please try again.' })
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    const defaultConfig = {
      selectedCameraId: null,
      resolution: { width: 1920, height: 1080 },
      fps: 30,
      autoExposure: true,
      exposure: 0.5,
      brightness: 0.5,
      contrast: 0.5
    }

    setSelectedCameraId(defaultConfig.selectedCameraId || '')
    setResolution(defaultConfig.resolution)
    setFps(defaultConfig.fps)
    setAutoExposure(defaultConfig.autoExposure)
    setExposure(defaultConfig.exposure)
    setBrightness(defaultConfig.brightness)
    setContrast(defaultConfig.contrast)
    setWhiteBalance(5000)
    setSaturation(1.0)
    setSharpness(0.5)
    setEnableHDR(false)
    setDenoise(true)
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

      {/* Camera Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Camera Selection</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Select
              label="Available Cameras"
              value={selectedCameraId}
              onChange={(e) => setSelectedCameraId(e.target.value)}
              options={cameraOptions}
              error={errors.camera}
              fullWidth
            />

            <div className="flex items-end">
              <Button
                onClick={handleTestCamera}
                loading={loading}
                disabled={!selectedCameraId || loading}
                variant="outline"
              >
                Test Camera
              </Button>
            </div>
          </div>

          {previewEnabled && (
            <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-md">
              <p className="text-sm text-green-800">
                Camera test successful! The camera is working with current settings.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Resolution and Frame Rate */}
      <Card>
        <CardHeader>
          <CardTitle>Video Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Select
              label="Resolution"
              value={getCurrentResolutionString()}
              onChange={(e) => handleResolutionChange(e.target.value)}
              options={resolutionOptions}
              fullWidth
            />

            <Select
              label="Frame Rate"
              value={fps.toString()}
              onChange={(e) => setFps(Number(e.target.value))}
              options={fpsOptions}
              error={errors.fps}
              fullWidth
            />
          </div>
        </CardContent>
      </Card>

      {/* Image Quality Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Image Quality</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Checkbox
              label="Auto Exposure"
              checked={autoExposure}
              onChange={(e) => setAutoExposure(e.target.checked)}
              hint="Automatically adjust exposure based on lighting conditions"
            />

            <Slider
              label="Manual Exposure"
              value={exposure}
              onChange={(e) => setExposure(Number(e.target.value))}
              min={0}
              max={1}
              step={0.01}
              disabled={autoExposure}
              error={errors.exposure}
              formatValue={(value) => `${Math.round(value * 100)}%`}
              fullWidth
            />

            <Slider
              label="Brightness"
              value={brightness}
              onChange={(e) => setBrightness(Number(e.target.value))}
              min={0}
              max={1}
              step={0.01}
              error={errors.brightness}
              formatValue={(value) => `${Math.round(value * 100)}%`}
              fullWidth
            />

            <Slider
              label="Contrast"
              value={contrast}
              onChange={(e) => setContrast(Number(e.target.value))}
              min={0}
              max={1}
              step={0.01}
              error={errors.contrast}
              formatValue={(value) => `${Math.round(value * 100)}%`}
              fullWidth
            />
          </div>
        </CardContent>
      </Card>

      {/* Advanced Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Advanced Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Slider
              label="White Balance (K)"
              value={whiteBalance}
              onChange={(e) => setWhiteBalance(Number(e.target.value))}
              min={2000}
              max={8000}
              step={100}
              formatValue={(value) => `${value}K`}
              fullWidth
            />

            <Slider
              label="Saturation"
              value={saturation}
              onChange={(e) => setSaturation(Number(e.target.value))}
              min={0}
              max={2}
              step={0.01}
              formatValue={(value) => `${Math.round(value * 100)}%`}
              fullWidth
            />

            <Slider
              label="Sharpness"
              value={sharpness}
              onChange={(e) => setSharpness(Number(e.target.value))}
              min={0}
              max={1}
              step={0.01}
              formatValue={(value) => `${Math.round(value * 100)}%`}
              fullWidth
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Checkbox
                label="Enable HDR"
                checked={enableHDR}
                onChange={(e) => setEnableHDR(e.target.checked)}
                hint="High Dynamic Range for better lighting"
              />

              <Checkbox
                label="Enable Noise Reduction"
                checked={denoise}
                onChange={(e) => setDenoise(e.target.checked)}
                hint="Reduce image noise in low light conditions"
              />
            </div>
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
