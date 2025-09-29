import React, { useState } from 'react'
import { observer } from 'mobx-react-lite'
import { useStores } from '../../stores/context'
import { Card, CardHeader, CardTitle, CardContent, Button, Slider, Checkbox, Select } from '../ui'
import type { SelectOption } from '../ui'

export const VisionConfig = observer(() => {
  const { configStore } = useStores()
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})

  // Get detection config from store
  const detectionConfig = configStore.detectionConfig

  // Ball detection
  const [ballThreshold, setBallThreshold] = useState(detectionConfig.ballDetectionThreshold)
  const [ballMinRadius, setBallMinRadius] = useState(10)
  const [ballMaxRadius, setBallMaxRadius] = useState(50)
  const [ballColorRange, setBallColorRange] = useState(0.15)
  const [enableBallFiltering, setEnableBallFiltering] = useState(true)

  // Cue detection
  const [cueThreshold, setCueThreshold] = useState(detectionConfig.cueDetectionThreshold)
  const [cueMinLength, setCueMinLength] = useState(200)
  const [cueMaxLength, setCueMaxLength] = useState(1500)
  const [cueAngleTolerance, setCueAngleTolerance] = useState(15)
  const [enableCueTracking, setEnableCueTracking] = useState(true)

  // Motion detection
  const [motionThreshold, setMotionThreshold] = useState(detectionConfig.motionDetectionThreshold)
  const [motionSensitivity, setMotionSensitivity] = useState(0.5)
  const [backgroundSubtraction, setBackgroundSubtraction] = useState(true)
  const [motionBlur, setMotionBlur] = useState(0.3)

  // Tracking settings
  const [enableTracking, setEnableTracking] = useState(detectionConfig.enableTracking)
  const [trackingAlgorithm, setTrackingAlgorithm] = useState('kalman')
  const [maxTrackingDistance, setMaxTrackingDistance] = useState(100)
  const [trackingConfidence, setTrackingConfidence] = useState(0.7)
  const [lostTrackTimeout, setLostTrackTimeout] = useState(30)

  // Prediction settings
  const [enablePrediction, setEnablePrediction] = useState(detectionConfig.enablePrediction)
  const [predictionHorizon, setPredictionHorizon] = useState(2.0)
  const [predictionAccuracy, setPredictionAccuracy] = useState(0.8)
  const [collisionPrediction, setCollisionPrediction] = useState(true)

  // Image processing
  const [enableDenoising, setEnableDenoising] = useState(true)
  const [contrastEnhancement, setContrastEnhancement] = useState(0.2)
  const [edgeDetection, setEdgeDetection] = useState(true)
  const [morphologyKernel, setMorphologyKernel] = useState(3)

  // Calibration
  const [stabilizationFrames, setStabilizationFrames] = useState(detectionConfig.stabilizationFrames)
  const [cameraMatrix, setCameraMatrix] = useState('[1920, 0, 960; 0, 1080, 540; 0, 0, 1]')
  const [distortionCoeffs, setDistortionCoeffs] = useState('[0.1, -0.2, 0, 0, 0]')

  const trackingAlgorithmOptions: SelectOption[] = [
    { value: 'kalman', label: 'Kalman Filter' },
    { value: 'particle', label: 'Particle Filter' },
    { value: 'optical_flow', label: 'Optical Flow' },
    { value: 'centroid', label: 'Centroid Tracking' }
  ]

  const handleTestDetection = async () => {
    setLoading(true)
    try {
      console.log('Testing vision detection with current settings')
      // Simulate detection test
      await new Promise(resolve => setTimeout(resolve, 3000))
    } catch (error) {
      console.error('Detection test failed:', error)
      setErrors({ detection: 'Detection test failed. Check camera and lighting conditions.' })
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    setLoading(true)
    setErrors({})

    try {
      const newErrors: Record<string, string> = {}

      // Validation
      if (ballThreshold < 0.1 || ballThreshold > 1.0) {
        newErrors.ballThreshold = 'Ball detection threshold must be between 0.1 and 1.0'
      }

      if (cueThreshold < 0.1 || cueThreshold > 1.0) {
        newErrors.cueThreshold = 'Cue detection threshold must be between 0.1 and 1.0'
      }

      if (motionThreshold < 0.01 || motionThreshold > 1.0) {
        newErrors.motionThreshold = 'Motion detection threshold must be between 0.01 and 1.0'
      }

      if (stabilizationFrames < 1 || stabilizationFrames > 60) {
        newErrors.stabilizationFrames = 'Stabilization frames must be between 1 and 60'
      }

      if (predictionHorizon < 0.1 || predictionHorizon > 10.0) {
        newErrors.predictionHorizon = 'Prediction horizon must be between 0.1 and 10.0 seconds'
      }

      if (Object.keys(newErrors).length > 0) {
        setErrors(newErrors)
        return
      }

      // Update config store
      configStore.updateDetectionConfig({
        ballDetectionThreshold: ballThreshold,
        cueDetectionThreshold: cueThreshold,
        motionDetectionThreshold: motionThreshold,
        stabilizationFrames,
        enableTracking,
        enablePrediction
      })

      const visionConfig = {
        detection: {
          ball: {
            threshold: ballThreshold,
            minRadius: ballMinRadius,
            maxRadius: ballMaxRadius,
            colorRange: ballColorRange,
            enableFiltering: enableBallFiltering
          },
          cue: {
            threshold: cueThreshold,
            minLength: cueMinLength,
            maxLength: cueMaxLength,
            angleTolerance: cueAngleTolerance,
            enableTracking: enableCueTracking
          },
          motion: {
            threshold: motionThreshold,
            sensitivity: motionSensitivity,
            backgroundSubtraction,
            motionBlur
          }
        },
        tracking: {
          enabled: enableTracking,
          algorithm: trackingAlgorithm,
          maxDistance: maxTrackingDistance,
          confidence: trackingConfidence,
          lostTrackTimeout
        },
        prediction: {
          enabled: enablePrediction,
          horizon: predictionHorizon,
          accuracy: predictionAccuracy,
          collisionPrediction
        },
        imageProcessing: {
          denoising: enableDenoising,
          contrastEnhancement,
          edgeDetection,
          morphologyKernel
        },
        calibration: {
          stabilizationFrames,
          cameraMatrix,
          distortionCoeffs
        }
      }

      console.log('Saving vision configuration:', visionConfig)
      await new Promise(resolve => setTimeout(resolve, 1000))

    } catch (error) {
      console.error('Failed to save vision settings:', error)
      setErrors({ general: 'Failed to save vision settings. Please try again.' })
    } finally {
      setLoading(false)
    }
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

      {/* Ball Detection */}
      <Card>
        <CardHeader>
          <CardTitle>Ball Detection</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Slider
              label="Detection Threshold"
              value={ballThreshold}
              onChange={(e) => setBallThreshold(Number(e.target.value))}
              min={0.1}
              max={1.0}
              step={0.01}
              error={errors.ballThreshold}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="Higher values = more strict detection"
              fullWidth
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Slider
                label="Minimum Ball Radius (px)"
                value={ballMinRadius}
                onChange={(e) => setBallMinRadius(Number(e.target.value))}
                min={5}
                max={30}
                step={1}
                formatValue={(value) => `${value}px`}
                fullWidth
              />

              <Slider
                label="Maximum Ball Radius (px)"
                value={ballMaxRadius}
                onChange={(e) => setBallMaxRadius(Number(e.target.value))}
                min={30}
                max={100}
                step={1}
                formatValue={(value) => `${value}px`}
                fullWidth
              />
            </div>

            <Slider
              label="Color Range Tolerance"
              value={ballColorRange}
              onChange={(e) => setBallColorRange(Number(e.target.value))}
              min={0.05}
              max={0.5}
              step={0.01}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="How much color variation to allow"
              fullWidth
            />

            <Checkbox
              label="Enable Ball Filtering"
              checked={enableBallFiltering}
              onChange={(e) => setEnableBallFiltering(e.target.checked)}
              hint="Filter out false ball detections"
            />
          </div>
        </CardContent>
      </Card>

      {/* Cue Detection */}
      <Card>
        <CardHeader>
          <CardTitle>Cue Detection</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Slider
              label="Detection Threshold"
              value={cueThreshold}
              onChange={(e) => setCueThreshold(Number(e.target.value))}
              min={0.1}
              max={1.0}
              step={0.01}
              error={errors.cueThreshold}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              fullWidth
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Slider
                label="Minimum Cue Length (px)"
                value={cueMinLength}
                onChange={(e) => setCueMinLength(Number(e.target.value))}
                min={100}
                max={500}
                step={10}
                formatValue={(value) => `${value}px`}
                fullWidth
              />

              <Slider
                label="Maximum Cue Length (px)"
                value={cueMaxLength}
                onChange={(e) => setCueMaxLength(Number(e.target.value))}
                min={500}
                max={2000}
                step={50}
                formatValue={(value) => `${value}px`}
                fullWidth
              />
            </div>

            <Slider
              label="Angle Tolerance (degrees)"
              value={cueAngleTolerance}
              onChange={(e) => setCueAngleTolerance(Number(e.target.value))}
              min={5}
              max={45}
              step={1}
              formatValue={(value) => `${value}°`}
              hint="Angular deviation allowed for cue detection"
              fullWidth
            />

            <Checkbox
              label="Enable Cue Tracking"
              checked={enableCueTracking}
              onChange={(e) => setEnableCueTracking(e.target.checked)}
              hint="Track cue position and angle continuously"
            />
          </div>
        </CardContent>
      </Card>

      {/* Motion Detection */}
      <Card>
        <CardHeader>
          <CardTitle>Motion Detection</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Slider
              label="Motion Threshold"
              value={motionThreshold}
              onChange={(e) => setMotionThreshold(Number(e.target.value))}
              min={0.01}
              max={1.0}
              step={0.01}
              error={errors.motionThreshold}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              fullWidth
            />

            <Slider
              label="Motion Sensitivity"
              value={motionSensitivity}
              onChange={(e) => setMotionSensitivity(Number(e.target.value))}
              min={0.1}
              max={1.0}
              step={0.01}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="Sensitivity to small movements"
              fullWidth
            />

            <Slider
              label="Motion Blur Compensation"
              value={motionBlur}
              onChange={(e) => setMotionBlur(Number(e.target.value))}
              min={0.0}
              max={1.0}
              step={0.01}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="Compensate for motion blur in images"
              fullWidth
            />

            <Checkbox
              label="Background Subtraction"
              checked={backgroundSubtraction}
              onChange={(e) => setBackgroundSubtraction(e.target.checked)}
              hint="Remove static background to highlight moving objects"
            />
          </div>
        </CardContent>
      </Card>

      {/* Tracking Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Object Tracking</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Checkbox
              label="Enable Object Tracking"
              checked={enableTracking}
              onChange={(e) => setEnableTracking(e.target.checked)}
              hint="Track objects across multiple frames"
            />

            <Select
              label="Tracking Algorithm"
              value={trackingAlgorithm}
              onChange={(e) => setTrackingAlgorithm(e.target.value)}
              options={trackingAlgorithmOptions}
              disabled={!enableTracking}
              fullWidth
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Slider
                label="Max Tracking Distance (px)"
                value={maxTrackingDistance}
                onChange={(e) => setMaxTrackingDistance(Number(e.target.value))}
                min={10}
                max={200}
                step={5}
                disabled={!enableTracking}
                formatValue={(value) => `${value}px`}
                fullWidth
              />

              <Slider
                label="Tracking Confidence"
                value={trackingConfidence}
                onChange={(e) => setTrackingConfidence(Number(e.target.value))}
                min={0.1}
                max={1.0}
                step={0.01}
                disabled={!enableTracking}
                formatValue={(value) => `${(value * 100).toFixed(0)}%`}
                fullWidth
              />
            </div>

            <Slider
              label="Lost Track Timeout (frames)"
              value={lostTrackTimeout}
              onChange={(e) => setLostTrackTimeout(Number(e.target.value))}
              min={5}
              max={120}
              step={5}
              disabled={!enableTracking}
              formatValue={(value) => `${value} frames`}
              hint="Frames to wait before declaring an object lost"
              fullWidth
            />
          </div>
        </CardContent>
      </Card>

      {/* Prediction Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Motion Prediction</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Checkbox
              label="Enable Motion Prediction"
              checked={enablePrediction}
              onChange={(e) => setEnablePrediction(e.target.checked)}
              hint="Predict future object positions"
            />

            <Slider
              label="Prediction Horizon (seconds)"
              value={predictionHorizon}
              onChange={(e) => setPredictionHorizon(Number(e.target.value))}
              min={0.1}
              max={10.0}
              step={0.1}
              disabled={!enablePrediction}
              error={errors.predictionHorizon}
              formatValue={(value) => `${value.toFixed(1)}s`}
              hint="How far into the future to predict"
              fullWidth
            />

            <Slider
              label="Prediction Accuracy"
              value={predictionAccuracy}
              onChange={(e) => setPredictionAccuracy(Number(e.target.value))}
              min={0.5}
              max={1.0}
              step={0.01}
              disabled={!enablePrediction}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="Required confidence for predictions"
              fullWidth
            />

            <Checkbox
              label="Collision Prediction"
              checked={collisionPrediction}
              onChange={(e) => setCollisionPrediction(e.target.checked)}
              disabled={!enablePrediction}
              hint="Predict ball-to-ball collisions"
            />
          </div>
        </CardContent>
      </Card>

      {/* Image Processing */}
      <Card>
        <CardHeader>
          <CardTitle>Image Processing</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Slider
              label="Stabilization Frames"
              value={stabilizationFrames}
              onChange={(e) => setStabilizationFrames(Number(e.target.value))}
              min={1}
              max={60}
              step={1}
              error={errors.stabilizationFrames}
              formatValue={(value) => `${value} frames`}
              hint="Frames to average for stable detection"
              fullWidth
            />

            <Slider
              label="Contrast Enhancement"
              value={contrastEnhancement}
              onChange={(e) => setContrastEnhancement(Number(e.target.value))}
              min={0.0}
              max={1.0}
              step={0.01}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="Enhance image contrast for better detection"
              fullWidth
            />

            <Slider
              label="Morphology Kernel Size"
              value={morphologyKernel}
              onChange={(e) => setMorphologyKernel(Number(e.target.value))}
              min={1}
              max={15}
              step={2}
              formatValue={(value) => `${value}x${value}`}
              hint="Size of morphological operations kernel"
              fullWidth
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Checkbox
                label="Enable Denoising"
                checked={enableDenoising}
                onChange={(e) => setEnableDenoising(e.target.checked)}
                hint="Reduce image noise"
              />

              <Checkbox
                label="Edge Detection"
                checked={edgeDetection}
                onChange={(e) => setEdgeDetection(e.target.checked)}
                hint="Enhance object edges"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Test Detection */}
      <Card>
        <CardHeader>
          <CardTitle>Detection Test</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Test the current detection settings with the live camera feed.
            </p>
            <Button
              onClick={handleTestDetection}
              loading={loading}
              variant="outline"
            >
              {loading ? 'Testing Detection...' : 'Test Detection'}
            </Button>
            {errors.detection && (
              <p className="text-sm text-red-600">{errors.detection}</p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <div className="flex justify-end space-x-4">
        <Button
          variant="outline"
          onClick={() => window.location.reload()}
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
