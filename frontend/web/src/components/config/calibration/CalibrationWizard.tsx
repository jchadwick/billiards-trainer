import React, { useState, useRef, useEffect, useCallback } from 'react'
import { observer } from 'mobx-react-lite'
import { Card, CardHeader, CardTitle, CardContent, Button } from '../../ui'
import { useStores } from '../../../hooks/useStores'
import { CalibrationStartRequest, CalibrationPointRequest } from '../../../types/api'

// Types for calibration workflow
interface CalibrationStep {
  id: string
  title: string
  description: string
  component: React.ComponentType<CalibrationStepProps>
  requirements: string[]
}

interface CalibrationStepProps {
  onNext: () => void
  onPrevious: () => void
  onComplete: () => void
  isFirstStep: boolean
  isLastStep: boolean
  data?: any
  onDataChange?: (data: any) => void
}

interface Point {
  x: number
  y: number
  id: string
  screenX: number
  screenY: number
  worldX: number
  worldY: number
}

interface HSVRange {
  hMin: number
  hMax: number
  sMin: number
  sMax: number
  vMin: number
  vMax: number
}

interface ColorProfile {
  name: string
  hsv: HSVRange
  ballType: 'cue' | 'solid' | 'stripe' | 'eight'
}

// Interactive video feed with point selection
const VideoFeedCanvas: React.FC<{
  onPointSelect: (x: number, y: number) => void
  points: Point[]
  width: number
  height: number
  overlayVisible: boolean
}> = ({ onPointSelect, points, width, height, overlayVisible }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const videoImageRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const { videoStore } = useStores()

  // Draw video frame and overlay on canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw video frame if available
    if (videoStore.currentFrameImage && videoImageRef.current && videoImageRef.current.complete) {
      // Draw the video frame scaled to fit canvas
      ctx.drawImage(videoImageRef.current, 0, 0, width, height)
    } else {
      // Draw placeholder if no video feed is available
      ctx.fillStyle = '#1f2937'
      ctx.fillRect(0, 0, width, height)

      // Draw grid lines for reference when no video is available
      ctx.strokeStyle = '#374151'
      ctx.lineWidth = 1
      const gridSize = 50
      for (let x = 0; x <= width; x += gridSize) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, height)
        ctx.stroke()
      }
      for (let y = 0; y <= height; y += gridSize) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(width, y)
        ctx.stroke()
      }

      // Draw center indicator
      ctx.strokeStyle = '#6b7280'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(width/2 - 20, height/2)
      ctx.lineTo(width/2 + 20, height/2)
      ctx.moveTo(width/2, height/2 - 20)
      ctx.lineTo(width/2, height/2 + 20)
      ctx.stroke()

      // Show "waiting for video" message
      ctx.fillStyle = '#9ca3af'
      ctx.font = '16px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('Waiting for video feed...', width/2, height/2 + 50)
    }

    // Draw calibration points overlay if visible
    if (overlayVisible) {
      points.forEach((point, index) => {
        ctx.fillStyle = '#3b82f6'
        ctx.strokeStyle = '#1e40af'
        ctx.lineWidth = 2

        // Draw point circle
        ctx.beginPath()
        ctx.arc(point.screenX, point.screenY, 8, 0, 2 * Math.PI)
        ctx.fill()
        ctx.stroke()

        // Draw point label
        ctx.fillStyle = '#ffffff'
        ctx.font = '12px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText((index + 1).toString(), point.screenX, point.screenY + 4)
      })

      // Draw connecting lines for table outline
      if (points.length >= 4) {
        ctx.strokeStyle = '#ef4444'
        ctx.lineWidth = 2
        ctx.beginPath()
        points.forEach((point, index) => {
          if (index === 0) {
            ctx.moveTo(point.screenX, point.screenY)
          } else {
            ctx.lineTo(point.screenX, point.screenY)
          }
        })
        ctx.closePath()
        ctx.stroke()
      }
    }

    // Add click instruction text at bottom
    ctx.fillStyle = '#9ca3af'
    ctx.font = '14px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('Click on the video feed to set calibration points', width/2, height - 20)
  }, [points, width, height, overlayVisible, videoStore.currentFrameImage])

  // Load video frame into image element for drawing
  useEffect(() => {
    if (videoStore.currentFrameImage && videoImageRef.current) {
      videoImageRef.current.src = videoStore.currentFrameImage
    }
  }, [videoStore.currentFrameImage])

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    onPointSelect(x, y)
  }

  return (
    <div ref={containerRef} className="relative">
      {/* Hidden image element for loading video frames */}
      <img
        ref={videoImageRef}
        style={{ display: 'none' }}
        alt="Video frame"
        onLoad={() => {
          // Trigger canvas redraw when new frame loads
          const canvas = canvasRef.current
          if (canvas) {
            const event = new Event('frameLoaded')
            canvas.dispatchEvent(event)
          }
        }}
      />

      {/* Canvas for video display and overlay */}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onClick={handleCanvasClick}
        className="border border-gray-300 dark:border-gray-600 rounded-lg cursor-crosshair bg-gray-900"
      />

      {/* Connection status indicator */}
      {!videoStore.isConnected && (
        <div className="absolute top-2 left-2 px-2 py-1 bg-red-600 text-white text-xs rounded shadow">
          Video Not Connected
        </div>
      )}

      {videoStore.isConnected && !videoStore.currentFrameImage && (
        <div className="absolute top-2 left-2 px-2 py-1 bg-yellow-600 text-white text-xs rounded shadow">
          Waiting for video stream...
        </div>
      )}
    </div>
  )
}

// HSV Color Range Slider Component
const HSVSlider: React.FC<{
  label: string
  value: number
  min: number
  max: number
  onChange: (value: number) => void
}> = ({ label, value, min, max, onChange }) => (
  <div className="space-y-2">
    <div className="flex justify-between">
      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</label>
      <span className="text-sm text-gray-500 dark:text-gray-400">{value}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      value={value}
      onChange={(e) => onChange(parseInt(e.target.value))}
      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
    />
  </div>
)

// Camera Calibration Step - Interactive geometric calibration
const CameraCalibrationStep: React.FC<CalibrationStepProps> = ({
  onNext,
  onPrevious,
  isFirstStep,
  data,
  onDataChange
}) => {
  const [points, setPoints] = useState<Point[]>(data?.points || [])
  const [currentPointIndex, setCurrentPointIndex] = useState(0)
  const [showInstructions, setShowInstructions] = useState(true)
  const [videoConnectionError, setVideoConnectionError] = useState<string | null>(null)
  const { calibrationStore, videoStore } = useStores()

  // Ensure video connection is established when component mounts
  useEffect(() => {
    const connectVideo = async () => {
      if (!videoStore.isConnected) {
        try {
          setVideoConnectionError(null)
          await videoStore.connect('http://localhost:8080')
        } catch (error) {
          console.error('Failed to connect video stream:', error)
          setVideoConnectionError(
            error instanceof Error ? error.message : 'Failed to connect to video stream'
          )
        }
      }
    }

    connectVideo()

    // Cleanup: don't disconnect on unmount as other components may be using the video
    // The video store will handle connection lifecycle
  }, [videoStore])

  const pointInstructions = [
    "Click on the top-left corner of the billiard table",
    "Click on the top-right corner of the billiard table",
    "Click on the bottom-right corner of the billiard table",
    "Click on the bottom-left corner of the billiard table"
  ]

  const handlePointSelect = (screenX: number, screenY: number) => {
    if (currentPointIndex >= pointInstructions.length) return

    // Convert screen coordinates to world coordinates (simplified)
    const worldX = (screenX / 640) * 2.84 - 1.42 // Standard pool table width
    const worldY = (screenY / 360) * 1.42 - 0.71 // Standard pool table height

    const newPoint: Point = {
      x: screenX,
      y: screenY,
      id: `corner_${currentPointIndex}`,
      screenX,
      screenY,
      worldX,
      worldY
    }

    const newPoints = [...points]
    newPoints[currentPointIndex] = newPoint
    setPoints(newPoints)
    setCurrentPointIndex(Math.min(currentPointIndex + 1, pointInstructions.length))

    onDataChange?.({ points: newPoints })
  }

  const removeLastPoint = () => {
    if (points.length > 0) {
      const newPoints = points.slice(0, -1)
      setPoints(newPoints)
      setCurrentPointIndex(Math.max(0, currentPointIndex - 1))
      onDataChange?.({ points: newPoints })
    }
  }

  const canProceed = points.length >= 4

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          Camera Geometric Calibration
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Click on the four corners of the billiard table to establish geometric calibration
        </p>
      </div>

      {videoConnectionError && (
        <Card>
          <CardContent padding="sm">
            <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
              <h4 className="font-medium text-red-900 dark:text-red-200 mb-2">Video Connection Error:</h4>
              <p className="text-red-800 dark:text-red-300 text-sm">
                {videoConnectionError}
              </p>
              <p className="text-red-700 dark:text-red-400 text-xs mt-2">
                Please ensure the camera is connected and the backend service is running.
              </p>
              <Button
                variant="outline"
                size="sm"
                onClick={async () => {
                  setVideoConnectionError(null)
                  try {
                    await videoStore.connect('http://localhost:8080')
                  } catch (error) {
                    setVideoConnectionError(
                      error instanceof Error ? error.message : 'Failed to connect to video stream'
                    )
                  }
                }}
                className="mt-2"
              >
                Retry Connection
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {showInstructions && (
        <Card>
          <CardContent padding="sm">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <h4 className="font-medium text-blue-900 dark:text-blue-200 mb-2">Current Step:</h4>
              <p className="text-blue-800 dark:text-blue-300">
                {currentPointIndex < pointInstructions.length
                  ? pointInstructions[currentPointIndex]
                  : "All corner points captured. Review and proceed to next step."
                }
              </p>
              <div className="mt-2 text-sm text-blue-700 dark:text-blue-400">
                Points captured: {points.length} / {pointInstructions.length}
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowInstructions(false)}
              className="mt-2"
            >
              Hide Instructions
            </Button>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardContent>
          <div className="flex justify-center">
            <VideoFeedCanvas
              onPointSelect={handlePointSelect}
              points={points}
              width={640}
              height={360}
              overlayVisible={true}
            />
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        <Button
          variant="outline"
          onClick={removeLastPoint}
          disabled={points.length === 0}
        >
          Remove Last Point
        </Button>
        <Button
          variant="outline"
          onClick={() => setShowInstructions(true)}
        >
          Show Instructions
        </Button>
      </div>

      <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
        <h4 className="font-medium text-gray-900 dark:text-white mb-2">Captured Points:</h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          {points.map((point, index) => (
            <div key={point.id} className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">
                Corner {index + 1}:
              </span>
              <span className="font-mono text-gray-800 dark:text-gray-200">
                ({point.screenX.toFixed(0)}, {point.screenY.toFixed(0)})
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex justify-between">
        <Button
          variant="outline"
          onClick={onPrevious}
          disabled={isFirstStep}
        >
          Previous
        </Button>
        <Button
          onClick={onNext}
          disabled={!canProceed}
          variant={canProceed ? "primary" : "outline"}
        >
          Next: Table Configuration
        </Button>
      </div>
    </div>
  )
}

// Table Configuration Step - ROI and perspective correction
const TableConfigurationStep: React.FC<CalibrationStepProps> = ({
  onNext,
  onPrevious,
  data,
  onDataChange
}) => {
  const [roiBounds, setRoiBounds] = useState(data?.roiBounds || { x: 50, y: 50, width: 540, height: 260 })
  const [perspectivePoints, setPerspectivePoints] = useState(data?.perspectivePoints || [])
  const [mode, setMode] = useState<'roi' | 'perspective'>('roi')

  const handleROIChange = (bounds: typeof roiBounds) => {
    setRoiBounds(bounds)
    onDataChange?.({ ...data, roiBounds: bounds })
  }

  const handlePerspectivePointSelect = (x: number, y: number) => {
    if (perspectivePoints.length >= 4) return

    const newPoints = [...perspectivePoints, { x, y, id: `perspective_${perspectivePoints.length}` }]
    setPerspectivePoints(newPoints)
    onDataChange?.({ ...data, perspectivePoints: newPoints })
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          Table Configuration
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Configure region of interest and perspective correction
        </p>
      </div>

      <div className="flex space-x-2">
        <Button
          variant={mode === 'roi' ? 'primary' : 'outline'}
          onClick={() => setMode('roi')}
        >
          ROI Selection
        </Button>
        <Button
          variant={mode === 'perspective' ? 'primary' : 'outline'}
          onClick={() => setMode('perspective')}
        >
          Perspective Correction
        </Button>
      </div>

      <Card>
        <CardContent>
          <div className="flex justify-center">
            {mode === 'roi' ? (
              <div className="relative">
                <VideoFeedCanvas
                  onPointSelect={(x, y) => {}}
                  points={[]}
                  width={640}
                  height={360}
                  overlayVisible={false}
                />
                <div
                  className="absolute border-2 border-yellow-400 bg-yellow-400/20"
                  style={{
                    left: roiBounds.x,
                    top: roiBounds.y,
                    width: roiBounds.width,
                    height: roiBounds.height
                  }}
                />
              </div>
            ) : (
              <VideoFeedCanvas
                onPointSelect={handlePerspectivePointSelect}
                points={perspectivePoints.map(p => ({
                  ...p,
                  screenX: p.x,
                  screenY: p.y,
                  worldX: 0,
                  worldY: 0
                }))}
                width={640}
                height={360}
                overlayVisible={true}
              />
            )}
          </div>
        </CardContent>
      </Card>

      {mode === 'roi' && (
        <Card>
          <CardHeader>
            <CardTitle>ROI Bounds</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">X Position</label>
                <input
                  type="range"
                  min={0}
                  max={640}
                  value={roiBounds.x}
                  onChange={(e) => handleROIChange({ ...roiBounds, x: parseInt(e.target.value) })}
                  className="w-full"
                />
                <span className="text-xs text-gray-500">{roiBounds.x}px</span>
              </div>
              <div>
                <label className="text-sm font-medium">Y Position</label>
                <input
                  type="range"
                  min={0}
                  max={360}
                  value={roiBounds.y}
                  onChange={(e) => handleROIChange({ ...roiBounds, y: parseInt(e.target.value) })}
                  className="w-full"
                />
                <span className="text-xs text-gray-500">{roiBounds.y}px</span>
              </div>
              <div>
                <label className="text-sm font-medium">Width</label>
                <input
                  type="range"
                  min={100}
                  max={640}
                  value={roiBounds.width}
                  onChange={(e) => handleROIChange({ ...roiBounds, width: parseInt(e.target.value) })}
                  className="w-full"
                />
                <span className="text-xs text-gray-500">{roiBounds.width}px</span>
              </div>
              <div>
                <label className="text-sm font-medium">Height</label>
                <input
                  type="range"
                  min={100}
                  max={360}
                  value={roiBounds.height}
                  onChange={(e) => handleROIChange({ ...roiBounds, height: parseInt(e.target.value) })}
                  className="w-full"
                />
                <span className="text-xs text-gray-500">{roiBounds.height}px</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="flex justify-between">
        <Button variant="outline" onClick={onPrevious}>
          Previous
        </Button>
        <Button onClick={onNext}>
          Next: Color Calibration
        </Button>
      </div>
    </div>
  )
}

// Color Calibration Step - HSV range controls
const ColorCalibrationStep: React.FC<CalibrationStepProps> = ({
  onNext,
  onPrevious,
  data,
  onDataChange
}) => {
  const [selectedProfile, setSelectedProfile] = useState<ColorProfile>(
    data?.selectedProfile || {
      name: 'Cue Ball',
      ballType: 'cue',
      hsv: { hMin: 0, hMax: 180, sMin: 0, sMax: 30, vMin: 200, vMax: 255 }
    }
  )

  const [profiles, setProfiles] = useState<ColorProfile[]>(data?.profiles || [])
  const [livePreview, setLivePreview] = useState(true)

  const ballProfiles: ColorProfile[] = [
    {
      name: 'Cue Ball',
      ballType: 'cue',
      hsv: { hMin: 0, hMax: 180, sMin: 0, sMax: 30, vMin: 200, vMax: 255 }
    },
    {
      name: 'Solid Balls',
      ballType: 'solid',
      hsv: { hMin: 0, hMax: 180, sMin: 50, sMax: 255, vMin: 50, vMax: 255 }
    },
    {
      name: 'Stripe Balls',
      ballType: 'stripe',
      hsv: { hMin: 0, hMax: 180, sMin: 50, sMax: 255, vMin: 50, vMax: 255 }
    },
    {
      name: 'Eight Ball',
      ballType: 'eight',
      hsv: { hMin: 0, hMax: 180, sMin: 0, sMax: 30, vMin: 0, vMax: 50 }
    }
  ]

  const updateHSVValue = (key: keyof HSVRange, value: number) => {
    const newProfile = {
      ...selectedProfile,
      hsv: { ...selectedProfile.hsv, [key]: value }
    }
    setSelectedProfile(newProfile)
    onDataChange?.({ selectedProfile: newProfile, profiles })
  }

  const saveProfile = () => {
    const newProfiles = [...profiles.filter(p => p.name !== selectedProfile.name), selectedProfile]
    setProfiles(newProfiles)
    onDataChange?.({ selectedProfile, profiles: newProfiles })
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          Color Calibration
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Configure HSV color ranges for accurate ball detection
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Ball Type Selection</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {ballProfiles.map((profile) => (
                <Button
                  key={profile.name}
                  variant={selectedProfile.name === profile.name ? 'primary' : 'outline'}
                  onClick={() => setSelectedProfile(profile)}
                  className="w-full justify-start"
                >
                  {profile.name}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Live Preview</CardTitle>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setLivePreview(!livePreview)}
              >
                {livePreview ? 'Disable' : 'Enable'} Preview
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-48 bg-gray-100 dark:bg-gray-800 rounded-lg flex items-center justify-center">
              {livePreview ? (
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-2 bg-gradient-to-br from-blue-400 to-blue-600 rounded-full"></div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Color detection preview
                  </p>
                </div>
              ) : (
                <p className="text-gray-500 dark:text-gray-400">Preview disabled</p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>HSV Range Controls - {selectedProfile.name}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="font-medium text-gray-900 dark:text-white">Hue Range</h4>
              <HSVSlider
                label="Hue Min"
                value={selectedProfile.hsv.hMin}
                min={0}
                max={180}
                onChange={(value) => updateHSVValue('hMin', value)}
              />
              <HSVSlider
                label="Hue Max"
                value={selectedProfile.hsv.hMax}
                min={0}
                max={180}
                onChange={(value) => updateHSVValue('hMax', value)}
              />
            </div>

            <div className="space-y-4">
              <h4 className="font-medium text-gray-900 dark:text-white">Saturation Range</h4>
              <HSVSlider
                label="Saturation Min"
                value={selectedProfile.hsv.sMin}
                min={0}
                max={255}
                onChange={(value) => updateHSVValue('sMin', value)}
              />
              <HSVSlider
                label="Saturation Max"
                value={selectedProfile.hsv.sMax}
                min={0}
                max={255}
                onChange={(value) => updateHSVValue('sMax', value)}
              />
            </div>

            <div className="space-y-4">
              <h4 className="font-medium text-gray-900 dark:text-white">Value Range</h4>
              <HSVSlider
                label="Value Min"
                value={selectedProfile.hsv.vMin}
                min={0}
                max={255}
                onChange={(value) => updateHSVValue('vMin', value)}
              />
              <HSVSlider
                label="Value Max"
                value={selectedProfile.hsv.vMax}
                min={0}
                max={255}
                onChange={(value) => updateHSVValue('vMax', value)}
              />
            </div>

            <div className="space-y-4">
              <h4 className="font-medium text-gray-900 dark:text-white">Actions</h4>
              <div className="space-y-2">
                <Button variant="outline" onClick={saveProfile} className="w-full">
                  Save Profile
                </Button>
                <Button variant="outline" className="w-full">
                  Load Default
                </Button>
                <Button variant="outline" className="w-full">
                  Test Detection
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-between">
        <Button variant="outline" onClick={onPrevious}>
          Previous
        </Button>
        <Button onClick={onNext}>
          Next: Projector Alignment
        </Button>
      </div>
    </div>
  )
}

// Projector Alignment Step - Keystone correction
const ProjectorAlignmentStep: React.FC<CalibrationStepProps> = ({
  onNext,
  onPrevious,
  data,
  onDataChange
}) => {
  const [keystoneValues, setKeystoneValues] = useState(data?.keystone || {
    topLeft: { x: 0, y: 0 },
    topRight: { x: 0, y: 0 },
    bottomLeft: { x: 0, y: 0 },
    bottomRight: { x: 0, y: 0 }
  })

  const [brightness, setBrightness] = useState(data?.brightness || 75)
  const [contrast, setContrast] = useState(data?.contrast || 50)
  const [testPattern, setTestPattern] = useState<'grid' | 'crosshair' | 'corners' | 'none'>('none')

  const updateKeystone = (corner: string, axis: 'x' | 'y', value: number) => {
    const newKeystone = {
      ...keystoneValues,
      [corner]: { ...keystoneValues[corner], [axis]: value }
    }
    setKeystoneValues(newKeystone)
    onDataChange?.({ keystone: newKeystone, brightness, contrast })
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          Projector Alignment
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Align projector overlay with keystone correction and brightness controls
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Test Patterns</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2">
              {['grid', 'crosshair', 'corners', 'none'].map((pattern) => (
                <Button
                  key={pattern}
                  variant={testPattern === pattern ? 'primary' : 'outline'}
                  onClick={() => setTestPattern(pattern as any)}
                  size="sm"
                >
                  {pattern.charAt(0).toUpperCase() + pattern.slice(1)}
                </Button>
              ))}
            </div>

            <div className="mt-4 h-32 bg-gray-900 rounded-lg flex items-center justify-center border">
              {testPattern === 'grid' && (
                <div className="grid grid-cols-4 grid-rows-3 gap-1 w-full h-full p-2">
                  {Array.from({ length: 12 }).map((_, i) => (
                    <div key={i} className="border border-white/30"></div>
                  ))}
                </div>
              )}
              {testPattern === 'crosshair' && (
                <div className="relative w-full h-full">
                  <div className="absolute inset-x-0 top-1/2 h-px bg-white"></div>
                  <div className="absolute inset-y-0 left-1/2 w-px bg-white"></div>
                </div>
              )}
              {testPattern === 'corners' && (
                <div className="relative w-full h-full">
                  <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-white"></div>
                  <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-white"></div>
                  <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-white"></div>
                  <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-white"></div>
                </div>
              )}
              {testPattern === 'none' && (
                <span className="text-gray-500">No test pattern</span>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Brightness & Contrast</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-2">
                  <label className="text-sm font-medium">Brightness</label>
                  <span className="text-sm text-gray-500">{brightness}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={brightness}
                  onChange={(e) => setBrightness(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>

              <div>
                <div className="flex justify-between mb-2">
                  <label className="text-sm font-medium">Contrast</label>
                  <span className="text-sm text-gray-500">{contrast}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={contrast}
                  onChange={(e) => setContrast(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="grid grid-cols-2 gap-2 pt-2">
                <Button variant="outline" size="sm">
                  Reset to Default
                </Button>
                <Button variant="outline" size="sm">
                  Auto Adjust
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Keystone Correction</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="font-medium">Top Corners</h4>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm">Top Left X</label>
                  <input
                    type="range"
                    min={-50}
                    max={50}
                    value={keystoneValues.topLeft.x}
                    onChange={(e) => updateKeystone('topLeft', 'x', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <span className="text-xs text-gray-500">{keystoneValues.topLeft.x}</span>
                </div>
                <div>
                  <label className="text-sm">Top Left Y</label>
                  <input
                    type="range"
                    min={-50}
                    max={50}
                    value={keystoneValues.topLeft.y}
                    onChange={(e) => updateKeystone('topLeft', 'y', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <span className="text-xs text-gray-500">{keystoneValues.topLeft.y}</span>
                </div>
                <div>
                  <label className="text-sm">Top Right X</label>
                  <input
                    type="range"
                    min={-50}
                    max={50}
                    value={keystoneValues.topRight.x}
                    onChange={(e) => updateKeystone('topRight', 'x', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <span className="text-xs text-gray-500">{keystoneValues.topRight.x}</span>
                </div>
                <div>
                  <label className="text-sm">Top Right Y</label>
                  <input
                    type="range"
                    min={-50}
                    max={50}
                    value={keystoneValues.topRight.y}
                    onChange={(e) => updateKeystone('topRight', 'y', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <span className="text-xs text-gray-500">{keystoneValues.topRight.y}</span>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="font-medium">Bottom Corners</h4>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm">Bottom Left X</label>
                  <input
                    type="range"
                    min={-50}
                    max={50}
                    value={keystoneValues.bottomLeft.x}
                    onChange={(e) => updateKeystone('bottomLeft', 'x', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <span className="text-xs text-gray-500">{keystoneValues.bottomLeft.x}</span>
                </div>
                <div>
                  <label className="text-sm">Bottom Left Y</label>
                  <input
                    type="range"
                    min={-50}
                    max={50}
                    value={keystoneValues.bottomLeft.y}
                    onChange={(e) => updateKeystone('bottomLeft', 'y', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <span className="text-xs text-gray-500">{keystoneValues.bottomLeft.y}</span>
                </div>
                <div>
                  <label className="text-sm">Bottom Right X</label>
                  <input
                    type="range"
                    min={-50}
                    max={50}
                    value={keystoneValues.bottomRight.x}
                    onChange={(e) => updateKeystone('bottomRight', 'x', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <span className="text-xs text-gray-500">{keystoneValues.bottomRight.x}</span>
                </div>
                <div>
                  <label className="text-sm">Bottom Right Y</label>
                  <input
                    type="range"
                    min={-50}
                    max={50}
                    value={keystoneValues.bottomRight.y}
                    onChange={(e) => updateKeystone('bottomRight', 'y', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <span className="text-xs text-gray-500">{keystoneValues.bottomRight.y}</span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-between">
        <Button variant="outline" onClick={onPrevious}>
          Previous
        </Button>
        <Button onClick={onNext}>
          Next: Validation
        </Button>
      </div>
    </div>
  )
}

// Calibration Validation Step - Accuracy testing
const CalibrationValidationStep: React.FC<CalibrationStepProps> = ({
  onPrevious,
  onComplete,
  isLastStep,
  data
}) => {
  const [validationResults, setValidationResults] = useState<any>(null)
  const [isValidating, setIsValidating] = useState(false)
  const [testPoints, setTestPoints] = useState<Point[]>([])
  const { calibrationStore } = useStores()

  const runValidation = async () => {
    setIsValidating(true)
    try {
      // Simulate validation process
      await new Promise(resolve => setTimeout(resolve, 2000))

      const mockResults = {
        accuracy: 0.92,
        maxError: 2.3,
        meanError: 1.1,
        testResults: testPoints.map((point, index) => ({
          pointId: `test_${index}`,
          screenPosition: [point.screenX, point.screenY],
          worldPosition: [point.worldX, point.worldY],
          transformedPosition: [point.worldX + Math.random() * 0.1, point.worldY + Math.random() * 0.1],
          errorPixels: Math.random() * 3,
          errorMm: Math.random() * 1.5
        })),
        recommendations: [
          "Calibration accuracy is good for general use",
          "Consider adding more calibration points for higher precision",
          "Check lighting conditions for optimal ball detection"
        ]
      }

      setValidationResults(mockResults)
    } catch (error) {
      console.error('Validation failed:', error)
    } finally {
      setIsValidating(false)
    }
  }

  const addTestPoint = (x: number, y: number) => {
    const newPoint: Point = {
      x, y,
      id: `test_${testPoints.length}`,
      screenX: x,
      screenY: y,
      worldX: (x / 640) * 2.84 - 1.42,
      worldY: (y / 360) * 1.42 - 0.71
    }
    setTestPoints([...testPoints, newPoint])
  }

  const clearTestPoints = () => {
    setTestPoints([])
    setValidationResults(null)
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          Calibration Validation
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Test calibration accuracy and validate system performance
        </p>
      </div>

      {!validationResults ? (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Add Test Points</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Click on the video feed to add test points for validation
              </p>
              <div className="flex justify-center">
                <VideoFeedCanvas
                  onPointSelect={addTestPoint}
                  points={testPoints}
                  width={640}
                  height={360}
                  overlayVisible={true}
                />
              </div>
              <div className="flex justify-between mt-4">
                <div className="text-sm text-gray-500">
                  Test points: {testPoints.length}
                </div>
                <Button variant="outline" size="sm" onClick={clearTestPoints}>
                  Clear Points
                </Button>
              </div>
            </CardContent>
          </Card>

          <div className="flex justify-center">
            <Button
              onClick={runValidation}
              disabled={isValidating || testPoints.length === 0}
              variant="primary"
              className="px-8"
            >
              {isValidating ? 'Validating...' : 'Run Validation'}
            </Button>
          </div>
        </>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Validation Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {(validationResults.accuracy * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-green-700 dark:text-green-300">Overall Accuracy</div>
                </div>
                <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {validationResults.meanError.toFixed(2)}px
                  </div>
                  <div className="text-sm text-blue-700 dark:text-blue-300">Mean Error</div>
                </div>
                <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                    {validationResults.maxError.toFixed(2)}px
                  </div>
                  <div className="text-sm text-yellow-700 dark:text-yellow-300">Max Error</div>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">Recommendations</h4>
                  <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                    {validationResults.recommendations.map((rec: string, index: number) => (
                      <li key={index} className="flex items-start">
                        <span className="text-green-500 mr-2">•</span>
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">Test Point Results</h4>
                  <div className="max-h-40 overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2">Point</th>
                          <th className="text-left py-2">Error (px)</th>
                          <th className="text-left py-2">Error (mm)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {validationResults.testResults.map((result: any, index: number) => (
                          <tr key={index} className="border-b">
                            <td className="py-1">{index + 1}</td>
                            <td className="py-1">{result.errorPixels.toFixed(2)}</td>
                            <td className="py-1">{result.errorMm.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="flex justify-center space-x-4">
            <Button variant="outline" onClick={() => setValidationResults(null)}>
              Run Again
            </Button>
            <Button variant="outline">
              Export Results
            </Button>
          </div>
        </>
      )}

      <div className="flex justify-between">
        <Button variant="outline" onClick={onPrevious}>
          Previous
        </Button>
        <Button
          onClick={onComplete}
          variant="primary"
          disabled={!validationResults}
        >
          Complete Calibration
        </Button>
      </div>
    </div>
  )
}

// Main CalibrationWizard component
export const CalibrationWizard = observer(() => {
  const [currentStep, setCurrentStep] = useState(0)
  const [loading, setLoading] = useState(false)
  const [stepData, setStepData] = useState<Record<string, any>>({})
  const [sessionId, setSessionId] = useState<string | null>(null)
  const { calibrationStore } = useStores()

  const steps: CalibrationStep[] = [
    {
      id: 'camera',
      title: 'Camera Calibration',
      description: 'Interactive geometric calibration with point selection',
      component: CameraCalibrationStep,
      requirements: ['Camera access', 'Clear view of table', 'Stable lighting']
    },
    {
      id: 'table',
      title: 'Table Configuration',
      description: 'ROI selection and perspective correction',
      component: TableConfigurationStep,
      requirements: ['Visible table boundaries', 'All pockets in view']
    },
    {
      id: 'color',
      title: 'Color Calibration',
      description: 'HSV range calibration for ball detection',
      component: ColorCalibrationStep,
      requirements: ['Representative ball samples', 'Consistent lighting']
    },
    {
      id: 'projector',
      title: 'Projector Alignment',
      description: 'Keystone correction and alignment',
      component: ProjectorAlignmentStep,
      requirements: ['Projector connected', 'Clear projection surface']
    },
    {
      id: 'validation',
      title: 'Validation',
      description: 'Accuracy testing and validation',
      component: CalibrationValidationStep,
      requirements: ['All previous steps completed']
    }
  ]

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleDataChange = (data: any) => {
    setStepData(prev => ({
      ...prev,
      [steps[currentStep].id]: data
    }))
  }

  const handleComplete = async () => {
    setLoading(true)
    try {
      // Start calibration session if not already started
      if (!sessionId) {
        const request: CalibrationStartRequest = {
          calibration_type: 'advanced',
          timeout_seconds: 1800
        }
        const response = await calibrationStore.startCalibration(request)
        setSessionId(response.session.session_id)
      }

      // Process all captured points
      const cameraData = stepData.camera
      if (cameraData?.points) {
        for (const point of cameraData.points) {
          await calibrationStore.capturePoint(
            point.screenX,
            point.screenY,
            point.worldX,
            point.worldY
          )
        }
      }

      // Apply calibration
      await calibrationStore.applyCalibration()

      // Reset wizard
      setCurrentStep(0)
      setStepData({})
      setSessionId(null)

      alert('Calibration completed successfully!')
    } catch (error) {
      console.error('Calibration failed:', error)
      alert('Calibration failed: ' + (error instanceof Error ? error.message : 'Unknown error'))
    } finally {
      setLoading(false)
    }
  }

  const currentStepData = steps[currentStep]
  const StepComponent = currentStepData.component

  return (
    <div className="space-y-6">
      {/* Progress Indicator */}
      <Card>
        <CardContent padding="sm">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Interactive Calibration Wizard
            </h2>
            <span className="text-sm text-gray-500">
              Step {currentStep + 1} of {steps.length}
            </span>
          </div>

          {/* Step Progress Bar */}
          <div className="flex items-center space-x-2">
            {steps.map((step, index) => (
              <React.Fragment key={step.id}>
                <div className="flex items-center">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                      index < currentStep
                        ? 'bg-green-100 text-green-600 dark:bg-green-900 dark:text-green-400'
                        : index === currentStep
                        ? 'bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-400'
                        : 'bg-gray-100 text-gray-400 dark:bg-gray-800 dark:text-gray-600'
                    }`}
                  >
                    {index < currentStep ? (
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    ) : (
                      index + 1
                    )}
                  </div>
                  <div className="ml-2 hidden sm:block">
                    <p className={`text-sm font-medium ${
                      index === currentStep ? 'text-gray-900 dark:text-white' : 'text-gray-500 dark:text-gray-400'
                    }`}>
                      {step.title}
                    </p>
                    <p className="text-xs text-gray-400 dark:text-gray-500">
                      {step.description}
                    </p>
                  </div>
                </div>
                {index < steps.length - 1 && (
                  <div className={`flex-1 h-0.5 ${
                    index < currentStep ? 'bg-green-300' : 'bg-gray-200 dark:bg-gray-700'
                  }`} />
                )}
              </React.Fragment>
            ))}
          </div>

          {/* Requirements for current step */}
          <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">Requirements:</h4>
            <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
              {currentStepData.requirements.map((req, index) => (
                <li key={index} className="flex items-center">
                  <span className="text-green-500 mr-2">✓</span>
                  {req}
                </li>
              ))}
            </ul>
          </div>
        </CardContent>
      </Card>

      {/* Current Step */}
      <Card>
        <CardContent>
          {loading ? (
            <div className="text-center py-12">
              <div className="inline-flex items-center space-x-2">
                <svg className="animate-spin h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span className="text-gray-600 dark:text-gray-400">Completing calibration...</span>
              </div>
            </div>
          ) : (
            <StepComponent
              onNext={handleNext}
              onPrevious={handlePrevious}
              onComplete={handleComplete}
              isFirstStep={currentStep === 0}
              isLastStep={currentStep === steps.length - 1}
              data={stepData[currentStepData.id]}
              onDataChange={handleDataChange}
            />
          )}
        </CardContent>
      </Card>
    </div>
  )
})
