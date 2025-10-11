import React, { useState, useRef, useEffect, useCallback } from 'react'
import { observer } from 'mobx-react-lite'
import { Card, CardHeader, CardTitle, CardContent, Button } from '../../ui'
import { useStores } from '../../../hooks/useStores'

// Types for calibration workflow
interface Point {
  x: number
  y: number
  id: string
  screenX: number
  screenY: number
  worldX: number
  worldY: number
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
  const imgRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const animationFrameRef = useRef<number>()
  const [streamConnected, setStreamConnected] = useState(false)
  const [streamError, setStreamError] = useState<string | null>(null)

  // Get the video stream URL from the API
  const videoStreamUrl = `${window.location.origin}/api/v1/stream/video`

  // Continuous canvas drawing loop for live video
  const drawFrame = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw video frame if available
    if (img.complete && img.naturalWidth > 0) {
      ctx.drawImage(img, 0, 0, width, height)
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
        ctx.lineWidth = 3
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

      // Add click instruction text at bottom
      ctx.fillStyle = '#9ca3af'
      ctx.font = '14px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('Click on corners to manually adjust, or use auto-detect', width/2, height - 20)
    }

    // Request next frame
    animationFrameRef.current = requestAnimationFrame(drawFrame)
  }, [points, width, height, overlayVisible])

  // Start animation loop
  useEffect(() => {
    animationFrameRef.current = requestAnimationFrame(drawFrame)

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [drawFrame])

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
      {/* MJPEG stream image (hidden, used as source for canvas) */}
      <img
        ref={imgRef}
        src={videoStreamUrl}
        alt="Camera stream"
        style={{ display: 'none' }}
        onLoad={() => setStreamConnected(true)}
        onError={(e) => {
          setStreamError('Failed to load video stream')
          setStreamConnected(false)
        }}
      />

      {/* Canvas for overlay */}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onClick={handleCanvasClick}
        className="border border-gray-300 dark:border-gray-600 rounded-lg cursor-crosshair bg-gray-900"
      />

      {/* Connection status indicator */}
      {!streamConnected && (
        <div className="absolute top-2 left-2 px-2 py-1 bg-yellow-600 text-white text-xs rounded shadow">
          Connecting to camera stream...
        </div>
      )}

      {streamError && (
        <div className="absolute top-2 left-2 px-2 py-1 bg-red-600 text-white text-xs rounded shadow">
          {streamError}
        </div>
      )}

      {streamConnected && (
        <div className="absolute top-2 left-2 px-2 py-1 bg-green-600 text-white text-xs rounded shadow">
          Stream Connected
        </div>
      )}
    </div>
  )
}

// Playing Area Calibration Step - Single step for table detection
const PlayingAreaCalibrationStep: React.FC<{
  onComplete: () => void
}> = ({ onComplete }) => {
  const [points, setPoints] = useState<Point[]>([])
  const [isDetecting, setIsDetecting] = useState(false)
  const [isApplying, setIsApplying] = useState(false)
  const [detectionResult, setDetectionResult] = useState<any>(null)

  const pointInstructions = [
    "Top-left corner of the playing area",
    "Top-right corner of the playing area",
    "Bottom-right corner of the playing area",
    "Bottom-left corner of the playing area"
  ]

  const autoDetectPlayingArea = async () => {
    setIsDetecting(true)
    try {
      // Use the existing table detection endpoint
      const response = await fetch('/api/v1/vision/detection/table', {
        method: 'GET'
      })

      if (!response.ok) {
        throw new Error('Failed to detect table boundaries')
      }

      const data = await response.json()

      if (data.table_corners && data.table_corners.length === 4) {
        // Convert the detected corners to our Point format
        const detectedPoints: Point[] = data.table_corners.map((corner: [number, number], index: number) => ({
          x: corner[0],
          y: corner[1],
          id: `corner_${index}`,
          screenX: corner[0],
          screenY: corner[1],
          worldX: 0, // Will be calculated when applied
          worldY: 0
        }))

        setPoints(detectedPoints)
        setDetectionResult(data)
        alert('Playing area detected successfully! Review the corners and click Apply.')
      } else {
        alert('Could not detect all 4 corners. Please try again or set them manually.')
      }
    } catch (error) {
      console.error('Auto-detection failed:', error)
      alert('Auto-detection failed: ' + (error instanceof Error ? error.message : 'Unknown error'))
    } finally {
      setIsDetecting(false)
    }
  }

  const handlePointSelect = (screenX: number, screenY: number) => {
    // Find the nearest corner and update it
    if (points.length === 0) {
      // No points yet, just add sequentially
      const newPoint: Point = {
        x: screenX,
        y: screenY,
        id: `corner_${points.length}`,
        screenX,
        screenY,
        worldX: 0,
        worldY: 0
      }
      setPoints([newPoint])
      return
    }

    if (points.length < 4) {
      // Add next point
      const newPoint: Point = {
        x: screenX,
        y: screenY,
        id: `corner_${points.length}`,
        screenX,
        screenY,
        worldX: 0,
        worldY: 0
      }
      setPoints([...points, newPoint])
      return
    }

    // Find nearest corner to update
    let nearestIndex = 0
    let minDistance = Number.MAX_VALUE

    points.forEach((point, index) => {
      const distance = Math.sqrt(
        Math.pow(point.screenX - screenX, 2) + Math.pow(point.screenY - screenY, 2)
      )
      if (distance < minDistance) {
        minDistance = distance
        nearestIndex = index
      }
    })

    // Update the nearest corner
    const newPoints = [...points]
    newPoints[nearestIndex] = {
      ...newPoints[nearestIndex],
      screenX,
      screenY,
      x: screenX,
      y: screenY
    }
    setPoints(newPoints)
  }

  const removeLastPoint = () => {
    if (points.length > 0) {
      setPoints(points.slice(0, -1))
    }
  }

  const clearAllPoints = () => {
    setPoints([])
    setDetectionResult(null)
  }

  const applyCalibration = async () => {
    if (points.length !== 4) {
      alert('Please set all 4 corners of the playing area')
      return
    }

    setIsApplying(true)
    try {
      // Send the corners to the backend to save in table configuration
      const response = await fetch('/api/v1/config/table/playing-area', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          corners: points.map(p => ({
            x: p.screenX,
            y: p.screenY
          }))
        })
      })

      if (!response.ok) {
        throw new Error('Failed to apply playing area calibration')
      }

      alert('Playing area calibration applied successfully!')
      onComplete()
    } catch (error) {
      console.error('Failed to apply calibration:', error)
      alert('Failed to apply calibration: ' + (error instanceof Error ? error.message : 'Unknown error'))
    } finally {
      setIsApplying(false)
    }
  }

  const canProceed = points.length === 4

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          Playing Area Calibration
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Identify the 4 corners of the playing surface for accurate trajectory calculations
        </p>
      </div>

      <Card>
        <CardContent padding="sm">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <h4 className="font-medium text-blue-900 dark:text-blue-200 mb-2">How it works:</h4>
            <ul className="text-sm text-blue-800 dark:text-blue-300 space-y-2">
              <li className="flex items-start">
                <span className="text-blue-500 mr-2">1.</span>
                Click "Auto-Detect" to automatically find the playing area, or manually click the 4 corners
              </li>
              <li className="flex items-start">
                <span className="text-blue-500 mr-2">2.</span>
                Click on any corner to adjust its position if needed
              </li>
              <li className="flex items-start">
                <span className="text-blue-500 mr-2">3.</span>
                Click "Apply Calibration" to save the playing area for accurate trajectories
              </li>
            </ul>
            <div className="mt-3 text-sm text-blue-700 dark:text-blue-400">
              Corners set: {points.length} / 4
              {points.length > 0 && points.length < 4 && (
                <span className="ml-2">
                  - Next: {pointInstructions[points.length]}
                </span>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

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

      <div className="grid grid-cols-3 gap-4">
        <Button
          variant="primary"
          onClick={autoDetectPlayingArea}
          disabled={isDetecting}
        >
          {isDetecting ? 'Detecting...' : 'Auto-Detect'}
        </Button>
        <Button
          variant="outline"
          onClick={removeLastPoint}
          disabled={points.length === 0}
        >
          Remove Last
        </Button>
        <Button
          variant="outline"
          onClick={clearAllPoints}
          disabled={points.length === 0}
        >
          Clear All
        </Button>
      </div>

      <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
        <h4 className="font-medium text-gray-900 dark:text-white mb-2">Detected Corners:</h4>
        {points.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            No corners set. Click "Auto-Detect" or manually click on the video to set corners.
          </p>
        ) : (
          <div className="grid grid-cols-2 gap-4 text-sm">
            {points.map((point, index) => (
              <div key={point.id} className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">
                  {pointInstructions[index]}:
                </span>
                <span className="font-mono text-gray-800 dark:text-gray-200">
                  ({point.screenX.toFixed(0)}, {point.screenY.toFixed(0)})
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="flex justify-center">
        <Button
          onClick={applyCalibration}
          disabled={!canProceed || isApplying}
          variant={canProceed ? "primary" : "outline"}
          className="px-8"
        >
          {isApplying ? 'Applying...' : 'Apply Calibration'}
        </Button>
      </div>
    </div>
  )
}

// Main CalibrationWizard component
export const CalibrationWizard = observer(() => {
  const [completed, setCompleted] = useState(false)
  const { calibrationStore } = useStores()

  const handleComplete = () => {
    setCompleted(true)
  }

  const handleReset = () => {
    setCompleted(false)
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardContent>
          {!completed ? (
            <PlayingAreaCalibrationStep onComplete={handleComplete} />
          ) : (
            <div className="text-center py-12 space-y-4">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 dark:bg-green-900/30 rounded-full mb-4">
                <svg className="w-10 h-10 text-green-600 dark:text-green-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
                Calibration Complete!
              </h3>
              <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto">
                Your playing area has been calibrated successfully. Trajectory calculations will now
                use the defined boundaries for accurate predictions.
              </p>
              <div className="flex justify-center gap-4 pt-4">
                <Button variant="outline" onClick={handleReset}>
                  Recalibrate
                </Button>
                <Button variant="primary" onClick={() => window.location.reload()}>
                  Done
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
})
