import React, { useState } from 'react'
import { observer } from 'mobx-react-lite'
import { Card, CardHeader, CardTitle, CardContent, Input, Select, Button, Slider, Checkbox } from '../ui'
import type { SelectOption } from '../ui'

export const ProjectorConfig = observer(() => {
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})

  // Display settings
  const [resolution, setResolution] = useState('1920x1080')
  const [refreshRate, setRefreshRate] = useState(60)
  const [brightness, setBrightness] = useState(0.8)
  const [contrast, setContrast] = useState(0.7)
  const [colorTemp, setColorTemp] = useState(6500)

  // Positioning and calibration
  const [distanceFromTable, setDistanceFromTable] = useState(150) // cm
  const [projectionAngle, setProjectionAngle] = useState(90) // degrees
  const [keystoneH, setKeystoneH] = useState(0) // horizontal keystone
  const [keystoneV, setKeystoneV] = useState(0) // vertical keystone
  const [offsetX, setOffsetX] = useState(0)
  const [offsetY, setOffsetY] = useState(0)

  // Overlay settings
  const [enableOverlays, setEnableOverlays] = useState(true)
  const [overlayOpacity, setOverlayOpacity] = useState(0.7)
  const [ballTrails, setBallTrails] = useState(true)
  const [trajectoryLines, setTrajectoryLines] = useState(true)
  const [aimingAssist, setAimingAssist] = useState(false)
  const [ghostBalls, setGhostBalls] = useState(true)

  // Visual effects
  const [enableAnimations, setEnableAnimations] = useState(true)
  const [animationSpeed, setAnimationSpeed] = useState(1.0)
  const [particleEffects, setParticleEffects] = useState(true)
  const [showFPS, setShowFPS] = useState(false)
  const [debugMode, setDebugMode] = useState(false)

  // Color schemes
  const [ballTrailColor, setBallTrailColor] = useState('#ff6b35')
  const [trajectoryColor, setTrajectoryColor] = useState('#4ecdc4')
  const [aimLineColor, setAimLineColor] = useState('#45b7d1')
  const [ghostBallColor, setGhostBallColor] = useState('#ffffff')

  const resolutionOptions: SelectOption[] = [
    { value: '1280x720', label: '1280 × 720 (HD)' },
    { value: '1920x1080', label: '1920 × 1080 (Full HD)' },
    { value: '2560x1440', label: '2560 × 1440 (QHD)' },
    { value: '3840x2160', label: '3840 × 2160 (4K)' }
  ]

  const refreshRateOptions: SelectOption[] = [
    { value: '30', label: '30 Hz' },
    { value: '60', label: '60 Hz' },
    { value: '120', label: '120 Hz' },
    { value: '144', label: '144 Hz' }
  ]

  const handleTestProjection = async () => {
    setLoading(true)
    try {
      // Test projection by showing a calibration pattern
      console.log('Testing projector with current settings')
      await new Promise(resolve => setTimeout(resolve, 2000))
    } catch (error) {
      console.error('Projector test failed:', error)
      setErrors({ projection: 'Failed to test projection. Check projector connection.' })
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    setLoading(true)
    setErrors({})

    try {
      const newErrors: Record<string, string> = {}

      if (distanceFromTable < 50 || distanceFromTable > 500) {
        newErrors.distance = 'Distance from table must be between 50cm and 500cm'
      }

      if (projectionAngle < 45 || projectionAngle > 135) {
        newErrors.angle = 'Projection angle must be between 45° and 135°'
      }

      if (Object.keys(newErrors).length > 0) {
        setErrors(newErrors)
        return
      }

      const projectorConfig = {
        display: {
          resolution,
          refreshRate,
          brightness,
          contrast,
          colorTemperature: colorTemp
        },
        positioning: {
          distanceFromTable,
          projectionAngle,
          keystoneCorrection: {
            horizontal: keystoneH,
            vertical: keystoneV
          },
          offset: {
            x: offsetX,
            y: offsetY
          }
        },
        overlays: {
          enabled: enableOverlays,
          opacity: overlayOpacity,
          ballTrails,
          trajectoryLines,
          aimingAssist,
          ghostBalls
        },
        effects: {
          animations: enableAnimations,
          animationSpeed,
          particleEffects,
          showFPS,
          debugMode
        },
        colors: {
          ballTrail: ballTrailColor,
          trajectory: trajectoryColor,
          aimLine: aimLineColor,
          ghostBall: ghostBallColor
        }
      }

      console.log('Saving projector configuration:', projectorConfig)
      await new Promise(resolve => setTimeout(resolve, 1000))

    } catch (error) {
      console.error('Failed to save projector settings:', error)
      setErrors({ general: 'Failed to save projector settings. Please try again.' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Display Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Display Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Select
              label="Resolution"
              value={resolution}
              onChange={(e) => setResolution(e.target.value)}
              options={resolutionOptions}
              fullWidth
            />

            <Select
              label="Refresh Rate"
              value={refreshRate.toString()}
              onChange={(e) => setRefreshRate(Number(e.target.value))}
              options={refreshRateOptions}
              fullWidth
            />
          </div>

          <div className="mt-4 space-y-4">
            <Slider
              label="Brightness"
              value={brightness}
              onChange={(e) => setBrightness(Number(e.target.value))}
              min={0.1}
              max={1.0}
              step={0.01}
              formatValue={(value) => `${Math.round(value * 100)}%`}
              fullWidth
            />

            <Slider
              label="Contrast"
              value={contrast}
              onChange={(e) => setContrast(Number(e.target.value))}
              min={0.1}
              max={1.0}
              step={0.01}
              formatValue={(value) => `${Math.round(value * 100)}%`}
              fullWidth
            />

            <Slider
              label="Color Temperature (K)"
              value={colorTemp}
              onChange={(e) => setColorTemp(Number(e.target.value))}
              min={3000}
              max={10000}
              step={100}
              formatValue={(value) => `${value}K`}
              fullWidth
            />
          </div>
        </CardContent>
      </Card>

      {/* Positioning */}
      <Card>
        <CardHeader>
          <CardTitle>Positioning & Calibration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Distance from Table (cm)"
              type="number"
              value={distanceFromTable}
              onChange={(e) => setDistanceFromTable(Number(e.target.value))}
              error={errors.distance}
              min={50}
              max={500}
              fullWidth
            />

            <Input
              label="Projection Angle (degrees)"
              type="number"
              value={projectionAngle}
              onChange={(e) => setProjectionAngle(Number(e.target.value))}
              error={errors.angle}
              min={45}
              max={135}
              fullWidth
            />
          </div>

          <div className="mt-4 space-y-4">
            <Slider
              label="Horizontal Keystone"
              value={keystoneH}
              onChange={(e) => setKeystoneH(Number(e.target.value))}
              min={-50}
              max={50}
              step={1}
              formatValue={(value) => `${value > 0 ? '+' : ''}${value}`}
              hint="Correct horizontal perspective distortion"
              fullWidth
            />

            <Slider
              label="Vertical Keystone"
              value={keystoneV}
              onChange={(e) => setKeystoneV(Number(e.target.value))}
              min={-50}
              max={50}
              step={1}
              formatValue={(value) => `${value > 0 ? '+' : ''}${value}`}
              hint="Correct vertical perspective distortion"
              fullWidth
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Slider
                label="Horizontal Offset"
                value={offsetX}
                onChange={(e) => setOffsetX(Number(e.target.value))}
                min={-100}
                max={100}
                step={1}
                formatValue={(value) => `${value > 0 ? '+' : ''}${value} px`}
                fullWidth
              />

              <Slider
                label="Vertical Offset"
                value={offsetY}
                onChange={(e) => setOffsetY(Number(e.target.value))}
                min={-100}
                max={100}
                step={1}
                formatValue={(value) => `${value > 0 ? '+' : ''}${value} px`}
                fullWidth
              />
            </div>
          </div>

          <div className="mt-4">
            <Button
              onClick={handleTestProjection}
              loading={loading}
              variant="outline"
            >
              Test Projection
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Overlay Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Overlay Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Checkbox
              label="Enable Overlays"
              checked={enableOverlays}
              onChange={(e) => setEnableOverlays(e.target.checked)}
              hint="Show visual overlays on the table"
            />

            <Slider
              label="Overlay Opacity"
              value={overlayOpacity}
              onChange={(e) => setOverlayOpacity(Number(e.target.value))}
              min={0.1}
              max={1.0}
              step={0.01}
              disabled={!enableOverlays}
              formatValue={(value) => `${Math.round(value * 100)}%`}
              fullWidth
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Checkbox
                label="Ball Trails"
                checked={ballTrails}
                onChange={(e) => setBallTrails(e.target.checked)}
                disabled={!enableOverlays}
                hint="Show path trails behind moving balls"
              />

              <Checkbox
                label="Trajectory Lines"
                checked={trajectoryLines}
                onChange={(e) => setTrajectoryLines(e.target.checked)}
                disabled={!enableOverlays}
                hint="Show predicted ball paths"
              />

              <Checkbox
                label="Aiming Assist"
                checked={aimingAssist}
                onChange={(e) => setAimingAssist(e.target.checked)}
                disabled={!enableOverlays}
                hint="Show aiming guidelines"
              />

              <Checkbox
                label="Ghost Balls"
                checked={ghostBalls}
                onChange={(e) => setGhostBalls(e.target.checked)}
                disabled={!enableOverlays}
                hint="Show target ball positions"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Visual Effects */}
      <Card>
        <CardHeader>
          <CardTitle>Visual Effects</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Checkbox
              label="Enable Animations"
              checked={enableAnimations}
              onChange={(e) => setEnableAnimations(e.target.checked)}
              hint="Smooth transitions and animations"
            />

            <Slider
              label="Animation Speed"
              value={animationSpeed}
              onChange={(e) => setAnimationSpeed(Number(e.target.value))}
              min={0.1}
              max={3.0}
              step={0.1}
              disabled={!enableAnimations}
              formatValue={(value) => `${value.toFixed(1)}x`}
              fullWidth
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Checkbox
                label="Particle Effects"
                checked={particleEffects}
                onChange={(e) => setParticleEffects(e.target.checked)}
                hint="Visual effects for collisions"
              />

              <Checkbox
                label="Show FPS Counter"
                checked={showFPS}
                onChange={(e) => setShowFPS(e.target.checked)}
                hint="Display frames per second"
              />

              <Checkbox
                label="Debug Mode"
                checked={debugMode}
                onChange={(e) => setDebugMode(e.target.checked)}
                hint="Show technical debugging information"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Color Customization */}
      <Card>
        <CardHeader>
          <CardTitle>Color Customization</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Ball Trail Color
              </label>
              <div className="flex items-center space-x-3">
                <input
                  type="color"
                  value={ballTrailColor}
                  onChange={(e) => setBallTrailColor(e.target.value)}
                  className="w-12 h-8 border border-gray-300 rounded cursor-pointer"
                  disabled={!ballTrails || !enableOverlays}
                />
                <Input
                  value={ballTrailColor}
                  onChange={(e) => setBallTrailColor(e.target.value)}
                  disabled={!ballTrails || !enableOverlays}
                  className="flex-1"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Trajectory Color
              </label>
              <div className="flex items-center space-x-3">
                <input
                  type="color"
                  value={trajectoryColor}
                  onChange={(e) => setTrajectoryColor(e.target.value)}
                  className="w-12 h-8 border border-gray-300 rounded cursor-pointer"
                  disabled={!trajectoryLines || !enableOverlays}
                />
                <Input
                  value={trajectoryColor}
                  onChange={(e) => setTrajectoryColor(e.target.value)}
                  disabled={!trajectoryLines || !enableOverlays}
                  className="flex-1"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Aim Line Color
              </label>
              <div className="flex items-center space-x-3">
                <input
                  type="color"
                  value={aimLineColor}
                  onChange={(e) => setAimLineColor(e.target.value)}
                  className="w-12 h-8 border border-gray-300 rounded cursor-pointer"
                  disabled={!aimingAssist || !enableOverlays}
                />
                <Input
                  value={aimLineColor}
                  onChange={(e) => setAimLineColor(e.target.value)}
                  disabled={!aimingAssist || !enableOverlays}
                  className="flex-1"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Ghost Ball Color
              </label>
              <div className="flex items-center space-x-3">
                <input
                  type="color"
                  value={ghostBallColor}
                  onChange={(e) => setGhostBallColor(e.target.value)}
                  className="w-12 h-8 border border-gray-300 rounded cursor-pointer"
                  disabled={!ghostBalls || !enableOverlays}
                />
                <Input
                  value={ghostBallColor}
                  onChange={(e) => setGhostBallColor(e.target.value)}
                  disabled={!ghostBalls || !enableOverlays}
                  className="flex-1"
                />
              </div>
            </div>
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
