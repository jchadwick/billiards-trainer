import React, { useState } from 'react'
import { observer } from 'mobx-react-lite'
import { Card, CardHeader, CardTitle, CardContent, Button, Slider, Checkbox } from '../ui'

export const PhysicsConfig = observer(() => {
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})

  // Ball physics
  const [ballMass, setBallMass] = useState(163) // grams (standard pool ball)
  const [ballRadius, setBallRadius] = useState(2.25) // inches
  const [ballFriction, setBallFriction] = useState(0.02)
  const [rollingResistance, setRollingResistance] = useState(0.01)
  const [airResistance, setAirResistance] = useState(0.001)

  // Table physics
  const [tableFriction, setTableFriction] = useState(0.15)
  const [cushionElasticity, setCushionElasticity] = useState(0.85)
  const [spinDecay, setSpinDecay] = useState(0.95)
  const [bounceThreshold, setBounceThreshold] = useState(0.1)

  // Collision physics
  const [restitutionCoeff, setRestitutionCoeff] = useState(0.92)
  const [impactThreshold, setImpactThreshold] = useState(0.05)
  const [energyLoss, setEnergyLoss] = useState(0.08)
  const [soundEnabled, setSoundEnabled] = useState(true)

  // Advanced physics
  const [gravityStrength, setGravityStrength] = useState(9.81)
  const [timeStep, setTimeStep] = useState(0.016) // 60fps
  const [maxVelocity, setMaxVelocity] = useState(500) // cm/s
  const [enableMagnus, setEnableMagnus] = useState(true)
  const [enableCoriolis, setEnableCoriolis] = useState(false)

  // Simulation quality
  const [physicsIterations, setPhysicsIterations] = useState(8)
  const [collisionIterations, setCollisionIterations] = useState(4)
  const [enableSubstepping, setEnableSubstepping] = useState(true)
  const [adaptiveTimeStep, setAdaptiveTimeStep] = useState(true)

  const handleSave = async () => {
    setLoading(true)
    setErrors({})

    try {
      // Validate inputs
      const newErrors: Record<string, string> = {}

      if (ballMass < 100 || ballMass > 200) {
        newErrors.ballMass = 'Ball mass must be between 100g and 200g'
      }

      if (ballRadius < 2.0 || ballRadius > 3.0) {
        newErrors.ballRadius = 'Ball radius must be between 2.0 and 3.0 inches'
      }

      if (timeStep < 0.001 || timeStep > 0.1) {
        newErrors.timeStep = 'Time step must be between 0.001 and 0.1 seconds'
      }

      if (maxVelocity < 100 || maxVelocity > 1000) {
        newErrors.maxVelocity = 'Max velocity must be between 100 and 1000 cm/s'
      }

      if (physicsIterations < 1 || physicsIterations > 20) {
        newErrors.physicsIterations = 'Physics iterations must be between 1 and 20'
      }

      if (collisionIterations < 1 || collisionIterations > 10) {
        newErrors.collisionIterations = 'Collision iterations must be between 1 and 10'
      }

      if (Object.keys(newErrors).length > 0) {
        setErrors(newErrors)
        return
      }

      const physicsConfig = {
        balls: {
          mass: ballMass,
          radius: ballRadius,
          friction: ballFriction,
          rollingResistance,
          airResistance
        },
        table: {
          friction: tableFriction,
          cushionElasticity,
          spinDecay,
          bounceThreshold
        },
        collisions: {
          restitutionCoefficient: restitutionCoeff,
          impactThreshold,
          energyLoss,
          soundEnabled
        },
        advanced: {
          gravity: gravityStrength,
          timeStep,
          maxVelocity,
          enableMagnusEffect: enableMagnus,
          enableCoriolisEffect: enableCoriolis
        },
        simulation: {
          physicsIterations,
          collisionIterations,
          enableSubstepping,
          adaptiveTimeStep
        }
      }

      console.log('Saving physics configuration:', physicsConfig)
      await new Promise(resolve => setTimeout(resolve, 1000))

    } catch (error) {
      console.error('Failed to save physics settings:', error)
      setErrors({ general: 'Failed to save physics settings. Please try again.' })
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setBallMass(163)
    setBallRadius(2.25)
    setBallFriction(0.02)
    setRollingResistance(0.01)
    setAirResistance(0.001)
    setTableFriction(0.15)
    setCushionElasticity(0.85)
    setSpinDecay(0.95)
    setBounceThreshold(0.1)
    setRestitutionCoeff(0.92)
    setImpactThreshold(0.05)
    setEnergyLoss(0.08)
    setSoundEnabled(true)
    setGravityStrength(9.81)
    setTimeStep(0.016)
    setMaxVelocity(500)
    setEnableMagnus(true)
    setEnableCoriolis(false)
    setPhysicsIterations(8)
    setCollisionIterations(4)
    setEnableSubstepping(true)
    setAdaptiveTimeStep(true)
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

      {/* Ball Physics */}
      <Card>
        <CardHeader>
          <CardTitle>Ball Physics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Slider
              label="Ball Mass (grams)"
              value={ballMass}
              onChange={(e) => setBallMass(Number(e.target.value))}
              min={100}
              max={200}
              step={1}
              error={errors.ballMass}
              formatValue={(value) => `${value}g`}
              hint="Standard pool ball mass is 163g"
              fullWidth
            />

            <Slider
              label="Ball Radius (inches)"
              value={ballRadius}
              onChange={(e) => setBallRadius(Number(e.target.value))}
              min={2.0}
              max={3.0}
              step={0.01}
              error={errors.ballRadius}
              formatValue={(value) => `${value}"`}
              hint="Standard pool ball radius is 2.25 inches"
              fullWidth
            />

            <Slider
              label="Ball Friction Coefficient"
              value={ballFriction}
              onChange={(e) => setBallFriction(Number(e.target.value))}
              min={0.001}
              max={0.1}
              step={0.001}
              formatValue={(value) => value.toFixed(3)}
              hint="Friction between ball and table surface"
              fullWidth
            />

            <Slider
              label="Rolling Resistance"
              value={rollingResistance}
              onChange={(e) => setRollingResistance(Number(e.target.value))}
              min={0.001}
              max={0.05}
              step={0.001}
              formatValue={(value) => value.toFixed(3)}
              hint="Resistance to rolling motion"
              fullWidth
            />

            <Slider
              label="Air Resistance"
              value={airResistance}
              onChange={(e) => setAirResistance(Number(e.target.value))}
              min={0.0001}
              max={0.01}
              step={0.0001}
              formatValue={(value) => value.toFixed(4)}
              hint="Air drag coefficient"
              fullWidth
            />
          </div>
        </CardContent>
      </Card>

      {/* Table Physics */}
      <Card>
        <CardHeader>
          <CardTitle>Table Physics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Slider
              label="Table Friction"
              value={tableFriction}
              onChange={(e) => setTableFriction(Number(e.target.value))}
              min={0.05}
              max={0.5}
              step={0.01}
              formatValue={(value) => value.toFixed(2)}
              hint="Sliding friction on table surface"
              fullWidth
            />

            <Slider
              label="Cushion Elasticity"
              value={cushionElasticity}
              onChange={(e) => setCushionElasticity(Number(e.target.value))}
              min={0.5}
              max={1.0}
              step={0.01}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="How bouncy the cushions are"
              fullWidth
            />

            <Slider
              label="Spin Decay Rate"
              value={spinDecay}
              onChange={(e) => setSpinDecay(Number(e.target.value))}
              min={0.8}
              max={1.0}
              step={0.01}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="How quickly spin energy dissipates"
              fullWidth
            />

            <Slider
              label="Bounce Threshold"
              value={bounceThreshold}
              onChange={(e) => setBounceThreshold(Number(e.target.value))}
              min={0.01}
              max={0.5}
              step={0.01}
              formatValue={(value) => value.toFixed(2)}
              hint="Minimum velocity for bouncing"
              fullWidth
            />
          </div>
        </CardContent>
      </Card>

      {/* Collision Physics */}
      <Card>
        <CardHeader>
          <CardTitle>Collision Physics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Slider
              label="Restitution Coefficient"
              value={restitutionCoeff}
              onChange={(e) => setRestitutionCoeff(Number(e.target.value))}
              min={0.5}
              max={1.0}
              step={0.01}
              formatValue={(value) => value.toFixed(2)}
              hint="Energy preserved in ball-to-ball collisions"
              fullWidth
            />

            <Slider
              label="Impact Threshold"
              value={impactThreshold}
              onChange={(e) => setImpactThreshold(Number(e.target.value))}
              min={0.01}
              max={0.2}
              step={0.01}
              formatValue={(value) => value.toFixed(2)}
              hint="Minimum velocity for collision detection"
              fullWidth
            />

            <Slider
              label="Energy Loss Factor"
              value={energyLoss}
              onChange={(e) => setEnergyLoss(Number(e.target.value))}
              min={0.01}
              max={0.2}
              step={0.01}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="Energy lost per collision"
              fullWidth
            />

            <Checkbox
              label="Enable Collision Sounds"
              checked={soundEnabled}
              onChange={(e) => setSoundEnabled(e.target.checked)}
              hint="Play audio for ball collisions"
            />
          </div>
        </CardContent>
      </Card>

      {/* Advanced Physics */}
      <Card>
        <CardHeader>
          <CardTitle>Advanced Physics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Slider
              label="Gravity Strength (m/s²)"
              value={gravityStrength}
              onChange={(e) => setGravityStrength(Number(e.target.value))}
              min={5}
              max={15}
              step={0.1}
              formatValue={(value) => `${value.toFixed(1)} m/s²`}
              hint="Earth gravity is 9.81 m/s²"
              fullWidth
            />

            <Slider
              label="Physics Time Step (seconds)"
              value={timeStep}
              onChange={(e) => setTimeStep(Number(e.target.value))}
              min={0.001}
              max={0.1}
              step={0.001}
              error={errors.timeStep}
              formatValue={(value) => `${value.toFixed(3)}s`}
              hint="Smaller values = more accurate simulation"
              fullWidth
            />

            <Slider
              label="Maximum Velocity (cm/s)"
              value={maxVelocity}
              onChange={(e) => setMaxVelocity(Number(e.target.value))}
              min={100}
              max={1000}
              step={10}
              error={errors.maxVelocity}
              formatValue={(value) => `${value} cm/s`}
              hint="Velocity cap for stability"
              fullWidth
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Checkbox
                label="Enable Magnus Effect"
                checked={enableMagnus}
                onChange={(e) => setEnableMagnus(e.target.checked)}
                hint="Spin affects ball trajectory"
              />

              <Checkbox
                label="Enable Coriolis Effect"
                checked={enableCoriolis}
                onChange={(e) => setEnableCoriolis(e.target.checked)}
                hint="Earth rotation effects (minimal)"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Simulation Quality */}
      <Card>
        <CardHeader>
          <CardTitle>Simulation Quality</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Slider
              label="Physics Iterations"
              value={physicsIterations}
              onChange={(e) => setPhysicsIterations(Number(e.target.value))}
              min={1}
              max={20}
              step={1}
              error={errors.physicsIterations}
              formatValue={(value) => `${value} iterations`}
              hint="More iterations = higher accuracy, lower performance"
              fullWidth
            />

            <Slider
              label="Collision Iterations"
              value={collisionIterations}
              onChange={(e) => setCollisionIterations(Number(e.target.value))}
              min={1}
              max={10}
              step={1}
              error={errors.collisionIterations}
              formatValue={(value) => `${value} iterations`}
              hint="Collision resolution accuracy"
              fullWidth
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Checkbox
                label="Enable Substepping"
                checked={enableSubstepping}
                onChange={(e) => setEnableSubstepping(e.target.checked)}
                hint="Break large time steps into smaller ones"
              />

              <Checkbox
                label="Adaptive Time Step"
                checked={adaptiveTimeStep}
                onChange={(e) => setAdaptiveTimeStep(e.target.checked)}
                hint="Automatically adjust time step based on complexity"
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
