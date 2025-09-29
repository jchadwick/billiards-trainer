import React, { useState } from 'react'
import { observer } from 'mobx-react-lite'
import { Card, CardHeader, CardTitle, CardContent, Input, Select, Button, Slider } from '../ui'
import type { SelectOption } from '../ui'

interface TableDimensions {
  length: number
  width: number
  height: number
}

interface PocketPosition {
  x: number
  y: number
  radius: number
}

export const TableConfig = observer(() => {
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})

  // Table dimensions
  const [tableType, setTableType] = useState('full')
  const [dimensions, setDimensions] = useState<TableDimensions>({
    length: 254, // cm (9-foot table)
    width: 127,
    height: 80
  })

  // Surface properties
  const [clothColor, setClothColor] = useState('#1e7f47') // Traditional green
  const [clothTexture, setClothTexture] = useState('worsted')
  const [railHeight, setRailHeight] = useState(6.35)
  const [cushionRebound, setCushionRebound] = useState(0.75)

  // Pocket configuration
  const [pocketSize, setPocketSize] = useState(11.5) // cm
  const [cornerPocketSize, setCornerPocketSize] = useState(11.5)
  const [sidePocketSize, setSidePocketSize] = useState(12.7)
  const [pocketDepth, setPocketDepth] = useState(4.5)

  // Lighting and environment
  const [ambientLight, setAmbientLight] = useState(0.3)
  const [directLight, setDirectLight] = useState(0.8)
  const [shadowStrength, setShadowStrength] = useState(0.5)
  const [reflectivity, setReflectivity] = useState(0.2)

  const tableTypeOptions: SelectOption[] = [
    { value: 'full', label: '9-foot Table (254 × 127 cm)' },
    { value: 'tournament', label: '8-foot Tournament (244 × 122 cm)' },
    { value: 'home', label: '7-foot Home (213 × 107 cm)' },
    { value: 'bar', label: '6-foot Bar Table (183 × 91 cm)' },
    { value: 'custom', label: 'Custom Dimensions' }
  ]

  const clothTextureOptions: SelectOption[] = [
    { value: 'worsted', label: 'Worsted Wool (Fast)' },
    { value: 'woolen', label: 'Woolen (Medium)' },
    { value: 'napped', label: 'Napped (Slow)' },
    { value: 'synthetic', label: 'Synthetic' }
  ]

  const handleTableTypeChange = (type: string) => {
    setTableType(type)

    // Update dimensions based on table type
    switch (type) {
      case 'full':
        setDimensions({ length: 254, width: 127, height: 80 })
        break
      case 'tournament':
        setDimensions({ length: 244, width: 122, height: 80 })
        break
      case 'home':
        setDimensions({ length: 213, width: 107, height: 80 })
        break
      case 'bar':
        setDimensions({ length: 183, width: 91, height: 70 })
        break
      default:
        // Keep current dimensions for custom
        break
    }
  }

  const calculatePocketPositions = (): PocketPosition[] => {
    const { length, width } = dimensions
    const cornerRadius = cornerPocketSize / 2
    const sideRadius = sidePocketSize / 2

    return [
      // Corner pockets
      { x: 0, y: 0, radius: cornerRadius }, // Top-left
      { x: length, y: 0, radius: cornerRadius }, // Top-right
      { x: 0, y: width, radius: cornerRadius }, // Bottom-left
      { x: length, y: width, radius: cornerRadius }, // Bottom-right

      // Side pockets
      { x: length / 2, y: 0, radius: sideRadius }, // Top-center
      { x: length / 2, y: width, radius: sideRadius } // Bottom-center
    ]
  }

  const handleSave = async () => {
    setLoading(true)
    setErrors({})

    try {
      // Validate inputs
      const newErrors: Record<string, string> = {}

      if (dimensions.length < 150 || dimensions.length > 300) {
        newErrors.length = 'Table length must be between 150cm and 300cm'
      }

      if (dimensions.width < 75 || dimensions.width > 150) {
        newErrors.width = 'Table width must be between 75cm and 150cm'
      }

      if (dimensions.height < 60 || dimensions.height > 100) {
        newErrors.height = 'Table height must be between 60cm and 100cm'
      }

      if (railHeight < 3 || railHeight > 10) {
        newErrors.railHeight = 'Rail height must be between 3cm and 10cm'
      }

      if (pocketSize < 8 || pocketSize > 15) {
        newErrors.pocketSize = 'Pocket size must be between 8cm and 15cm'
      }

      if (pocketDepth < 2 || pocketDepth > 8) {
        newErrors.pocketDepth = 'Pocket depth must be between 2cm and 8cm'
      }

      if (Object.keys(newErrors).length > 0) {
        setErrors(newErrors)
        return
      }

      // Calculate pocket positions
      const pocketPositions = calculatePocketPositions()

      const tableConfig = {
        type: tableType,
        dimensions,
        surface: {
          clothColor,
          clothTexture,
          railHeight,
          cushionRebound
        },
        pockets: {
          cornerSize: cornerPocketSize,
          sideSize: sidePocketSize,
          depth: pocketDepth,
          positions: pocketPositions
        },
        lighting: {
          ambient: ambientLight,
          direct: directLight,
          shadowStrength,
          reflectivity
        }
      }

      console.log('Saving table configuration:', tableConfig)

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))

    } catch (error) {
      console.error('Failed to save table settings:', error)
      setErrors({ general: 'Failed to save table settings. Please try again.' })
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setTableType('full')
    setDimensions({ length: 254, width: 127, height: 80 })
    setClothColor('#1e7f47')
    setClothTexture('worsted')
    setRailHeight(6.35)
    setCushionRebound(0.75)
    setPocketSize(11.5)
    setCornerPocketSize(11.5)
    setSidePocketSize(12.7)
    setPocketDepth(4.5)
    setAmbientLight(0.3)
    setDirectLight(0.8)
    setShadowStrength(0.5)
    setReflectivity(0.2)
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

      {/* Table Type and Dimensions */}
      <Card>
        <CardHeader>
          <CardTitle>Table Dimensions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Select
              label="Table Type"
              value={tableType}
              onChange={(e) => handleTableTypeChange(e.target.value)}
              options={tableTypeOptions}
              fullWidth
            />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Input
                label="Length (cm)"
                type="number"
                value={dimensions.length}
                onChange={(e) => setDimensions({ ...dimensions, length: Number(e.target.value) })}
                error={errors.length}
                disabled={tableType !== 'custom'}
                min={150}
                max={300}
                fullWidth
              />

              <Input
                label="Width (cm)"
                type="number"
                value={dimensions.width}
                onChange={(e) => setDimensions({ ...dimensions, width: Number(e.target.value) })}
                error={errors.width}
                disabled={tableType !== 'custom'}
                min={75}
                max={150}
                fullWidth
              />

              <Input
                label="Height (cm)"
                type="number"
                value={dimensions.height}
                onChange={(e) => setDimensions({ ...dimensions, height: Number(e.target.value) })}
                error={errors.height}
                disabled={tableType !== 'custom'}
                min={60}
                max={100}
                fullWidth
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Surface Properties */}
      <Card>
        <CardHeader>
          <CardTitle>Surface Properties</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Cloth Color
                </label>
                <div className="flex items-center space-x-3">
                  <input
                    type="color"
                    value={clothColor}
                    onChange={(e) => setClothColor(e.target.value)}
                    className="w-12 h-8 border border-gray-300 rounded cursor-pointer"
                  />
                  <Input
                    value={clothColor}
                    onChange={(e) => setClothColor(e.target.value)}
                    placeholder="#1e7f47"
                    className="flex-1"
                  />
                </div>
              </div>

              <Select
                label="Cloth Texture"
                value={clothTexture}
                onChange={(e) => setClothTexture(e.target.value)}
                options={clothTextureOptions}
                fullWidth
              />
            </div>

            <Slider
              label="Rail Height (cm)"
              value={railHeight}
              onChange={(e) => setRailHeight(Number(e.target.value))}
              min={3}
              max={10}
              step={0.1}
              error={errors.railHeight}
              formatValue={(value) => `${value.toFixed(1)} cm`}
              fullWidth
            />

            <Slider
              label="Cushion Rebound Coefficient"
              value={cushionRebound}
              onChange={(e) => setCushionRebound(Number(e.target.value))}
              min={0.5}
              max={1.0}
              step={0.01}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="Higher values create more energetic bounces"
              fullWidth
            />
          </div>
        </CardContent>
      </Card>

      {/* Pocket Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Pocket Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Input
                label="Corner Pocket Size (cm)"
                type="number"
                value={cornerPocketSize}
                onChange={(e) => setCornerPocketSize(Number(e.target.value))}
                min={8}
                max={15}
                step={0.1}
                fullWidth
              />

              <Input
                label="Side Pocket Size (cm)"
                type="number"
                value={sidePocketSize}
                onChange={(e) => setSidePocketSize(Number(e.target.value))}
                min={8}
                max={15}
                step={0.1}
                fullWidth
              />
            </div>

            <Input
              label="Pocket Depth (cm)"
              type="number"
              value={pocketDepth}
              onChange={(e) => setPocketDepth(Number(e.target.value))}
              error={errors.pocketDepth}
              min={2}
              max={8}
              step={0.1}
              hint="Depth from table surface to bottom of pocket"
              fullWidth
            />
          </div>
        </CardContent>
      </Card>

      {/* Lighting and Environment */}
      <Card>
        <CardHeader>
          <CardTitle>Lighting and Environment</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Slider
              label="Ambient Light Level"
              value={ambientLight}
              onChange={(e) => setAmbientLight(Number(e.target.value))}
              min={0}
              max={1}
              step={0.01}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="General room lighting level"
              fullWidth
            />

            <Slider
              label="Direct Light Level"
              value={directLight}
              onChange={(e) => setDirectLight(Number(e.target.value))}
              min={0}
              max={1}
              step={0.01}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="Table lighting intensity"
              fullWidth
            />

            <Slider
              label="Shadow Strength"
              value={shadowStrength}
              onChange={(e) => setShadowStrength(Number(e.target.value))}
              min={0}
              max={1}
              step={0.01}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="How pronounced shadows appear"
              fullWidth
            />

            <Slider
              label="Surface Reflectivity"
              value={reflectivity}
              onChange={(e) => setReflectivity(Number(e.target.value))}
              min={0}
              max={1}
              step={0.01}
              formatValue={(value) => `${(value * 100).toFixed(0)}%`}
              hint="How much light the table surface reflects"
              fullWidth
            />
          </div>
        </CardContent>
      </Card>

      {/* Table Preview */}
      <Card>
        <CardHeader>
          <CardTitle>Table Preview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6 flex items-center justify-center">
            <div
              className="border-4 border-amber-800 rounded-lg relative"
              style={{
                width: '300px',
                height: `${(dimensions.width / dimensions.length) * 300}px`,
                backgroundColor: clothColor,
                minHeight: '150px'
              }}
            >
              {/* Pocket indicators */}
              {calculatePocketPositions().map((pocket, index) => (
                <div
                  key={index}
                  className="absolute bg-black rounded-full border border-gray-600"
                  style={{
                    left: `${(pocket.x / dimensions.length) * 100}%`,
                    top: `${(pocket.y / dimensions.width) * 100}%`,
                    width: `${(pocket.radius * 2 / dimensions.length) * 300}px`,
                    height: `${(pocket.radius * 2 / dimensions.length) * 300}px`,
                    transform: 'translate(-50%, -50%)'
                  }}
                />
              ))}

              {/* Table center line */}
              <div
                className="absolute border-l border-white opacity-50"
                style={{
                  left: '50%',
                  top: '0',
                  height: '100%'
                }}
              />
            </div>
          </div>
          <p className="text-sm text-gray-500 text-center mt-2">
            {dimensions.length} × {dimensions.width} cm table preview
          </p>
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
