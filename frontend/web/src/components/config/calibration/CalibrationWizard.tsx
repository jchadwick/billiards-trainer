import React, { useState } from 'react'
import { observer } from 'mobx-react-lite'
import { Card, CardHeader, CardTitle, CardContent, Button } from '../../ui'

interface CalibrationStep {
  id: string
  title: string
  description: string
  component: React.ComponentType<CalibrationStepProps>
}

interface CalibrationStepProps {
  onNext: () => void
  onPrevious: () => void
  onComplete: () => void
  isFirstStep: boolean
  isLastStep: boolean
}

const CameraCalibrationStep: React.FC<CalibrationStepProps> = ({
  onNext,
  onPrevious,
  isFirstStep,
  isLastStep
}) => (
  <div className="space-y-6">
    <div className="text-center">
      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
        Camera Calibration
      </h3>
      <p className="text-gray-600 dark:text-gray-400">
        Position the camera to get a clear view of the entire table.
      </p>
    </div>

    <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6 text-center">
      <div className="w-16 h-16 mx-auto mb-4 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
        <svg className="w-8 h-8 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
      </div>
      <p className="text-gray-600 dark:text-gray-400">
        Camera feed would be displayed here during actual calibration
      </p>
    </div>

    <div className="space-y-4">
      <h4 className="font-medium text-gray-900 dark:text-white">Instructions:</h4>
      <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
        <li>• Ensure the camera has a clear, unobstructed view of the entire table</li>
        <li>• Position the camera at a stable height above the table</li>
        <li>• Adjust lighting to minimize shadows and glare</li>
        <li>• Check that all table corners and pockets are visible</li>
      </ul>
    </div>

    <div className="flex justify-between">
      <Button
        variant="outline"
        onClick={onPrevious}
        disabled={isFirstStep}
      >
        Previous
      </Button>
      <Button onClick={onNext}>
        Next
      </Button>
    </div>
  </div>
)

const TableCalibrationStep: React.FC<CalibrationStepProps> = ({
  onNext,
  onPrevious,
  isLastStep
}) => (
  <div className="space-y-6">
    <div className="text-center">
      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
        Table Boundaries
      </h3>
      <p className="text-gray-600 dark:text-gray-400">
        Mark the table boundaries and pocket positions.
      </p>
    </div>

    <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
      <div className="border-4 border-amber-800 rounded-lg relative bg-green-600 h-48 mx-auto max-w-md">
        {/* Table representation with clickable areas */}
        <div className="absolute top-0 left-0 w-4 h-4 bg-black rounded-full transform -translate-x-2 -translate-y-2 cursor-pointer"></div>
        <div className="absolute top-0 right-0 w-4 h-4 bg-black rounded-full transform translate-x-2 -translate-y-2 cursor-pointer"></div>
        <div className="absolute bottom-0 left-0 w-4 h-4 bg-black rounded-full transform -translate-x-2 translate-y-2 cursor-pointer"></div>
        <div className="absolute bottom-0 right-0 w-4 h-4 bg-black rounded-full transform translate-x-2 translate-y-2 cursor-pointer"></div>
        <div className="absolute top-0 left-1/2 w-4 h-4 bg-black rounded-full transform -translate-x-2 -translate-y-2 cursor-pointer"></div>
        <div className="absolute bottom-0 left-1/2 w-4 h-4 bg-black rounded-full transform -translate-x-2 translate-y-2 cursor-pointer"></div>
      </div>
    </div>

    <div className="space-y-4">
      <h4 className="font-medium text-gray-900 dark:text-white">Instructions:</h4>
      <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
        <li>• Click on each corner of the table to mark the boundaries</li>
        <li>• Mark the center of each pocket by clicking on them</li>
        <li>• Ensure all markings are accurate for proper tracking</li>
        <li>• Double-check that the table outline matches the actual table</li>
      </ul>
    </div>

    <div className="flex justify-between">
      <Button
        variant="outline"
        onClick={onPrevious}
      >
        Previous
      </Button>
      <Button onClick={onNext}>
        Next
      </Button>
    </div>
  </div>
)

const ProjectorCalibrationStep: React.FC<CalibrationStepProps> = ({
  onNext,
  onPrevious,
  onComplete,
  isLastStep
}) => (
  <div className="space-y-6">
    <div className="text-center">
      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
        Projector Calibration
      </h3>
      <p className="text-gray-600 dark:text-gray-400">
        Align the projector overlay with the physical table.
      </p>
    </div>

    <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6 text-center">
      <div className="w-16 h-16 mx-auto mb-4 bg-purple-100 dark:bg-purple-900 rounded-full flex items-center justify-center">
        <svg className="w-8 h-8 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m3 0H4a1 1 0 00-1 1v10a1 1 0 001 1h16a1 1 0 001-1V5a1 1 0 00-1-1z" />
        </svg>
      </div>
      <p className="text-gray-600 dark:text-gray-400">
        Projector calibration interface would be displayed here
      </p>
    </div>

    <div className="space-y-4">
      <h4 className="font-medium text-gray-900 dark:text-white">Instructions:</h4>
      <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
        <li>• Adjust keystone correction to align projected image</li>
        <li>• Calibrate corner points to match table boundaries</li>
        <li>• Fine-tune brightness and contrast for visibility</li>
        <li>• Test overlay accuracy with sample projections</li>
      </ul>
    </div>

    <div className="grid grid-cols-2 gap-4 mb-6">
      <Button variant="outline" size="sm">
        Test Projection
      </Button>
      <Button variant="outline" size="sm">
        Reset Keystone
      </Button>
    </div>

    <div className="flex justify-between">
      <Button
        variant="outline"
        onClick={onPrevious}
      >
        Previous
      </Button>
      <Button
        onClick={isLastStep ? onComplete : onNext}
        variant={isLastStep ? 'primary' : 'outline'}
      >
        {isLastStep ? 'Complete Calibration' : 'Next'}
      </Button>
    </div>
  </div>
)

export const CalibrationWizard = observer(() => {
  const [currentStep, setCurrentStep] = useState(0)
  const [loading, setLoading] = useState(false)

  const steps: CalibrationStep[] = [
    {
      id: 'camera',
      title: 'Camera Setup',
      description: 'Position and configure the camera',
      component: CameraCalibrationStep
    },
    {
      id: 'table',
      title: 'Table Boundaries',
      description: 'Define table boundaries and pockets',
      component: TableCalibrationStep
    },
    {
      id: 'projector',
      title: 'Projector Alignment',
      description: 'Calibrate projector overlay',
      component: ProjectorCalibrationStep
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

  const handleComplete = async () => {
    setLoading(true)
    try {
      // Save calibration data
      console.log('Completing calibration...')
      await new Promise(resolve => setTimeout(resolve, 2000))

      // Reset wizard
      setCurrentStep(0)
    } catch (error) {
      console.error('Calibration failed:', error)
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
              System Calibration Wizard
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
            />
          )}
        </CardContent>
      </Card>
    </div>
  )
})
