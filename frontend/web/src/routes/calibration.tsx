import { createFileRoute } from '@tanstack/react-router'
import { PageContainer } from '../components/layout'
import { Card, CardContent, CardHeader, CardTitle, Button } from '../components/ui'

export const Route = createFileRoute('/calibration')({
  component: CalibrationPage,
})

function CalibrationPage() {
  return (
    <PageContainer
      title="Calibration"
      description="Calibrate your billiards table detection and tracking system"
      actions={
        <Button variant="primary">
          Start Calibration
        </Button>
      }
    >
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Camera Calibration</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-secondary-600 dark:text-secondary-400 mb-4">
              Begin by calibrating the camera position and angle for optimal table detection.
            </p>
            <Button variant="secondary">Configure Camera</Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Table Detection</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-secondary-600 dark:text-secondary-400 mb-4">
              Set up the table boundaries and pocket detection for accurate ball tracking.
            </p>
            <Button variant="secondary">Configure Table</Button>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
