import { createFileRoute } from '@tanstack/react-router'
import { PageContainer } from '../components/layout'
import { CalibrationWizard } from '../components/config/calibration/CalibrationWizard'

export const Route = createFileRoute('/calibration')({
  component: CalibrationPage,
})

function CalibrationPage() {
  return (
    <PageContainer
      title="System Calibration"
      description="Complete interactive calibration wizard for camera, table detection, and projector alignment"
    >
      <CalibrationWizard />
    </PageContainer>
  )
}
