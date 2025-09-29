import { createFileRoute } from '@tanstack/react-router'
import { observer } from 'mobx-react-lite'
import { PageContainer, Section } from '../components/layout'
import { SystemManagement } from '../components/system-management/SystemManagement'

export const Route = createFileRoute('/system-management')({
  component: SystemManagementPage,
})

const SystemManagementPage = observer(() => {
  return (
    <PageContainer
      title="System Management"
      description="Monitor and control backend modules and system health"
      actions={
        <div className="space-x-2">
          {/* Action buttons will be added here if needed */}
        </div>
      }
    >
      <SystemManagement />
    </PageContainer>
  )
})