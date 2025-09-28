import { createFileRoute } from '@tanstack/react-router'
import { observer } from 'mobx-react-lite'
import { PageContainer, Section } from '../components/layout'
import { Card, CardContent, CardHeader, CardTitle, Button } from '../components/ui'
import { useUIStore, useSystemStore } from '../hooks/useStores'

export const Route = createFileRoute('/')({
  component: Dashboard,
})

const Dashboard = observer(() => {
  const uiStore = useUIStore()
  const systemStore = useSystemStore()

  const handleTestNotification = () => {
    uiStore.showSuccess(
      'Test Notification',
      'This is a test notification to demonstrate the notification system.'
    )
  }

  const handleTestError = () => {
    uiStore.showError(
      'Error Notification',
      'This is an error notification example.'
    )
  }

  return (
    <PageContainer
      title="Dashboard"
      description="Welcome to the Billiards Trainer application"
      actions={
        <div className="space-x-2">
          <Button variant="secondary" onClick={handleTestNotification}>
            Test Notification
          </Button>
          <Button variant="primary">
            Start Training
          </Button>
        </div>
      }
    >
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* System Status Card */}
        <Card hover>
          <CardHeader>
            <CardTitle>System Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-secondary-600 dark:text-secondary-400">
                  Connection
                </span>
                <span className={`text-sm font-medium ${systemStore.status.isConnected ? 'text-success-600' : 'text-error-600'}`}>
                  {systemStore.status.isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-secondary-600 dark:text-secondary-400">
                  Theme
                </span>
                <span className="text-sm font-medium text-secondary-900 dark:text-secondary-100 capitalize">
                  Light
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-secondary-600 dark:text-secondary-400">
                  Notifications
                </span>
                <span className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                  {uiStore.uiState.notifications.length} total
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Quick Actions Card */}
        <Card hover>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <Button variant="ghost" className="w-full justify-start">
                <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>
                Start Calibration
              </Button>
              <Button variant="ghost" className="w-full justify-start">
                <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Open Settings
              </Button>
              <Button variant="ghost" className="w-full justify-start">
                <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                View Diagnostics
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Recent Activity Card */}
        <Card hover>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="text-sm text-secondary-600 dark:text-secondary-400">
                No recent activity to display.
              </div>
              <Button variant="outline" size="sm" className="w-full">
                View All Activity
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Testing Section */}
      <Section
        title="Testing & Development"
        description="Tools for testing the new layout system"
        className="mt-8"
      >
        <Card>
          <CardContent>
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                <Button onClick={handleTestNotification}>
                  Test Success Notification
                </Button>
                <Button onClick={handleTestError} variant="danger">
                  Test Error Notification
                </Button>
                <Button
                  onClick={() => {
                    uiStore.setGlobalLoading(true)
                    setTimeout(() => uiStore.setGlobalLoading(false), 3000)
                  }}
                  variant="secondary"
                >
                  Test Loading (3s)
                </Button>
              </div>
              {uiStore.isGlobalLoading && (
                <div className="mt-4">
                  <Button
                    onClick={() => uiStore.setGlobalLoading(false)}
                    variant="ghost"
                  >
                    Stop Loading
                  </Button>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </Section>
    </PageContainer>
  )
})
