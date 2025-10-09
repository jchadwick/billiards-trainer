import { TanstackDevtools } from '@tanstack/react-devtools'
import { Outlet, createRootRoute } from '@tanstack/react-router'
import { TanStackRouterDevtoolsPanel } from '@tanstack/react-router-devtools'
import { observer } from 'mobx-react-lite'
import { ErrorBoundary } from '../components/common'
import { AppLayout } from '../components/layout'
import { StoreProvider, rootStore } from '../hooks/useStores'

const RootComponent = observer(() => {
  const systemInfo = {
    version: '1.0.0',
    buildDate: '2024-01-01',
    environment: process.env.NODE_ENV || 'development',
    apiVersion: '1.0.0',
  }

  return (
    <StoreProvider value={rootStore}>
      <ErrorBoundary>
        <AppLayout systemInfo={systemInfo}>
          <Outlet />
        </AppLayout>
        {process.env.NODE_ENV === 'development' && (
          <TanstackDevtools
            config={{
              position: 'bottom-left',
            }}
            plugins={[
              {
                name: 'Tanstack Router',
                render: <TanStackRouterDevtoolsPanel />,
              },
            ]}
          />
        )}
      </ErrorBoundary>
    </StoreProvider>
  )
})

export const Route = createRootRoute({
  component: RootComponent,
})
