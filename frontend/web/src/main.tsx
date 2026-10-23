import { RouterProvider, createRouter } from '@tanstack/react-router';
import React, { StrictMode } from 'react';
import ReactDOM from 'react-dom/client';
import { routeTree } from './routeTree.gen';
import { StoreProvider } from './stores/context';
import { getRootStore } from './stores/RootStore';
import './styles.css';

const rootStore = getRootStore();
const router = createRouter({
  routeTree,
  defaultPreload: 'intent',
  scrollRestoration: true,
  defaultStructuralSharing: true,
  defaultPreloadStaleTime: 0,
});

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}

/**
 * Keep a small failure boundary around the live surface so an unexpected
 * rendering error does not leave the browser on a blank page.
 */
class LiveAppErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error): void {
    console.error('Live view failed to render:', error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <main className="flex min-h-screen items-center justify-center bg-gray-950 p-6 text-center text-gray-100">
          <div>
            <h1 className="text-xl font-semibold">Live view unavailable</h1>
            <p className="mt-2 text-sm text-gray-300">
              Refresh the page to reconnect to the live table.
            </p>
            <button
              type="button"
              className="mt-4 rounded bg-white px-4 py-2 text-gray-950"
              onClick={() => window.location.reload()}
            >
              Refresh
            </button>
          </div>
        </main>
      );
    }

    return this.props.children;
  }
}

const rootElement = document.getElementById('app');
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <StrictMode>
      <StoreProvider store={rootStore}>
        <LiveAppErrorBoundary>
          <RouterProvider router={router} />
        </LiveAppErrorBoundary>
      </StoreProvider>
    </StrictMode>,
  );
}

window.addEventListener('beforeunload', () => {
  void rootStore.shutdown();
});
