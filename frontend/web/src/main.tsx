import React, { StrictMode } from 'react'
import ReactDOM from 'react-dom/client'
import { RouterProvider, createRouter } from '@tanstack/react-router'

// Import the generated route tree
import { routeTree } from './routeTree.gen'

// Import MobX store context and utilities
import { StoreProvider, rootStore } from './stores'
import { PersistenceManager } from './stores/persistence'
import { initializeDevTools } from './stores/dev-tools'

import './styles.css'
import reportWebVitals from './reportWebVitals.ts'

// Initialize persistence system
const persistenceManager = new PersistenceManager(rootStore);
persistenceManager.initialize();

// Initialize development tools
const devTools = initializeDevTools(rootStore);

// Create a new router instance with store context
const router = createRouter({
  routeTree,
  context: {
    stores: rootStore,
    auth: rootStore.auth,
  },
  defaultPreload: 'intent',
  scrollRestoration: true,
  defaultStructuralSharing: true,
  defaultPreloadStaleTime: 0,
})

// Register the router instance for type safety
declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

// Error boundary for the entire app
class AppErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Application error caught by boundary:', error, errorInfo);

    // Log error to system store
    rootStore.system.addCritical(
      'React',
      `Unhandled error: ${error.message}`,
      { stack: error.stack, errorInfo }
    );

    // Show error notification
    rootStore.ui.showError(
      'Application Error',
      'An unexpected error occurred. Please refresh the page.',
      { autoHide: false }
    );
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-100">
          <div className="text-center p-8">
            <h1 className="text-2xl font-bold text-red-600 mb-4">
              Something went wrong
            </h1>
            <p className="text-gray-600 mb-4">
              An unexpected error occurred. Please refresh the page to continue.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Refresh Page
            </button>
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="mt-4 text-left">
                <summary className="cursor-pointer text-red-500">
                  Error Details (Development)
                </summary>
                <pre className="mt-2 p-4 bg-gray-800 text-white text-sm overflow-auto">
                  {this.state.error.stack}
                </pre>
              </details>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// App component with store provider
function App() {
  return (
    <StoreProvider store={rootStore}>
      <AppErrorBoundary>
        <RouterProvider router={router} />
      </AppErrorBoundary>
    </StoreProvider>
  );
}

// Cleanup function for graceful shutdown
function setupCleanup() {
  const cleanup = async () => {
    try {
      console.log('Cleaning up application...');

      // Save current state
      await persistenceManager.forceSave();

      // Cleanup stores
      await rootStore.shutdown();

      // Cleanup persistence manager
      persistenceManager.destroy();

      // Cleanup dev tools
      if (devTools) {
        devTools.destroy();
      }

      console.log('Cleanup complete');
    } catch (error) {
      console.error('Error during cleanup:', error);
    }
  };

  // Handle page unload
  window.addEventListener('beforeunload', cleanup);

  // Handle React unmounting (for development)
  return cleanup;
}

// Initialize cleanup
const cleanup = setupCleanup();

// Render the app
const rootElement = document.getElementById('app')
if (rootElement && !rootElement.innerHTML) {
  const root = ReactDOM.createRoot(rootElement)
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  )

  // Store cleanup function for development hot reloading
  if (process.env.NODE_ENV === 'development') {
    (window as any).__APP_CLEANUP__ = cleanup;
  }
}

// Performance monitoring
reportWebVitals((metric) => {
  // Log performance metrics to system store
  if (metric.name === 'CLS' && metric.value > 0.1) {
    rootStore.system.addWarning(
      'Performance',
      `High Cumulative Layout Shift: ${metric.value.toFixed(3)}`
    );
  } else if (metric.name === 'FCP' && metric.value > 3000) {
    rootStore.system.addWarning(
      'Performance',
      `Slow First Contentful Paint: ${metric.value.toFixed(0)}ms`
    );
  } else if (metric.name === 'LCP' && metric.value > 4000) {
    rootStore.system.addWarning(
      'Performance',
      `Slow Largest Contentful Paint: ${metric.value.toFixed(0)}ms`
    );
  }

  // Log all metrics in development
  if (process.env.NODE_ENV === 'development') {
    console.log('Performance metric:', metric);
  }
});

// Global error handling
window.addEventListener('error', (event) => {
  rootStore.system.addError(
    'JavaScript',
    `Unhandled error: ${event.error?.message || event.message}`,
    {
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      stack: event.error?.stack
    }
  );
});

window.addEventListener('unhandledrejection', (event) => {
  rootStore.system.addError(
    'Promise',
    `Unhandled promise rejection: ${event.reason}`,
    { reason: event.reason }
  );
});

// Log successful initialization
console.log('🎱 Billiards Trainer initialized successfully');
if (process.env.NODE_ENV === 'development') {
  console.log('🔧 Development mode - DevTools available');
  console.log('📊 Stores available at window.__MOBX_STORES__');
  console.log('🐛 Debug tools available at window.__MOBX_DEBUG__');
}
