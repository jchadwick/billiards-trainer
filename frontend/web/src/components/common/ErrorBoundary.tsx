import React, { Component, ReactNode } from 'react'
import { Button } from '../ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card'

export interface ErrorBoundaryState {
  hasError: boolean
  error?: Error
  errorInfo?: React.ErrorInfo
}

export interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      error,
    }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.setState({
      error,
      errorInfo,
    })

    // Call the optional error handler
    this.props.onError?.(error, errorInfo)

    // Log error to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('ErrorBoundary caught an error:', error, errorInfo)
    }
  }

  handleReset = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined })
  }

  handleReload = () => {
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback
      }

      // Default error UI
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 p-4">
          <Card className="max-w-lg w-full">
            <CardHeader>
              <CardTitle className="text-error-600 dark:text-error-400">
                Something went wrong
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-secondary-700 dark:text-secondary-300">
                  An unexpected error has occurred. This has been logged and we're working to fix it.
                </p>

                {process.env.NODE_ENV === 'development' && this.state.error && (
                  <details className="mt-4">
                    <summary className="cursor-pointer text-sm font-medium text-secondary-600 dark:text-secondary-400 hover:text-secondary-800 dark:hover:text-secondary-200">
                      Error Details (Development)
                    </summary>
                    <div className="mt-2 p-3 bg-secondary-100 dark:bg-secondary-800 rounded-md">
                      <pre className="text-xs text-secondary-800 dark:text-secondary-200 whitespace-pre-wrap overflow-auto max-h-40">
                        {this.state.error.toString()}
                        {this.state.errorInfo?.componentStack}
                      </pre>
                    </div>
                  </details>
                )}

                <div className="flex space-x-3">
                  <Button onClick={this.handleReset} variant="primary">
                    Try Again
                  </Button>
                  <Button onClick={this.handleReload} variant="secondary">
                    Reload Page
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )
    }

    return this.props.children
  }
}

// Hook-based error boundary component for functional components
export interface ErrorFallbackProps {
  error: Error
  resetError: () => void
}

export const ErrorFallback: React.FC<ErrorFallbackProps> = ({ error, resetError }) => (
  <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 p-4">
    <Card className="max-w-lg w-full">
      <CardHeader>
        <CardTitle className="text-error-600 dark:text-error-400">
          Something went wrong
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <p className="text-secondary-700 dark:text-secondary-300">
            An unexpected error has occurred. Please try again.
          </p>

          {process.env.NODE_ENV === 'development' && (
            <details className="mt-4">
              <summary className="cursor-pointer text-sm font-medium text-secondary-600 dark:text-secondary-400 hover:text-secondary-800 dark:hover:text-secondary-200">
                Error Details (Development)
              </summary>
              <div className="mt-2 p-3 bg-secondary-100 dark:bg-secondary-800 rounded-md">
                <pre className="text-xs text-secondary-800 dark:text-secondary-200 whitespace-pre-wrap overflow-auto max-h-40">
                  {error.toString()}
                </pre>
              </div>
            </details>
          )}

          <div className="flex space-x-3">
            <Button onClick={resetError} variant="primary">
              Try Again
            </Button>
            <Button onClick={() => window.location.reload()} variant="secondary">
              Reload Page
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  </div>
)
