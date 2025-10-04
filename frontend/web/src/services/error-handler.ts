/**
 * Comprehensive error handling and connection monitoring service
 */

import { ApiError, isApiError } from './api-client';
import { ConnectionState } from './websocket-client';

export interface ErrorContext {
  component?: string;
  action?: string;
  userId?: string;
  timestamp: Date;
  url?: string;
  metadata?: Record<string, any>;
}

export interface ErrorReport {
  id: string;
  type: ErrorType;
  severity: ErrorSeverity;
  message: string;
  code?: string;
  context: ErrorContext;
  stack?: string;
  userAgent?: string;
  resolved: boolean;
  resolvedAt?: Date;
  occurrenceCount: number;
  firstOccurrence: Date;
  lastOccurrence: Date;
}

export type ErrorType =
  | 'api'
  | 'auth'
  | 'websocket'
  | 'validation'
  | 'network'
  | 'runtime'
  | 'permission'
  | 'timeout'
  | 'unknown';

export type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical';

export interface ConnectionHealth {
  status: 'excellent' | 'good' | 'poor' | 'disconnected';
  apiLatency: number;
  wsLatency: number;
  apiErrors: number;
  wsErrors: number;
  lastApiCall: Date | null;
  lastWsMessage: Date | null;
  uptime: number;
  qualityScore: number;
}

export interface RetryConfig {
  maxAttempts: number;
  initialDelay: number;
  maxDelay: number;
  backoffFactor: number;
  retryCondition?: (error: any) => boolean;
}

export type ErrorHandler = (error: ErrorReport) => void;
export type ConnectionHealthHandler = (health: ConnectionHealth) => void;

export class ErrorHandlingService {
  private errorReports = new Map<string, ErrorReport>();
  private errorHandlers = new Set<ErrorHandler>();
  private healthHandlers = new Set<ConnectionHealthHandler>();

  // Connection monitoring
  private connectionHealth: ConnectionHealth = {
    status: 'disconnected',
    apiLatency: 0,
    wsLatency: 0,
    apiErrors: 0,
    wsErrors: 0,
    lastApiCall: null,
    lastWsMessage: null,
    uptime: 0,
    qualityScore: 1.0,
  };

  // Error tracking
  private errorCounts = new Map<string, number>();
  private recentErrors: ErrorReport[] = [];
  private maxRecentErrors = 50;

  // Retry mechanisms
  private retryQueues = new Map<string, Array<() => Promise<any>>>();
  private defaultRetryConfig: RetryConfig = {
    maxAttempts: 3,
    initialDelay: 1000,
    maxDelay: 30000,
    backoffFactor: 2,
  };

  // Monitoring timers
  private healthCheckTimer: NodeJS.Timeout | null = null;
  private startTime = Date.now();

  constructor() {
    this.setupGlobalErrorHandlers();
    this.startHealthMonitoring();
  }

  // =============================================================================
  // Error Reporting
  // =============================================================================

  reportError(
    error: any,
    context: Partial<ErrorContext> = {},
    severity?: ErrorSeverity
  ): string {
    const errorReport = this.createErrorReport(error, context, severity);

    // Check for duplicate errors
    const existingReport = this.findSimilarError(errorReport);
    if (existingReport) {
      this.updateExistingError(existingReport, errorReport);
      return existingReport.id;
    }

    // Add new error report
    this.errorReports.set(errorReport.id, errorReport);
    this.recentErrors.unshift(errorReport);

    // Limit recent errors
    if (this.recentErrors.length > this.maxRecentErrors) {
      this.recentErrors.pop();
    }

    // Update error counts
    const errorKey = this.getErrorKey(errorReport);
    this.errorCounts.set(errorKey, (this.errorCounts.get(errorKey) || 0) + 1);

    // Notify handlers
    this.notifyErrorHandlers(errorReport);

    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('Error reported:', errorReport);
    }

    return errorReport.id;
  }

  reportApiError(error: ApiError, context: Partial<ErrorContext> = {}): string {
    return this.reportError(error, {
      ...context,
      component: context.component || 'api-client',
    }, this.getApiErrorSeverity(error));
  }

  // Auth functionality removed - authentication not implemented per specs
  // reportAuthError(error: AuthError, context: Partial<ErrorContext> = {}): string {
  //   return this.reportError(error, {
  //     ...context,
  //     component: context.component || 'auth-service',
  //   }, 'high');
  // }

  reportWebSocketError(
    error: any,
    connectionState: ConnectionState,
    context: Partial<ErrorContext> = {}
  ): string {
    return this.reportError(error, {
      ...context,
      component: context.component || 'websocket-client',
      metadata: {
        ...context.metadata,
        connectionState,
      },
    }, this.getWebSocketErrorSeverity(connectionState));
  }

  reportNetworkError(error: any, context: Partial<ErrorContext> = {}): string {
    return this.reportError(error, {
      ...context,
      component: context.component || 'network',
    }, 'medium');
  }

  // =============================================================================
  // Error Resolution
  // =============================================================================

  resolveError(errorId: string): boolean {
    const error = this.errorReports.get(errorId);
    if (error && !error.resolved) {
      error.resolved = true;
      error.resolvedAt = new Date();
      return true;
    }
    return false;
  }

  resolveErrorsByType(type: ErrorType): number {
    let resolved = 0;
    this.errorReports.forEach(error => {
      if (error.type === type && !error.resolved) {
        error.resolved = true;
        error.resolvedAt = new Date();
        resolved++;
      }
    });
    return resolved;
  }

  clearResolvedErrors(): void {
    const unresolvedErrors = new Map<string, ErrorReport>();
    this.errorReports.forEach((error, id) => {
      if (!error.resolved) {
        unresolvedErrors.set(id, error);
      }
    });
    this.errorReports = unresolvedErrors;

    // Also filter recent errors
    this.recentErrors = this.recentErrors.filter(error => !error.resolved);
  }

  // =============================================================================
  // Connection Health Monitoring
  // =============================================================================

  updateApiHealth(latency: number, isError: boolean = false): void {
    this.connectionHealth.lastApiCall = new Date();
    this.connectionHealth.apiLatency = latency;

    if (isError) {
      this.connectionHealth.apiErrors++;
    }

    this.updateConnectionStatus();
  }

  updateWebSocketHealth(latency: number, isError: boolean = false): void {
    this.connectionHealth.lastWsMessage = new Date();
    this.connectionHealth.wsLatency = latency;

    if (isError) {
      this.connectionHealth.wsErrors++;
    }

    this.updateConnectionStatus();
  }

  private updateConnectionStatus(): void {
    const now = Date.now();
    this.connectionHealth.uptime = now - this.startTime;

    // Calculate quality score based on various factors
    let qualityScore = 1.0;

    // Factor in latency
    const avgLatency = (this.connectionHealth.apiLatency + this.connectionHealth.wsLatency) / 2;
    if (avgLatency > 1000) qualityScore -= 0.3;
    else if (avgLatency > 500) qualityScore -= 0.1;

    // Factor in error rates
    const totalRequests = Math.max(1, this.connectionHealth.apiErrors + 100); // Assume 100 successful requests
    const errorRate = this.connectionHealth.apiErrors / totalRequests;
    if (errorRate > 0.1) qualityScore -= 0.4;
    else if (errorRate > 0.05) qualityScore -= 0.2;

    // Factor in connection recency
    const timeSinceLastActivity = Math.min(
      this.connectionHealth.lastApiCall ? now - this.connectionHealth.lastApiCall.getTime() : Infinity,
      this.connectionHealth.lastWsMessage ? now - this.connectionHealth.lastWsMessage.getTime() : Infinity
    );

    if (timeSinceLastActivity > 60000) qualityScore -= 0.3; // 1 minute
    else if (timeSinceLastActivity > 30000) qualityScore -= 0.1; // 30 seconds

    this.connectionHealth.qualityScore = Math.max(0, qualityScore);

    // Determine overall status
    if (qualityScore >= 0.9) this.connectionHealth.status = 'excellent';
    else if (qualityScore >= 0.7) this.connectionHealth.status = 'good';
    else if (qualityScore >= 0.3) this.connectionHealth.status = 'poor';
    else this.connectionHealth.status = 'disconnected';

    // Notify health handlers
    this.notifyHealthHandlers();
  }

  // =============================================================================
  // Retry Mechanisms
  // =============================================================================

  async withRetry<T>(
    operation: () => Promise<T>,
    config: Partial<RetryConfig> = {},
    context: Partial<ErrorContext> = {}
  ): Promise<T> {
    const retryConfig = { ...this.defaultRetryConfig, ...config };
    let lastError: any;

    for (let attempt = 1; attempt <= retryConfig.maxAttempts; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;

        // Check if we should retry this error
        if (retryConfig.retryCondition && !retryConfig.retryCondition(error)) {
          break;
        }

        // Don't retry on last attempt
        if (attempt === retryConfig.maxAttempts) {
          break;
        }

        // Calculate delay for next attempt
        const delay = Math.min(
          retryConfig.initialDelay * Math.pow(retryConfig.backoffFactor, attempt - 1),
          retryConfig.maxDelay
        );

        // Wait before retry
        await this.delay(delay);

        // Report retry attempt
        this.reportError(error, {
          ...context,
          action: `retry_attempt_${attempt}`,
          metadata: {
            ...context.metadata,
            attempt,
            maxAttempts: retryConfig.maxAttempts,
            nextDelay: delay,
          },
        }, 'low');
      }
    }

    // All retries failed, report final error
    this.reportError(lastError, {
      ...context,
      action: 'retry_exhausted',
    });

    throw lastError;
  }

  // =============================================================================
  // Event Handlers
  // =============================================================================

  onError(handler: ErrorHandler): void {
    this.errorHandlers.add(handler);
  }

  offError(handler: ErrorHandler): void {
    this.errorHandlers.delete(handler);
  }

  onConnectionHealth(handler: ConnectionHealthHandler): void {
    this.healthHandlers.add(handler);
  }

  offConnectionHealth(handler: ConnectionHealthHandler): void {
    this.healthHandlers.delete(handler);
  }

  // =============================================================================
  // Error Analysis
  // =============================================================================

  getErrorStats(): {
    total: number;
    unresolved: number;
    byType: Record<ErrorType, number>;
    bySeverity: Record<ErrorSeverity, number>;
    recentCount: number;
  } {
    const total = this.errorReports.size;
    const unresolved = Array.from(this.errorReports.values()).filter(e => !e.resolved).length;

    const byType: Record<ErrorType, number> = {
      api: 0, auth: 0, websocket: 0, validation: 0,
      network: 0, runtime: 0, permission: 0, timeout: 0, unknown: 0
    };

    const bySeverity: Record<ErrorSeverity, number> = {
      low: 0, medium: 0, high: 0, critical: 0
    };

    this.errorReports.forEach(error => {
      byType[error.type]++;
      bySeverity[error.severity]++;
    });

    const recentCount = this.recentErrors.filter(
      error => error.lastOccurrence.getTime() > Date.now() - 3600000 // Last hour
    ).length;

    return { total, unresolved, byType, bySeverity, recentCount };
  }

  getFrequentErrors(limit: number = 10): Array<{
    error: ErrorReport;
    count: number;
  }> {
    const errorCounts = new Map<string, { error: ErrorReport; count: number }>();

    this.errorReports.forEach(error => {
      const key = this.getErrorKey(error);
      const existing = errorCounts.get(key);
      if (existing) {
        existing.count = Math.max(existing.count, error.occurrenceCount);
      } else {
        errorCounts.set(key, { error, count: error.occurrenceCount });
      }
    });

    return Array.from(errorCounts.values())
      .sort((a, b) => b.count - a.count)
      .slice(0, limit);
  }

  getCriticalErrors(): ErrorReport[] {
    return Array.from(this.errorReports.values())
      .filter(error => error.severity === 'critical' && !error.resolved)
      .sort((a, b) => b.lastOccurrence.getTime() - a.lastOccurrence.getTime());
  }

  // =============================================================================
  // Private Helper Methods
  // =============================================================================

  private createErrorReport(
    error: any,
    context: Partial<ErrorContext>,
    severity?: ErrorSeverity
  ): ErrorReport {
    const id = `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const now = new Date();

    return {
      id,
      type: this.determineErrorType(error),
      severity: severity || this.determineSeverity(error),
      message: this.extractErrorMessage(error),
      code: this.extractErrorCode(error),
      context: {
        timestamp: now,
        url: window.location.href,
        ...context,
      },
      stack: error?.stack,
      userAgent: navigator.userAgent,
      resolved: false,
      occurrenceCount: 1,
      firstOccurrence: now,
      lastOccurrence: now,
    };
  }

  private findSimilarError(newError: ErrorReport): ErrorReport | null {
    const errorKey = this.getErrorKey(newError);

    for (const existing of this.errorReports.values()) {
      if (this.getErrorKey(existing) === errorKey) {
        return existing;
      }
    }

    return null;
  }

  private updateExistingError(existing: ErrorReport, newError: ErrorReport): void {
    existing.occurrenceCount++;
    existing.lastOccurrence = newError.lastOccurrence;

    // Update context with latest information
    existing.context = { ...existing.context, ...newError.context };
  }

  private getErrorKey(error: ErrorReport): string {
    return `${error.type}:${error.code || 'unknown'}:${error.message}`;
  }

  private determineErrorType(error: any): ErrorType {
    if (isApiError(error)) return 'api';
    // Auth removed - authentication not implemented per specs
    // if (error instanceof AuthError) return 'auth';
    if (error.name === 'NetworkError') return 'network';
    if (error.name === 'TimeoutError') return 'timeout';
    if (error.name === 'ValidationError') return 'validation';
    if (error.name === 'PermissionError') return 'permission';
    if (error.message?.includes('WebSocket')) return 'websocket';
    return 'runtime';
  }

  private determineSeverity(error: any): ErrorSeverity {
    if (isApiError(error)) {
      return this.getApiErrorSeverity(error);
    }
    // Auth removed - authentication not implemented per specs
    // if (error instanceof AuthError) {
    //   return 'high';
    // }
    if (error.name === 'NetworkError') {
      return 'medium';
    }
    return 'low';
  }

  private getApiErrorSeverity(error: ApiError): ErrorSeverity {
    if (error.status >= 500) return 'high';
    if (error.status === 401 || error.status === 403) return 'high';
    if (error.status === 404) return 'medium';
    if (error.status === 429) return 'medium';
    return 'low';
  }

  private getWebSocketErrorSeverity(connectionState: ConnectionState): ErrorSeverity {
    switch (connectionState) {
      case 'error':
        return 'high';
      case 'disconnected':
        return 'medium';
      case 'reconnecting':
        return 'low';
      default:
        return 'low';
    }
  }

  private extractErrorMessage(error: any): string {
    if (typeof error === 'string') return error;
    if (error?.message) return error.message;
    if (error?.toString) return error.toString();
    return 'Unknown error';
  }

  private extractErrorCode(error: any): string | undefined {
    if (isApiError(error)) return error.code;
    // Auth removed - authentication not implemented per specs
    // if (error instanceof AuthError) return error.code;
    if (error?.code) return error.code;
    return undefined;
  }

  private notifyErrorHandlers(error: ErrorReport): void {
    this.errorHandlers.forEach(handler => {
      try {
        handler(error);
      } catch (err) {
        console.error('Error in error handler:', err);
      }
    });
  }

  private notifyHealthHandlers(): void {
    this.healthHandlers.forEach(handler => {
      try {
        handler(this.connectionHealth);
      } catch (err) {
        console.error('Error in health handler:', err);
      }
    });
  }

  private setupGlobalErrorHandlers(): void {
    // Handle unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.reportError(event.reason, {
        component: 'global',
        action: 'unhandled_promise_rejection',
      }, 'high');
    });

    // Handle global JavaScript errors
    window.addEventListener('error', (event) => {
      this.reportError(event.error || event.message, {
        component: 'global',
        action: 'javascript_error',
        metadata: {
          filename: event.filename,
          lineno: event.lineno,
          colno: event.colno,
        },
      }, 'medium');
    });
  }

  private startHealthMonitoring(): void {
    this.healthCheckTimer = setInterval(() => {
      this.updateConnectionStatus();
    }, 30000); // Check every 30 seconds
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // =============================================================================
  // Public Accessors
  // =============================================================================

  get connectionHealthStatus(): ConnectionHealth {
    return { ...this.connectionHealth };
  }

  getErrorReport(id: string): ErrorReport | null {
    return this.errorReports.get(id) || null;
  }

  getRecentErrors(limit?: number): ErrorReport[] {
    return limit ? this.recentErrors.slice(0, limit) : [...this.recentErrors];
  }

  getAllErrors(): ErrorReport[] {
    return Array.from(this.errorReports.values());
  }

  // =============================================================================
  // Cleanup
  // =============================================================================

  destroy(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
      this.healthCheckTimer = null;
    }

    this.errorReports.clear();
    this.errorHandlers.clear();
    this.healthHandlers.clear();
    this.errorCounts.clear();
    this.recentErrors.length = 0;
    this.retryQueues.clear();
  }
}

// =============================================================================
// Factory Function and Default Instance
// =============================================================================

export function createErrorHandler(): ErrorHandlingService {
  return new ErrorHandlingService();
}

// Default instance
export const errorHandler = createErrorHandler();

// Global error reporting functions
export function reportError(error: any, context?: Partial<ErrorContext>, severity?: ErrorSeverity): string {
  return errorHandler.reportError(error, context, severity);
}

export function reportApiError(error: ApiError, context?: Partial<ErrorContext>): string {
  return errorHandler.reportApiError(error, context);
}

// Auth functionality removed - authentication not implemented per specs
// export function reportAuthError(error: AuthError, context?: Partial<ErrorContext>): string {
//   return errorHandler.reportAuthError(error, context);
// }

export function withRetry<T>(
  operation: () => Promise<T>,
  config?: Partial<RetryConfig>,
  context?: Partial<ErrorContext>
): Promise<T> {
  return errorHandler.withRetry(operation, config, context);
}
