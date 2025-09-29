/**
 * REST API client with authentication and comprehensive error handling
 */

import {
  BaseResponse,
  ErrorResponse,
  SuccessResponse,
  HealthResponse,
  ConfigResponse,
  ConfigUpdateResponse,
  ConfigUpdateRequest,
  LoginRequest,
  LoginResponse,
  TokenRefreshResponse,
  UserCreateResponse,
  GameStateResponse,
  CalibrationStartRequest,
  CalibrationStartResponse,
  CalibrationPointRequest,
  CalibrationPointResponse,
  CalibrationApplyResponse,
  WebSocketConnectionResponse,
  isErrorResponse,
  isSuccessResponse,
} from '../types/api';

export interface ApiClientConfig {
  baseUrl: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
}

export interface RequestOptions {
  timeout?: number;
  retryAttempts?: number;
  skipAuth?: boolean;
  signal?: AbortSignal;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: Record<string, any>
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export class ApiClient {
  private config: Required<ApiClientConfig>;
  private authToken: string | null = null;
  private refreshToken: string | null = null;
  private refreshPromise: Promise<string> | null = null;
  private requestInterceptors: ((config: RequestInit) => RequestInit | Promise<RequestInit>)[] = [];
  private responseInterceptors: ((response: Response) => Response | Promise<Response>)[] = [];

  constructor(config: ApiClientConfig) {
    this.config = {
      timeout: 10000,
      retryAttempts: 3,
      retryDelay: 1000,
      ...config,
    };
  }

  // =============================================================================
  // Authentication Management
  // =============================================================================

  setAuthToken(token: string): void {
    this.authToken = token;
  }

  setRefreshToken(token: string): void {
    this.refreshToken = token;
  }

  clearAuth(): void {
    this.authToken = null;
    this.refreshToken = null;
  }

  get isAuthenticated(): boolean {
    return !!this.authToken;
  }

  // =============================================================================
  // Request/Response Interceptors
  // =============================================================================

  addRequestInterceptor(interceptor: (config: RequestInit) => RequestInit | Promise<RequestInit>): void {
    this.requestInterceptors.push(interceptor);
  }

  addResponseInterceptor(interceptor: (response: Response) => Response | Promise<Response>): void {
    this.responseInterceptors.push(interceptor);
  }

  // =============================================================================
  // Core HTTP Methods
  // =============================================================================

  private async request<T = any>(
    endpoint: string,
    options: RequestInit & RequestOptions = {}
  ): Promise<T> {
    const { skipAuth, retryAttempts = this.config.retryAttempts, timeout = this.config.timeout, ...init } = options;

    const url = `${this.config.baseUrl}${endpoint}`;
    let config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...init.headers,
      },
      ...init,
    };

    // Add authentication header
    if (!skipAuth && this.authToken) {
      config.headers = {
        ...config.headers,
        Authorization: `Bearer ${this.authToken}`,
      };
    }

    // Apply request interceptors
    for (const interceptor of this.requestInterceptors) {
      config = await interceptor(config);
    }

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      config.signal = config.signal || controller.signal;

      const response = await this.executeRequest(url, config, retryAttempts);
      clearTimeout(timeoutId);

      // Apply response interceptors
      let processedResponse = response;
      for (const interceptor of this.responseInterceptors) {
        processedResponse = await interceptor(processedResponse);
      }

      return await this.handleResponse<T>(processedResponse);

    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error && error.name === 'AbortError') {
        throw new ApiError(408, 'TIMEOUT', 'Request timeout');
      }

      // Handle authentication errors and retry with token refresh
      if (error instanceof ApiError && error.status === 401 && !skipAuth && this.refreshToken) {
        try {
          await this.refreshAuthToken();
          // Retry the original request once with new token
          return this.request<T>(endpoint, { ...options, retryAttempts: 0 });
        } catch (refreshError) {
          // If refresh fails, clear auth and throw original error
          this.clearAuth();
          throw error;
        }
      }

      throw error;
    }
  }

  private async executeRequest(url: string, config: RequestInit, retryAttempts: number): Promise<Response> {
    let lastError: Error;

    for (let attempt = 0; attempt <= retryAttempts; attempt++) {
      try {
        const response = await fetch(url, config);

        // Don't retry on client errors (4xx) except for rate limiting
        if (response.status >= 400 && response.status < 500 && response.status !== 429) {
          return response;
        }

        // Retry on server errors (5xx) and rate limiting
        if (response.status >= 500 || response.status === 429) {
          if (attempt < retryAttempts) {
            await this.delay(this.config.retryDelay * Math.pow(2, attempt));
            continue;
          }
        }

        return response;

      } catch (error) {
        lastError = error as Error;
        if (attempt < retryAttempts) {
          await this.delay(this.config.retryDelay * Math.pow(2, attempt));
        }
      }
    }

    throw lastError!;
  }

  private async handleResponse<T>(response: Response): Promise<T> {
    let data: any;

    try {
      const text = await response.text();
      data = text ? JSON.parse(text) : {};
    } catch (error) {
      throw new ApiError(response.status, 'PARSE_ERROR', 'Failed to parse response JSON');
    }

    if (!response.ok) {
      if (isErrorResponse(data)) {
        throw new ApiError(response.status, data.code, data.message, data.details);
      } else {
        throw new ApiError(
          response.status,
          `HTTP_${response.status}`,
          `HTTP ${response.status}: ${response.statusText}`
        );
      }
    }

    return data;
  }

  private async refreshAuthToken(): Promise<string> {
    // Prevent concurrent refresh requests
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = (async () => {
      try {
        const response = await this.request<TokenRefreshResponse>('/auth/refresh', {
          method: 'POST',
          body: JSON.stringify({ refresh_token: this.refreshToken }),
          skipAuth: true,
        });

        if (response.success && response.access_token) {
          this.setAuthToken(response.access_token);
          return response.access_token;
        } else {
          throw new Error('Token refresh failed');
        }
      } finally {
        this.refreshPromise = null;
      }
    })();

    return this.refreshPromise;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // =============================================================================
  // HTTP Method Helpers
  // =============================================================================

  async get<T = any>(endpoint: string, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET', ...options });
  }

  async post<T = any>(endpoint: string, data?: any, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
      ...options,
    });
  }

  async put<T = any>(endpoint: string, data?: any, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
      ...options,
    });
  }

  async patch<T = any>(endpoint: string, data?: any, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
      ...options,
    });
  }

  async delete<T = any>(endpoint: string, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE', ...options });
  }

  // =============================================================================
  // Authentication Endpoints
  // =============================================================================

  async login(credentials: LoginRequest): Promise<LoginResponse> {
    const response = await this.post<LoginResponse>('/auth/login', credentials, { skipAuth: true });

    if (response.success && response.access_token) {
      this.setAuthToken(response.access_token);
      if (response.refresh_token) {
        this.setRefreshToken(response.refresh_token);
      }
    }

    return response;
  }

  async logout(): Promise<SuccessResponse> {
    try {
      const response = await this.post<SuccessResponse>('/auth/logout');
      return response;
    } finally {
      this.clearAuth();
    }
  }

  async createUser(userData: {
    username: string;
    password: string;
    role?: string;
  }): Promise<UserCreateResponse> {
    return this.post<UserCreateResponse>('/auth/users', userData);
  }

  // =============================================================================
  // Health and System Endpoints
  // =============================================================================

  async getHealth(): Promise<HealthResponse> {
    return this.get<HealthResponse>('/health', { skipAuth: true });
  }

  async getVersion(): Promise<{ version: string; build_date: string }> {
    return this.get('/version', { skipAuth: true });
  }

  // =============================================================================
  // Configuration Endpoints
  // =============================================================================

  async getConfig(section?: string): Promise<ConfigResponse> {
    const endpoint = section ? `/config/${section}` : '/config';
    return this.get<ConfigResponse>(endpoint);
  }

  async updateConfig(request: ConfigUpdateRequest): Promise<ConfigUpdateResponse> {
    const endpoint = request.section ? `/config/${request.section}` : '/config';
    return this.put<ConfigUpdateResponse>(endpoint, request);
  }

  async validateConfig(values: Record<string, any>): Promise<ConfigUpdateResponse> {
    return this.post<ConfigUpdateResponse>('/config/validate', { values, validate_only: true });
  }

  async exportConfig(format: string = 'json'): Promise<Blob> {
    const response = await fetch(`${this.config.baseUrl}/config/export?format=${format}`, {
      headers: this.authToken ? { Authorization: `Bearer ${this.authToken}` } : {},
    });

    if (!response.ok) {
      throw new ApiError(response.status, 'EXPORT_FAILED', 'Config export failed');
    }

    return response.blob();
  }

  // =============================================================================
  // Game State Endpoints
  // =============================================================================

  async getCurrentGameState(): Promise<GameStateResponse> {
    return this.get<GameStateResponse>('/game/state');
  }

  async getGameHistory(
    limit?: number,
    offset?: number,
    startTime?: Date,
    endTime?: Date
  ): Promise<GameStateResponse[]> {
    const params = new URLSearchParams();
    if (limit) params.set('limit', limit.toString());
    if (offset) params.set('offset', offset.toString());
    if (startTime) params.set('start_time', startTime.toISOString());
    if (endTime) params.set('end_time', endTime.toISOString());

    const query = params.toString();
    return this.get<GameStateResponse[]>(`/game/history${query ? `?${query}` : ''}`);
  }

  async resetGameState(): Promise<SuccessResponse> {
    return this.post<SuccessResponse>('/game/reset');
  }

  // =============================================================================
  // Calibration Endpoints
  // =============================================================================

  async startCalibration(request: CalibrationStartRequest): Promise<CalibrationStartResponse> {
    return this.post<CalibrationStartResponse>('/calibration/start', request);
  }

  async captureCalibrationPoint(
    sessionId: string,
    point: CalibrationPointRequest
  ): Promise<CalibrationPointResponse> {
    return this.post<CalibrationPointResponse>(`/calibration/${sessionId}/points`, point);
  }

  async applyCalibration(sessionId: string): Promise<CalibrationApplyResponse> {
    return this.post<CalibrationApplyResponse>(`/calibration/${sessionId}/apply`);
  }

  async cancelCalibration(sessionId: string): Promise<SuccessResponse> {
    return this.delete<SuccessResponse>(`/calibration/${sessionId}`);
  }

  // =============================================================================
  // WebSocket Management Endpoints
  // =============================================================================

  async getWebSocketConnections(): Promise<Record<string, any>> {
    return this.get('/ws/connections');
  }

  async getWebSocketHealth(): Promise<Record<string, any>> {
    return this.get('/ws/health');
  }

  async broadcastTestFrame(width: number = 1920, height: number = 1080): Promise<SuccessResponse> {
    return this.post<SuccessResponse>('/ws/broadcast/frame', { width, height });
  }

  async broadcastTestAlert(
    level: string = 'info',
    message: string = 'Test alert'
  ): Promise<SuccessResponse> {
    return this.post<SuccessResponse>('/ws/broadcast/alert', { level, message });
  }

  // =============================================================================
  // Hardware Control Endpoints
  // =============================================================================

  async getCameraStatus(): Promise<Record<string, any>> {
    return this.get('/hardware/camera/status');
  }

  async setCameraSettings(settings: Record<string, any>): Promise<SuccessResponse> {
    return this.put<SuccessResponse>('/hardware/camera/settings', settings);
  }

  async getProjectorStatus(): Promise<Record<string, any>> {
    return this.get('/hardware/projector/status');
  }

  async setProjectorSettings(settings: Record<string, any>): Promise<SuccessResponse> {
    return this.put<SuccessResponse>('/hardware/projector/settings', settings);
  }
}

// =============================================================================
// Factory Function and Default Instance
// =============================================================================

export function createApiClient(config: ApiClientConfig): ApiClient {
  return new ApiClient(config);
}

// Default instance for development
export const apiClient = createApiClient({
  baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
});

// =============================================================================
// Request/Response Interceptor Helpers
// =============================================================================

export function createRequestLogger() {
  return (config: RequestInit): RequestInit => {
    console.log('API Request:', config);
    return config;
  };
}

export function createResponseLogger() {
  return (response: Response): Response => {
    console.log('API Response:', response.status, response.statusText);
    return response;
  };
}

export function createRequestTimestampInterceptor() {
  return (config: RequestInit): RequestInit => {
    return {
      ...config,
      headers: {
        ...config.headers,
        'X-Request-Timestamp': new Date().toISOString(),
      },
    };
  };
}

// =============================================================================
// Error Handling Utilities
// =============================================================================

export function isApiError(error: any): error is ApiError {
  return error instanceof ApiError;
}

export function getApiErrorMessage(error: any): string {
  if (isApiError(error)) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unknown error occurred';
}

export function getApiErrorCode(error: any): string {
  if (isApiError(error)) {
    return error.code;
  }
  return 'UNKNOWN_ERROR';
}
