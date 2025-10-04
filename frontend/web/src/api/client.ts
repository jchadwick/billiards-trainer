/**
 * API Client for communicating with the backend server
 * Provides a centralized interface for all HTTP requests and WebSocket connections
 */

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}

export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_io: {
    bytes_sent: number;
    bytes_recv: number;
    packets_sent: number;
    packets_recv: number;
  };
  api_requests_per_second: number;
  websocket_connections: number;
  average_response_time: number;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  uptime: number;
  version: string;
  components?: Record<string, {
    name: string;
    status: 'healthy' | 'degraded' | 'unhealthy';
    message: string;
    last_check: string;
    uptime?: number;
    error_count?: number;
  }>;
  metrics?: SystemMetrics;
}

export interface ConfigResponse {
  timestamp: string;
  values: Record<string, any>;
  schema_version: string;
  last_modified: string;
  is_valid: boolean;
  validation_errors: string[];
}

export interface WebSocketMessage {
  type: string;
  timestamp: Date;
  data: any;
}

export class ApiClient {
  private baseURL: string;
  private token: string | null = null;
  private wsUrl: string;

  constructor(baseURL?: string, wsUrl?: string) {
    // Auto-detect base URL from current window location if not provided
    if (!baseURL) {
      if (typeof window !== 'undefined') {
        // Use current origin when running in browser
        baseURL = window.location.origin;
      } else {
        // Fallback for SSR or testing
        baseURL = 'http://localhost:8000';
      }
    }

    // Auto-detect WebSocket URL from current window location if not provided
    if (!wsUrl) {
      if (typeof window !== 'undefined') {
        // Use current origin with ws/wss protocol
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        wsUrl = `${protocol}//${window.location.host}`;
      } else {
        // Fallback for SSR or testing
        wsUrl = 'ws://localhost:8000';
      }
    }

    this.baseURL = baseURL.replace(/\/$/, ''); // Remove trailing slash
    this.wsUrl = wsUrl.replace(/\/$/, '');
  }

  setAuthToken(token: string | null): void {
    this.token = token;
  }

  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    return headers;
  }

  private async handleResponse<T>(response: Response): Promise<ApiResponse<T>> {
    try {
      const data = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error: data.detail?.message || data.message || `HTTP ${response.status}`,
          timestamp: new Date().toISOString(),
        };
      }

      return {
        success: true,
        data,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to parse response',
        timestamp: new Date().toISOString(),
      };
    }
  }

  // Health and System endpoints
  async getHealth(includeDetails = false, includeMetrics = false): Promise<ApiResponse<HealthResponse>> {
    try {
      const params = new URLSearchParams();
      if (includeDetails) params.append('include_details', 'true');
      if (includeMetrics) params.append('include_metrics', 'true');

      const response = await fetch(`${this.baseURL}/api/v1/health?${params}`, {
        method: 'GET',
        headers: this.getHeaders(),
      });

      return this.handleResponse<HealthResponse>(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async getMetrics(timeRange = '5m'): Promise<ApiResponse<SystemMetrics>> {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/health/metrics?time_range=${timeRange}`, {
        method: 'GET',
        headers: this.getHeaders(),
      });

      return this.handleResponse<SystemMetrics>(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async getVersion(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/health/version`, {
        method: 'GET',
        headers: this.getHeaders(),
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  // Configuration endpoints
  async getConfiguration(section?: string, includeMetadata = true): Promise<ApiResponse<ConfigResponse>> {
    try {
      const params = new URLSearchParams();
      if (section) params.append('section', section);
      if (includeMetadata) params.append('include_metadata', 'true');

      const response = await fetch(`${this.baseURL}/api/v1/config?${params}`, {
        method: 'GET',
        headers: this.getHeaders(),
      });

      return this.handleResponse<ConfigResponse>(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async updateConfiguration(
    configData: Record<string, any>,
    section?: string,
    validateOnly = false,
    forceUpdate = false
  ): Promise<ApiResponse<any>> {
    try {
      const params = new URLSearchParams();
      if (section) params.append('section', section);
      if (validateOnly) params.append('validate_only', 'true');
      if (forceUpdate) params.append('force_update', 'true');

      const response = await fetch(`${this.baseURL}/api/v1/config?${params}`, {
        method: 'PUT',
        headers: this.getHeaders(),
        body: JSON.stringify(configData),
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async resetConfiguration(
    confirm = true,
    backupCurrent = true,
    resetType = 'all',
    sections?: string[]
  ): Promise<ApiResponse<any>> {
    try {
      const params = new URLSearchParams({
        confirm: confirm.toString(),
        backup_current: backupCurrent.toString(),
        reset_type: resetType,
      });

      if (sections) {
        sections.forEach(section => params.append('sections', section));
      }

      const response = await fetch(`${this.baseURL}/api/v1/config/reset?${params}`, {
        method: 'POST',
        headers: this.getHeaders(),
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async exportConfiguration(
    format = 'json',
    sections?: string[],
    includeDefaults = false,
    includeMetadata = true
  ): Promise<ApiResponse<any>> {
    try {
      const params = new URLSearchParams({
        format,
        include_defaults: includeDefaults.toString(),
        include_metadata: includeMetadata.toString(),
      });

      if (sections) {
        sections.forEach(section => params.append('sections', section));
      }

      const response = await fetch(`${this.baseURL}/api/v1/config/export?${params}`, {
        method: 'GET',
        headers: this.getHeaders(),
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async importConfiguration(
    file: File,
    mergeStrategy = 'replace',
    validateOnly = false
  ): Promise<ApiResponse<any>> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const params = new URLSearchParams({
        merge_strategy: mergeStrategy,
        validate_only: validateOnly.toString(),
      });

      const headers: HeadersInit = {};
      if (this.token) {
        headers.Authorization = `Bearer ${this.token}`;
      }

      const response = await fetch(`${this.baseURL}/api/v1/config/import?${params}`, {
        method: 'POST',
        headers,
        body: formData,
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  // Video streaming endpoints
  getVideoStreamUrl(quality = 80, fps = 30, width?: number, height?: number): string {
    const params = new URLSearchParams({
      quality: quality.toString(),
      fps: fps.toString(),
    });

    if (width) params.append('width', width.toString());
    if (height) params.append('height', height.toString());

    return `${this.baseURL}/api/v1/stream/video?${params}`;
  }

  getSingleFrameUrl(quality = 90, width?: number, height?: number): string {
    const params = new URLSearchParams({
      quality: quality.toString(),
    });

    if (width) params.append('width', width.toString());
    if (height) params.append('height', height.toString());

    return `${this.baseURL}/api/v1/stream/video/frame?${params}`;
  }

  async getStreamStatus(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/stream/video/status`, {
        method: 'GET',
        headers: this.getHeaders(),
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async startVideoCapture(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/stream/video/start`, {
        method: 'POST',
        headers: this.getHeaders(),
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async stopVideoCapture(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/stream/video/stop`, {
        method: 'POST',
        headers: this.getHeaders(),
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  // WebSocket connection
  createWebSocket(token?: string): WebSocket {
    const wsUrl = `${this.wsUrl}/api/v1/ws`;
    const url = token ? `${wsUrl}?token=${encodeURIComponent(token)}` : wsUrl;
    return new WebSocket(url);
  }

  // Vision and Calibration endpoints
  async performCalibration(calibrationData: any): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/vision/calibration`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(calibrationData),
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async getCalibrationData(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/vision/calibration`, {
        method: 'GET',
        headers: this.getHeaders(),
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async getVisionStatus(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/vision/status`, {
        method: 'GET',
        headers: this.getHeaders(),
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async startDetection(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/vision/detection/start`, {
        method: 'POST',
        headers: this.getHeaders(),
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }

  async stopDetection(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/vision/detection/stop`, {
        method: 'POST',
        headers: this.getHeaders(),
      });

      return this.handleResponse(response);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
        timestamp: new Date().toISOString(),
      };
    }
  }
}

// Create singleton instance
export const apiClient = new ApiClient();
