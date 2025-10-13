/**
 * API Client for communicating with the backend server
 * Provides a centralized interface for all HTTP requests and WebSocket connections
 */

import axiosClient from './axios-client';
import { AxiosError } from './axios-client';
import type { ApiErrorResponse, AxiosResponse } from './axios-client';

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
  status: "healthy" | "degraded" | "unhealthy";
  timestamp: string;
  uptime: number;
  version: string;
  components?: Record<
    string,
    {
      name: string;
      status: "healthy" | "degraded" | "unhealthy";
      message: string;
      last_check: string;
      uptime?: number;
      error_count?: number;
    }
  >;
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
  setAuthToken(token: string | null): void {
    axiosClient.setAuthToken(token);
  }

  /**
   * Transform AxiosResponse to ApiResponse format
   */
  private handleResponse<T>(response: AxiosResponse<T>): ApiResponse<T> {
    return {
      success: true,
      data: response.data,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Transform AxiosError to ApiResponse error format
   */
  private handleError(error: unknown): ApiResponse {
    if (error instanceof AxiosError) {
      const errorData = error.response?.data as ApiErrorResponse | undefined;
      const errorMessage =
        errorData?.detail?.message ||
        errorData?.message ||
        errorData?.error ||
        error.message ||
        'Network error';

      return {
        success: false,
        error: errorMessage,
        timestamp: new Date().toISOString(),
      };
    }

    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString(),
    };
  }

  // Health and System endpoints
  async getHealth(
    includeDetails = false,
    includeMetrics = false
  ): Promise<ApiResponse<HealthResponse>> {
    try {
      const params: Record<string, string> = {};
      if (includeDetails) params.include_details = 'true';
      if (includeMetrics) params.include_metrics = 'true';

      const response = await axiosClient.get<HealthResponse>('/api/v1/health', { params });
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async healthCheck(): Promise<ApiResponse<void>> {
    try {
      const response = await axiosClient.head('/api/health');
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getMetrics(timeRange = "5m"): Promise<ApiResponse<SystemMetrics>> {
    try {
      const response = await axiosClient.get<SystemMetrics>('/api/v1/health/metrics', {
        params: { time_range: timeRange },
      });
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getVersion(): Promise<ApiResponse<any>> {
    try {
      const response = await axiosClient.get('/api/v1/health/version');
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Configuration endpoints
  async getConfiguration(
    section?: string,
    includeMetadata = true
  ): Promise<ApiResponse<ConfigResponse>> {
    try {
      const params: Record<string, string> = {};
      if (section) params.section = section;
      if (includeMetadata) params.include_metadata = 'true';

      const response = await axiosClient.get<ConfigResponse>('/api/v1/config', { params });
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async updateConfiguration(
    configData: Record<string, any>,
    section?: string,
    validateOnly = false,
    forceUpdate = false
  ): Promise<ApiResponse<any>> {
    try {
      const params: Record<string, string> = {};
      if (section) params.section = section;
      if (validateOnly) params.validate_only = 'true';
      if (forceUpdate) params.force_update = 'true';

      const response = await axiosClient.put('/api/v1/config', configData, { params });
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async resetConfiguration(
    confirm = true,
    backupCurrent = true,
    resetType = "all",
    sections?: string[]
  ): Promise<ApiResponse<any>> {
    try {
      const params: Record<string, string | string[]> = {
        confirm: confirm.toString(),
        backup_current: backupCurrent.toString(),
        reset_type: resetType,
      };

      if (sections) {
        params.sections = sections;
      }

      const response = await axiosClient.post('/api/v1/config/reset', null, { params });
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async exportConfiguration(
    format = "json",
    sections?: string[],
    includeDefaults = false,
    includeMetadata = true
  ): Promise<ApiResponse<any>> {
    try {
      const params: Record<string, string | string[]> = {
        format,
        include_defaults: includeDefaults.toString(),
        include_metadata: includeMetadata.toString(),
      };

      if (sections) {
        params.sections = sections;
      }

      const response = await axiosClient.get('/api/v1/config/export', { params });
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async importConfiguration(
    file: File,
    mergeStrategy = "replace",
    validateOnly = false
  ): Promise<ApiResponse<any>> {
    try {
      const formData = new FormData();
      formData.append("file", file);

      const params: Record<string, string> = {
        merge_strategy: mergeStrategy,
        validate_only: validateOnly.toString(),
      };

      const response = await axiosClient.upload('/api/v1/config/import', formData, { params });
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Video streaming endpoints
  getVideoStreamUrl(
    quality = 80,
    fps = 30,
    width?: number,
    height?: number
  ): string {
    const params = new URLSearchParams({
      quality: quality.toString(),
      fps: fps.toString(),
    });

    if (width) params.append("width", width.toString());
    if (height) params.append("height", height.toString());

    return `${axiosClient.getBaseURL()}/api/v1/stream/video?${params}`;
  }

  getSingleFrameUrl(quality = 90, width?: number, height?: number): string {
    const params = new URLSearchParams({
      quality: quality.toString(),
    });

    if (width) params.append("width", width.toString());
    if (height) params.append("height", height.toString());

    return `${axiosClient.getBaseURL()}/api/v1/stream/video/frame?${params}`;
  }

  async getStreamStatus(): Promise<ApiResponse<any>> {
    try {
      const response = await axiosClient.get('/api/v1/stream/video/status');
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async startVideoCapture(): Promise<ApiResponse<any>> {
    try {
      const response = await axiosClient.post('/api/v1/stream/video/start');
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async stopVideoCapture(): Promise<ApiResponse<any>> {
    try {
      const response = await axiosClient.post('/api/v1/stream/video/stop');
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  // WebSocket connection
  createWebSocket(token?: string): WebSocket {
    return axiosClient.createWebSocket('/api/v1/ws', token);
  }

  // Vision and Calibration endpoints
  async performCalibration(calibrationData: any): Promise<ApiResponse<any>> {
    try {
      const response = await axiosClient.post('/api/v1/vision/calibration', calibrationData);
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getCalibrationData(): Promise<ApiResponse<any>> {
    try {
      const response = await axiosClient.get('/api/v1/vision/calibration');
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getVisionStatus(): Promise<ApiResponse<any>> {
    try {
      const response = await axiosClient.get('/api/v1/vision/status');
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getPlayingArea(): Promise<ApiResponse<any>> {
    try {
      const response = await axiosClient.get('/api/v1/config/table/playing-area');
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async updatePlayingArea(data: any): Promise<ApiResponse<any>> {
    try {
      const response = await axiosClient.post('/api/v1/config/table/playing-area', data);
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async startDetection(): Promise<ApiResponse<any>> {
    try {
      const response = await axiosClient.post('/api/v1/vision/detection/start');
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async stopDetection(): Promise<ApiResponse<any>> {
    try {
      const response = await axiosClient.post('/api/v1/vision/detection/stop');
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Test endpoints
  async testDownload(sizeMb: number): Promise<ApiResponse<any>> {
    try {
      const response = await axiosClient.get(`/api/test/download/${sizeMb}mb`);
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  async testUpload(data: any): Promise<ApiResponse<any>> {
    try {
      const response = await axiosClient.post('/api/test/upload', data);
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }
}

// Create singleton instance
export const apiClient = new ApiClient();
