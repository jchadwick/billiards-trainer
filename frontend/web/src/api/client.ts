/**
 * API Client for communicating with the backend server
 * Provides a centralized interface for all HTTP requests and WebSocket connections
 */

import axiosClient from './axios-client';
import { AxiosError } from './axios-client';
import type { ApiErrorResponse, AxiosResponse } from './axios-client';
import type { ApiGameStateResponse } from '../types/api';
import type {
  PerceptionRuntimeStatus,
  RustFramePreview,
  RustFramePreviewDescriptor,
} from '../types/perception';

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

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isRustTimestamp(value: unknown): value is Record<string, unknown> {
  if (!isObject(value)) return false;
  return (
    typeof value.ticks === 'string' &&
    /^[0-9]{1,20}$/.test(value.ticks) &&
    value.unit === 'nanoseconds' &&
    ['sensor', 'monotonic', 'unix_epoch'].includes(String(value.clock)) &&
    typeof value.domain_id === 'string' &&
    value.domain_id.length > 0 &&
    [
      'exposure',
      'camera_read_complete',
      'shared_memory_publication',
      'frame_arrival',
      'inference_started',
      'inference_complete',
      'batch_publication',
    ].includes(String(value.stage))
  );
}

function parseRustFramePreviewDescriptor(
  value: unknown
): RustFramePreviewDescriptor {
  if (!isObject(value)) throw new Error('Rust frame descriptor is not an object');
  if (
    value.schema_version !== 1 ||
    typeof value.engine_instance_id !== 'string' ||
    value.engine_instance_id.length === 0 ||
    typeof value.capture_session_id !== 'string' ||
    value.capture_session_id.length === 0 ||
    typeof value.source_frame_id !== 'string' ||
    value.source_frame_id.length === 0 ||
    typeof value.camera_profile_id !== 'string' ||
    value.camera_profile_id.length === 0 ||
    !Number.isSafeInteger(value.width_px) ||
    Number(value.width_px) <= 0 ||
    !Number.isSafeInteger(value.height_px) ||
    Number(value.height_px) <= 0 ||
    value.pixel_format !== 'rgb8' ||
    !['raw', 'rectified'].includes(String(value.image_space)) ||
    !Number.isSafeInteger(value.configuration_revision) ||
    Number(value.configuration_revision) <= 0 ||
    !Number.isSafeInteger(value.calibration_revision) ||
    Number(value.calibration_revision) < 0 ||
    !Number.isSafeInteger(value.rectification_revision) ||
    Number(value.rectification_revision) < 0
  ) {
    throw new Error('Rust frame descriptor identity or geometry is invalid');
  }

  const semantics = value.source_timestamp_semantics;
  if (!isObject(semantics)) {
    throw new Error('Rust frame timestamp semantics are missing');
  }
  const sourceTimestamp = value.source_timestamp;
  if (semantics.kind === 'read_completion_only') {
    if (Object.keys(semantics).length !== 1 || sourceTimestamp !== null) {
      throw new Error('Rust read-completion timestamp semantics are inconsistent');
    }
  } else if (semantics.kind === 'source_timestamp') {
    if (
      !isRustTimestamp(sourceTimestamp) ||
      semantics.unit !== sourceTimestamp.unit ||
      semantics.clock !== sourceTimestamp.clock ||
      semantics.domain_id !== sourceTimestamp.domain_id ||
      semantics.stage !== sourceTimestamp.stage
    ) {
      throw new Error('Rust source timestamp disagrees with its clock semantics');
    }
  } else {
    throw new Error('Rust frame timestamp semantics are unsupported');
  }

  const hostRead = value.host_read_completed;
  if (
    !isRustTimestamp(hostRead) ||
    hostRead.unit !== 'nanoseconds' ||
    hostRead.clock !== 'monotonic' ||
    hostRead.domain_id !== value.engine_instance_id ||
    hostRead.stage !== 'camera_read_complete' ||
    hostRead.semantics !== 'read_completion_only'
  ) {
    throw new Error('Rust host timestamp is not a labeled read-completion time');
  }
  return value as unknown as RustFramePreviewDescriptor;
}

export class ApiClient {
  /** Read a versioned API resource for monitoring views that consume raw payloads. */
  async get<T = unknown>(path: string): Promise<T> {
    const response = await axiosClient.get<T>(`/api/v1${path}`);
    return response.data;
  }

  /** Read which runtime owns capture before choosing the calibration preview path. */
  async getPerceptionRuntimeStatus(): Promise<ApiResponse<PerceptionRuntimeStatus>> {
    try {
      const response = await axiosClient.get<PerceptionRuntimeStatus>(
        '/api/v1/health/perception'
      );
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
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
      // The Rust gateway exposes liveness independently from capture readiness.
      // The live view should connect while camera/inference startup is ongoing.
      const response = await axiosClient.get('/api/v1/health/live');
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }

  /** Fetch the current Core or scene-localized state without coercing unknown fields. */
  async getCurrentGameState(): Promise<ApiResponse<ApiGameStateResponse>> {
    try {
      const response = await axiosClient.get<ApiGameStateResponse>(
        '/api/v1/game/state'
      );
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

  /**
   * Fetch one Rust-owned image and its same-response source-frame descriptor.
   * The binary body stays a Blob; only the bounded descriptor travels in a
   * response header so no image is base64-expanded or decoded by the gateway.
   */
  async getRustCalibrationFrame(): Promise<ApiResponse<RustFramePreview>> {
    try {
      const response = await axiosClient.get<Blob>(
        '/api/v1/stream/calibration-frame',
        { responseType: 'blob', headers: { 'Cache-Control': 'no-cache' } }
      );
      const encodedDescriptor = response.headers['x-frame-preview-descriptor'];
      if (typeof encodedDescriptor !== 'string' || response.data.size === 0) {
        throw new Error('Rust frame preview did not include its image descriptor');
      }
      const normalized = encodedDescriptor.replace(/-/g, '+').replace(/_/g, '/');
      const padded = normalized + '='.repeat((4 - (normalized.length % 4)) % 4);
      const binary = atob(padded);
      const descriptorText = new TextDecoder().decode(
        Uint8Array.from(binary, (character) => character.charCodeAt(0))
      );
      const descriptor = parseRustFramePreviewDescriptor(
        JSON.parse(descriptorText) as unknown
      );
      if (response.data.type && response.data.type !== 'image/jpeg') {
        throw new Error('Rust frame preview returned a non-JPEG image');
      }
      return this.handleResponse({
        ...response,
        data: { image: response.data, descriptor },
      });
    } catch (error) {
      return this.handleError(error);
    }
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
  createWebSocket(): WebSocket {
    return axiosClient.createWebSocket('/api/v1/websocket/ws');
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
