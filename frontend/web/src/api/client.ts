import axiosClient, { AxiosError } from './axios-client';
import type { ApiErrorResponse, AxiosResponse } from './axios-client';

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

export interface ConfigResponse {
  values: Record<string, any>;
}

/** HTTP surface used by /live to discover its direct camera viewer URL. */
export class ApiClient {
  private handleResponse<T>(response: AxiosResponse<T>): ApiResponse<T> {
    return {
      success: true,
      data: response.data,
      timestamp: new Date().toISOString(),
    };
  }

  private handleError(error: unknown): ApiResponse<never> {
    let message = error instanceof Error ? error.message : 'Unknown error';
    if (error instanceof AxiosError) {
      const errorData = error.response?.data as ApiErrorResponse | undefined;
      message = errorData?.detail?.message
        || errorData?.message
        || errorData?.error
        || error.message
        || 'Network error';
    }
    return { success: false, error: message, timestamp: new Date().toISOString() };
  }

  async getConfiguration(): Promise<ApiResponse<ConfigResponse>> {
    try {
      const response = await axiosClient.get<ConfigResponse>('/api/v1/config', {
        params: { include_metadata: 'true' },
      });
      return this.handleResponse(response);
    } catch (error) {
      return this.handleError(error);
    }
  }
}

export const apiClient = new ApiClient();
