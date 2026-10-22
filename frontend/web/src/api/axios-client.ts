/**
 * Axios-based HTTP client with interceptors, retry logic, and centralized configuration
 * This module provides the core axios instance used by the API client
 * Also includes WebSocket factory for centralized WebSocket management
 */

import axios from "axios";
import type {
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
  AxiosError,
  InternalAxiosRequestConfig,
} from "axios";

// Configuration interface
export interface ApiClientConfig {
  baseURL: string;
  wsBaseURL: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  enableLogging?: boolean;
}

// Error response structure
export interface ApiErrorResponse {
  detail?: {
    message?: string;
    code?: string;
  };
  message?: string;
  error?: string;
  status?: number;
}

// Create axios instance with default configuration
const createAxiosInstance = (config: ApiClientConfig): AxiosInstance => {
  const {
    baseURL,
    timeout = 30000, // 30 seconds default
    enableLogging = import.meta.env.DEV,
  } = config;

  const instance = axios.create({
    baseURL,
    timeout,
    headers: {
      "Content-Type": "application/json",
    },
    // Keep the anonymous single-user API from sending browser cookies.
    withCredentials: false,
  });

  // Request interceptor - runs before every request
  instance.interceptors.request.use(
    (config: InternalAxiosRequestConfig) => {
      // Log request in development
      if (enableLogging) {
        console.log(
          `[API Request] ${config.method?.toUpperCase()} ${config.url}`,
          {
            params: config.params,
            data: config.data,
          }
        );
      }

      // Add timestamp for request tracking
      (config as any).metadata = { startTime: Date.now() };

      return config;
    },
    (error: AxiosError) => {
      if (enableLogging) {
        console.error("[API Request Error]", error);
      }
      return Promise.reject(error);
    }
  );

  // Response interceptor - runs after every response
  instance.interceptors.response.use(
    (response: AxiosResponse) => {
      // Calculate request duration
      const config = response.config as any;
      const duration = Date.now() - (config.metadata?.startTime || 0);

      if (enableLogging) {
        console.log(
          `[API Response] ${config.method?.toUpperCase()} ${config.url} - ${response.status} (${duration}ms)`,
          response.data
        );
      }

      return response;
    },
    async (error: AxiosError<ApiErrorResponse>) => {
      const config = error.config as any;
      const duration = Date.now() - (config?.metadata?.startTime || 0);

      if (enableLogging) {
        console.error(
          `[API Error] ${config?.method?.toUpperCase()} ${config?.url} - ${error.response?.status || "Network Error"} (${duration}ms)`,
          {
            status: error.response?.status,
            data: error.response?.data,
            message: error.message,
          }
        );
      }

      // Handle specific error cases
      if (error.response) {
        // Server responded with error status
        const status = error.response.status;

        switch (status) {
          case 404:
            // Not found
            console.warn(`[API] Resource not found: ${config?.url}`);
            break;
          case 429:
            // Rate limited
            console.warn("[API] Rate limit exceeded");
            break;
          case 500:
          case 502:
          case 503:
          case 504:
            // Server errors - might want to retry
            console.error("[API] Server error occurred");
            break;
        }
      } else if (error.request) {
        // Request made but no response received
        console.error("[API] No response received - possible network issue");
      } else {
        // Error in request configuration
        console.error("[API] Request configuration error:", error.message);
      }

      return Promise.reject(error);
    }
  );

  return instance;
};

// Retry logic for failed requests
export const createRetryableRequest = <T = any>(
  instance: AxiosInstance,
  config: AxiosRequestConfig,
  retryAttempts = 3,
  retryDelay = 1000
): Promise<AxiosResponse<T>> => {
  return new Promise((resolve, reject) => {
    const attempt = async (attemptNumber: number) => {
      try {
        const response = await instance.request<T>(config);
        resolve(response);
      } catch (error) {
        const axiosError = error as AxiosError;

        // Only retry on network errors or 5xx errors
        const shouldRetry =
          attemptNumber < retryAttempts &&
          (!axiosError.response ||
            (axiosError.response.status >= 500 &&
              axiosError.response.status < 600));

        if (shouldRetry) {
          const delay = retryDelay * Math.pow(2, attemptNumber - 1); // Exponential backoff
          console.log(
            `[API Retry] Attempt ${attemptNumber + 1}/${retryAttempts} after ${delay}ms`
          );
          setTimeout(() => attempt(attemptNumber + 1), delay);
        } else {
          reject(error);
        }
      }
    };

    attempt(1);
  });
};

// Main axios client class
export class AxiosClient {
  private instance: AxiosInstance;
  private config: ApiClientConfig;

  constructor(config: ApiClientConfig) {
    this.config = config;
    this.instance = createAxiosInstance(config);
  }

  /**
   * Get the configuration
   */
  getConfig(): ApiClientConfig {
    return { ...this.config };
  }

  /**
   * Get the base URL
   */
  getBaseURL(): string {
    return this.config.baseURL;
  }

  /**
   * Get the WebSocket base URL
   */
  getWsBaseURL(): string {
    return this.config.wsBaseURL;
  }

  /**
   * Get the raw axios instance for advanced usage
   */
  getInstance(): AxiosInstance {
    return this.instance;
  }

  /**
   * GET request
   */
  async get<T = any>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<AxiosResponse<T>> {
    return this.instance.get<T>(url, config);
  }

  /**
   * GET request with retry
   */
  async getWithRetry<T = any>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<AxiosResponse<T>> {
    return createRetryableRequest<T>(
      this.instance,
      { ...config, method: "GET", url },
      this.config.retryAttempts,
      this.config.retryDelay
    );
  }

  /**
   * POST request
   */
  async post<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<AxiosResponse<T>> {
    return this.instance.post<T>(url, data, config);
  }

  /**
   * POST request with retry
   */
  async postWithRetry<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<AxiosResponse<T>> {
    return createRetryableRequest<T>(
      this.instance,
      { ...config, method: "POST", url, data },
      this.config.retryAttempts,
      this.config.retryDelay
    );
  }

  /**
   * PUT request
   */
  async put<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<AxiosResponse<T>> {
    return this.instance.put<T>(url, data, config);
  }

  /**
   * PUT request with retry
   */
  async putWithRetry<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<AxiosResponse<T>> {
    return createRetryableRequest<T>(
      this.instance,
      { ...config, method: "PUT", url, data },
      this.config.retryAttempts,
      this.config.retryDelay
    );
  }

  /**
   * PATCH request
   */
  async patch<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<AxiosResponse<T>> {
    return this.instance.patch<T>(url, data, config);
  }

  /**
   * DELETE request
   */
  async delete<T = any>(
    url: string,
    config?: AxiosRequestConfig
  ): Promise<AxiosResponse<T>> {
    return this.instance.delete<T>(url, config);
  }

  /**
   * HEAD request (useful for health checks)
   */
  async head(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse> {
    return this.instance.head(url, config);
  }

  /**
   * Upload file(s) with FormData
   */
  async upload<T = any>(
    url: string,
    formData: FormData,
    config?: AxiosRequestConfig
  ): Promise<AxiosResponse<T>> {
    return this.instance.post<T>(url, formData, {
      ...config,
      headers: {
        ...config?.headers,
        "Content-Type": "multipart/form-data",
      },
    });
  }

  /**
   * Create an AbortController for request cancellation
   */
  createAbortController(): AbortController {
    return new AbortController();
  }

  /**
   * Update timeout
   */
  setTimeout(timeout: number): void {
    this.instance.defaults.timeout = timeout;
  }

  // ============================================
  // WebSocket Factory Methods
  // ============================================

  /**
   * Create a WebSocket connection to the specified endpoint
   * @param path - The WebSocket path (e.g., '/api/v1/websocket/ws')
   * @returns WebSocket instance
   */
  createWebSocket(path: string): WebSocket {
    return new WebSocket(this.getWebSocketUrl(path));
  }

  /**
   * Convert HTTP(S) URL to WebSocket URL
   * @param httpUrl - HTTP or HTTPS URL
   * @returns WebSocket URL (ws:// or wss://)
   */
  static httpToWsUrl(httpUrl: string): string {
    return httpUrl.replace(/^http/, "ws");
  }

  /**
   * Get the full WebSocket URL for a given path
   * @param path - The WebSocket path
   * @returns Full WebSocket URL
   */
  getWebSocketUrl(path: string): string {
    return `${this.config.wsBaseURL}${path}`;
  }
}

/**
 * Prefer an explicit API origin for split frontend/backend development. A
 * prebuilt client served by the Rust gateway defaults to its own origin so a
 * bundle can move between a developer machine and the target without baking
 * either machine's address into the JavaScript assets.
 */
export const resolveApiBaseUrl = (
  configuredBaseUrl: string | undefined,
  browserOrigin: string
): string => {
  const configured = configuredBaseUrl?.trim();
  return configured || browserOrigin;
};

const apiBaseUrl = resolveApiBaseUrl(
  import.meta.env.VITE_API_BASE_URL,
  typeof window === "undefined" ? "" : window.location.origin
);

if (!apiBaseUrl) {
  throw new Error(
    "API origin is unavailable; configure VITE_API_BASE_URL when building outside a browser"
  );
}

// Create and export singleton instance
const axiosClient = new AxiosClient({
  baseURL: apiBaseUrl,
  wsBaseURL: apiBaseUrl.replace(/^http/, "ws"),
  timeout: 30000,
  retryAttempts: 3,
  retryDelay: 1000,
  enableLogging: import.meta.env.DEV,
});

export default axiosClient;

// Re-export AxiosError as value (needed for instanceof checks)
export { AxiosError } from "axios";

// Re-export axios types for convenience
export type { AxiosResponse, AxiosRequestConfig } from "axios";
