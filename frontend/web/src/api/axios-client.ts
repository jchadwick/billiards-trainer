import axios from 'axios';
import type { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';

export interface ApiClientConfig {
  baseURL: string;
  wsBaseURL: string;
  timeout?: number;
}

export interface ApiErrorResponse {
  detail?: { message?: string };
  message?: string;
  error?: string;
}

/** Small anonymous HTTP/WebSocket address provider used by the live surface. */
export class AxiosClient {
  private readonly instance: AxiosInstance;
  private readonly config: ApiClientConfig;

  constructor(config: ApiClientConfig) {
    this.config = { timeout: 30_000, ...config };
    this.instance = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: { 'Content-Type': 'application/json' },
      // The single-user local gateway has no authentication or session cookies.
      withCredentials: false,
    });
  }

  getInstance(): AxiosInstance {
    return this.instance;
  }

  get<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.get<T>(url, config);
  }

  getWebSocketUrl(path: string): string {
    return `${this.config.wsBaseURL}${path}`;
  }
}

/** A prebuilt bundle follows its serving origin unless a split dev API is set. */
export function resolveApiBaseUrl(
  configuredBaseUrl: string | undefined,
  browserOrigin: string,
): string {
  return configuredBaseUrl?.trim() || browserOrigin;
}

const apiBaseUrl = resolveApiBaseUrl(
  import.meta.env.VITE_API_BASE_URL,
  typeof window === 'undefined' ? '' : window.location.origin,
);

if (!apiBaseUrl) {
  throw new Error(
    'API origin is unavailable; configure VITE_API_BASE_URL when building outside a browser',
  );
}

const axiosClient = new AxiosClient({
  baseURL: apiBaseUrl,
  wsBaseURL: apiBaseUrl.replace(/^http/, 'ws'),
});

export default axiosClient;
export { AxiosError } from 'axios';
export type { AxiosResponse, AxiosRequestConfig } from 'axios';
