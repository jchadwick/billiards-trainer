/**
 * Authentication service with token management and session handling
 */

import { ApiClient, ApiError, isApiError } from './api-client';
import {
  LoginRequest,
  LoginResponse,
  TokenRefreshResponse,
  UserCreateResponse,
} from '../types/api';

export interface User {
  id: string;
  username: string;
  role: string;
  permissions: string[];
  loginTimestamp: Date;
  lastActivity: Date;
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  expiresAt: Date | null;
  isRefreshing: boolean;
  lastError: string | null;
}

export interface AuthConfig {
  autoRefresh?: boolean;
  refreshThreshold?: number; // Refresh token when this many seconds before expiry
  inactivityTimeout?: number; // Auto-logout after this many seconds of inactivity
  persistAuth?: boolean; // Store auth state in localStorage
  storageKey?: string;
}

export type AuthStateChangeHandler = (state: AuthState) => void;
export type AuthErrorHandler = (error: AuthError) => void;

export class AuthError extends Error {
  constructor(
    public code: string,
    message: string,
    public details?: Record<string, any>
  ) {
    super(message);
    this.name = 'AuthError';
  }
}

export class AuthService {
  private apiClient: ApiClient;
  private config: Required<AuthConfig>;
  private state: AuthState;
  private refreshTimer: NodeJS.Timeout | null = null;
  private activityTimer: NodeJS.Timeout | null = null;
  private stateChangeHandlers = new Set<AuthStateChangeHandler>();
  private errorHandlers = new Set<AuthErrorHandler>();

  constructor(apiClient: ApiClient, config: AuthConfig = {}) {
    this.apiClient = apiClient;
    this.config = {
      autoRefresh: true,
      refreshThreshold: 300, // 5 minutes
      inactivityTimeout: 3600, // 1 hour
      persistAuth: true,
      storageKey: 'billiards_auth',
      ...config,
    };

    this.state = {
      isAuthenticated: false,
      user: null,
      accessToken: null,
      refreshToken: null,
      expiresAt: null,
      isRefreshing: false,
      lastError: null,
    };

    this.initializeFromStorage();
    this.setupActivityTracking();
  }

  // =============================================================================
  // Public API
  // =============================================================================

  async login(credentials: LoginRequest): Promise<User> {
    try {
      this.updateState({ lastError: null });

      const response = await this.apiClient.login(credentials);

      if (!response.success || !response.access_token) {
        throw new AuthError('LOGIN_FAILED', 'Login failed - invalid response');
      }

      const user: User = {
        id: response.user_id || 'unknown',
        username: credentials.username,
        role: response.role || 'viewer',
        permissions: response.permissions || [],
        loginTimestamp: new Date(),
        lastActivity: new Date(),
      };

      const expiresAt = response.expires_in
        ? new Date(Date.now() + response.expires_in * 1000)
        : null;

      this.updateState({
        isAuthenticated: true,
        user,
        accessToken: response.access_token,
        refreshToken: response.refresh_token || null,
        expiresAt,
      });

      this.startAutoRefresh();
      this.resetActivityTimer();

      return user;

    } catch (error) {
      const authError = this.createAuthError(error);
      this.updateState({ lastError: authError.message });
      this.handleError(authError);
      throw authError;
    }
  }

  async logout(): Promise<void> {
    try {
      // Attempt to notify the server
      if (this.state.isAuthenticated) {
        await this.apiClient.logout().catch(console.warn);
      }
    } finally {
      this.clearAuth();
    }
  }

  async refreshAccessToken(): Promise<string> {
    if (this.state.isRefreshing) {
      // Wait for existing refresh to complete
      return new Promise((resolve, reject) => {
        const checkRefresh = () => {
          if (!this.state.isRefreshing) {
            if (this.state.accessToken) {
              resolve(this.state.accessToken);
            } else {
              reject(new AuthError('REFRESH_FAILED', 'Token refresh failed'));
            }
          } else {
            setTimeout(checkRefresh, 100);
          }
        };
        checkRefresh();
      });
    }

    if (!this.state.refreshToken) {
      throw new AuthError('NO_REFRESH_TOKEN', 'No refresh token available');
    }

    try {
      this.updateState({ isRefreshing: true, lastError: null });

      // Set refresh token on API client
      this.apiClient.setRefreshToken(this.state.refreshToken);

      const response = await this.apiClient.request<TokenRefreshResponse>('/auth/refresh', {
        method: 'POST',
        body: JSON.stringify({ refresh_token: this.state.refreshToken }),
        skipAuth: true,
      });

      if (!response.success || !response.access_token) {
        throw new AuthError('REFRESH_FAILED', 'Token refresh failed');
      }

      const expiresAt = response.expires_in
        ? new Date(Date.now() + response.expires_in * 1000)
        : null;

      this.updateState({
        accessToken: response.access_token,
        expiresAt,
        isRefreshing: false,
      });

      this.apiClient.setAuthToken(response.access_token);
      this.startAutoRefresh();

      return response.access_token;

    } catch (error) {
      this.updateState({ isRefreshing: false });
      const authError = this.createAuthError(error);

      // If refresh fails, clear auth state
      if (authError.code === 'REFRESH_FAILED' || authError.code === 'AUTH_TOKEN_EXPIRED') {
        this.clearAuth();
      }

      this.handleError(authError);
      throw authError;
    }
  }

  async createUser(userData: {
    username: string;
    password: string;
    role?: string;
  }): Promise<UserCreateResponse> {
    try {
      return await this.apiClient.createUser(userData);
    } catch (error) {
      const authError = this.createAuthError(error);
      this.handleError(authError);
      throw authError;
    }
  }

  // =============================================================================
  // State Management
  // =============================================================================

  getState(): AuthState {
    return { ...this.state };
  }

  getUser(): User | null {
    return this.state.user;
  }

  isAuthenticated(): boolean {
    return this.state.isAuthenticated && !!this.state.accessToken;
  }

  hasPermission(permission: string): boolean {
    if (!this.state.user) return false;
    return this.state.user.permissions.includes(permission) ||
           this.state.user.permissions.includes('*') ||
           this.state.user.role === 'admin';
  }

  hasRole(role: string): boolean {
    return this.state.user?.role === role;
  }

  isTokenExpiringSoon(): boolean {
    if (!this.state.expiresAt) return false;
    const threshold = this.config.refreshThreshold * 1000; // Convert to milliseconds
    return (this.state.expiresAt.getTime() - Date.now()) < threshold;
  }

  getTimeUntilExpiry(): number {
    if (!this.state.expiresAt) return Infinity;
    return Math.max(0, this.state.expiresAt.getTime() - Date.now());
  }

  // =============================================================================
  // Event Handlers
  // =============================================================================

  onStateChange(handler: AuthStateChangeHandler): void {
    this.stateChangeHandlers.add(handler);
  }

  offStateChange(handler: AuthStateChangeHandler): void {
    this.stateChangeHandlers.delete(handler);
  }

  onError(handler: AuthErrorHandler): void {
    this.errorHandlers.add(handler);
  }

  offError(handler: AuthErrorHandler): void {
    this.errorHandlers.delete(handler);
  }

  // =============================================================================
  // Activity Tracking
  // =============================================================================

  recordActivity(): void {
    if (this.state.user) {
      this.state.user.lastActivity = new Date();
      this.resetActivityTimer();
    }
  }

  // =============================================================================
  // Private Methods
  // =============================================================================

  private updateState(updates: Partial<AuthState>): void {
    const previousState = { ...this.state };
    this.state = { ...this.state, ...updates };

    // Update API client token
    if (updates.accessToken !== undefined) {
      if (updates.accessToken) {
        this.apiClient.setAuthToken(updates.accessToken);
      } else {
        this.apiClient.clearAuth();
      }
    }

    // Persist state if enabled
    if (this.config.persistAuth) {
      this.saveToStorage();
    }

    // Notify handlers if state actually changed
    if (JSON.stringify(previousState) !== JSON.stringify(this.state)) {
      this.stateChangeHandlers.forEach(handler => {
        try {
          handler(this.state);
        } catch (error) {
          console.error('Error in auth state change handler:', error);
        }
      });
    }
  }

  private clearAuth(): void {
    this.stopAutoRefresh();
    this.stopActivityTimer();

    this.updateState({
      isAuthenticated: false,
      user: null,
      accessToken: null,
      refreshToken: null,
      expiresAt: null,
      isRefreshing: false,
    });

    if (this.config.persistAuth) {
      this.removeFromStorage();
    }
  }

  private startAutoRefresh(): void {
    this.stopAutoRefresh();

    if (!this.config.autoRefresh || !this.state.expiresAt) {
      return;
    }

    const timeUntilRefresh = this.getTimeUntilExpiry() - (this.config.refreshThreshold * 1000);

    if (timeUntilRefresh > 0) {
      this.refreshTimer = setTimeout(() => {
        this.refreshAccessToken().catch(console.error);
      }, timeUntilRefresh);
    } else {
      // Token is already close to expiry, refresh immediately
      this.refreshAccessToken().catch(console.error);
    }
  }

  private stopAutoRefresh(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
      this.refreshTimer = null;
    }
  }

  private setupActivityTracking(): void {
    // Track mouse and keyboard activity
    const events = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart', 'click'];

    const activityHandler = () => {
      this.recordActivity();
    };

    events.forEach(event => {
      document.addEventListener(event, activityHandler, { passive: true });
    });
  }

  private resetActivityTimer(): void {
    this.stopActivityTimer();

    if (this.config.inactivityTimeout > 0 && this.state.isAuthenticated) {
      this.activityTimer = setTimeout(() => {
        console.log('Session expired due to inactivity');
        this.logout();
      }, this.config.inactivityTimeout * 1000);
    }
  }

  private stopActivityTimer(): void {
    if (this.activityTimer) {
      clearTimeout(this.activityTimer);
      this.activityTimer = null;
    }
  }

  private createAuthError(error: any): AuthError {
    if (error instanceof AuthError) {
      return error;
    }

    if (isApiError(error)) {
      return new AuthError(error.code, error.message, error.details);
    }

    if (error instanceof Error) {
      return new AuthError('UNKNOWN_ERROR', error.message);
    }

    return new AuthError('UNKNOWN_ERROR', 'An unknown authentication error occurred');
  }

  private handleError(error: AuthError): void {
    this.errorHandlers.forEach(handler => {
      try {
        handler(error);
      } catch (err) {
        console.error('Error in auth error handler:', err);
      }
    });
  }

  private initializeFromStorage(): void {
    if (!this.config.persistAuth) return;

    try {
      const stored = localStorage.getItem(this.config.storageKey);
      if (!stored) return;

      const data = JSON.parse(stored);

      // Validate stored data
      if (data.accessToken && data.user) {
        const expiresAt = data.expiresAt ? new Date(data.expiresAt) : null;

        // Check if token is expired
        if (expiresAt && expiresAt <= new Date()) {
          // Token is expired, try to refresh if we have refresh token
          if (data.refreshToken) {
            this.state.refreshToken = data.refreshToken;
            this.refreshAccessToken().catch(() => {
              this.removeFromStorage();
            });
          } else {
            this.removeFromStorage();
          }
          return;
        }

        this.updateState({
          isAuthenticated: true,
          user: {
            ...data.user,
            loginTimestamp: new Date(data.user.loginTimestamp),
            lastActivity: new Date(data.user.lastActivity),
          },
          accessToken: data.accessToken,
          refreshToken: data.refreshToken,
          expiresAt,
        });

        this.startAutoRefresh();
        this.resetActivityTimer();
      }

    } catch (error) {
      console.warn('Failed to load auth state from storage:', error);
      this.removeFromStorage();
    }
  }

  private saveToStorage(): void {
    if (!this.config.persistAuth) return;

    try {
      const data = {
        accessToken: this.state.accessToken,
        refreshToken: this.state.refreshToken,
        expiresAt: this.state.expiresAt?.toISOString(),
        user: this.state.user,
      };

      localStorage.setItem(this.config.storageKey, JSON.stringify(data));
    } catch (error) {
      console.warn('Failed to save auth state to storage:', error);
    }
  }

  private removeFromStorage(): void {
    try {
      localStorage.removeItem(this.config.storageKey);
    } catch (error) {
      console.warn('Failed to remove auth state from storage:', error);
    }
  }

  // =============================================================================
  // Cleanup
  // =============================================================================

  destroy(): void {
    this.stopAutoRefresh();
    this.stopActivityTimer();
    this.stateChangeHandlers.clear();
    this.errorHandlers.clear();
    this.clearAuth();
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createAuthService(apiClient: ApiClient, config?: AuthConfig): AuthService {
  return new AuthService(apiClient, config);
}

// =============================================================================
// Auth Utilities
// =============================================================================

export function parseJwtPayload(token: string): Record<string, any> | null {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    return JSON.parse(jsonPayload);
  } catch (error) {
    console.error('Failed to parse JWT payload:', error);
    return null;
  }
}

export function isTokenExpired(token: string): boolean {
  const payload = parseJwtPayload(token);
  if (!payload || !payload.exp) return true;

  return payload.exp * 1000 <= Date.now();
}

export function getTokenExpiryTime(token: string): Date | null {
  const payload = parseJwtPayload(token);
  if (!payload || !payload.exp) return null;

  return new Date(payload.exp * 1000);
}

// =============================================================================
// Permission Helpers
// =============================================================================

export const PERMISSIONS = {
  // Stream permissions
  STREAM_FRAME: 'stream:frame',
  STREAM_STATE: 'stream:state',
  STREAM_TRAJECTORY: 'stream:trajectory',
  STREAM_ALL: 'stream:*',

  // Control permissions
  CONTROL_BASIC: 'control:basic',
  CONTROL_ADVANCED: 'control:advanced',
  CONTROL_ALL: 'control:*',

  // Configuration permissions
  CONFIG_READ: 'config:read',
  CONFIG_WRITE: 'config:write',
  CONFIG_ALL: 'config:*',

  // Admin permissions
  ADMIN_USERS: 'admin:users',
  ADMIN_SYSTEM: 'admin:system',
  ADMIN_ALL: 'admin:*',
} as const;

export const ROLES = {
  VIEWER: 'viewer',
  OPERATOR: 'operator',
  ADMIN: 'admin',
} as const;

export function getRolePermissions(role: string): string[] {
  switch (role) {
    case ROLES.VIEWER:
      return [PERMISSIONS.STREAM_FRAME, PERMISSIONS.STREAM_STATE];

    case ROLES.OPERATOR:
      return [
        PERMISSIONS.STREAM_ALL,
        PERMISSIONS.CONTROL_BASIC,
        PERMISSIONS.CONFIG_READ,
      ];

    case ROLES.ADMIN:
      return ['*']; // All permissions

    default:
      return [];
  }
}
