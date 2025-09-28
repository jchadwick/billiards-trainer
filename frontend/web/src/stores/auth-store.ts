/**
 * MobX store for authentication state management
 */

import { makeAutoObservable, runInAction, flow } from 'mobx';
import { AuthState, User, AuthError } from '../services/auth-service';
import { LoginRequest } from '../types/api';
import type { RootStore } from './index';

export class AuthStore {
  private rootStore: RootStore;

  // Authentication state
  isAuthenticated = false;
  user: User | null = null;
  isLoading = false;
  error: string | null = null;
  lastLoginAttempt: Date | null = null;

  // Token information
  tokenExpiresAt: Date | null = null;
  isTokenExpiringSoon = false;

  // Session information
  loginTimestamp: Date | null = null;
  lastActivity: Date | null = null;
  sessionTimeoutWarning = false;

  constructor(rootStore: RootStore) {
    makeAutoObservable(this, {}, { autoBind: true });
    this.rootStore = rootStore;
  }

  async initialize(): Promise<void> {
    // Subscribe to auth service state changes
    this.rootStore.apiService.authService.onStateChange(this.handleAuthStateChange);
    this.rootStore.apiService.authService.onError(this.handleAuthError);

    // Get initial auth state
    const initialState = this.rootStore.apiService.authService.getState();
    this.handleAuthStateChange(initialState);
  }

  // =============================================================================
  // Authentication Actions
  // =============================================================================

  login = flow(function* (this: AuthStore, credentials: LoginRequest) {
    this.isLoading = true;
    this.error = null;
    this.lastLoginAttempt = new Date();

    try {
      yield this.rootStore.apiService.login(credentials);

      // Login success - state will be updated via handleAuthStateChange
      runInAction(() => {
        this.isLoading = false;
      });

    } catch (error) {
      runInAction(() => {
        this.isLoading = false;
        this.error = error instanceof Error ? error.message : 'Login failed';
      });
      throw error;
    }
  });

  logout = flow(function* (this: AuthStore) {
    this.isLoading = true;
    this.error = null;

    try {
      yield this.rootStore.apiService.logout();

      // Logout success - state will be updated via handleAuthStateChange
      runInAction(() => {
        this.isLoading = false;
      });

    } catch (error) {
      runInAction(() => {
        this.isLoading = false;
        this.error = error instanceof Error ? error.message : 'Logout failed';
      });
    }
  });

  clearError(): void {
    this.error = null;
  }

  // =============================================================================
  // Permission Helpers
  // =============================================================================

  hasPermission(permission: string): boolean {
    return this.rootStore.apiService.hasPermission(permission);
  }

  hasRole(role: string): boolean {
    return this.user?.role === role;
  }

  get isAdmin(): boolean {
    return this.hasRole('admin');
  }

  get isOperator(): boolean {
    return this.hasRole('operator') || this.isAdmin;
  }

  get isViewer(): boolean {
    return this.hasRole('viewer') || this.isOperator;
  }

  // =============================================================================
  // Computed Properties
  // =============================================================================

  get sessionDuration(): number {
    if (!this.loginTimestamp) return 0;
    return Date.now() - this.loginTimestamp.getTime();
  }

  get timeUntilExpiry(): number {
    if (!this.tokenExpiresAt) return Infinity;
    return Math.max(0, this.tokenExpiresAt.getTime() - Date.now());
  }

  get timeUntilExpiryFormatted(): string {
    const ms = this.timeUntilExpiry;
    if (ms === Infinity) return 'Never';

    const minutes = Math.floor(ms / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ${hours % 24}h`;
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    return `${minutes}m`;
  }

  get sessionInfo(): {
    duration: string;
    lastActivity: string;
    expiresIn: string;
  } {
    const formatDuration = (ms: number): string => {
      const minutes = Math.floor(ms / 60000);
      const hours = Math.floor(minutes / 60);
      if (hours > 0) return `${hours}h ${minutes % 60}m`;
      return `${minutes}m`;
    };

    const formatLastActivity = (): string => {
      if (!this.lastActivity) return 'Unknown';
      const ms = Date.now() - this.lastActivity.getTime();
      const minutes = Math.floor(ms / 60000);
      if (minutes === 0) return 'Just now';
      if (minutes < 60) return `${minutes}m ago`;
      const hours = Math.floor(minutes / 60);
      return `${hours}h ago`;
    };

    return {
      duration: formatDuration(this.sessionDuration),
      lastActivity: formatLastActivity(),
      expiresIn: this.timeUntilExpiryFormatted,
    };
  }

  // =============================================================================
  // Event Handlers
  // =============================================================================

  private handleAuthStateChange = (authState: AuthState): void => {
    runInAction(() => {
      this.isAuthenticated = authState.isAuthenticated;
      this.user = authState.user;
      this.tokenExpiresAt = authState.expiresAt;

      if (authState.user) {
        this.loginTimestamp = authState.user.loginTimestamp;
        this.lastActivity = authState.user.lastActivity;
      } else {
        this.loginTimestamp = null;
        this.lastActivity = null;
      }

      // Update token expiry warning
      this.isTokenExpiringSoon = this.tokenExpiresAt
        ? (this.tokenExpiresAt.getTime() - Date.now()) < 300000 // 5 minutes
        : false;

      // Clear error on successful state change
      if (authState.isAuthenticated && this.error) {
        this.error = null;
      }
    });
  };

  private handleAuthError = (error: AuthError): void => {
    runInAction(() => {
      this.error = error.message;
      this.isLoading = false;
    });
  };

  // =============================================================================
  // Activity Tracking
  // =============================================================================

  recordActivity(): void {
    runInAction(() => {
      this.lastActivity = new Date();
    });

    // Record activity in auth service
    this.rootStore.apiService.authService.recordActivity();
  }

  // =============================================================================
  // Store Lifecycle
  // =============================================================================

  reset(): void {
    this.isAuthenticated = false;
    this.user = null;
    this.isLoading = false;
    this.error = null;
    this.lastLoginAttempt = null;
    this.tokenExpiresAt = null;
    this.isTokenExpiringSoon = false;
    this.loginTimestamp = null;
    this.lastActivity = null;
    this.sessionTimeoutWarning = false;
  }

  destroy(): void {
    // Clean up event handlers
    this.rootStore.apiService.authService.offStateChange(this.handleAuthStateChange);
    this.rootStore.apiService.authService.offError(this.handleAuthError);
  }
}
