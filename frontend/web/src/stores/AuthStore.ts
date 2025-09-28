import { makeAutoObservable, runInAction } from 'mobx';
import type {
  AuthState,
  User,
  UserPreferences,
  ActionResult
} from './types';

interface LoginCredentials {
  username: string;
  password: string;
}

interface RegisterData {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
}

interface TokenRefreshResponse {
  token: string;
  refreshToken: string;
  expiresIn: number;
}

export class AuthStore {
  // Observable state
  authState: AuthState = {
    isAuthenticated: false,
    user: null,
    token: null,
    refreshToken: null,
    expiresAt: null
  };

  // Loading states
  isLoggingIn: boolean = false;
  isRegistering: boolean = false;
  isRefreshingToken: boolean = false;
  isLoggingOut: boolean = false;

  // Error states
  loginError: string | null = null;
  registrationError: string | null = null;

  private refreshTokenTimer: NodeJS.Timeout | null = null;

  constructor() {
    makeAutoObservable(this, {}, { autoBind: true });

    // Check for existing session on initialization
    this.initializeFromStorage();
  }

  // Computed values
  get isAuthenticated(): boolean {
    return this.authState.isAuthenticated && this.authState.user !== null;
  }

  get currentUser(): User | null {
    return this.authState.user;
  }

  get userRoles(): string[] {
    return this.authState.user?.roles || [];
  }

  get userPreferences(): UserPreferences | null {
    return this.authState.user?.preferences || null;
  }

  get isTokenExpired(): boolean {
    if (!this.authState.expiresAt) return true;
    return new Date() >= this.authState.expiresAt;
  }

  get timeUntilExpiry(): number {
    if (!this.authState.expiresAt) return 0;
    return Math.max(0, this.authState.expiresAt.getTime() - Date.now());
  }

  get hasRole(): (role: string) => boolean {
    return (role: string) => this.userRoles.includes(role);
  }

  get isAdmin(): boolean {
    return this.hasRole('admin');
  }

  get canCalibrate(): boolean {
    return this.hasRole('admin') || this.hasRole('calibrator');
  }

  get canManageGames(): boolean {
    return this.hasRole('admin') || this.hasRole('game_manager');
  }

  // Actions
  async login(credentials: LoginCredentials): Promise<ActionResult<User>> {
    runInAction(() => {
      this.isLoggingIn = true;
      this.loginError = null;
    });

    try {
      // Validate credentials
      if (!credentials.username.trim() || !credentials.password.trim()) {
        throw new Error('Username and password are required');
      }

      // Mock API call - in real implementation, this would be an HTTP request
      const response = await this.mockLoginAPI(credentials);

      if (!response.success) {
        throw new Error(response.error || 'Login failed');
      }

      const { user, token, refreshToken, expiresIn } = response.data;
      const expiresAt = new Date(Date.now() + expiresIn * 1000);

      runInAction(() => {
        this.authState = {
          isAuthenticated: true,
          user,
          token,
          refreshToken,
          expiresAt
        };
        this.isLoggingIn = false;
      });

      // Store in localStorage for persistence
      this.saveToStorage();

      // Schedule token refresh
      this.scheduleTokenRefresh();

      return {
        success: true,
        data: user,
        timestamp: new Date()
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Login failed';

      runInAction(() => {
        this.isLoggingIn = false;
        this.loginError = errorMessage;
      });

      return {
        success: false,
        error: errorMessage,
        timestamp: new Date()
      };
    }
  }

  async register(data: RegisterData): Promise<ActionResult<User>> {
    runInAction(() => {
      this.isRegistering = true;
      this.registrationError = null;
    });

    try {
      // Validate registration data
      const validationError = this.validateRegistrationData(data);
      if (validationError) {
        throw new Error(validationError);
      }

      // Mock API call
      const response = await this.mockRegisterAPI(data);

      if (!response.success) {
        throw new Error(response.error || 'Registration failed');
      }

      const { user, token, refreshToken, expiresIn } = response.data;
      const expiresAt = new Date(Date.now() + expiresIn * 1000);

      runInAction(() => {
        this.authState = {
          isAuthenticated: true,
          user,
          token,
          refreshToken,
          expiresAt
        };
        this.isRegistering = false;
      });

      // Store in localStorage for persistence
      this.saveToStorage();

      // Schedule token refresh
      this.scheduleTokenRefresh();

      return {
        success: true,
        data: user,
        timestamp: new Date()
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Registration failed';

      runInAction(() => {
        this.isRegistering = false;
        this.registrationError = errorMessage;
      });

      return {
        success: false,
        error: errorMessage,
        timestamp: new Date()
      };
    }
  }

  async logout(): Promise<ActionResult> {
    runInAction(() => {
      this.isLoggingOut = true;
    });

    try {
      // Cancel token refresh
      this.cancelTokenRefresh();

      // Mock API call to logout (invalidate tokens on server)
      if (this.authState.token) {
        await this.mockLogoutAPI(this.authState.token);
      }

      // Clear state
      runInAction(() => {
        this.authState = {
          isAuthenticated: false,
          user: null,
          token: null,
          refreshToken: null,
          expiresAt: null
        };
        this.isLoggingOut = false;
        this.loginError = null;
        this.registrationError = null;
      });

      // Clear storage
      this.clearStorage();

      return {
        success: true,
        timestamp: new Date()
      };
    } catch (error) {
      runInAction(() => {
        this.isLoggingOut = false;
      });

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Logout failed',
        timestamp: new Date()
      };
    }
  }

  async refreshToken(): Promise<ActionResult> {
    if (!this.authState.refreshToken) {
      return {
        success: false,
        error: 'No refresh token available',
        timestamp: new Date()
      };
    }

    runInAction(() => {
      this.isRefreshingToken = true;
    });

    try {
      const response = await this.mockRefreshTokenAPI(this.authState.refreshToken);

      if (!response.success) {
        throw new Error(response.error || 'Token refresh failed');
      }

      const { token, refreshToken, expiresIn } = response.data;
      const expiresAt = new Date(Date.now() + expiresIn * 1000);

      runInAction(() => {
        this.authState.token = token;
        this.authState.refreshToken = refreshToken;
        this.authState.expiresAt = expiresAt;
        this.isRefreshingToken = false;
      });

      // Update storage
      this.saveToStorage();

      // Schedule next refresh
      this.scheduleTokenRefresh();

      return {
        success: true,
        timestamp: new Date()
      };
    } catch (error) {
      runInAction(() => {
        this.isRefreshingToken = false;
      });

      // If refresh fails, logout user
      await this.logout();

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Token refresh failed',
        timestamp: new Date()
      };
    }
  }

  async updateProfile(updates: Partial<User>): Promise<ActionResult<User>> {
    if (!this.authState.user) {
      return {
        success: false,
        error: 'No authenticated user',
        timestamp: new Date()
      };
    }

    try {
      // Mock API call
      const response = await this.mockUpdateProfileAPI(this.authState.user.id, updates);

      if (!response.success) {
        throw new Error(response.error || 'Profile update failed');
      }

      runInAction(() => {
        if (this.authState.user) {
          Object.assign(this.authState.user, response.data);
        }
      });

      // Update storage
      this.saveToStorage();

      return {
        success: true,
        data: this.authState.user,
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Profile update failed',
        timestamp: new Date()
      };
    }
  }

  async updatePreferences(preferences: Partial<UserPreferences>): Promise<ActionResult<UserPreferences>> {
    if (!this.authState.user) {
      return {
        success: false,
        error: 'No authenticated user',
        timestamp: new Date()
      };
    }

    try {
      const updatedPreferences = { ...this.authState.user.preferences, ...preferences };

      // Mock API call
      const response = await this.mockUpdatePreferencesAPI(this.authState.user.id, updatedPreferences);

      if (!response.success) {
        throw new Error(response.error || 'Preferences update failed');
      }

      runInAction(() => {
        if (this.authState.user) {
          this.authState.user.preferences = response.data;
        }
      });

      // Update storage
      this.saveToStorage();

      return {
        success: true,
        data: response.data,
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Preferences update failed',
        timestamp: new Date()
      };
    }
  }

  clearErrors(): void {
    runInAction(() => {
      this.loginError = null;
      this.registrationError = null;
    });
  }

  // Private methods
  private initializeFromStorage(): void {
    try {
      const stored = localStorage.getItem('billiards_auth');
      if (!stored) return;

      const data = JSON.parse(stored);
      const expiresAt = data.expiresAt ? new Date(data.expiresAt) : null;

      // Check if token is expired
      if (expiresAt && new Date() >= expiresAt) {
        this.clearStorage();
        return;
      }

      runInAction(() => {
        this.authState = {
          ...data,
          expiresAt
        };
      });

      // Schedule token refresh if authenticated
      if (this.authState.isAuthenticated) {
        this.scheduleTokenRefresh();
      }
    } catch (error) {
      console.error('Failed to initialize auth from storage:', error);
      this.clearStorage();
    }
  }

  private saveToStorage(): void {
    try {
      const data = {
        ...this.authState,
        expiresAt: this.authState.expiresAt?.toISOString()
      };
      localStorage.setItem('billiards_auth', JSON.stringify(data));
    } catch (error) {
      console.error('Failed to save auth to storage:', error);
    }
  }

  private clearStorage(): void {
    localStorage.removeItem('billiards_auth');
  }

  private scheduleTokenRefresh(): void {
    this.cancelTokenRefresh();

    if (!this.authState.expiresAt) return;

    // Refresh 5 minutes before expiry
    const refreshTime = this.authState.expiresAt.getTime() - Date.now() - (5 * 60 * 1000);

    if (refreshTime > 0) {
      this.refreshTokenTimer = setTimeout(() => {
        this.refreshToken();
      }, refreshTime);
    } else {
      // Token expires soon, refresh immediately
      this.refreshToken();
    }
  }

  private cancelTokenRefresh(): void {
    if (this.refreshTokenTimer) {
      clearTimeout(this.refreshTokenTimer);
      this.refreshTokenTimer = null;
    }
  }

  private validateRegistrationData(data: RegisterData): string | null {
    if (!data.username.trim()) return 'Username is required';
    if (data.username.length < 3) return 'Username must be at least 3 characters';
    if (!data.email.trim()) return 'Email is required';
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(data.email)) return 'Invalid email format';
    if (!data.password) return 'Password is required';
    if (data.password.length < 6) return 'Password must be at least 6 characters';
    if (data.password !== data.confirmPassword) return 'Passwords do not match';
    return null;
  }

  // Mock API methods (in real implementation, these would be actual HTTP calls)
  private async mockLoginAPI(credentials: LoginCredentials): Promise<any> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Mock user data
    if (credentials.username === 'admin' && credentials.password === 'admin') {
      return {
        success: true,
        data: {
          user: {
            id: '1',
            username: 'admin',
            email: 'admin@billiards-trainer.com',
            roles: ['admin', 'calibrator', 'game_manager'],
            preferences: {
              theme: 'dark' as const,
              language: 'en',
              notifications: {
                gameEvents: true,
                systemAlerts: true,
                achievements: true
              }
            },
            lastLogin: new Date()
          },
          token: 'mock-jwt-token-' + Date.now(),
          refreshToken: 'mock-refresh-token-' + Date.now(),
          expiresIn: 3600 // 1 hour
        }
      };
    } else if (credentials.username === 'user' && credentials.password === 'user') {
      return {
        success: true,
        data: {
          user: {
            id: '2',
            username: 'user',
            email: 'user@billiards-trainer.com',
            roles: ['user'],
            preferences: {
              theme: 'light' as const,
              language: 'en',
              notifications: {
                gameEvents: true,
                systemAlerts: false,
                achievements: true
              }
            },
            lastLogin: new Date()
          },
          token: 'mock-jwt-token-' + Date.now(),
          refreshToken: 'mock-refresh-token-' + Date.now(),
          expiresIn: 3600
        }
      };
    }

    return {
      success: false,
      error: 'Invalid credentials'
    };
  }

  private async mockRegisterAPI(data: RegisterData): Promise<any> {
    await new Promise(resolve => setTimeout(resolve, 1200));

    // Check if username already exists (mock)
    if (data.username === 'admin' || data.username === 'user') {
      return {
        success: false,
        error: 'Username already exists'
      };
    }

    return {
      success: true,
      data: {
        user: {
          id: 'new-user-' + Date.now(),
          username: data.username,
          email: data.email,
          roles: ['user'],
          preferences: {
            theme: 'auto' as const,
            language: 'en',
            notifications: {
              gameEvents: true,
              systemAlerts: true,
              achievements: true
            }
          },
          lastLogin: new Date()
        },
        token: 'mock-jwt-token-' + Date.now(),
        refreshToken: 'mock-refresh-token-' + Date.now(),
        expiresIn: 3600
      }
    };
  }

  private async mockLogoutAPI(token: string): Promise<any> {
    await new Promise(resolve => setTimeout(resolve, 500));
    return { success: true };
  }

  private async mockRefreshTokenAPI(refreshToken: string): Promise<any> {
    await new Promise(resolve => setTimeout(resolve, 500));
    return {
      success: true,
      data: {
        token: 'mock-jwt-token-refreshed-' + Date.now(),
        refreshToken: 'mock-refresh-token-refreshed-' + Date.now(),
        expiresIn: 3600
      }
    };
  }

  private async mockUpdateProfileAPI(userId: string, updates: Partial<User>): Promise<any> {
    await new Promise(resolve => setTimeout(resolve, 800));
    return {
      success: true,
      data: updates
    };
  }

  private async mockUpdatePreferencesAPI(userId: string, preferences: UserPreferences): Promise<any> {
    await new Promise(resolve => setTimeout(resolve, 600));
    return {
      success: true,
      data: preferences
    };
  }

  // Cleanup
  destroy(): void {
    this.cancelTokenRefresh();
    this.clearStorage();
  }
}
