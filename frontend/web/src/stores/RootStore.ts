import { makeAutoObservable } from 'mobx';
import { SystemStore } from './SystemStore';
import { GameStore } from './GameStore';
import { VisionStore } from './VisionStore';
import { ConfigStore } from './ConfigStore';
import { AuthStore } from './AuthStore';
import { UIStore } from './UIStore';
import { ConnectionStore } from './ConnectionStore';
import { CalibrationStore } from './calibration-store';
import { VideoStore } from './VideoStore';
import { ApiService, createApiService } from '../services/api-service';
import type { PersistedState } from './types';
import type { CalibrationStartRequest, CalibrationStartResponse, CalibrationPointResponse, CalibrationApplyResponse } from '../types/api';

export class RootStore {
  // Core stores
  system: SystemStore;
  game: GameStore;
  vision: VisionStore;
  config: ConfigStore;
  auth: AuthStore;
  ui: UIStore;
  connection: ConnectionStore;
  calibrationStore: CalibrationStore;
  videoStore: VideoStore;

  // Real API Service for calibration and other backend operations
  apiService: ApiService;

  // Store initialization state
  isInitialized: boolean = false;
  initializationError: string | null = null;

  constructor() {
    // Initialize API service first as other stores may depend on it
    this.apiService = createApiService({
      apiBaseUrl: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8080/api/v1',
      wsBaseUrl: process.env.REACT_APP_WS_BASE_URL || 'ws://localhost:8080/ws',
      enableCaching: true,
      enableRequestDeduplication: true,
      autoConnectWebSocket: true,
      defaultStreamSubscriptions: ['game_state', 'detection_frame', 'system_alerts']
    });

    // Initialize all stores
    this.system = new SystemStore();
    this.game = new GameStore();
    this.vision = new VisionStore();
    this.config = new ConfigStore();
    this.auth = new AuthStore(this);
    this.ui = new UIStore();
    this.connection = new ConnectionStore();
    this.calibrationStore = new CalibrationStore(this);
    this.videoStore = new VideoStore();

    makeAutoObservable(this, {
      system: false,
      game: false,
      vision: false,
      config: false,
      auth: false,
      ui: false,
      connection: false,
      calibrationStore: false,
      videoStore: false
    });

    // Initialize the root store
    this.initialize();
  }

  // Computed values
  get isHealthy(): boolean {
    return this.system.isHealthy &&
           this.isInitialized &&
           !this.initializationError;
  }

  get allStores() {
    return {
      system: this.system,
      game: this.game,
      vision: this.vision,
      config: this.config,
      auth: this.auth,
      ui: this.ui
    };
  }

  // Initialization
  async initialize(): Promise<void> {
    try {
      this.ui.setGlobalLoading(true);

      // Load persisted state
      await this.loadPersistedState();

      // Initialize stores in dependency order
      await this.initializeStores();

      // Set up store connections and event handlers
      this.setupStoreConnections();

      this.isInitialized = true;
      this.ui.showInfo('System Ready', 'Billiards trainer initialized successfully');
    } catch (error) {
      this.initializationError = error instanceof Error ? error.message : 'Unknown initialization error';
      this.ui.showError('Initialization Failed', this.initializationError);
      console.error('Failed to initialize root store:', error);
    } finally {
      this.ui.setGlobalLoading(false);
    }
  }

  private async initializeStores(): Promise<void> {
    // Apply theme from config
    const theme = this.config.theme;
    this.ui.applyTheme(theme);

    // Connect to backend system if configured
    const websocketUrl = 'ws://localhost:8080/ws'; // This should come from config
    if (this.auth.isAuthenticated) {
      await this.system.connect(websocketUrl);
    }
  }

  private setupStoreConnections(): void {
    // Set up WebSocket message routing
    this.setupWebSocketMessageHandlers();

    // Set up MobX reactions for cross-store dependencies
    this.setupMobXReactions();
  }

  private setupWebSocketMessageHandlers(): void {
    // Set up message routing by extending the system store's message handling
    // We'll add a custom message router that hooks into the WebSocket connection

    // Add a message router function to handle cross-store communication
    this.system.addMessageRouter = (message: any) => {
      try {
        // Route messages to appropriate stores based on type
        switch (message.type) {
          case 'game_update':
            if (this.game.handleGameUpdate) {
              this.game.handleGameUpdate(message);
            }
            break;

          case 'detection_frame':
            if (this.vision.updateDetectionFrame) {
              this.vision.updateDetectionFrame(message.data);
            }
            // Also update video store with frame data if it has setCurrentFrame method
            if ('setCurrentFrame' in this.vision) {
              const videoFrame = {
                ...message.data,
                timestamp: Date.now()
              };
              (this.vision as any).setCurrentFrame(videoFrame);
            }
            break;

          case 'config_update':
            // Reload configuration when backend config changes
            if (this.config.loadFromBackend) {
              this.config.loadFromBackend().catch(console.error);
            }
            break;

          case 'system_alert':
            this.ui.showNotification(
              message.data.level || 'info',
              'System Alert',
              message.data.message,
              { autoHide: true, duration: 5000 }
            );
            break;

          case 'vision_status':
            if (message.data.camera_status) {
              // Update vision store camera status
              if ('updateCameraStatus' in this.vision) {
                (this.vision as any).updateCameraStatus(message.data.camera_status);
              }
            }
            break;

          default:
            // Unknown message types are handled by SystemStore
            break;
        }
      } catch (error) {
        console.error('Error routing WebSocket message:', error);
      }
    };
  }

  private setupMobXReactions(): void {
    // Import reaction from mobx
    import('mobx').then(({ reaction }) => {
      // React to theme changes and apply them to UI
      reaction(
        () => this.config.theme,
        (theme) => {
          this.ui.applyTheme(theme);
        }
      );

      // React to auth state changes and manage system connections
      reaction(
        () => this.auth.isAuthenticated,
        (isAuthenticated) => {
          if (isAuthenticated && !this.system.status.isConnected) {
            // Auto-connect when authenticated
            const websocketUrl = 'ws://localhost:8080/api/v1/ws';
            this.system.connect(websocketUrl).catch(error => {
              console.error('Auto-connect failed:', error);
            });
          } else if (!isAuthenticated && this.system.status.isConnected) {
            // Disconnect when logged out
            this.system.disconnect();
          }
        }
      );

      // React to config changes and mark them as dirty
      reaction(
        () => [
          this.config.config.camera,
          this.config.config.detection,
          this.config.config.game,
          this.config.config.ui
        ],
        () => {
          // Auto-save configuration changes to backend after a delay
          this.debounceConfigSave();
        }
      );

      // React to system errors and show notifications
      reaction(
        () => this.system.status.errors.length,
        () => {
          const latestError = this.system.status.errors[this.system.status.errors.length - 1];
          if (latestError && latestError.level === 'critical') {
            this.ui.showError('Critical System Error', latestError.message);
          }
        }
      );
    }).catch(console.error);
  }

  private configSaveTimeout: NodeJS.Timeout | null = null;

  private debounceConfigSave(): void {
    if (this.configSaveTimeout) {
      clearTimeout(this.configSaveTimeout);
    }

    this.configSaveTimeout = setTimeout(() => {
      if (this.config.hasUnsavedChanges && this.config.isValid) {
        this.config.saveToBackend().catch(error => {
          console.error('Auto-save configuration failed:', error);
        });
      }
    }, 2000); // Save after 2 seconds of no changes
  }

  // State persistence
  async loadPersistedState(): Promise<void> {
    try {
      const persistedJson = localStorage.getItem('billiards_app_state');
      if (!persistedJson) return;

      const persisted: PersistedState = JSON.parse(persistedJson);

      // Validate version compatibility
      if (persisted.version !== '1.0.0') {
        console.warn('Persisted state version mismatch, using defaults');
        return;
      }

      // Restore state to stores
      if (persisted.config) {
        await this.config.importConfig(persisted.config);
      }

      if (persisted.ui) {
        this.ui.restoreUIState(persisted.ui);
      }

      // Note: Auth tokens are restored automatically by AuthStore constructor
    } catch (error) {
      console.error('Failed to load persisted state:', error);
      // Clear corrupted state
      localStorage.removeItem('billiards_app_state');
    }
  }

  async savePersistedState(): Promise<void> {
    try {
      const persistedState: PersistedState = {
        config: this.config.exportConfig(),
        auth: {
          token: this.auth.authState.token,
          refreshToken: this.auth.authState.refreshToken,
          expiresAt: this.auth.authState.expiresAt
        },
        ui: this.ui.getPersistedUIState(),
        version: '1.0.0'
      };

      localStorage.setItem('billiards_app_state', JSON.stringify(persistedState));
    } catch (error) {
      console.error('Failed to save persisted state:', error);
      this.ui.showWarning('Save Failed', 'Could not save application state');
    }
  }

  // Cleanup and shutdown
  async shutdown(): Promise<void> {
    try {
      // Save current state
      await this.savePersistedState();

      // Cleanup all stores
      this.system.destroy();
      this.vision.destroy();
      this.auth.destroy();
      this.ui.destroy();

      this.ui.showInfo('Shutdown', 'Application shutdown complete');
    } catch (error) {
      console.error('Error during shutdown:', error);
    }
  }

  // Development helpers
  getDebugInfo() {
    return {
      initialized: this.isInitialized,
      healthy: this.isHealthy,
      stores: {
        system: {
          connected: this.system.status.isConnected,
          errors: this.system.status.errors.length
        },
        game: {
          active: this.game.gameState.isActive,
          ballCount: this.game.balls.length
        },
        vision: {
          connected: this.vision.isConnected,
          calibrated: this.vision.isCalibrated,
          detecting: this.vision.isDetecting
        },
        config: {
          profile: this.config.currentProfile,
          valid: this.config.isValid
        },
        auth: {
          authenticated: this.auth.isAuthenticated,
          user: this.auth.currentUser?.username
        },
        ui: {
          loading: this.ui.isAnyLoading,
          notifications: this.ui.unreadNotificationCount,
          modals: this.ui.isAnyModalOpen
        }
      }
    };
  }

  // Reset application state (for development/testing)
  async reset(): Promise<void> {
    await this.ui.withLoading('global', async () => {
      // Clear persisted state
      localStorage.clear();

      // Reinitialize all stores
      Object.values(this.allStores).forEach(store => {
        if ('destroy' in store && typeof store.destroy === 'function') {
          store.destroy();
        }
      });

      // Recreate stores
      this.system = new SystemStore();
      this.game = new GameStore();
      this.vision = new VisionStore();
      this.config = new ConfigStore();
      this.auth = new AuthStore(this);
      this.ui = new UIStore();

      this.isInitialized = false;
      this.initializationError = null;

      // Reinitialize
      await this.initialize();
    });
  }
}

// Create singleton instance
let rootStoreInstance: RootStore | null = null;

export function getRootStore(): RootStore {
  if (!rootStoreInstance) {
    rootStoreInstance = new RootStore();
  }
  return rootStoreInstance;
}

// Export singleton instance
export const rootStore = getRootStore();

// Export individual stores for convenience
export const systemStore = rootStore.system;
export const gameStore = rootStore.game;
export const visionStore = rootStore.vision;
export const configStore = rootStore.config;
export const authStore = rootStore.auth;
export const uiStore = rootStore.ui;
