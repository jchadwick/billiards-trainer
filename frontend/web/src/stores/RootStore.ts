import { makeAutoObservable } from 'mobx';
import { SystemStore } from './SystemStore';
import { GameStore } from './GameStore';
import { VisionStore } from './VisionStore';
import { ConfigStore } from './ConfigStore';
import { AuthStore } from './AuthStore';
import { UIStore } from './UIStore';
import type { PersistedState } from './types';

export class RootStore {
  // Core stores
  system: SystemStore;
  game: GameStore;
  vision: VisionStore;
  config: ConfigStore;
  auth: AuthStore;
  ui: UIStore;

  // Store initialization state
  isInitialized: boolean = false;
  initializationError: string | null = null;

  constructor() {
    // Initialize all stores
    this.system = new SystemStore();
    this.game = new GameStore();
    this.vision = new VisionStore();
    this.config = new ConfigStore();
    this.auth = new AuthStore();
    this.ui = new UIStore();

    makeAutoObservable(this, {
      system: false,
      game: false,
      vision: false,
      config: false,
      auth: false,
      ui: false
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
    // When system receives messages, route them to appropriate stores
    // This would be expanded based on message types

    // Example: Game updates from backend
    // this.system.onMessage('game_update', (message) => {
    //   this.game.handleGameUpdate(message);
    // });

    // Example: Vision updates from backend
    // this.system.onMessage('detection_frame', (message) => {
    //   this.vision.updateDetectionFrame(message.data);
    // });

    // Config changes should trigger UI theme updates
    // Note: This would be implemented with MobX reactions in a real app

    // Auth changes should trigger system connection updates
    // Note: This would be implemented with MobX reactions in a real app
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
      this.auth = new AuthStore();
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
