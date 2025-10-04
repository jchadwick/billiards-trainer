import { reaction } from 'mobx';
import { RootStore } from './RootStore';
import type { PersistedState, AppConfig } from './types';

/**
 * Store persistence system for maintaining state across browser sessions
 */
export class PersistenceManager {
  private rootStore: RootStore;
  private storageKey = 'billiards_app_state';
  private version = '1.0.0';
  private autoSaveDelay = 1000; // 1 second debounce
  private autoSaveTimer: NodeJS.Timeout | null = null;
  private disposers: (() => void)[] = [];

  constructor(rootStore: RootStore) {
    this.rootStore = rootStore;
  }

  /**
   * Initialize persistence system and set up auto-save reactions
   */
  initialize(): void {
    // Load persisted state on startup
    this.loadPersistedState();

    // Set up auto-save reactions for specific state changes
    this.setupAutoSave();

    // Save state on window beforeunload
    if (typeof window !== 'undefined') {
      const handleBeforeUnload = () => {
        this.savePersistedStateSync();
      };
      window.addEventListener('beforeunload', handleBeforeUnload);

      // Store disposal function
      this.disposers.push(() => {
        window.removeEventListener('beforeunload', handleBeforeUnload);
      });
    }
  }

  /**
   * Set up MobX reactions for auto-saving state changes
   */
  private setupAutoSave(): void {
    // Auto-save config changes
    const configDisposer = reaction(
      () => this.rootStore.config.config,
      () => this.scheduleAutoSave(),
      { fireImmediately: false }
    );

    // Auto-save UI preferences
    const uiDisposer = reaction(
      () => this.rootStore.ui.getPersistedUIState(),
      () => this.scheduleAutoSave(),
      { fireImmediately: false }
    );

    // Store disposers for cleanup
    this.disposers.push(configDisposer, uiDisposer);
  }

  /**
   * Schedule auto-save with debouncing to avoid excessive saves
   */
  private scheduleAutoSave(): void {
    if (this.autoSaveTimer) {
      clearTimeout(this.autoSaveTimer);
    }

    this.autoSaveTimer = setTimeout(() => {
      this.savePersistedState();
      this.autoSaveTimer = null;
    }, this.autoSaveDelay);
  }

  /**
   * Load persisted state from localStorage
   */
  loadPersistedState(): boolean {
    try {
      const persistedJson = localStorage.getItem(this.storageKey);
      if (!persistedJson) {
        console.log('No persisted state found');
        return false;
      }

      const persisted: PersistedState = JSON.parse(persistedJson);

      // Validate version compatibility
      if (persisted.version !== this.version) {
        console.warn(
          `Persisted state version mismatch. Expected: ${this.version}, Found: ${persisted.version}`
        );
        this.migrateState(persisted);
        return false;
      }

      // Restore config
      if (persisted.config) {
        this.rootStore.config.importConfig(persisted.config);
        console.log('Restored config from persisted state');
      }

      // Restore UI state
      if (persisted.ui) {
        this.rootStore.ui.restoreUIState(persisted.ui);
        console.log('Restored UI state from persisted state');
      }

      console.log('Successfully loaded persisted state');
      return true;
    } catch (error) {
      console.error('Failed to load persisted state:', error);
      this.clearPersistedState();
      return false;
    }
  }

  /**
   * Save current state to localStorage (async)
   */
  async savePersistedState(): Promise<boolean> {
    try {
      const persistedState = this.gatherPersistedState();
      const serialized = JSON.stringify(persistedState, null, 2);

      localStorage.setItem(this.storageKey, serialized);
      console.log('Persisted state saved successfully');
      return true;
    } catch (error) {
      console.error('Failed to save persisted state:', error);
      this.rootStore.ui.showWarning(
        'Save Failed',
        'Could not save application state to local storage'
      );
      return false;
    }
  }

  /**
   * Save current state to localStorage (synchronous for beforeunload)
   */
  private savePersistedStateSync(): boolean {
    try {
      const persistedState = this.gatherPersistedState();
      const serialized = JSON.stringify(persistedState);

      localStorage.setItem(this.storageKey, serialized);
      return true;
    } catch (error) {
      console.error('Failed to save persisted state synchronously:', error);
      return false;
    }
  }

  /**
   * Gather current state that should be persisted
   */
  private gatherPersistedState(): PersistedState {
    return {
      config: this.rootStore.config.exportConfig(),
      ui: this.rootStore.ui.getPersistedUIState(),
      version: this.version
    };
  }

  /**
   * Clear all persisted state
   */
  clearPersistedState(): void {
    try {
      localStorage.removeItem(this.storageKey);
      console.log('Persisted state cleared');
    } catch (error) {
      console.error('Failed to clear persisted state:', error);
    }
  }

  /**
   * Export current state as JSON string
   */
  exportState(): string {
    const persistedState = this.gatherPersistedState();
    return JSON.stringify(persistedState, null, 2);
  }

  /**
   * Import state from JSON string
   */
  async importState(jsonString: string): Promise<boolean> {
    try {
      const imported: PersistedState = JSON.parse(jsonString);

      // Validate structure
      if (!this.validatePersistedState(imported)) {
        throw new Error('Invalid state structure');
      }

      // Import config
      if (imported.config) {
        const result = await this.rootStore.config.importConfig(imported.config);
        if (!result.success) {
          throw new Error(`Config import failed: ${result.error}`);
        }
      }

      // Import UI state
      if (imported.ui) {
        this.rootStore.ui.restoreUIState(imported.ui);
      }

      // Save the imported state
      await this.savePersistedState();

      this.rootStore.ui.showSuccess(
        'Import Successful',
        'Application state imported successfully'
      );

      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      this.rootStore.ui.showError(
        'Import Failed',
        `Failed to import state: ${message}`
      );
      return false;
    }
  }

  /**
   * Validate persisted state structure
   */
  private validatePersistedState(state: any): state is PersistedState {
    return (
      typeof state === 'object' &&
      state !== null &&
      typeof state.version === 'string' &&
      (state.config === undefined || typeof state.config === 'object') &&
      (state.ui === undefined || typeof state.ui === 'object')
    );
  }

  /**
   * Migrate state from older versions
   */
  private migrateState(oldState: any): void {
    console.log('State migration not implemented for version:', oldState.version);
    // Clear old state to avoid conflicts
    this.clearPersistedState();
  }

  /**
   * Get storage usage information
   */
  getStorageInfo(): {
    used: number;
    available: number;
    percentage: number;
    storageSize: number;
  } {
    try {
      const testKey = 'storage_test';
      const testValue = 'x';
      let available = 0;
      let storageSize = 0;

      // Estimate available storage by trying to store data
      try {
        while (true) {
          localStorage.setItem(testKey, testValue.repeat(available + 1));
          available++;
          if (available > 1000000) break; // Reasonable limit
        }
      } catch {
        // Storage full
      } finally {
        localStorage.removeItem(testKey);
      }

      // Get current usage
      const currentState = this.exportState();
      const used = new Blob([currentState]).size;

      storageSize = available * testValue.length;
      const percentage = storageSize > 0 ? (used / storageSize) * 100 : 0;

      return {
        used,
        available: storageSize - used,
        percentage,
        storageSize
      };
    } catch (error) {
      console.error('Failed to get storage info:', error);
      return {
        used: 0,
        available: 0,
        percentage: 0,
        storageSize: 0
      };
    }
  }

  /**
   * Create a backup of current state
   */
  createBackup(): { name: string; data: string; timestamp: Date } {
    const timestamp = new Date();
    const name = `billiards_backup_${timestamp.toISOString().slice(0, 19).replace(/:/g, '-')}`;
    const data = this.exportState();

    return {
      name,
      data,
      timestamp
    };
  }

  /**
   * Restore from backup
   */
  async restoreFromBackup(backupData: string): Promise<boolean> {
    return this.importState(backupData);
  }

  /**
   * Clean up persistence manager
   */
  destroy(): void {
    // Cancel any pending auto-save
    if (this.autoSaveTimer) {
      clearTimeout(this.autoSaveTimer);
      this.autoSaveTimer = null;
    }

    // Dispose of all MobX reactions
    this.disposers.forEach(dispose => dispose());
    this.disposers = [];

    // Final save
    this.savePersistedStateSync();

    console.log('Persistence manager destroyed');
  }

  /**
   * Force an immediate save (useful for testing or manual triggers)
   */
  forceSave(): Promise<boolean> {
    if (this.autoSaveTimer) {
      clearTimeout(this.autoSaveTimer);
      this.autoSaveTimer = null;
    }
    return this.savePersistedState();
  }
}
