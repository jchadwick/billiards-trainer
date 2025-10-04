import { configure, trace, spy } from 'mobx';
import { RootStore } from './RootStore';

/**
 * Development tools and debugging utilities for MobX stores
 */
export class DevTools {
  private rootStore: RootStore;
  private spyDisposer: (() => void) | null = null;
  private isEnabled: boolean = false;

  constructor(rootStore: RootStore) {
    this.rootStore = rootStore;
  }

  /**
   * Initialize development tools
   */
  initialize(): void {
    if (process.env.NODE_ENV !== 'development') {
      return;
    }

    this.isEnabled = true;

    // Configure MobX for development
    configure({
      enforceActions: 'never', // More lenient for development
      computedRequiresReaction: false,
      reactionRequiresObservable: false,
      observableRequiresReaction: false,
      disableErrorBoundaries: true
    });

    // Set up global debugging methods
    this.setupGlobalDebugging();

    // Set up MobX spy for action logging
    this.setupMobXSpy();

    console.log('🔧 MobX DevTools initialized');
    console.log('Available commands:');
    console.log('  - __MOBX_DEBUG__.inspect() - Inspect all store states');
    console.log('  - __MOBX_DEBUG__.reset() - Reset all stores');
    console.log('  - __MOBX_DEBUG__.export() - Export current state');
    console.log('  - __MOBX_DEBUG__.import(state) - Import state');
    console.log('  - __MOBX_DEBUG__.trace() - Start tracing computations');
    console.log('  - __MOBX_DEBUG__.stopTrace() - Stop tracing');
  }

  /**
   * Set up global debugging interface
   */
  private setupGlobalDebugging(): void {
    const globalDebug = {
      // Store inspection
      inspect: () => this.inspectStores(),
      stores: this.rootStore.allStores,
      root: this.rootStore,

      // State management
      reset: () => this.resetAllStores(),
      export: () => this.exportState(),
      import: (state: any) => this.importState(state),

      // Performance monitoring
      performance: () => this.getPerformanceInfo(),
      memory: () => this.getMemoryInfo(),

      // Debugging tools
      trace: () => this.startTracing(),
      stopTrace: () => this.stopTracing(),
      spy: () => this.startSpying(),
      stopSpy: () => this.stopSpying(),

      // Utility functions
      simulate: {
        gameStart: () => this.simulateGameStart(),
        detection: () => this.simulateDetection(),
        error: () => this.simulateError(),
        notification: () => this.simulateNotification()
      },

      // Store-specific helpers
      system: {
        connect: () => this.rootStore.system.connect('ws://localhost:8080/ws'),
        disconnect: () => this.rootStore.system.disconnect(),
        addError: (msg: string) => this.rootStore.system.addError('DevTools', msg),
        clearErrors: () => this.rootStore.system.clearErrors()
      },

      game: {
        start: () => this.rootStore.game.startNewGame('practice', [{
          name: 'Player 1',
          ballGroup: null,
          score: 0,
          isActive: true
        }]),
        reset: () => this.rootStore.game.resetTable(),
        addBall: (id: number, x: number, y: number) =>
          this.rootStore.game.setBallPosition(id, { x, y })
      },

      vision: {
        startDetection: () => this.rootStore.vision.startDetection(),
        stopDetection: () => this.rootStore.vision.stopDetection(),
        calibrate: () => this.rootStore.vision.startCalibration()
      },

      ui: {
        showSuccess: (msg: string) => this.rootStore.ui.showSuccess('DevTools', msg),
        showError: (msg: string) => this.rootStore.ui.showError('DevTools', msg),
        openModal: (modal: string) => this.rootStore.ui.openModal(modal as any),
        closeAllModals: () => this.rootStore.ui.closeAllModals()
      }
    };

    // Make available globally
    (window as any).__MOBX_DEBUG__ = globalDebug;
    (window as any).__STORES__ = this.rootStore.allStores;
  }

  /**
   * Set up MobX spy for action logging
   */
  private setupMobXSpy(): void {
    if (this.spyDisposer) {
      this.spyDisposer();
    }

    this.spyDisposer = spy((event) => {
      if (event.type === 'action') {
        console.group(`🎯 Action: ${event.name}`);
        console.log('Object:', event.object);
        if (event.arguments && event.arguments.length > 0) {
          console.log('Arguments:', event.arguments);
        }
        console.groupEnd();
      } else if (event.type === 'reaction') {
        console.log(`⚡ Reaction: ${event.name || 'anonymous'}`);
      }
    });
  }

  /**
   * Inspect all store states
   */
  inspectStores(): any {
    const inspection = {
      system: {
        status: this.rootStore.system.status,
        isHealthy: this.rootStore.system.isHealthy,
        errors: this.rootStore.system.status.errors.length
      },
      game: {
        isActive: this.rootStore.game.gameState.isActive,
        ballCount: this.rootStore.game.balls.length,
        activeBalls: this.rootStore.game.activeBalls.length,
        gameType: this.rootStore.game.gameState.gameType,
        shotCount: this.rootStore.game.gameState.shotCount
      },
      vision: {
        isConnected: this.rootStore.vision.isConnected,
        isCalibrated: this.rootStore.vision.isCalibrated,
        isDetecting: this.rootStore.vision.isDetecting,
        selectedCamera: this.rootStore.vision.selectedCamera?.name,
        frameCount: this.rootStore.vision.frameCount
      },
      config: {
        currentProfile: this.rootStore.config.currentProfile,
        isValid: this.rootStore.config.isValid,
        hasUnsavedChanges: this.rootStore.config.hasUnsavedChanges,
        profiles: this.rootStore.config.availableProfiles
      },
      ui: {
        activeTab: this.rootStore.ui.uiState.activeTab,
        modalsOpen: Object.entries(this.rootStore.ui.uiState.modals)
          .filter(([, isOpen]) => isOpen)
          .map(([name]) => name),
        notifications: this.rootStore.ui.unreadNotificationCount,
        isLoading: this.rootStore.ui.isAnyLoading,
        viewport: {
          width: this.rootStore.ui.windowWidth,
          height: this.rootStore.ui.windowHeight,
          isMobile: this.rootStore.ui.isMobile
        }
      }
    };

    console.table(inspection);
    return inspection;
  }

  /**
   * Reset all stores to initial state
   */
  async resetAllStores(): Promise<void> {
    console.log('🔄 Resetting all stores...');
    await this.rootStore.reset();
    console.log('✅ All stores reset');
  }

  /**
   * Export current state as JSON
   */
  exportState(): string {
    const state = this.rootStore.savePersistedState();
    const json = JSON.stringify(state, null, 2);
    console.log('📦 Exported state:');
    console.log(json);
    return json;
  }

  /**
   * Import state from JSON
   */
  async importState(stateJson: string): Promise<void> {
    try {
      const state = JSON.parse(stateJson);
      await this.rootStore.loadPersistedState();
      console.log('📥 State imported successfully');
    } catch (error) {
      console.error('❌ Failed to import state:', error);
    }
  }

  /**
   * Get performance information
   */
  getPerformanceInfo(): any {
    const info = {
      detectionRate: this.rootStore.vision.detectionRate,
      averageProcessingTime: this.rootStore.vision.averageProcessingTime,
      frameCount: this.rootStore.vision.frameCount,
      systemUptime: this.rootStore.system.connectionUptime,
      memoryUsage: this.getMemoryInfo(),
      storeCounts: {
        systemErrors: this.rootStore.system.status.errors.length,
        notifications: this.rootStore.ui.uiState.notifications.length,
        shotHistory: this.rootStore.game.shotHistory.length,
        balls: this.rootStore.game.balls.length
      }
    };

    console.table(info);
    return info;
  }

  /**
   * Get memory usage information
   */
  getMemoryInfo(): any {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      return {
        usedJSHeapSize: Math.round(memory.usedJSHeapSize / 1024 / 1024) + ' MB',
        totalJSHeapSize: Math.round(memory.totalJSHeapSize / 1024 / 1024) + ' MB',
        jsHeapSizeLimit: Math.round(memory.jsHeapSizeLimit / 1024 / 1024) + ' MB'
      };
    }
    return { message: 'Memory info not available' };
  }

  /**
   * Start tracing MobX computations
   */
  startTracing(): void {
    console.log('🔍 Starting MobX tracing...');
    console.log('Call trace() within your component to see dependencies');
    (window as any).__MOBX_TRACE__ = trace;
  }

  /**
   * Stop tracing MobX computations
   */
  stopTracing(): void {
    console.log('🛑 Stopping MobX tracing...');
    delete (window as any).__MOBX_TRACE__;
  }

  /**
   * Start spying on MobX events
   */
  startSpying(): void {
    if (!this.spyDisposer) {
      this.setupMobXSpy();
      console.log('👁️ Started spying on MobX events');
    }
  }

  /**
   * Stop spying on MobX events
   */
  stopSpying(): void {
    if (this.spyDisposer) {
      this.spyDisposer();
      this.spyDisposer = null;
      console.log('🛑 Stopped spying on MobX events');
    }
  }

  // Simulation methods for testing
  private async simulateGameStart(): Promise<void> {
    console.log('🧪 Simulating game start...');
    await this.rootStore.game.startNewGame('eightball', [
      { name: 'Player 1', ballGroup: null, score: 0, isActive: true },
      { name: 'Player 2', ballGroup: null, score: 0, isActive: false }
    ]);
  }

  private simulateDetection(): void {
    console.log('🧪 Simulating detection frame...');
    const mockFrame = {
      timestamp: new Date(),
      frameNumber: Math.floor(Math.random() * 1000),
      balls: this.rootStore.game.balls.map(ball => ({
        ...ball,
        confidence: Math.random() * 0.3 + 0.7
      })),
      cue: null,
      confidence: Math.random() * 0.3 + 0.7,
      processingTimeMs: Math.random() * 50 + 10
    };
    this.rootStore.vision.updateDetectionFrame(mockFrame);
  }

  private simulateError(): void {
    console.log('🧪 Simulating system error...');
    this.rootStore.system.addError('DevTools', 'Simulated error for testing');
  }

  private simulateNotification(): void {
    console.log('🧪 Simulating notification...');
    const types = ['info', 'success', 'warning', 'error'] as const;
    const type = types[Math.floor(Math.random() * types.length)];
    this.rootStore.ui.showNotification(
      type,
      'Test Notification',
      'This is a simulated notification for testing purposes'
    );
  }

  /**
   * Cleanup development tools
   */
  destroy(): void {
    if (this.spyDisposer) {
      this.spyDisposer();
    }

    // Clean up global objects
    delete (window as any).__MOBX_DEBUG__;
    delete (window as any).__STORES__;
    delete (window as any).__MOBX_TRACE__;

    console.log('🔧 DevTools destroyed');
  }
}

/**
 * Initialize development tools if in development mode
 */
export function initializeDevTools(rootStore: RootStore): DevTools | null {
  if (process.env.NODE_ENV === 'development') {
    const devTools = new DevTools(rootStore);
    devTools.initialize();
    return devTools;
  }
  return null;
}
