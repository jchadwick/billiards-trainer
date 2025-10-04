import React, { createContext, useContext, ReactNode } from 'react';
import { observer } from 'mobx-react-lite';
import { RootStore, getRootStore } from './RootStore';
import { SystemStore } from './SystemStore';
import { GameStore } from './GameStore';
import { VisionStore } from './VisionStore';
import { ConfigStore } from './ConfigStore';
import { UIStore } from './UIStore';

// Create contexts
const RootStoreContext = createContext<RootStore | null>(null);

// Store provider component
interface StoreProviderProps {
  children: ReactNode;
  store?: RootStore;
}

export const StoreProvider: React.FC<StoreProviderProps> = ({ children, store }) => {
  const rootStore = store || getRootStore();

  return (
    <RootStoreContext.Provider value={rootStore}>
      {children}
    </RootStoreContext.Provider>
  );
};

// Root store hook
export function useRootStore(): RootStore {
  const context = useContext(RootStoreContext);
  if (!context) {
    throw new Error('useRootStore must be used within a StoreProvider');
  }
  return context;
}

// All stores hook for convenience
export function useStores() {
  const rootStore = useRootStore();
  return {
    rootStore,
    systemStore: rootStore.system,
    gameStore: rootStore.game,
    visionStore: rootStore.vision,
    configStore: rootStore.config,
    uiStore: rootStore.ui,
    connectionStore: rootStore.connection,
  };
}

// Individual store hooks
export function useSystemStore(): SystemStore {
  const rootStore = useRootStore();
  return rootStore.system;
}

export function useGameStore(): GameStore {
  const rootStore = useRootStore();
  return rootStore.game;
}

export function useVisionStore(): VisionStore {
  const rootStore = useRootStore();
  return rootStore.vision;
}

export function useConfigStore(): ConfigStore {
  const rootStore = useRootStore();
  return rootStore.config;
}

export function useUIStore(): UIStore {
  const rootStore = useRootStore();
  return rootStore.ui;
}

// Specialized hooks for common use cases
export function useNotifications() {
  const uiStore = useUIStore();
  return {
    notifications: uiStore.uiState.notifications,
    unreadCount: uiStore.unreadNotificationCount,
    showSuccess: uiStore.showSuccess.bind(uiStore),
    showError: uiStore.showError.bind(uiStore),
    showWarning: uiStore.showWarning.bind(uiStore),
    showInfo: uiStore.showInfo.bind(uiStore),
    markAsRead: uiStore.markNotificationAsRead.bind(uiStore),
    markAllAsRead: uiStore.markAllNotificationsAsRead.bind(uiStore),
    remove: uiStore.removeNotification.bind(uiStore),
    clearAll: uiStore.clearAllNotifications.bind(uiStore)
  };
}

export function useModals() {
  const uiStore = useUIStore();
  return {
    modals: uiStore.uiState.modals,
    isAnyOpen: uiStore.isAnyModalOpen,
    open: uiStore.openModal.bind(uiStore),
    close: uiStore.closeModal.bind(uiStore),
    closeAll: uiStore.closeAllModals.bind(uiStore),
    toggle: uiStore.toggleModal.bind(uiStore)
  };
}

export function useLoading() {
  const uiStore = useUIStore();
  return {
    loading: uiStore.uiState.loading,
    isGlobalLoading: uiStore.isGlobalLoading,
    isAnyLoading: uiStore.isAnyLoading,
    setLoading: uiStore.setLoading.bind(uiStore),
    setGlobalLoading: uiStore.setGlobalLoading.bind(uiStore),
    withLoading: uiStore.withLoading.bind(uiStore)
  };
}

export function useSystemStatus() {
  const systemStore = useSystemStore();
  return {
    status: systemStore.status,
    isHealthy: systemStore.isHealthy,
    isConnected: systemStore.status.isConnected,
    errors: systemStore.status.errors,
    criticalErrors: systemStore.criticalErrors,
    connect: systemStore.connect.bind(systemStore),
    disconnect: systemStore.disconnect.bind(systemStore),
    clearErrors: systemStore.clearErrors.bind(systemStore)
  };
}

export function useGameState() {
  const gameStore = useGameStore();
  return {
    gameState: gameStore.gameState,
    balls: gameStore.balls,
    cue: gameStore.cue,
    table: gameStore.table,
    activeBalls: gameStore.activeBalls,
    cueBall: gameStore.cueBall,
    isGameActive: gameStore.gameState.isActive,
    isGameOver: gameStore.isGameOver,
    currentPlayer: gameStore.currentPlayer,
    shotHistory: gameStore.shotHistory,
    startNewGame: gameStore.startNewGame.bind(gameStore),
    pauseGame: gameStore.pauseGame.bind(gameStore),
    resumeGame: gameStore.resumeGame.bind(gameStore),
    endGame: gameStore.endGame.bind(gameStore),
    resetTable: gameStore.resetTable.bind(gameStore)
  };
}

export function useVision() {
  const visionStore = useVisionStore();
  return {
    cameras: visionStore.availableCameras,
    selectedCamera: visionStore.selectedCamera,
    isConnected: visionStore.isConnected,
    isCalibrated: visionStore.isCalibrated,
    isDetecting: visionStore.isDetecting,
    calibrationData: visionStore.calibrationData,
    currentFrame: visionStore.currentFrame,
    detectionRate: visionStore.detectionRate,
    discoverCameras: visionStore.discoverCameras.bind(visionStore),
    selectCamera: visionStore.selectCamera.bind(visionStore),
    startStream: visionStore.startStream.bind(visionStore),
    stopStream: visionStore.stopStream.bind(visionStore),
    startDetection: visionStore.startDetection.bind(visionStore),
    stopDetection: visionStore.stopDetection.bind(visionStore),
    startCalibration: visionStore.startCalibration.bind(visionStore),
    cancelCalibration: visionStore.cancelCalibration.bind(visionStore)
  };
}

export function useConfig() {
  const configStore = useConfigStore();
  return {
    config: configStore.config,
    profiles: configStore.availableProfiles,
    currentProfile: configStore.currentProfile,
    hasUnsavedChanges: configStore.hasUnsavedChanges,
    isValid: configStore.isValid,
    validationErrors: configStore.validationErrors,
    updateCameraConfig: configStore.updateCameraConfig.bind(configStore),
    updateDetectionConfig: configStore.updateDetectionConfig.bind(configStore),
    updateGameConfig: configStore.updateGameConfig.bind(configStore),
    updateUIConfig: configStore.updateUIConfig.bind(configStore),
    saveProfile: configStore.saveProfile.bind(configStore),
    loadProfile: configStore.loadProfile.bind(configStore),
    deleteProfile: configStore.deleteProfile.bind(configStore),
    resetToDefaults: configStore.resetToDefaults.bind(configStore),
    exportConfig: configStore.exportConfig.bind(configStore),
    importConfig: configStore.importConfig.bind(configStore)
  };
}

// HOC for automatic re-rendering with MobX
export function withStores<P extends object>(
  Component: React.ComponentType<P>
): React.ComponentType<P> {
  return observer(Component);
}

// Utility hook for responsive design
export function useResponsive() {
  const uiStore = useUIStore();
  return {
    windowWidth: uiStore.windowWidth,
    windowHeight: uiStore.windowHeight,
    isMobile: uiStore.isMobile,
    isTablet: uiStore.isTablet,
    isDesktop: uiStore.isDesktop,
    shouldCollapseSidebar: uiStore.shouldCollapseSidebar
  };
}

// Debug hook for development
export function useDebugInfo() {
  const rootStore = useRootStore();
  return {
    debugInfo: rootStore.getDebugInfo(),
    reset: rootStore.reset.bind(rootStore),
    shutdown: rootStore.shutdown.bind(rootStore),
    saveState: rootStore.savePersistedState.bind(rootStore)
  };
}

// Performance monitoring hook
export function usePerformance() {
  const systemStore = useSystemStore();
  const visionStore = useVisionStore();

  return {
    systemUptime: systemStore.connectionUptime,
    detectionRate: visionStore.detectionRate,
    averageProcessingTime: visionStore.averageProcessingTime,
    frameCount: visionStore.frameCount,
    lastFrameTimestamp: visionStore.lastFrameTimestamp
  };
}

// Error boundary hook
export function useErrorHandler() {
  const uiStore = useUIStore();
  const systemStore = useSystemStore();

  return {
    handleError: (error: Error, context?: string) => {
      console.error('Application error:', error, context);
      systemStore.addError('Frontend', error.message, { context, stack: error.stack });
      uiStore.showError(
        'Application Error',
        `${context ? `${context}: ` : ''}${error.message}`
      );
    },
    handleWarning: (message: string, context?: string) => {
      systemStore.addWarning('Frontend', message, { context });
      uiStore.showWarning('Warning', message);
    }
  };
}
