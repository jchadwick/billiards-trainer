/**
 * MobX stores for reactive state management
 * This is the main store index that exports all stores and provides store context
 */

import { configure } from 'mobx';

// Configure MobX for production-ready settings
configure({
  enforceActions: 'never', // Allow direct state mutations for simplicity
  computedRequiresReaction: false, // Allow computed values without reactions
  reactionRequiresObservable: false, // Allow reactions on non-observable values
  observableRequiresReaction: false, // Allow observable access without reactions
  disableErrorBoundaries: true // Let React handle error boundaries
});

// Export all stores
export { SystemStore } from './SystemStore';
export { GameStore } from './GameStore';
export { VisionStore } from './VisionStore';
export { ConfigStore } from './ConfigStore';
export { AuthStore } from './AuthStore';
export { UIStore } from './UIStore';
export { RootStore, getRootStore, rootStore } from './RootStore';

// Export individual store instances for convenience
export {
  systemStore,
  gameStore,
  visionStore,
  configStore,
  authStore,
  uiStore
} from './RootStore';

// Export all types
export * from './types';

// Export React context and hooks
export * from './context';

// Export utility functions
export { observer } from 'mobx-react-lite';

// Development tools
if (process.env.NODE_ENV === 'development') {
  // Make stores available globally for debugging
  (window as any).__MOBX_STORES__ = {
    system: systemStore,
    game: gameStore,
    vision: visionStore,
    config: configStore,
    auth: authStore,
    ui: uiStore,
    root: rootStore
  };

  console.log('MobX stores initialized and available at window.__MOBX_STORES__');
}
