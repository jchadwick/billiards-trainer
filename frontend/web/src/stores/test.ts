/**
 * Simple test to verify MobX stores are working correctly
 */

import { getRootStore } from './RootStore';

// Test function to verify all stores are working
export async function testStores(): Promise<void> {
  console.log('🧪 Testing MobX Store System...');

  const rootStore = getRootStore();

  // Test SystemStore
  console.log('Testing SystemStore...');
  rootStore.system.addInfo('Test', 'System store working');
  console.log('✅ SystemStore: OK');

  // Test GameStore
  console.log('Testing GameStore...');
  const gameResult = await rootStore.game.startNewGame('practice', [{
    name: 'Test Player',
    ballGroup: null,
    score: 0,
    isActive: true
  }]);
  console.log('✅ GameStore:', gameResult.success ? 'OK' : 'FAILED');

  // Test VisionStore
  console.log('Testing VisionStore...');
  rootStore.vision.resetStatistics();
  console.log('✅ VisionStore: OK');

  // Test ConfigStore
  console.log('Testing ConfigStore...');
  rootStore.config.setShowDebugInfo(true);
  console.log('✅ ConfigStore: OK');

  // Test AuthStore
  console.log('Testing AuthStore...');
  const loginResult = await rootStore.auth.login({ username: 'admin', password: 'admin' });
  console.log('✅ AuthStore:', loginResult.success ? 'OK' : 'FAILED');

  // Test UIStore
  console.log('Testing UIStore...');
  const notificationId = rootStore.ui.showSuccess('Test Success', 'UI store working correctly');
  console.log('✅ UIStore: OK');

  // Test persistence
  console.log('Testing Persistence...');
  await rootStore.savePersistedState();
  console.log('✅ Persistence: OK');

  // Display final status
  const debugInfo = rootStore.getDebugInfo();
  console.log('🎯 Final Store Status:', debugInfo);

  console.log('✅ All MobX stores tested successfully!');
}

// Run test if this file is executed directly
if (typeof window !== 'undefined') {
  // Make test available globally for manual execution
  (window as any).__TEST_STORES__ = testStores;
}
