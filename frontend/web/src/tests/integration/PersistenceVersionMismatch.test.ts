import { PersistenceManager } from '../../stores/persistence';
import type { RootStore } from '../../stores/RootStore';

describe('persisted state version handling', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  test('clears incompatible state instead of invoking a migration shim', () => {
    localStorage.setItem(
      'billiards_app_state',
      JSON.stringify({ version: '0.9.0', config: {}, ui: {} }),
    );
    const manager = new PersistenceManager({} as RootStore);

    expect(manager.loadPersistedState()).toBe(false);
    expect(localStorage.getItem('billiards_app_state')).toBeNull();
  });
});
