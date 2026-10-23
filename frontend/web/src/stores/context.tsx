import React, { createContext, useContext, type ReactNode } from 'react';
import { getRootStore, type RootStore } from './RootStore';

const LiveStoreContext = createContext<RootStore | null>(null);

export function StoreProvider({
  children,
  store = getRootStore(),
}: {
  children: ReactNode;
  store?: RootStore;
}): React.JSX.Element {
  return <LiveStoreContext.Provider value={store}>{children}</LiveStoreContext.Provider>;
}

export function useRootStore(): RootStore {
  const store = useContext(LiveStoreContext);
  if (!store) throw new Error('useRootStore must be used within a StoreProvider');
  return store;
}
