import { createContext, useContext } from 'react'
import { getRootStore, type RootStore } from '../stores/RootStore'

const rootStore = getRootStore()

const StoreContext = createContext<RootStore>(rootStore)

export const useStores = () => {
  const stores = useContext(StoreContext)
  if (!stores) {
    throw new Error('useStores must be used within a StoreProvider')
  }
  return stores
}

export const useUIStore = () => useStores().ui
export const useConnectionStore = () => useStores().system
export const useSystemStore = () => useStores().system
export const useGameStore = () => useStores().game
export const useVisionStore = () => useStores().vision
export const useConfigStore = () => useStores().config

export const StoreProvider = StoreContext.Provider
export { rootStore }
