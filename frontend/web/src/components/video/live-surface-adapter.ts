import type { RootStore } from '../../stores/RootStore';
import type { DisplayMjpegFrame } from './use-frame-aware-mjpeg';
import type { GameStateData } from '../../types/api';

/** Stable route surface: only latest producer state exists in the committed core. */
export interface LiveSurface {
  system: { status: { isConnected: boolean } };
  videoStore: {
    latestGameState: GameStateData | null;
    latestFrame?: DisplayMjpegFrame | null;
    delivery?: { mode?: string | null; sourceGeneration?: number; sourceConfigurationRevision?: number; configured?: boolean; error?: string | null };
  };
  nativeVideoElement?: HTMLImageElement | null;
  setVideoDimensions?: (width: number, height: number) => void;
}

/**
 * Administration work adds an app-owned camera receiver beyond the versioned
 * RootStore. Read those additions optionally so LiveView also compiles against
 * the committed core and remains usable with its own camera fallback.
 */
export function adaptLiveSurface(store: RootStore): LiveSurface {
  return store as unknown as LiveSurface;
}

/** Bind an untagged native feed to the configuration that selected its source. */
export function sourceConfigurationRevision(values: unknown): number | undefined {
  const revision = (values as { metadata?: { configuration_revision?: unknown } } | undefined)?.metadata?.configuration_revision;
  return typeof revision === 'number' && Number.isSafeInteger(revision) && revision >= 0 ? revision : undefined;
}
