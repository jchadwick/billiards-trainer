import { useCallback, useEffect, useRef, useState } from "react";
import { observer } from "mobx-react-lite";
import { apiClient } from "../../api/client";
import axiosClient from "../../api/axios-client";
import { sourceConfigurationRevision } from "./live-surface-adapter";
import { useRootStore } from "../../stores/context";
import { DEFAULT_MAX_STATE_OBSERVATION_AGE_SECONDS, positiveFreshnessLimit } from "./freshness";
import {
  getConfiguredLiveStreamUrl,
} from "./live-stream-url";
import type { FramePairingIdentity } from "./use-frame-aware-mjpeg";

const DEFAULT_RECONNECT_DELAY_MS = 1_000;
const DEFAULT_MAX_RECONNECT_DELAY_MS = 30_000;

function configuredDuration(value: string | undefined, fallback: number, minimum: number, maximum: number): number {
  if (value === undefined || value.trim() === "") return fallback;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed >= minimum && parsed <= maximum
    ? parsed
    : fallback;
}

export const VIDEO_RECEIVER_POLICY = {
  reconnectInitialDelayMs: configuredDuration(import.meta.env.VITE_VIDEO_RECONNECT_INITIAL_MS, DEFAULT_RECONNECT_DELAY_MS, 100, 30_000),
  reconnectMaximumDelayMs: configuredDuration(import.meta.env.VITE_VIDEO_RECONNECT_MAX_MS, DEFAULT_MAX_RECONNECT_DELAY_MS, 1_000, 120_000),
};

// A verified stationary layout can accompany a newer cue measurement.
export function cuePairingIdentity(
  scene: FramePairingIdentity | null,
  perception: {
    layout_trusted?: boolean;
    cue?: { source_frame_id?: string | null; observation_age_ms?: number } | null;
    cue_evidence_age_ms?: number | null;
    cue_evidence_max_age_ms?: number;
  } | null | undefined,
): FramePairingIdentity | null {
  const frameId = perception?.cue?.source_frame_id;
  const age = perception?.cue_evidence_age_ms ?? perception?.cue?.observation_age_ms;
  const limit = perception?.cue_evidence_max_age_ms;
  return scene && perception?.layout_trusted && typeof frameId === "string"
    && typeof age === "number" && age >= 0 && typeof limit === "number" && age <= limit
    ? { ...scene, frameId }
    : null;
}

function safeError(error: string | null): string | null {
  if (!error) return null;
  return error.replace(/https?:\/\/\S+/gi, "camera stream").slice(0, 240);
}

/** Owns the sole camera connection for the complete administration workspace. */
export const VideoReceiverHost = observer(function VideoReceiverHost() {
  const store = useRootStore();
  const { videoStore } = store;
  const [streamUrl, setStreamUrl] = useState<string | undefined>();
  const nativeImageRef = useRef<HTMLImageElement | null>(null);
  const attemptRef = useRef(0);
  const retryTimerRef = useRef<number | null>(null);
  const retryScheduledRef = useRef(false);
  const startedAtRef = useRef<string | null>(null);
  const currentSourceKeyRef = useRef<string | undefined>(undefined);
  const configurationRequestRef = useRef<AbortController | null>(null);
  const loadConfiguration = useCallback(async (signal: AbortSignal) => {
      const response = await apiClient.getConfiguration(signal);
      if (signal.aborted) return;
      if (!response.success || !response.data) throw new Error(response.error ?? "Configuration unavailable");
      const values = response.data.values;
      const apiOrigin = axiosClient.getInstance().defaults.baseURL;
      const url = getConfiguredLiveStreamUrl(values?.api?.stream?.viewer_url, apiOrigin);
      setStreamUrl(url);
      const maximumAge = positiveFreshnessLimit(values?.perception?.freshness?.maximum_observation_age_seconds);
      const sourceKey = url;
      if (currentSourceKeyRef.current !== sourceKey) {
        currentSourceKeyRef.current = sourceKey;
        if (retryTimerRef.current !== null) window.clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
        retryScheduledRef.current = false;
        attemptRef.current = 0;
        videoStore.latestFrame = null;
        videoStore.matchedFrame = null;
        videoStore.delivery = {
          ...videoStore.delivery,
          sourceGeneration: videoStore.delivery.sourceGeneration + 1,
          sourceConfigurationRevision: sourceConfigurationRevision(values),
          dimensions: null,
          lastFrameIdentity: null,
          lastFrameReceiptAt: null,
        };
      }
      videoStore.maximumObservationAgeSeconds = maximumAge;
      startedAtRef.current = new Date().toISOString();
      videoStore.delivery = {
        ...videoStore.delivery,
        state: url ? "connecting" : "unconfigured",
        mode: url ? "native" : null,
        configured: !!url,
        configurationSynchronized: true,
        startedAt: startedAtRef.current,
        error: url ? null : "Set a browser-reachable camera stream URL in Configuration.",
      };
  }, [videoStore]);

  useEffect(() => {
    const controller = new AbortController();
    configurationRequestRef.current = controller;
    void loadConfiguration(controller.signal).catch(() => {
      if (controller.signal.aborted) return;
      startedAtRef.current = new Date().toISOString();
      videoStore.delivery = {
        ...videoStore.delivery,
        state: "error",
        mode: null,
        configured: false,
        configurationSynchronized: false,
        startedAt: startedAtRef.current,
        error: "Camera configuration could not be loaded.",
      };
    });
    return () => {
      controller.abort();
      if (retryTimerRef.current !== null) window.clearTimeout(retryTimerRef.current);
      const image = nativeImageRef.current;
      if (image) image.removeAttribute("src");
    };
  }, [loadConfiguration, videoStore]);

  useEffect(() => store.registerVideoConfigurationReload(() => {
    configurationRequestRef.current?.abort();
    const controller = new AbortController();
    configurationRequestRef.current = controller;
    videoStore.delivery = {
      ...videoStore.delivery,
      configurationSynchronized: false,
    };
    void loadConfiguration(controller.signal).catch(() => {
      if (controller.signal.aborted) return;
      videoStore.delivery = {
        ...videoStore.delivery,
        configurationSynchronized: false,
        error: "Active configuration reloaded unsuccessfully; video source may need reconciliation.",
      };
    });
  }), [loadConfiguration, store, videoStore]);

  useEffect(() => {
    const unregister = store.registerVideoReceiverShutdown(() => {
      configurationRequestRef.current?.abort();
      if (retryTimerRef.current !== null) window.clearTimeout(retryTimerRef.current);
      const image = nativeImageRef.current;
      if (image) image.removeAttribute("src");
    });
    return unregister;
  }, [store]);

  const scheduleRetry = (message: string) => {
    if (retryScheduledRef.current) return;
    retryScheduledRef.current = true;
    videoStore.delivery = { ...videoStore.delivery, state: "error", mode: "native", error: safeError(message) };
    const delay = Math.min(
      VIDEO_RECEIVER_POLICY.reconnectMaximumDelayMs,
      VIDEO_RECEIVER_POLICY.reconnectInitialDelayMs * (2 ** attemptRef.current),
    );
    retryTimerRef.current = window.setTimeout(() => {
      attemptRef.current += 1;
      retryScheduledRef.current = false;
      videoStore.delivery = { ...videoStore.delivery, state: "connecting", error: null };
      const image = nativeImageRef.current;
      if (image && streamUrl) {
        image.removeAttribute("src");
        image.src = streamUrl;
      }
    }, delay);
  };

  function nativeLoaded(): void {
    attemptRef.current = 0;
    retryScheduledRef.current = false;
    const image = nativeImageRef.current;
    if (image) store.setVideoDimensions(image.naturalWidth, image.naturalHeight);
    videoStore.delivery = {
      ...videoStore.delivery,
      state: "loaded",
      mode: "native",
      configured: true,
      lastFrameReceiptAt: new Date().toISOString(),
      lastFrameIdentity: null,
      error: null,
    };
  }

  const setNativeImage = useCallback((image: HTMLImageElement | null) => {
    nativeImageRef.current = image;
    store.nativeVideoElement = image;
    if (image && streamUrl && image.src !== streamUrl) image.src = streamUrl;
  }, [store, streamUrl]);

  return (
    <div id="shared-video-receiver-parking" className="video-receiver-parking" aria-hidden="true">
      {streamUrl && (
        <img
          ref={setNativeImage}
          className="live-video-image"
          src={streamUrl}
          alt=""
          onLoad={nativeLoaded}
          onError={() => scheduleRetry("The browser could not load the camera image.")}
        />
      )}
    </div>
  );
});
