import { act, cleanup, render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cuePairingIdentity, VideoReceiverHost } from "./VideoReceiverHost";

const fixture = vi.hoisted(() => {
  const frame = {
    frameId: "frame-1",
    engineInstanceId: "engine-1",
    sourceSessionId: "session-1",
    configurationRevision: 7,
    calibrationRevision: 2,
    rectificationRevision: 3,
    imageUrl: "blob:frame-1",
  };
  return {
    getConfiguration: vi.fn(),
    reloadConfiguration: undefined as (() => void) | undefined,
    stopReceiver: undefined as (() => void) | undefined,
    stream: { latest: frame, matched: frame, error: null as string | null },
    store: {
      nativeVideoElement: null as HTMLImageElement | null,
      videoStore: {
        latestGameState: {
          observation_status: {
            source_frame_id: "frame-1",
            engine_instance_id: "engine-1",
            source_session_id: "session-1",
            configuration_revision: 7,
            calibration_revision: 2,
            rectification_revision: 3,
            perception: {
              decision_calibration_revision: 2,
              decision_rectification_revision: 3,
            },
          },
        },
        maximumObservationAgeSeconds: 0.5,
        framePolicy: { maximumFrameBytes: 1024, maximumPartHeaderBytes: 512, bufferedFrameCount: 4 },
        delivery: {
          state: "unconfigured",
          mode: null,
          configured: false,
          sourceGeneration: 0,
          startedAt: null,
          lastFrameReceiptAt: null,
          lastFrameIdentity: null,
          dimensions: null,
          error: null,
        },
      },
      registerVideoConfigurationReload: vi.fn((handler: () => void) => { fixture.reloadConfiguration = handler; return () => { fixture.reloadConfiguration = undefined; }; }),
      registerVideoReceiverShutdown: vi.fn((handler: () => void) => { fixture.stopReceiver = handler; return () => { fixture.stopReceiver = undefined; }; }),
    },
  };
});

vi.mock("../../api/client", () => ({
  apiClient: { getConfiguration: fixture.getConfiguration },
}));
vi.mock("../../stores/context", () => ({ useRootStore: () => fixture.store }));
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("shared video receiver", () => {
  it("shares one receiver, reloads active settings after apply, and aborts on shutdown", async () => {
    fixture.getConfiguration.mockResolvedValue({
      success: true,
      data: {
        values: {
          metadata: { configuration_revision: 7 },
          api: {
            stream: {
              viewer_url: "http://camera.test/live.mjpg",
              frame_pairing: {
                maximum_frame_bytes: 1024,
                maximum_part_header_bytes: 512,
                buffered_frame_count: 4,
              },
            },
          },
          perception: { freshness: { maximum_observation_age_seconds: 0.5 } },
        },
      },
    });

    render(<VideoReceiverHost />);

    await waitFor(() => expect(fixture.store.nativeVideoElement?.src).toBe("http://camera.test/live.mjpg"));
    expect(fixture.getConfiguration).toHaveBeenCalledTimes(1);
    expect(fixture.store.registerVideoConfigurationReload).toHaveBeenCalledTimes(1);
    expect(fixture.store.registerVideoReceiverShutdown).toHaveBeenCalledTimes(1);
    expect(fixture.store.videoStore.delivery.mode).toBe("native");
    expect(fixture.store.videoStore.delivery).toMatchObject({ sourceConfigurationRevision: 7 });

    const reload = fixture.reloadConfiguration;
    const stop = fixture.stopReceiver;
    expect(reload).toBeDefined();
    expect(stop).toBeDefined();
    fixture.getConfiguration.mockResolvedValueOnce({
      success: true,
      data: { values: { metadata: { configuration_revision: 8 }, api: { stream: { viewer_url: "http://camera.test/next.mjpg" } } } },
    });
    await act(async () => { reload?.(); });
    await waitFor(() => expect(fixture.getConfiguration).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(fixture.store.videoStore.delivery.configurationSynchronized).toBe(true));
    expect(fixture.store.nativeVideoElement?.src).toBe("http://camera.test/next.mjpg");
    expect(fixture.store.videoStore.delivery.sourceGeneration).toBe(2);
    expect(fixture.store.videoStore.delivery).toMatchObject({ sourceConfigurationRevision: 8 });
    expect(fixture.store.videoStore.latestFrame).toBeNull();

    act(() => { stop?.(); });
    expect(fixture.store.nativeVideoElement?.hasAttribute("src")).toBe(false);
  });
});

it("pairs a newer cue frame with a trusted layout and preserves its source revisions", () => {
  const perception = {
    layout_trusted: true,
    cue: { source_frame_id: "frame-2", observation_age_ms: 30 },
    cue_evidence_max_age_ms: 250,
  };

  expect(cuePairingIdentity(fixture.stream.latest, perception))
    .toEqual({ ...fixture.stream.latest, frameId: "frame-2" });
  expect(cuePairingIdentity(fixture.stream.latest, { ...perception, layout_trusted: false }))
    .toBeNull();
  expect(cuePairingIdentity(fixture.stream.latest, { ...perception, cue_evidence_age_ms: 251 }))
    .toBeNull();
});
