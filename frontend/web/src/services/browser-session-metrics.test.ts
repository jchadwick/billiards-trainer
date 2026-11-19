import { describe, expect, it } from "vitest";
import {
  BrowserSessionMetrics,
  DEFAULT_BROWSER_SESSION_METRICS_POLICY,
  getBrowserSessionMetricsPolicy,
  type BrowserSessionMetricsPolicy,
} from "./browser-session-metrics";

const testPolicy: BrowserSessionMetricsPolicy = {
  ...DEFAULT_BROWSER_SESSION_METRICS_POLICY,
  rateWindowMs: 10_000,
  sampleIntervalMs: 1_000,
  retentionBuckets: 3,
  maximumRateEvents: 100,
};

function frame(frameId: string, sessionId = "session-a") {
  return {
    engineInstanceId: "engine-a",
    sessionId,
    frameId,
  } as const;
}

const wallTime = (milliseconds: number) => new Date(1_800_000_000_000 + milliseconds);

describe("browser session metrics policy", () => {
  it("validates overrides and falls back for invalid values", () => {
    expect(getBrowserSessionMetricsPolicy({
      VITE_HEALTH_RATE_WINDOW_MS: "0",
      VITE_HEALTH_SAMPLE_INTERVAL_MS: "90000",
      VITE_HEALTH_SAMPLE_RETENTION_BUCKETS: "-1",
      VITE_HEALTH_MAX_RATE_EVENTS: "NaN",
    })).toEqual(DEFAULT_BROWSER_SESSION_METRICS_POLICY);

    expect(getBrowserSessionMetricsPolicy({
      VITE_HEALTH_RATE_WINDOW_MS: "20000",
      VITE_HEALTH_SAMPLE_INTERVAL_MS: "2000",
      VITE_HEALTH_SAMPLE_RETENTION_BUCKETS: "1200",
      VITE_HEALTH_MAX_RATE_EVENTS: "8000",
    })).toEqual({
      rateWindowMs: 20_000,
      sampleIntervalMs: 2_000,
      retentionBuckets: 1_200,
      maximumRateEvents: 8_000,
    });
  });
});

describe("browser session metrics", () => {
  it("uses elapsed startup duration instead of dividing by the full window", () => {
    const metrics = new BrowserSessionMetrics(testPolicy);
    metrics.setConnected(true, 0);
    metrics.recordStateMessageReceipt(200);
    metrics.recordAcceptedSourceFrame(frame("frame-1"), 200);

    const sample = metrics.sample(1_000, wallTime(1_000));
    expect(sample.receivedStateMessagesPerSecond).toBe(1);
    expect(sample.distinctObservedSourceFramesPerSecond).toBeCloseTo(1.25);
    expect(sample.sampledAt).toBe(wallTime(1_000).toISOString());
  });

  it("counts connected silence as zero and expires receipts from the rolling window", () => {
    const metrics = new BrowserSessionMetrics(testPolicy);
    metrics.setConnected(true, 0);
    metrics.recordStateMessageReceipt(100);
    metrics.recordAcceptedSourceFrame(frame("frame-1"), 100);
    expect(metrics.sample(1_000, wallTime(1_000)).receivedStateMessagesPerSecond).toBe(1);

    for (let at = 2_000; at <= 11_000; at += 1_000) {
      const sample = metrics.sample(at, wallTime(at));
      if (at === 11_000) {
        expect(sample.receivedStateMessagesPerSecond).toBe(0);
        expect(sample.distinctObservedSourceFramesPerSecond).toBe(0);
      }
    }
  });

  it("counts every valid receipt while de-duplicating accepted source identities", () => {
    const metrics = new BrowserSessionMetrics(testPolicy);
    metrics.setConnected(true, 0);
    metrics.recordStateMessageReceipt(100);
    metrics.recordAcceptedSourceFrame(frame("frame-10"), 100);
    metrics.recordStateMessageReceipt(200); // repeated publication
    metrics.recordAcceptedSourceFrame(frame("frame-10"), 200);
    metrics.recordStateMessageReceipt(300); // out-of-order receipt, rejected by RootStore
    metrics.recordStateMessageReceipt(400); // phase-only receipt, no source-frame call
    metrics.recordStateMessageReceipt(500);
    metrics.recordAcceptedSourceFrame(frame("frame-11"), 500);

    const sample = metrics.sample(1_000, wallTime(1_000));
    expect(sample.receivedStateMessagesPerSecond).toBe(5);
    expect(sample.distinctObservedSourceFramesPerSecond).toBeCloseTo(2.2222222);
  });

  it("keeps frame rate unknown for missing identity, then marks a producer restart gap", () => {
    const metrics = new BrowserSessionMetrics(testPolicy);
    metrics.setConnected(true, 0);
    metrics.recordStateMessageReceipt(100);
    metrics.recordAcceptedSourceFrame(frame("frame-1"), 100);
    metrics.recordStateMessageReceipt(200);
    metrics.recordAcceptedSourceFrame(null, 200);
    const missingIdentity = metrics.sample(1_000, wallTime(1_000));
    expect(missingIdentity.receivedStateMessagesPerSecond).toBe(2);
    expect(missingIdentity.distinctObservedSourceFramesPerSecond).toBeNull();
    expect(missingIdentity.frameRateUnknown).toBe("missing_identity");

    metrics.recordStateMessageReceipt(1_100);
    metrics.recordAcceptedSourceFrame(frame("frame-1", "session-b"), 1_100);
    expect(metrics.sample(2_000, wallTime(2_000)).gapReason).toBe("producer_changed");
    metrics.recordObservationAges({
      ballAgeMs: 80,
      cueAgeMs: 40,
      motionAgeMs: 100,
      ballEvidenceId: "ball-a",
      cueEvidenceId: "cue-a",
      motionEvidenceId: "motion-a",
    }, 2_100);
    const recovered = metrics.sample(3_000, wallTime(3_000));
    expect(recovered.distinctObservedSourceFramesPerSecond).toBeCloseTo(0.5263157);
    expect(recovered.ballObservationAgeMs).toBe(980);
  });

  it("does not rejuvenate repeated evidence without a changed source identity", () => {
    const metrics = new BrowserSessionMetrics(testPolicy);
    metrics.setConnected(true, 0);
    const original = {
      ballAgeMs: 100,
      cueAgeMs: 100,
      motionAgeMs: 100,
      ballEvidenceId: null,
      cueEvidenceId: null,
      motionEvidenceId: null,
    };
    metrics.recordObservationAges(original, 100);
    metrics.sample(1_000, wallTime(1_000));
    metrics.recordObservationAges({ ...original, motionAgeMs: 50 }, 1_100);
    const repeated = metrics.sample(2_000, wallTime(2_000));
    expect(repeated.motionObservationAgeMs).toBe(2_000);

    metrics.recordObservationAges({
      ...original,
      motionAgeMs: 20,
      motionEvidenceId: "motion-frame-2",
    }, 2_100);
    expect(metrics.sample(3_000, wallTime(3_000)).motionObservationAgeMs).toBe(920);
  });

  it("reports disconnected and suspended intervals as gaps without backfilling", () => {
    const metrics = new BrowserSessionMetrics(testPolicy);
    metrics.setConnected(true, 0);
    metrics.recordStateMessageReceipt(100);
    expect(metrics.sample(1_000, wallTime(1_000)).receivedStateMessagesPerSecond).toBe(1);
    expect(metrics.sample(4_000, wallTime(4_000)).gapReason).toBe("suspended");
    expect(metrics.sample(5_000, wallTime(5_000)).receivedStateMessagesPerSecond).toBe(0);

    metrics.setConnected(false, 5_100);
    expect(metrics.sample(6_000, wallTime(6_000)).gapReason).toBe("disconnected");
    metrics.setConnected(true, 7_000);
    const resumed = metrics.sample(8_000, wallTime(8_000));
    expect(resumed.receivedStateMessagesPerSecond).toBe(0);
    expect(resumed.distinctObservedSourceFramesPerSecond).toBeNull();
  });
});
