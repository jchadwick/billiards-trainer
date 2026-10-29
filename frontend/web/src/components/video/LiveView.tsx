/**
 * Simple live view component showing video stream with ball detection overlay
 */

import React, { useEffect, useRef, useState } from 'react';
import { observer } from 'mobx-react-lite';
import { apiClient } from '../../api/client';
import { observedBallOutline } from './ball-outline';
import { getConfiguredLiveStreamUrl } from './live-stream-url';
import {
  frameIdentityMatches,
  useFrameAwareMjpeg,
  type FrameAwareStreamPolicy,
  type FramePairingIdentity,
} from './use-frame-aware-mjpeg';
import {
  currentObservationAgeSeconds,
  DEFAULT_MAX_STATE_OBSERVATION_AGE_SECONDS,
  observationIsStale,
  positiveFreshnessLimit,
} from './freshness';
import { canonicalToImagePoint, normalizeImageExtentToCanonical, normalizeImagePointToCanonical } from '../../utils/coordinates';
import type {
  GameType,
  ImageOutline,
  ObservationIdentity,
  ObservationStatus,
  PerceptionState,
  PositionRetentionStatus,
} from '../../types/api';
import { useRootStore } from '../../stores/context';

interface Ball {
  outline?: ImageOutline;
  source_observation?: ObservationIdentity;
  id: string | number;
  position: { x: number; y: number; scale: [number, number] };
  radius: number;
  radius_scale?: [number, number];
  radius_extent?: { x: number; y: number; scale: [number, number] };
  number?: number | null;
  type?: 'cue' | 'solid' | 'stripe' | 'eight' | 'unknown';
  is_cue_ball?: boolean;
  is_observed?: boolean;
  position_kind?: 'measured' | 'filtered' | 'extrapolated' | 'last_observed';
  confidence: number;
  confidence_kind?: 'detector_observation' | 'unavailable';
}

interface Cue {
  tip_position?: { x: number; y: number; scale: [number, number] };
  butt_position?: { x: number; y: number; scale: [number, number] };
  angle?: number;
  confidence?: number;
  detected?: boolean;
  is_visible?: boolean;
}

interface LastKnownBallLayout {
  /** A new camera session or image geometry must never reuse old coordinates. */
  sourceContextKey: string;
  observationKey: string;
  observedAtMonotonicMs: number;
  ageAtCaptureMs: number;
  maximumDisplayAgeMs: number;
  balls: Ball[];
}

interface LiveViewProps {
  className?: string;
  autoConnect?: boolean;
}

const GAME_TYPE_LABELS: Record<GameType, string> = {
  practice: 'Practice',
  '8ball': '8-ball',
  '9ball': '9-ball',
  straight: 'Straight Pool',
};

// A standard pool set contains the cue ball and fifteen object balls. Keep the
// live layout stable even while only part of the set is visible to the camera.
const BALL_SLOT_COUNT = 16;
const IDENTIFIED_FRAME_STREAM_PATH = '/api/v1/video/identified.mjpg';
const DEFAULT_FRAME_PAIRING_POLICY: FrameAwareStreamPolicy = {
  maximumFrameBytes: 16 * 1024 * 1024,
  maximumPartHeaderBytes: 16 * 1024,
  bufferedFrameCount: 64,
};

function nonNegativeRevision(value: unknown): number | null {
  const candidate = typeof value === 'string' && /^\d+$/.test(value)
    ? Number(value)
    : value;
  return typeof candidate === 'number'
    && Number.isSafeInteger(candidate)
    && candidate >= 0
    ? candidate
    : null;
}

function configuredInteger(value: unknown, fallback: number, minimum: number, maximum: number): number {
  return typeof value === 'number'
    && Number.isSafeInteger(value)
    && value >= minimum
    && value <= maximum
    ? value
    : fallback;
}

function isGatewayIdentifiedFrameStream(url: string | undefined): boolean {
  if (!url) return false;
  try {
    return new URL(url).pathname.endsWith(IDENTIFIED_FRAME_STREAM_PATH);
  } catch {
    return false;
  }
}

/** Reject retained coordinates when a known producer/calibration context changed. */
function retainedLayoutMatchesCurrentContext(
  retention: PositionRetentionStatus,
  status: ObservationStatus | undefined,
): boolean {
  const source = retention.source_identity;
  if (!source || typeof source !== 'object') return false;

  const identity = status?.observation_identity;
  const currentContext: Record<string, unknown> = {
    engine_instance_id: identity?.engine_instance_id
      ?? status?.engine_instance_id
      ?? status?.vision_instance_id,
    source_session_id: identity?.session_id
      ?? status?.source_session_id
      ?? status?.source_process_id,
    configuration_revision: status?.configuration_revision,
    rectification_revision: status?.rectification_revision,
    calibration_revision: status?.calibration_revision,
    calibration_state: status?.calibration_state,
  };

  return Object.entries(currentContext).every(([field, currentValue]) => {
    const retainedValue = source[field];
    return currentValue === undefined
      || retainedValue === undefined
      || JSON.stringify(currentValue) === JSON.stringify(retainedValue);
  });
}

export const LiveView: React.FC<LiveViewProps> = observer(({ className = '', autoConnect = true }) => {
  const { system, videoStore } = useRootStore();
  const videoRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });
  const [viewerStreamUrl, setViewerStreamUrl] = useState<string | undefined>();
  const [viewerConfigLoaded, setViewerConfigLoaded] = useState(false);
  const [maximumObservationAgeSeconds, setMaximumObservationAgeSeconds] = useState(
    DEFAULT_MAX_STATE_OBSERVATION_AGE_SECONDS,
  );
  const [framePairingPolicy, setFramePairingPolicy] = useState(DEFAULT_FRAME_PAIRING_POLICY);
  const [ageTickMonotonicMs, setAgeTickMonotonicMs] = useState(0);
  const [perceptionAgeTickMonotonicMs, setPerceptionAgeTickMonotonicMs] = useState(0);
  const receivedAtMonotonicMsRef = useRef(0);
  const retentionDeliveryKeyRef = useRef<string | null>(null);
  const retentionReceivedAtMonotonicMsRef = useRef(0);
  const perceptionReceivedAtMonotonicMsRef = useRef(0);
  const ageAtLastUniqueObservationSecondsRef = useRef<number | null>(null);
  const lastUniqueObservationKeyRef = useRef<string | number | null>(null);
  const lastKnownBallLayoutRef = useRef<LastKnownBallLayout | null>(null);
  const lastCachedBallObservationKeyRef = useRef<string | null>(null);
  const lastSeenSourceContextKeyRef = useRef<string | null>(null);
  const latestState = videoStore.latestGameState;
  const balls: Ball[] = latestState?.balls ?? [];
  const cue: Cue | null = latestState?.cue ?? null;
  const gameType = latestState?.game_type ?? null;
  const observationStatus = latestState?.observation_status;
  const perception: PerceptionState | undefined = observationStatus?.perception;
  const observationIdentity = observationStatus?.observation_identity;
  const sourceFrameId = observationIdentity?.frame_id ?? observationStatus?.source_frame_id;
  const engineInstanceId = observationIdentity?.engine_instance_id
    ?? observationStatus?.engine_instance_id
    ?? observationStatus?.vision_instance_id;
  const sourceSessionIdValue = observationIdentity?.session_id
    ?? observationStatus?.source_session_id
    ?? observationStatus?.source_process_id;
  const sourceSessionId = typeof sourceSessionIdValue === 'string'
    || typeof sourceSessionIdValue === 'number'
    ? String(sourceSessionIdValue)
    : undefined;
  const configurationRevision = nonNegativeRevision(observationStatus?.configuration_revision);
  const calibrationRevision = nonNegativeRevision(observationStatus?.calibration_revision);
  const rectificationRevision = nonNegativeRevision(observationStatus?.rectification_revision);
  const decisionCalibrationRevision = nonNegativeRevision(
    perception?.decision_calibration_revision,
  );
  const decisionRectificationRevision = nonNegativeRevision(
    perception?.decision_rectification_revision,
  );
  const framePairingIdentity: FramePairingIdentity | null = (
    typeof sourceFrameId === 'string'
    && typeof engineInstanceId === 'string'
    && sourceSessionId !== undefined
    && configurationRevision !== null
    && decisionCalibrationRevision !== null
    && decisionRectificationRevision !== null
    && (calibrationRevision === null || calibrationRevision === decisionCalibrationRevision)
    && (rectificationRevision === null || rectificationRevision === decisionRectificationRevision)
  ) ? {
    // Ball coordinates belong to the scene observation frame. The perception
    // coordinator can publish newer motion/cue decisions between low-rate
    // Coral runs, so its decision_frame_id may be later than the image that
    // produced the currently published ball positions.
    frameId: sourceFrameId,
    engineInstanceId,
    sourceSessionId,
    configurationRevision,
    calibrationRevision: decisionCalibrationRevision,
    rectificationRevision: decisionRectificationRevision,
  } : null;
  const frameAwareStream = useFrameAwareMjpeg(
    autoConnect && viewerConfigLoaded ? viewerStreamUrl : undefined,
    framePairingPolicy,
    framePairingIdentity,
  );
  const displayedMjpegFrame = frameAwareStream.matched ?? frameAwareStream.latest;
  const positionRetention = observationStatus?.position_retention;
  const reportedAge = observationStatus?.frame_publication_age_seconds;
  const sourceIdentityParts = [
    observationIdentity?.engine_instance_id
      ?? observationStatus?.engine_instance_id
      ?? observationStatus?.vision_instance_id,
    observationIdentity?.session_id
      ?? observationStatus?.source_session_id
      ?? observationStatus?.source_process_id,
    observationIdentity?.frame_id
      ?? observationStatus?.source_frame_id
      ?? observationStatus?.source_frame_number
      ?? observationStatus?.source_timestamp
      ?? latestState?.frame_number,
    observationIdentity?.snapshot_revision ?? observationStatus?.snapshot_revision,
  ];
  const uniqueObservationKey = sourceIdentityParts.some(
    (part) => part !== undefined && part !== null && part !== '',
  )
    ? sourceIdentityParts.map((part) => part ?? '').join('\u0000')
    : latestState ? '__anonymous_observation__' : null;
  const sourceContextKey = [
    observationIdentity?.engine_instance_id
      ?? observationStatus?.engine_instance_id
      ?? observationStatus?.vision_instance_id
      ?? '',
    observationIdentity?.session_id
      ?? observationStatus?.source_session_id
      ?? observationStatus?.source_process_id
      ?? '',
    observationStatus?.calibration_revision ?? '',
    perception?.image_width_px ?? '',
    perception?.image_height_px ?? '',
  ].join('\u0000');
  const currentAgeSeconds = currentObservationAgeSeconds(
    reportedAge,
    ageAtLastUniqueObservationSecondsRef.current,
    receivedAtMonotonicMsRef.current,
    ageTickMonotonicMs,
  );
  const visionAgeMs = currentAgeSeconds === null
    ? null
    : Math.round(currentAgeSeconds * 1000);
  const freshnessStatusIsStale = observationIsStale(
    null,
    maximumObservationAgeSeconds,
    observationStatus,
  );
  const isFreshnessUnknown = latestState !== null
    && currentAgeSeconds === null
    && !freshnessStatusIsStale;
  const isObservationStale = latestState !== null && (
    isFreshnessUnknown
    || observationIsStale(
      currentAgeSeconds,
      maximumObservationAgeSeconds,
      observationStatus,
    )
  );
  const emptyLayoutVerified = perception?.empty_layout_verified === true;
  const isUnverifiedEmptyObservation = observationStatus?.ball_positions_status === 'observed'
    && balls.length === 0
    && !emptyLayoutVerified;
  const hasFreshMeasuredBallLayout = observationStatus?.ball_positions_status === 'observed'
    && !isObservationStale
    && !isUnverifiedEmptyObservation;
  const ballLayoutMaximumDisplayAgeMs = typeof perception?.layout_verification_max_age_ms === 'number'
    && Number.isFinite(perception.layout_verification_max_age_ms)
    && perception.layout_verification_max_age_ms > 0
    ? perception.layout_verification_max_age_ms
    : maximumObservationAgeSeconds * 1000;
  const isConnected = autoConnect && system.status.isConnected;
  const hasReportedAge = typeof reportedAge === 'number' && Number.isFinite(reportedAge);
  const hasLatestState = latestState !== null;
  const retentionDeliveryKey = positionRetention
    ? [
      observationStatus?.transport_sequence ?? '',
      perception?.event_sequence ?? '',
      positionRetention.retained,
      positionRetention.unavailable_age_ms ?? '',
      positionRetention.retained_age_ms ?? '',
    ].join('\u0000')
    : null;

  // Age keeps advancing without WebSocket messages. The origin changes only
  // for a new source observation, so cached/repeated publications cannot reset
  // the client freshness clock and make old positions look current.
  useEffect(() => {
    if (uniqueObservationKey === null) {
      lastUniqueObservationKeyRef.current = null;
      return;
    }
    if (
      typeof reportedAge !== 'number'
      || !Number.isFinite(reportedAge)
      || reportedAge < 0
    ) return;
    if (lastUniqueObservationKeyRef.current === uniqueObservationKey) return;
    lastUniqueObservationKeyRef.current = uniqueObservationKey;
    receivedAtMonotonicMsRef.current = performance.now();
    ageAtLastUniqueObservationSecondsRef.current = reportedAge;
    setAgeTickMonotonicMs(receivedAtMonotonicMsRef.current);
  }, [uniqueObservationKey, reportedAge]);

  // Advance server-provided retention ages from each delivered state. A silent
  // WebSocket must not leave a retained layout visible forever.
  useEffect(() => {
    if (retentionDeliveryKey === null) {
      retentionDeliveryKeyRef.current = null;
      retentionReceivedAtMonotonicMsRef.current = 0;
      return;
    }
    if (retentionDeliveryKeyRef.current === retentionDeliveryKey) return;
    retentionDeliveryKeyRef.current = retentionDeliveryKey;
    retentionReceivedAtMonotonicMsRef.current = performance.now();
    setAgeTickMonotonicMs(retentionReceivedAtMonotonicMsRef.current);
  }, [retentionDeliveryKey]);

  // Retain the last successful ball measurement across brief phase and
  // delivery gaps. This cache is only a visual aid: the trusted-layout and
  // guidance gates below still use the current backend evidence.
  useEffect(() => {
    if (!latestState) return;
    const cachedLayout = lastKnownBallLayoutRef.current;
    const sourceContextChanged = lastSeenSourceContextKeyRef.current !== null
      && lastSeenSourceContextKeyRef.current !== sourceContextKey;
    if (lastSeenSourceContextKeyRef.current !== sourceContextKey) {
      lastSeenSourceContextKeyRef.current = sourceContextKey;
    }
    if (sourceContextChanged || (cachedLayout && cachedLayout.sourceContextKey !== sourceContextKey)) {
      lastKnownBallLayoutRef.current = null;
      lastCachedBallObservationKeyRef.current = null;
    }

    // An initial page load can begin after the backend has already downgraded
    // its latest positions to cached/unavailable. Seed once from that payload
    // so the overlay does not start blank; source changes must wait for a new
    // successful measurement before displaying coordinates from that source.
    const seedFromDeliveredLastKnownLayout = !sourceContextChanged
      && !lastKnownBallLayoutRef.current
      && balls.length > 0;
    if (
      (!hasFreshMeasuredBallLayout && !seedFromDeliveredLastKnownLayout)
      || uniqueObservationKey === null
    ) return;
    const observationKey = `${sourceContextKey}\u0001${uniqueObservationKey}`;
    if (lastCachedBallObservationKeyRef.current === observationKey) {
      // Configuration may arrive after the first state. Keep the retained
      // display horizon aligned with its validated runtime value.
      if (lastKnownBallLayoutRef.current) {
        lastKnownBallLayoutRef.current.maximumDisplayAgeMs = ballLayoutMaximumDisplayAgeMs;
      }
      return;
    }

    const observedAtMonotonicMs = performance.now();
    const ageAtCaptureMs = Math.max(
      typeof reportedAge === 'number' && Number.isFinite(reportedAge) && reportedAge >= 0
        ? reportedAge * 1000
        : 0,
      typeof perception?.last_ball_observation_age_ms === 'number'
        && Number.isFinite(perception.last_ball_observation_age_ms)
        && perception.last_ball_observation_age_ms >= 0
        ? perception.last_ball_observation_age_ms
        : 0,
    );
    lastKnownBallLayoutRef.current = {
      sourceContextKey,
      observationKey,
      observedAtMonotonicMs,
      ageAtCaptureMs,
      maximumDisplayAgeMs: ballLayoutMaximumDisplayAgeMs,
      // Only an explicitly verified empty layout is authoritative enough to
      // clear a previously visible layout. Unverified empty payloads never
      // reach this write because they are not fresh measured layouts.
      balls: balls.map((ball) => ({ ...ball })),
    };
    lastCachedBallObservationKeyRef.current = observationKey;
    setAgeTickMonotonicMs(observedAtMonotonicMs);
  }, [
    ballLayoutMaximumDisplayAgeMs,
    balls,
    hasFreshMeasuredBallLayout,
    perception?.last_ball_observation_age_ms,
    reportedAge,
    sourceContextKey,
    uniqueObservationKey,
    latestState,
  ]);

  useEffect(() => {
    if (!hasReportedAge && !latestState) return;
    const interval = window.setInterval(
      () => setAgeTickMonotonicMs(performance.now()),
      100,
    );
    return () => window.clearInterval(interval);
  }, [hasReportedAge, hasLatestState]);

  // Phase, motion, layout, and cue evidence ages advance independently of
  // ordinary ball-observation freshness. A cached layout must not make an old
  // cue or motion check appear current while the source is silent.
  useEffect(() => {
    if (!perception) {
      perceptionReceivedAtMonotonicMsRef.current = 0;
      setPerceptionAgeTickMonotonicMs(0);
      return;
    }
    perceptionReceivedAtMonotonicMsRef.current = performance.now();
    setPerceptionAgeTickMonotonicMs(perceptionReceivedAtMonotonicMsRef.current);
  }, [perception?.event_sequence]);

  useEffect(() => {
    if (!perception) return;
    const interval = window.setInterval(
      () => setPerceptionAgeTickMonotonicMs(performance.now()),
      100,
    );
    return () => window.clearInterval(interval);
  }, [perception?.event_sequence]);

  const perceptionDeliveryAgeMs = perception && perceptionAgeTickMonotonicMs > 0
    ? Math.max(0, perceptionAgeTickMonotonicMs - perceptionReceivedAtMonotonicMsRef.current)
    : 0;
  const ballEvidenceAgeMs = perception?.last_ball_observation_age_ms == null
    ? null
    : perception.last_ball_observation_age_ms + perceptionDeliveryAgeMs;
  const motionEvidenceAgeMs = perception?.latest_motion_check_age_ms == null
    ? null
    : perception.latest_motion_check_age_ms + perceptionDeliveryAgeMs;
  const cueEvidenceAgeMs = perception?.cue_evidence_age_ms == null
    ? perception?.cue?.observation_age_ms == null
      ? null
      : perception.cue.observation_age_ms + perceptionDeliveryAgeMs
    : perception.cue_evidence_age_ms + perceptionDeliveryAgeMs;
  const tableVisibilityAgeMs = perception?.table_visibility
    ? perception.table_visibility.observation_age_ms + perceptionDeliveryAgeMs
    : null;
  const tableVisibilityIsCurrent = perception?.table_visibility?.status === 'visible'
    && tableVisibilityAgeMs !== null
    && tableVisibilityAgeMs <= perception.motion_evidence_max_age_ms;
  const framePairingRequired = autoConnect;
  const exactFramePairSelected = Boolean(
    frameAwareStream.matched
    && framePairingIdentity
    && frameIdentityMatches(frameAwareStream.matched, framePairingIdentity),
  );
  const imageGeometryMatchesState = Boolean(
    displayedMjpegFrame
    && perception?.image_width_px === videoSize.width
    && perception?.image_height_px === videoSize.height,
  );
  // Exact-frame detections remain useful for visual verification when table
  // visibility is uncertain; only coaching eligibility requires visibility.
  const spatialOverlayFrameValid = !framePairingRequired
    || (exactFramePairSelected && imageGeometryMatchesState);
  const cueFrameMatchesDisplay = !framePairingRequired
    || Boolean(
      displayedMjpegFrame
      && perception?.cue?.source_frame_id === displayedMjpegFrame.frameId,
    );
  const layoutEvidenceIsCurrent = perception
    ? perception.layout_trusted
      && ballEvidenceAgeMs !== null
      && ballEvidenceAgeMs <= perception.layout_verification_max_age_ms
      && motionEvidenceAgeMs !== null
      && motionEvidenceAgeMs <= perception.motion_evidence_max_age_ms
    : true;
  const retainedLayoutCanBeDisplayed = Boolean(perception && layoutEvidenceIsCurrent);
  const cachedLayout = lastKnownBallLayoutRef.current;
  const cachedLayoutAgeMs = cachedLayout
    ? Math.max(
      0,
      cachedLayout.ageAtCaptureMs
        + ageTickMonotonicMs
        - cachedLayout.observedAtMonotonicMs,
    )
    : null;
  const cachedLayoutIsUsable = Boolean(
    cachedLayout
    && cachedLayout.sourceContextKey === sourceContextKey
    && cachedLayoutAgeMs !== null
    && cachedLayoutAgeMs < cachedLayout.maximumDisplayAgeMs,
  );
  const retentionDeliveryAgeMs = retentionReceivedAtMonotonicMsRef.current > 0
    ? Math.max(0, ageTickMonotonicMs - retentionReceivedAtMonotonicMsRef.current)
    : 0;
  const serverRetentionWithinGrace = Boolean(
    positionRetention
    && typeof positionRetention.unavailable_age_ms === 'number'
    && Number.isFinite(positionRetention.unavailable_age_ms)
    && positionRetention.unavailable_age_ms >= 0
    && typeof positionRetention.grace_ms === 'number'
    && Number.isFinite(positionRetention.grace_ms)
    && positionRetention.grace_ms > 0
    && positionRetention.unavailable_age_ms + retentionDeliveryAgeMs < positionRetention.grace_ms,
  );
  const usingServerRetainedBallLayout = Boolean(
    positionRetention?.retained === true
    && observationStatus?.ball_positions_status === 'cached'
    && balls.length > 0
    && serverRetentionWithinGrace
    && retainedLayoutMatchesCurrentContext(positionRetention, observationStatus),
  );
  const serverRetainedLayoutAgeMs = positionRetention
    && typeof positionRetention.retained_age_ms === 'number'
    && Number.isFinite(positionRetention.retained_age_ms)
    && positionRetention.retained_age_ms >= 0
    ? positionRetention.retained_age_ms + retentionDeliveryAgeMs
    : null;
  const usingLastKnownBallLayout = !usingServerRetainedBallLayout
    && cachedLayoutIsUsable
    && !hasFreshMeasuredBallLayout
    && (!retainedLayoutCanBeDisplayed || balls.length === 0);
  const ballsForDisplay: Ball[] = usingServerRetainedBallLayout
    ? balls.map((ball) => ({
      ...ball,
      is_observed: false,
      position_kind: 'last_observed',
    }))
    : usingLastKnownBallLayout && cachedLayout
      ? cachedLayout.balls.map((ball) => ({
        ...ball,
        is_observed: false,
        position_kind: 'last_observed',
      }))
      : balls;
  const hideUnverifiedStalePositions = isObservationStale
    && !retainedLayoutCanBeDisplayed
    && !usingLastKnownBallLayout
    && !usingServerRetainedBallLayout;
  const observedBalls = hideUnverifiedStalePositions
    ? []
    : ballsForDisplay.filter((ball) => ball.is_observed !== false);
  const nonObservedBalls = hideUnverifiedStalePositions
    ? ballsForDisplay
    : ballsForDisplay.filter((ball) => ball.is_observed === false);
  const guidanceIsEligible = perception
    ? perception.guidance_valid
      && perception.phase === 'aiming'
      && !usingServerRetainedBallLayout
      && layoutEvidenceIsCurrent
      && cueEvidenceAgeMs !== null
      && cueEvidenceAgeMs <= perception.cue_evidence_max_age_ms
      && tableVisibilityIsCurrent
      && spatialOverlayFrameValid
      && cueFrameMatchesDisplay
    : false;
  const cueVisible = perception
    // Drawing a detected cue axis is useful for inspection even when the ball
    // layout is not trusted enough for coaching. Keep the visual detection
    // gate independent from guidance eligibility, while still aging cue data
    // against its own evidence clock.
    ? cueEvidenceAgeMs !== null
      && cueEvidenceAgeMs <= perception.cue_evidence_max_age_ms
      && Boolean(perception.cue)
      && spatialOverlayFrameValid
      && cueFrameMatchesDisplay
    : !isObservationStale && Boolean(cue && (cue.is_visible ?? cue.detected));
  const cueConfidence = perception?.cue?.confidence ?? cue?.confidence;
  const perceptionStatusLabel = perception
    ? `${perception.phase.replaceAll('_', ' ')} · ${layoutEvidenceIsCurrent
      ? `${(ballEvidenceAgeMs ?? 0) > 0 ? 'verified layout retained' : 'verified layout'} · ball measurement ${Math.round(ballEvidenceAgeMs ?? 0)} ms ago`
      : `layout not guidance-eligible · ${perception.trust_reason.replaceAll('_', ' ')}`}`
    : null;
  const displayedBalls = [
    ...observedBalls.map((ball) => ({
      ball,
      entireStateIsStale: isObservationStale,
    })),
    ...nonObservedBalls.map((ball) => ({
      ball,
      entireStateIsStale: isObservationStale
        || usingServerRetainedBallLayout
        || usingLastKnownBallLayout
        || !retainedLayoutCanBeDisplayed,
    })),
  ].slice(0, BALL_SLOT_COUNT);
  const ballCountLabel = usingServerRetainedBallLayout
    ? `${displayedBalls.length} ${displayedBalls.length === 1 ? 'ball' : 'balls'} last known`
    : retainedLayoutCanBeDisplayed
      ? `${displayedBalls.length} ${displayedBalls.length === 1 ? 'ball' : 'balls'} in verified layout`
      : usingLastKnownBallLayout
        ? `${displayedBalls.length} ${displayedBalls.length === 1 ? 'ball' : 'balls'} last known`
        : `${observedBalls.length} ${observedBalls.length === 1 ? 'ball' : 'balls'} detected`;
  const ballFreshnessLabel = usingServerRetainedBallLayout
    ? `0 balls freshly observed, ${displayedBalls.length} last known${serverRetainedLayoutAgeMs === null ? '' : ` · ${Math.round(serverRetainedLayoutAgeMs)} ms ago`}`
    : retainedLayoutCanBeDisplayed
      ? `${isObservationStale ? 0 : observedBalls.length} balls freshly observed, ${isObservationStale ? displayedBalls.length : nonObservedBalls.length} retained`
      : usingLastKnownBallLayout
        ? `0 balls freshly observed, ${displayedBalls.length} last known${cachedLayoutAgeMs === null ? '' : ` · ${Math.round(cachedLayoutAgeMs)} ms ago`}`
        : `${observedBalls.length} balls observed`;
  const ballCountForBadge = retainedLayoutCanBeDisplayed || usingLastKnownBallLayout || usingServerRetainedBallLayout
    ? displayedBalls.length
    : observedBalls.length;
  const ballCountBadgeLabel = `${ballCountForBadge} ${ballCountForBadge === 1 ? 'ball' : 'balls'}`;
  const videoAspectRatio = videoSize.width > 0 && videoSize.height > 0
    ? `${videoSize.width} / ${videoSize.height}`
    : '16 / 9';

  // The detector may subscribe to a loopback camera URI while browsers need a
  // LAN-reachable hostname. Keep the viewer URL as runtime application config.
  useEffect(() => {
    if (!autoConnect) {
      setViewerConfigLoaded(false);
      return;
    }

    let cancelled = false;
    void apiClient.getConfiguration().then((response) => {
      if (cancelled) return;
      const values = response.data?.values;
      const streamUrl = values?.api?.stream?.viewer_url;
      const perception = values?.perception;
      const rustLive = perception?.rust_live;
      const coralRuntime = rustLive?.coral_runtime;
      const configuredMaximumAge = rustLive?.provider === 'coral'
        ? coralRuntime?.maximum_observation_age_seconds
          ?? perception?.freshness?.maximum_observation_age_seconds
        : perception?.freshness?.maximum_observation_age_seconds;
      const pairing = values?.api?.stream?.frame_pairing;
      setMaximumObservationAgeSeconds(
        positiveFreshnessLimit(configuredMaximumAge),
      );
      setFramePairingPolicy({
        maximumFrameBytes: configuredInteger(
          pairing?.maximum_frame_bytes,
          DEFAULT_FRAME_PAIRING_POLICY.maximumFrameBytes,
          1,
          64 * 1024 * 1024,
        ),
        maximumPartHeaderBytes: configuredInteger(
          pairing?.maximum_part_header_bytes,
          DEFAULT_FRAME_PAIRING_POLICY.maximumPartHeaderBytes,
          128,
          65_536,
        ),
        bufferedFrameCount: configuredInteger(
          pairing?.buffered_frame_count,
          DEFAULT_FRAME_PAIRING_POLICY.bufferedFrameCount,
          2,
          64,
        ),
      });
      setViewerStreamUrl(getConfiguredLiveStreamUrl(streamUrl));
      setViewerConfigLoaded(true);
    }).catch(() => {
      if (cancelled) return;
      setViewerStreamUrl(undefined);
      setMaximumObservationAgeSeconds(DEFAULT_MAX_STATE_OBSERVATION_AGE_SECONDS);
      setFramePairingPolicy(DEFAULT_FRAME_PAIRING_POLICY);
      setViewerConfigLoaded(true);
    });

    return () => { cancelled = true; };
  }, [autoConnect]);

  const gameTypeLabel = gameType
    ? GAME_TYPE_LABELS[gameType as GameType] ?? gameType
    : 'Unknown';
  const diagnosticSummary = [
    isConnected ? 'Connected' : 'Disconnected',
    ballFreshnessLabel,
    `Cue stick: ${cueVisible
      ? `visible${typeof cueConfidence === 'number' ? ` (${Math.round(cueConfidence * 100)}%)` : ''}`
      : 'not detected'}`,
    `Frame alignment: ${exactFramePairSelected && imageGeometryMatchesState ? 'exact' : 'waiting for matching frame'}`,
    `Table visibility: ${perception?.table_visibility?.status ?? 'unknown'}`,
    perceptionStatusLabel,
    visionAgeMs !== null
      ? `State age: ${visionAgeMs} ms${isObservationStale ? ' · STALE' : ''}`
      : latestState ? 'State age unavailable' : null,
    `Mode: ${gameTypeLabel}`,
  ].filter(Boolean).join(' | ');

  const getBallLabel = (ball: Ball): string => {
    if (typeof ball.number === 'number') return `Ball ${ball.number}`;
    if (ball.is_cue_ball || ball.type === 'cue') return 'Cue ball';
    if (ball.type === 'eight') return '8-ball';
    return 'Ball';
  };

  const getBallFreshnessLabel = (
    ball: Ball,
    entireStateIsStale: boolean,
  ): string => {
    if (usingServerRetainedBallLayout) {
      return `Last known layout · ${Math.round(serverRetainedLayoutAgeMs ?? 0)} ms ago`;
    }
    if (usingLastKnownBallLayout) {
      return `Last known layout · ${Math.round(cachedLayoutAgeMs ?? 0)} ms ago`;
    }
    if (perception && layoutEvidenceIsCurrent) {
      return `Verified layout retained · ball measurement ${Math.round(ballEvidenceAgeMs ?? 0)} ms ago`;
    }
    if (entireStateIsStale) {
      return isFreshnessUnknown
        ? 'Last known state (freshness unknown)'
        : 'Last known state (stale)';
    }
    if (ball.position_kind === 'extrapolated') return 'Predicted track';
    if (ball.position_kind === 'filtered') return 'Filtered track';
    return 'Last observed';
  };

  // Track video size when image loads
  useEffect(() => {
    const img = videoRef.current;
    if (!img) return;

    const handleLoad = () => {
      setVideoSize({
        width: img.naturalWidth,
        height: img.naturalHeight
      });
    };

    img.addEventListener('load', handleLoad);
    return () => img.removeEventListener('load', handleLoad);
  }, [autoConnect]);

  // Draw balls on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video || !videoSize.width || !videoSize.height) return;

    // Match canvas size to video display size
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // The browser must draw only on the exact identified image selected by
    // the paired state event. A separate MJPEG or websocket clock is not a
    // valid substitute for acquisition identity.
    if (!spatialOverlayFrameValid) return;

    // Calculate scale factor from video coordinates to canvas coordinates
    const scaleX = rect.width / videoSize.width;
    const scaleY = rect.height / videoSize.height;

    // Never paint ball positions once both ordinary freshness and the
    // phase-aware retained-layout evidence have expired. Cue evidence has an
    // independent clock and is drawn below even when the ball overlay is
    // suppressed.
    if (!hideUnverifiedStalePositions) ballsForDisplay.forEach(ball => {
      const canonicalPosition = normalizeImagePointToCanonical(ball.position);
      const canonicalRadius = ball.radius_extent
        ? normalizeImagePointToCanonical(ball.radius_extent)
        : normalizeImageExtentToCanonical(ball.radius, ball.radius_scale ?? ball.position.scale);
      if (!canonicalPosition || !canonicalRadius) return;
      const imagePosition = canonicalToImagePoint(canonicalPosition, videoSize);
      const imageRadius = canonicalToImagePoint(canonicalRadius, videoSize);
      if (!imagePosition || !imageRadius) return;

      // The active image size is applied after canonical normalization so the
      // same physical detection aligns over 1080p and 4K source frames.
      const x = imagePosition.x * scaleX;
      const y = imagePosition.y * scaleY;
      const radiusX = imageRadius.x * scaleX;
      const radiusY = imageRadius.y * scaleY;
      // Coral boxes can be slightly asymmetric even for round pool balls.
      // Use the larger measured half-extent on both axes so the overlay stays
      // circular and encloses the detector's full box without moving its center.
      const radius = Math.max(radiusX, radiusY);
      const isObserved = ball.is_observed !== false && !isObservationStale;

      // Preserve an accepted off-axis silhouette instead of forcing a circle.
      const outline = isObserved && spatialOverlayFrameValid
        ? observedBallOutline(ball, displayedMjpegFrame ?? null, videoSize) : null;
      ctx.beginPath();
      if (outline) {
        ctx.moveTo(outline[0].x * scaleX, outline[0].y * scaleY);
        outline.slice(1).forEach((point) => ctx.lineTo(point.x * scaleX, point.y * scaleY));
        ctx.closePath();
      } else {
        ctx.ellipse(x, y, radius, radius, 0, 0, 2 * Math.PI);
      }

      // Set ball color based on type
      const colors = {
        cue: 'rgba(255, 255, 255, 0.7)',
        solid: 'rgba(255, 107, 107, 0.7)',
        stripe: 'rgba(78, 205, 196, 0.7)',
        eight: 'rgba(44, 62, 80, 0.7)',
        unknown: 'rgba(200, 200, 200, 0.45)',
      };
      const ballType = ball.is_cue_ball ? 'cue' : ball.type ?? 'unknown';
      ctx.globalAlpha = isObserved ? 1 : 0.45;
      ctx.fillStyle = colors[ballType] || colors.unknown;
      ctx.fill();

      // Draw outline
      ctx.strokeStyle = isObserved ? '#000000' : '#fbbf24';
      ctx.lineWidth = 2;
      ctx.setLineDash(isObserved ? [] : [5, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw ball number if available
      if (typeof ball.number === 'number') {
        ctx.fillStyle = ball.type === 'eight' ? '#FFFFFF' : '#000000';
        ctx.font = `${Math.max(14, radius * 0.8)}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(ball.number.toString(), x, y);
      }
      ctx.globalAlpha = 1;
    });

    // Phase-aware cue geometry is expressed in source-image pixels. Convert it
    // to the same canonical image plane used by ball observations. Endpoint
    // direction is marked only when the perception contract supports it.
    const cueGeometry = perception?.cue;
    const sourceWidth = perception?.image_width_px;
    const sourceHeight = perception?.image_height_px;
    const imagePoint = (point: [number, number]) => (sourceWidth && sourceHeight
      ? normalizeImagePointToCanonical({
        x: point[0],
        y: point[1],
        scale: [3840 / sourceWidth, 2160 / sourceHeight],
      })
      : null);
    const cueStart = cueGeometry && sourceWidth && sourceHeight
      ? imagePoint(cueGeometry.shaft_start_px)
      : cue?.butt_position ? normalizeImagePointToCanonical(cue.butt_position) : null;
    const cueEnd = cueGeometry && sourceWidth && sourceHeight
      ? imagePoint(cueGeometry.shaft_end_px)
      : cue?.tip_position ? normalizeImagePointToCanonical(cue.tip_position) : null;
    const startImage = cueStart ? canonicalToImagePoint(cueStart, videoSize) : null;
    const endImage = cueEnd ? canonicalToImagePoint(cueEnd, videoSize) : null;
    if (cueVisible && startImage && endImage) {
        const startX = startImage.x * scaleX;
        const startY = startImage.y * scaleY;
        const endX = endImage.x * scaleX;
        const endY = endImage.y * scaleY;
        ctx.save();
        ctx.globalAlpha = 1;
        ctx.lineCap = 'round';

        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.strokeStyle = 'rgba(15, 23, 42, 0.95)';
        ctx.lineWidth = 12;
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 6;
        ctx.stroke();

        const signedTip = cueGeometry?.signed_direction === 'toward_start'
          ? { x: startX, y: startY }
          : cueGeometry?.signed_direction === 'toward_end' || (!cueGeometry && cue)
            ? { x: endX, y: endY }
            : null;
        for (const endpoint of [
          { x: startX, y: startY },
          { x: endX, y: endY },
        ]) {
          const isTip = signedTip === endpoint
            || (signedTip !== null && endpoint.x === signedTip.x && endpoint.y === signedTip.y);
          ctx.beginPath();
          ctx.arc(endpoint.x, endpoint.y, isTip ? 7 : 5, 0, 2 * Math.PI);
          ctx.fillStyle = '#0f172a';
          ctx.fill();
          ctx.beginPath();
          ctx.arc(endpoint.x, endpoint.y, isTip ? 4.5 : 3, 0, 2 * Math.PI);
          ctx.fillStyle = isTip ? '#fb7185' : '#38bdf8';
          ctx.fill();
        }

        const cueConfidence = cueGeometry?.confidence ?? cue?.confidence;
        const confidenceLabel = typeof cueConfidence === 'number'
          ? `CUE AXIS · ${Math.round(cueConfidence * 100)}%`
          : 'CUE AXIS';
        ctx.font = '700 13px system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        const labelPaddingX = 9;
        const labelHeight = 26;
        const labelWidth = ctx.measureText(confidenceLabel).width + labelPaddingX * 2;
        const midpointX = (startX + endX) / 2;
        const midpointY = (startY + endY) / 2;
        const labelX = Math.max(
          4,
          Math.min(canvas.width - labelWidth - 4, midpointX - labelWidth / 2),
        );
        const labelY = Math.max(
          labelHeight / 2 + 4,
          Math.min(
            canvas.height - labelHeight / 2 - 4,
            midpointY - 20,
          ),
        );
        ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
        ctx.fillRect(labelX, labelY - labelHeight / 2, labelWidth, labelHeight);
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 2;
        ctx.strokeRect(labelX, labelY - labelHeight / 2, labelWidth, labelHeight);
        ctx.fillStyle = '#fef08a';
        ctx.fillText(confidenceLabel, labelX + labelPaddingX, labelY);

        ctx.restore();
    }
  }, [
    ballsForDisplay,
    cue,
    cueVisible,
    hideUnverifiedStalePositions,
    isObservationStale,
    perception,
    spatialOverlayFrameValid,
    videoSize,
  ]);

  return (
    <div className={`live-view ${className}`}>
      {/* Header */}
      <div className="live-header">
        <h1 className="live-title">Live View</h1>
        <div className="live-meta" role="group" aria-label={diagnosticSummary} title={diagnosticSummary}>
          <span
            className={`status-badge connection-badge ${isConnected ? 'status-badge-positive' : 'status-badge-negative'}`}
            aria-label={isConnected ? 'Connected' : 'Disconnected'}
          >
            <span
              className={`connection-indicator ${isConnected ? 'connection-indicator-connected' : 'connection-indicator-disconnected'}`}
              aria-hidden="true"
            />
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </span>
          <span
            className={`status-badge ball-count-badge ${usingLastKnownBallLayout || usingServerRetainedBallLayout ? 'status-badge-muted' : ''}`}
            aria-label={`${ballCountBadgeLabel}; ${ballFreshnessLabel}`}
            title={`${ballCountBadgeLabel} · ${ballFreshnessLabel}`}
          >
            <svg className="ball-count-icon" viewBox="0 0 20 20" aria-hidden="true">
              <circle cx="6" cy="7" r="4.25" fill="currentColor" />
              <circle cx="13.5" cy="12.5" r="4.25" fill="currentColor" />
              <circle cx="6" cy="7" r="1.1" fill="#1f2937" />
              <circle cx="13.5" cy="12.5" r="1.1" fill="#1f2937" />
            </svg>
            <span>{ballCountForBadge}</span>
          </span>
          <span
            className={`status-badge cue-badge ${cueVisible ? 'status-badge-positive' : 'status-badge-negative'}`}
            aria-label={cueVisible
              ? `Cue detected${typeof cueConfidence === 'number' ? ` at ${Math.round(cueConfidence * 100)} percent confidence` : ''}`
              : 'Cue not detected'}
            title={cueVisible
              ? `Cue detected${typeof cueConfidence === 'number' ? ` · ${Math.round(cueConfidence * 100)}% confidence` : ''}`
              : 'Cue not detected'}
          >
            <span>Cue</span>
            <span className="cue-status-indicator" aria-hidden="true" />
          </span>
        </div>
      </div>

      {/* Video container */}
      <div
        data-testid="live-video-pane"
        className="video-pane"
        style={{ aspectRatio: videoAspectRatio }}
      >
        {/* MJPEG video stream */}
        {autoConnect && viewerConfigLoaded && !viewerStreamUrl ? (
          <p role="status" className="stream-unavailable">
            Direct camera stream unavailable. Configure a browser-reachable
            <code className="stream-setting">api.stream.viewer_url</code> to display video.
          </p>
        ) : (
          <img
            ref={videoRef}
            src={displayedMjpegFrame?.imageUrl
              ?? (autoConnect && viewerConfigLoaded && !isGatewayIdentifiedFrameStream(viewerStreamUrl)
                ? viewerStreamUrl
                : undefined)}
            alt="Video stream"
            className="live-video-image"
            data-frame-id={displayedMjpegFrame?.frameId}
          />
        )}

        {/* Ball detection overlay canvas */}
        <canvas
          ref={canvasRef}
          className="live-overlay"
          aria-label={cueVisible ? 'Cue detection overlay' : 'Ball detection overlay; cue not detected'}
          data-testid="live-detection-overlay"
          data-cue-overlay={cueVisible ? 'visible' : 'hidden'}
          data-cue-overlay-mode={!cueVisible
            ? 'hidden'
            : guidanceIsEligible ? 'guidance' : 'detection-only'}
          data-freshness={isObservationStale || usingLastKnownBallLayout || usingServerRetainedBallLayout
            ? (isFreshnessUnknown ? 'unknown' : 'stale')
            : 'current'}
          data-frame-pairing={exactFramePairSelected && imageGeometryMatchesState ? 'exact' : 'unmatched'}
          style={{
            // The pane follows the source image aspect ratio, so the source and
            // overlay share the same origin and dimensions without letterbox
            // offsets or crop-dependent coordinate transforms.
            inset: 0,
            width: '100%',
            height: '100%',
            opacity: spatialOverlayFrameValid
              && !(hideUnverifiedStalePositions && !cueVisible)
              ? 1
              : 0,
          }}
        />

        {/* Status overlay when not connected */}
        {!isConnected && (
          <div className="connection-overlay">
            <div className="connection-overlay-message">
              <div className="connection-overlay-title">Connecting...</div>
              <div className="connection-overlay-detail">Waiting for video stream and ball detection</div>
            </div>
          </div>
        )}
      </div>

      {/* Ball list */}
      <div className="ball-panel">
        <div className="ball-panel-header">
          <h2 className="ball-panel-title">Detected Balls</h2>
          <span
            role="status"
            aria-live="polite"
            className="detected-ball-count"
          >
            {ballCountLabel}
          </span>
        </div>
        <div
          role="grid"
          aria-label="Detected ball slots"
          className="ball-grid"
        >
          {Array.from({ length: BALL_SLOT_COUNT }, (_, index) => {
            const entry = displayedBalls[index];
            if (!entry) {
              return (
                <div
                  key={`empty-${index}`}
                  role="gridcell"
                  aria-label={`Empty ball slot ${index + 1}`}
                  className="ball-slot ball-slot-empty"
                />
              );
            }

            const { ball, entireStateIsStale } = entry;
            const isObserved = ball.is_observed !== false && !entireStateIsStale;
            const label = isObserved
              ? getBallLabel(ball)
              : getBallFreshnessLabel(ball, entireStateIsStale);
            const confidence = ball.confidence_kind === 'unavailable'
              ? 'n/a'
              : `${Math.round(ball.confidence * 100)}%`;

            return (
              <div
                key={`${isObserved ? 'observed' : 'stale'}-${ball.id}`}
                role="gridcell"
                title={`${label} · ${confidence}`}
                className={`ball-slot ${
                  isObserved
                    ? 'ball-slot-observed'
                    : 'ball-slot-stale'
                }`}
              >
                <span className="ball-slot-label">{label}</span>
                <span className={isObserved ? 'ball-slot-confidence-observed' : 'ball-slot-confidence-stale'}>
                  {confidence}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
});

LiveView.displayName = 'LiveView';
