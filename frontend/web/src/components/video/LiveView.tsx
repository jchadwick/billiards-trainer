import React, { useEffect, useMemo, useRef, useState } from 'react';
import { observer } from 'mobx-react-lite';
import { useRootStore } from '../../stores/context';
import { apiClient } from '../../api/client';
import axiosClient from '../../api/axios-client';
import { adaptLiveSurface, sourceConfigurationRevision } from './live-surface-adapter';
import { getConfiguredLiveStreamUrl } from './live-stream-url';
import { PRACTICE_TABLE_DEFAULTS, suggestDirectPots, type PracticeBall } from './shot-prediction';
import { assessCenterBallAim, confirmTipNearestCueBall, getLiveAimPolicy, projectCueToTable } from './center-ball-coaching';
import { advanceEvidenceAgeMs } from './freshness';
import { canonicalToImagePoint, normalizeImageExtentToCanonical, normalizeImagePointToCanonical } from '../../utils/coordinates';
import { centerToTable, tableMmToImage, tableProjectionIsCompatible, tableToCenter } from './live-coaching-geometry';
import type { BallData, ObservationStatus, PerceptionState, TableProjection } from '../../types/api';
import type { DisplayMjpegFrame } from './use-frame-aware-mjpeg';
import './live-aim.css';

interface LiveViewProps { className?: string; autoConnect?: boolean }
const POCKET_IDS = ['top-left', 'top-middle', 'top-right', 'bottom-left', 'bottom-middle', 'bottom-right'];
const LIVE_AIM_POLICY = getLiveAimPolicy();

function asId(id: string | number): string { return String(id); }

function uniqueIdBalls(balls: BallData[]): Map<string, BallData> {
  const counts = new Map<string, number>();
  balls.forEach((ball) => counts.set(asId(ball.id), (counts.get(asId(ball.id)) ?? 0) + 1));
  return new Map(balls.filter((ball) => counts.get(asId(ball.id)) === 1).map((ball) => [asId(ball.id), ball]));
}

function validPoint(point: unknown): point is { x: number; y: number } {
  if (!point || typeof point !== 'object') return false;
  const candidate = point as { x?: unknown; y?: unknown };
  return typeof candidate.x === 'number' && Number.isFinite(candidate.x)
    && typeof candidate.y === 'number' && Number.isFinite(candidate.y);
}

function knownValuesAgree(left: unknown, right: unknown): boolean {
  return left === undefined || left === null || right === undefined || right === null
    || String(left) === String(right);
}

/** Compare available source/geometry identity, deliberately excluding frame IDs. */
function displayMatchesContext(frame: DisplayMjpegFrame | null | undefined, status: ObservationStatus | undefined): boolean {
  if (!frame || !status) return true;
  const identity = status.observation_identity;
  return [
    [frame.engineInstanceId, identity?.engine_instance_id ?? status.engine_instance_id ?? status.vision_instance_id],
    [frame.sourceSessionId, identity?.session_id ?? status.source_session_id ?? status.source_process_id],
    [frame.configurationRevision, status.configuration_revision ?? identity?.configuration_revision],
    [frame.calibrationRevision, status.calibration_revision ?? identity?.calibration_revision],
    [frame.rectificationRevision, status.rectification_revision ?? identity?.rectification_revision],
  ].every(([display, observation]) => knownValuesAgree(display, observation));
}

/** Source frame IDs may differ; known producer/session/calibration context may not. */
function ballMatchesContext(ball: BallData, status: ObservationStatus | undefined): boolean {
  const source = ball.source_observation;
  if (!source || !status) return true;
  const current = status.observation_identity;
  const currentValues: Record<string, unknown> = {
    engine_instance_id: current?.engine_instance_id ?? status.engine_instance_id ?? status.vision_instance_id,
    session_id: current?.session_id ?? status.source_session_id ?? status.source_process_id,
    configuration_revision: status.configuration_revision,
    calibration_revision: status.table_calibration_revision ?? status.calibration_revision,
    rectification_revision: status.rectification_revision,
  };
  const sourceValues: Record<string, unknown> = {
    engine_instance_id: source.engine_instance_id,
    session_id: source.session_id,
    configuration_revision: source.configuration_revision,
    calibration_revision: source.calibration_revision,
    rectification_revision: source.rectification_revision,
  };
  return Object.entries(sourceValues).every(([key, sourceValue]) => {
    const currentValue = currentValues[key];
    return sourceValue === undefined || currentValue === undefined
      || String(sourceValue) === String(currentValue);
  });
}

export const LiveView: React.FC<LiveViewProps> = observer(({ className = '', autoConnect = true }) => {
  const store = useRootStore();
  const surface = adaptLiveSurface(store);
  const { videoStore } = surface;
  const imageSlotRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });
  const [fallbackStreamUrl, setFallbackStreamUrl] = useState<string>();
  const [fallbackSourceRevision, setFallbackSourceRevision] = useState<number>();
  const [ageTick, setAgeTick] = useState(0);
  const [cueBallId, setCueBallId] = useState('');
  const [targetId, setTargetId] = useState('');
  const [pocketId, setPocketId] = useState('');
  const [tipConfirmationContext, setTipConfirmationContext] = useState<string | null>(null);
  const receiptRef = useRef<{ eventSequence: number; at: number; motionAge: number | null }>({ eventSequence: -1, at: 0, motionAge: null });
  const previousContextRef = useRef<string | null>(null);
  const cueEvidenceRef = useRef({ signature: '', reportedAge: 0, at: 0 });

  const state = videoStore.latestGameState;
  const balls = state?.balls ?? [];
  const status = state?.observation_status;
  const perception: PerceptionState | undefined = status?.perception;
  const displayedFrame = videoStore.latestFrame;
  const producerSourceKey = JSON.stringify([
    status?.observation_identity?.engine_instance_id ?? status?.engine_instance_id ?? status?.vision_instance_id,
    status?.observation_identity?.session_id ?? status?.source_session_id ?? status?.source_process_id,
    status?.configuration_revision ?? status?.observation_identity?.configuration_revision,
    status?.camera_profile_id,
  ]);
  // Only context is remembered here, never a second copy of ball positions.
  const sourceBindingRef = useRef({
    generation: videoStore.delivery?.sourceGeneration, producerKey: producerSourceKey,
    blockedKey: null as string | null, displayed: false,
  });
  const binding = sourceBindingRef.current;
  if (binding.generation !== videoStore.delivery?.sourceGeneration) {
    if (binding.displayed) binding.blockedKey = binding.producerKey;
    binding.generation = videoStore.delivery?.sourceGeneration;
  }
  if (binding.blockedKey !== null && binding.blockedKey !== producerSourceKey) binding.blockedKey = null;
  binding.producerKey = producerSourceKey;
  binding.displayed ||= videoSize.width > 0 && videoSize.height > 0;
  const expectedSourceRevision = videoStore.delivery?.sourceConfigurationRevision ?? fallbackSourceRevision;
  const producerRevision = status?.configuration_revision ?? status?.observation_identity?.configuration_revision;
  // The source-selection revision is a floor: later same-camera configuration
  // updates remain valid, while an older producer cannot describe the new feed.
  const sourceRevisionCompatible = expectedSourceRevision === undefined
    || ((typeof producerRevision === 'number' || typeof producerRevision === 'string')
      && Number.isSafeInteger(Number(producerRevision)) && Number(producerRevision) >= expectedSourceRevision);
  const displayCompatible = binding.blockedKey === null
    && sourceRevisionCompatible
    && displayMatchesContext(displayedFrame, status);
  const decisionCompatible = knownValuesAgree(perception?.decision_calibration_revision, status?.calibration_revision)
    && knownValuesAgree(perception?.decision_rectification_revision, status?.rectification_revision);
  const projectionCandidate = perception?.table_projection ?? status?.table_projection;
  const statusCalibration = (status?.table_calibration_revision ?? status?.calibration_revision) as number | string | undefined;
  const decisionCalibrationRevision = perception?.decision_calibration_revision;
  const projectionRevisionMismatch = Boolean(projectionCandidate && (
    (statusCalibration !== undefined && String(statusCalibration) !== String(projectionCandidate.calibration_revision))
    || (decisionCalibrationRevision != null && decisionCalibrationRevision !== projectionCandidate.calibration_revision)
  ));
  const projection = !projectionRevisionMismatch && tableProjectionIsCompatible(projectionCandidate, {
    calibrationRevision: statusCalibration,
    rectificationRevision: perception?.decision_rectification_revision ?? status?.rectification_revision as number | undefined,
    cameraProfileId: status?.camera_profile_id as string | undefined,
    imageSpace: status?.image_space as string | undefined,
  }) ? projectionCandidate as TableProjection : null;
  const projectionMismatch = Boolean(projectionCandidate && (!projection || projectionRevisionMismatch));
  const table = useMemo(() => projection ? {
    width: projection.table_width_mm,
    height: projection.table_height_mm,
    ballRadius: PRACTICE_TABLE_DEFAULTS.ballRadius,
  } : null, [projection]);
  const uniqueBalls = useMemo(() => uniqueIdBalls(balls), [balls]);
  const ids = useMemo(() => new Set(uniqueBalls.keys()), [uniqueBalls]);
  const compatibleBalls = useMemo(() => new Map([...uniqueBalls].filter(([, ball]) => ballMatchesContext(ball, status))), [uniqueBalls, status]);

  const cueSignature = perception?.cue ? JSON.stringify({
    source: producerSourceKey,
    start: perception.cue.shaft_start_px,
    end: perception.cue.shaft_end_px,
    signedDirection: perception.cue.signed_direction,
    captureFrame: perception.cue.source_frame_id,
  }) : '';
  useEffect(() => {
    if (!perception) {
      receiptRef.current = { eventSequence: -1, at: 0, motionAge: null };
      cueEvidenceRef.current = { signature: '', reportedAge: 0, at: 0 };
      return;
    }
    const now = performance.now();
    if (receiptRef.current.eventSequence !== perception.event_sequence) {
      receiptRef.current = {
        eventSequence: perception.event_sequence,
        at: now,
        motionAge: perception.latest_motion_check_age_ms ?? null,
      };
    }
    if (cueEvidenceRef.current.signature !== cueSignature) {
      cueEvidenceRef.current = {
        signature: cueSignature,
        reportedAge: perception.cue_evidence_age_ms ?? perception.cue?.observation_age_ms ?? 0,
        at: now,
      };
    }
  }, [perception?.event_sequence, cueSignature, perception]);

  useEffect(() => {
    if (!perception) return undefined;
    const timer = window.setInterval(() => setAgeTick(performance.now()), 100);
    return () => window.clearInterval(timer);
  }, [perception?.event_sequence]);

  const motionAge = advanceEvidenceAgeMs(
    perception?.latest_motion_check_age_ms,
    receiptRef.current.motionAge,
    receiptRef.current.at,
    ageTick || receiptRef.current.at,
  );
  const cueReportedAge = perception?.cue_evidence_age_ms ?? perception?.cue?.observation_age_ms;
  const cueAge = advanceEvidenceAgeMs(
    cueReportedAge,
    cueEvidenceRef.current.reportedAge,
    cueEvidenceRef.current.at,
    ageTick || cueEvidenceRef.current.at,
  );
  const layoutTrusted = Boolean(perception?.layout_trusted
    && motionAge !== null
    && motionAge <= perception.motion_evidence_max_age_ms);
  const cueCurrent = Boolean(displayCompatible && decisionCompatible && perception?.cue && cueAge !== null
    && cueAge <= perception.cue_evidence_max_age_ms);

  const sourceContext = [
    videoStore.delivery?.sourceGeneration ?? '',
    status?.observation_identity?.engine_instance_id ?? status?.engine_instance_id ?? status?.vision_instance_id ?? '',
    status?.observation_identity?.session_id ?? status?.source_session_id ?? status?.source_process_id ?? '',
    projection?.camera_profile_id ?? status?.camera_profile_id ?? '',
    projection?.calibration_revision ?? statusCalibration ?? '',
    projection?.rectification_revision ?? perception?.decision_rectification_revision ?? '',
    projection?.table_width_mm ?? '', projection?.table_height_mm ?? '',
    displayedFrame?.engineInstanceId ?? '', displayedFrame?.sourceSessionId ?? '',
    displayedFrame?.configurationRevision ?? '', displayedFrame?.calibrationRevision ?? '',
    displayedFrame?.rectificationRevision ?? '',
  ].join('|');
  useEffect(() => {
    if (previousContextRef.current !== null && previousContextRef.current !== sourceContext) {
      setCueBallId(''); setTargetId(''); setPocketId('');
    }
    previousContextRef.current = sourceContext;
  }, [sourceContext]);
  // A manual tip confirmation applies to one selected ball and settled layout,
  // never to a new camera, calibration, or post-shot arrangement.
  const manualTipContext = JSON.stringify([sourceContext, cueBallId, perception?.layout_revision]);
  const confirmNearestTip = tipConfirmationContext === manualTipContext;
  useEffect(() => { setTipConfirmationContext(null); }, [manualTipContext, layoutTrusted, cueCurrent]);
  // Detection labels never silently choose the live cue ball; the player confirms it here.
  const selectedCueId = cueBallId;
  const cueBall = selectedCueId ? compatibleBalls.get(selectedCueId) : undefined;
  useEffect(() => {
    if (cueBallId && !ids.has(cueBallId)) setCueBallId('');
    if (targetId && (!ids.has(targetId) || targetId === selectedCueId)) setTargetId('');
  }, [ids, cueBallId, targetId, selectedCueId]);
  const canAdvise = Boolean(layoutTrusted && displayCompatible && decisionCompatible && !projectionMismatch && projection && table && cueBall
    && validPoint(cueBall.table_position)
    && compatibleBalls.size === balls.length
    && [...compatibleBalls.values()].every((ball) => validPoint(ball.table_position)));
  const practiceBalls: PracticeBall[] = canAdvise && table ? [...compatibleBalls].map(([id, ball]) => {
    const positive = centerToTable(ball.table_position!, table.width, table.height);
    return { id, ...positive, radius: ball.table_radius_mm && ball.table_radius_mm > 0 ? ball.table_radius_mm : table.ballRadius };
  }) : [];
  const coachCueBall = canAdvise ? practiceBalls.find((ball) => ball.id === selectedCueId) ?? null : null;
  const objectBalls = canAdvise ? practiceBalls.filter((ball) => ball.id !== selectedCueId) : [];
  const candidates = canAdvise && table ? suggestDirectPots(coachCueBall, objectBalls, table, {
    ...(targetId ? { targetId } : {}), ...(pocketId ? { pocketId } : {}),
  }) : [];
  const suggestion = candidates[0] ?? null;
  const projectedCue = cueCurrent && projection && perception?.cue
    ? projectCueToTable(perception.cue, projection, perception.image_width_px ?? 0, perception.image_height_px ?? 0)
    : null;
  const manualTipUsed = Boolean(confirmNearestTip && canAdvise && cueCurrent && projectedCue && !projectedCue.direction && coachCueBall);
  const aim = assessCenterBallAim(
    manualTipUsed ? confirmTipNearestCueBall(projectedCue!, coachCueBall!) : projectedCue,
    canAdvise ? coachCueBall : null,
    objectBalls, table, suggestion, LIVE_AIM_POLICY,
  );
  const measuredPath = aim.prediction;
  const aimMessage = !canAdvise ? 'Alignment paused until a trusted layout and cue-ball selection are available.'
    : !cueCurrent ? 'Bring your cue into view behind the selected cue ball. Waiting for a fresh cue line.'
      : aim.state === 'direction_unknown' ? 'Cue direction unknown. Confirm the nearest end is your tip below to enable current-aim prediction.'
        : aim.state === 'away' ? 'The cue points away from the selected cue ball. Check the cue-ball selection and tip direction.'
          : aim.state === 'off_center' ? `Move the cue through the ball center: the detected shaft is ${aim.offsetMm!.toFixed(1)} mm off center.`
            : aim.state === 'aligned' ? `Lined up within ${LIVE_AIM_POLICY.angleToleranceDegrees}° of the suggested aim (${aim.angleDegrees!.toFixed(1)}° difference).`
              : aim.state === 'adjust' ? `${aim.angleDegrees!.toFixed(1)}° from the suggested aim.${aim.wrongTarget ? measuredPath?.target ? ` Current aim hits ball ${measuredPath.target.id} first.` : ' Current aim misses the object balls.' : ' Adjust toward the green line.'}`
                : aim.state === 'ready' ? 'Current-aim prediction ready. Choose a supported target and pocket to compare alignment.'
                  : 'Cannot assess this cue line. Move the tip closer to the selected ball and keep the shaft clearly visible.';

  useEffect(() => {
    const image = surface.nativeVideoElement;
    const slot = imageSlotRef.current;
    if (!autoConnect || videoStore.delivery?.mode !== 'native' || !image || !slot) return undefined;
    const parking = document.getElementById('shared-video-receiver-parking');
    const loaded = () => {
      setVideoSize({ width: image.naturalWidth, height: image.naturalHeight });
      surface.setVideoDimensions?.(image.naturalWidth, image.naturalHeight);
    };
    image.addEventListener('load', loaded);
    slot.appendChild(image);
    imageRef.current = image;
    loaded();
    return () => { image.removeEventListener('load', loaded); imageRef.current = null; parking?.appendChild(image); };
  }, [autoConnect, videoStore.delivery?.mode, surface.nativeVideoElement, surface.setVideoDimensions]);

  const canvasSize = (imageWidth: number, imageHeight: number, sourceWidth = projection?.source_width_px ?? videoSize.width, sourceHeight = projection?.source_height_px ?? videoSize.height) => ({
    x: videoSize.width > 0 ? imageWidth * videoSize.width / sourceWidth : imageWidth,
    y: videoSize.height > 0 ? imageHeight * videoSize.height / sourceHeight : imageHeight,
  });
  const imagePointFromBall = (ball: BallData): { point: { x: number; y: number }; radiusX: number; radiusY: number } | null => {
    if (!displayCompatible || !ballMatchesContext(ball, status)) return null;
    if (projection && table && validPoint(ball.table_position)) {
      const centerMm = ball.table_position;
      const centerPixel = tableMmToImage(projection, centerMm);
      const radius = ball.table_radius_mm && ball.table_radius_mm > 0 ? ball.table_radius_mm : table.ballRadius;
      const xPixel = tableMmToImage(projection, { x: centerMm.x + radius, y: centerMm.y });
      const yPixel = tableMmToImage(projection, { x: centerMm.x, y: centerMm.y + radius });
      if (!centerPixel || !xPixel || !yPixel) return null;
      const center = canvasSize(centerPixel.x, centerPixel.y);
      const xEdge = canvasSize(xPixel.x, xPixel.y);
      const yEdge = canvasSize(yPixel.x, yPixel.y);
      return { point: center, radiusX: Math.hypot(xEdge.x - center.x, xEdge.y - center.y), radiusY: Math.hypot(yEdge.x - center.x, yEdge.y - center.y) };
    }
    // Physical mapping is optional for showing a compatible image measurement.
    // An unusable world coordinate must disable advice, not erase that marker.
    const normalizedCenter = normalizeImagePointToCanonical(ball.position);
    const normalizedRadius = ball.radius_extent
      ? normalizeImagePointToCanonical(ball.radius_extent)
      : normalizeImageExtentToCanonical(ball.radius, ball.radius_scale ?? ball.position.scale);
    const center = normalizedCenter ? canonicalToImagePoint(normalizedCenter, videoSize) : null;
    const extent = normalizedRadius ? canonicalToImagePoint(normalizedRadius, videoSize) : null;
    return center && extent ? { point: center, radiusX: extent.x, radiusY: extent.y } : null;
  };

  const imagePointFromTableCoach = (point: { x: number; y: number }): { x: number; y: number } | null => {
    if (!projection || !table) return null;
    const center = tableToCenter(point, table.width, table.height);
    const image = tableMmToImage(projection, center);
    return image ? canvasSize(image.x, image.y) : null;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = imageRef.current;
    if (!canvas || !video || videoSize.width <= 0 || videoSize.height <= 0) return;
    const rect = video.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    const width = canvasRect.width || rect.width || videoSize.width;
    const height = canvasRect.height || rect.height || videoSize.height;
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, width, height);
    // The image is object-fit:contain inside a height-constrained pane. Account
    // for its actual content rectangle, including letterboxing and borders.
    const contentWidth = video.clientWidth || rect.width || width;
    const contentHeight = video.clientHeight || rect.height || height;
    const scaleX = Math.min(contentWidth / videoSize.width, contentHeight / videoSize.height);
    const scaleY = scaleX;
    context.save();
    context.translate(rect.left - canvasRect.left + video.clientLeft + (contentWidth - videoSize.width * scaleX) / 2,
      rect.top - canvasRect.top + video.clientTop + (contentHeight - videoSize.height * scaleY) / 2);
    for (const ball of balls) {
      const geometry = imagePointFromBall(ball);
      if (!geometry) continue;
      // Trust is a whole-table producer decision; legacy per-ball flags do not create extra product states.
      const trustedMarker = layoutTrusted;
      context.beginPath();
      context.ellipse(geometry.point.x * scaleX, geometry.point.y * scaleY,
        Math.max(2, geometry.radiusX * scaleX), Math.max(2, geometry.radiusY * scaleY), 0, 0, 2 * Math.PI);
      context.globalAlpha = trustedMarker ? 0.9 : 0.42;
      context.fillStyle = asId(ball.id) === selectedCueId || ball.is_cue_ball || ball.type === 'cue' ? '#f8fafc' : '#fb7185';
      context.strokeStyle = trustedMarker ? '#0f172a' : '#fde68a';
      context.lineWidth = 2;
      context.setLineDash(trustedMarker ? [] : [6, 5]);
      context.fill(); context.stroke();
      context.setLineDash([]); context.globalAlpha = 1;
      context.font = '12px system-ui'; context.textAlign = 'center'; context.textBaseline = 'bottom';
      context.fillStyle = '#fff'; context.strokeStyle = '#0f172a'; context.lineWidth = 3;
      const labelY = (geometry.point.y - geometry.radiusY) * scaleY;
      context.strokeText(asId(ball.id), geometry.point.x * scaleX, labelY);
      context.fillText(asId(ball.id), geometry.point.x * scaleX, labelY);
    }
    if (suggestion) {
      const cueStart = imagePointFromTableCoach(suggestion.cueBall);
      const contact = imagePointFromTableCoach(suggestion.contactCenter);
      const contactPoint = imagePointFromTableCoach(suggestion.contactPoint);
      const targetStart = imagePointFromTableCoach(suggestion.objectPathStart);
      const pocketEnd = imagePointFromTableCoach(suggestion.objectPathEnd);
      if (cueStart && contact && contactPoint && targetStart && pocketEnd) {
        context.save(); context.lineCap = 'round'; context.lineWidth = 4;
        context.strokeStyle = '#a3e635'; context.setLineDash([]);
        context.beginPath(); context.moveTo(cueStart.x * scaleX, cueStart.y * scaleY); context.lineTo(contact.x * scaleX, contact.y * scaleY); context.stroke();
        context.strokeStyle = '#fb923c'; context.setLineDash([10, 7]);
        context.beginPath(); context.moveTo(targetStart.x * scaleX, targetStart.y * scaleY); context.lineTo(pocketEnd.x * scaleX, pocketEnd.y * scaleY); context.stroke();
        const ghost = imagePointFromTableCoach(suggestion.contactCenter);
        if (ghost) {
          const ball = cueBall?.table_radius_mm ?? table?.ballRadius ?? PRACTICE_TABLE_DEFAULTS.ballRadius;
          const edge = imagePointFromTableCoach({ x: suggestion.contactCenter.x + ball, y: suggestion.contactCenter.y });
          context.beginPath(); context.arc(ghost.x * scaleX, ghost.y * scaleY,
            edge ? Math.hypot((edge.x - ghost.x) * scaleX, (edge.y - ghost.y) * scaleY) : 8, 0, Math.PI * 2);
          context.strokeStyle = '#a3e635'; context.setLineDash([4, 3]); context.stroke();
        }
        context.beginPath(); context.arc(contactPoint.x * scaleX, contactPoint.y * scaleY, 4, 0, Math.PI * 2);
        context.fillStyle = '#fff'; context.strokeStyle = '#0f172a'; context.lineWidth = 1;
        context.fill(); context.stroke();
        context.restore();
      }
    }
    const currentCue = cueCurrent ? perception?.cue : null;
    if (currentCue && perception?.image_width_px && perception.image_height_px) {
      const start = canvasSize(currentCue.shaft_start_px[0], currentCue.shaft_start_px[1], perception.image_width_px, perception.image_height_px);
      const end = canvasSize(currentCue.shaft_end_px[0], currentCue.shaft_end_px[1], perception.image_width_px, perception.image_height_px);
      context.save(); context.lineCap = 'round'; context.lineWidth = 5;
      context.strokeStyle = '#facc15'; context.setLineDash([]);
      context.beginPath(); context.moveTo(start.x * scaleX, start.y * scaleY); context.lineTo(end.x * scaleX, end.y * scaleY); context.stroke();
      context.restore();
    }
    if (measuredPath) {
      const drawTravel = (start: { x: number; y: number }, end: { x: number; y: number }, color: string) => {
        const from = imagePointFromTableCoach(start);
        const to = imagePointFromTableCoach(end);
        if (!from || !to) return;
        context.strokeStyle = color;
        context.beginPath(); context.moveTo(from.x * scaleX, from.y * scaleY);
        context.lineTo(to.x * scaleX, to.y * scaleY); context.stroke();
      };
      context.save(); context.lineCap = 'round'; context.lineWidth = 4; context.setLineDash([5, 5]);
      drawTravel(measuredPath.cueTravel.start, measuredPath.cueTravel.end, '#38bdf8');
      if (measuredPath.objectTravel) {
        drawTravel(measuredPath.objectTravel.start, measuredPath.objectTravel.end, '#e879f9');
        const contact = imagePointFromTableCoach(measuredPath.cueTravel.end);
        if (contact) {
          context.setLineDash([]); context.strokeStyle = '#38bdf8';
          context.beginPath(); context.arc(contact.x * scaleX, contact.y * scaleY, 6, 0, Math.PI * 2); context.stroke();
        }
      }
      context.restore();
    }
    context.restore();
  // imagePointFromBall uses only current producer coordinates and current projection.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [balls, videoSize, projection, projectionMismatch, displayCompatible, table, layoutTrusted, suggestion, selectedCueId, cueCurrent, perception?.cue, measuredPath, ageTick]);

  const nativeVideo = autoConnect && videoStore.delivery?.mode === 'native';
  const imageUsable = Boolean(videoSize.width && videoSize.height);
  const renderableCount = imageUsable ? balls.filter((ball) => Boolean(imagePointFromBall(ball))).length : 0;
  const emptyMessage = perception?.empty_layout_verified
    ? 'Clean scan confirmed an empty table.'
    : 'Waiting for the first clean table scan.';
  const coachMessage = !perception
    ? 'Waiting for the first table scan.'
      : !displayCompatible || !decisionCompatible
        ? 'Live overlays or advice are paused: the camera view and table data do not match yet.'
      : projectionMismatch
        ? 'Live aiming is paused: table projection does not match the current camera or calibration. Compatible image markers remain available.'
      : !projection
      ? 'Calibrate the table to enable live aiming. Available ball markers remain visible.'
      : !layoutTrusted
        ? `Live advice paused: ${perception.trust_reason.replaceAll('_', ' ')}. Last-known ball centers remain as dim ghosts.`
        : !cueBall
          ? 'Confirm the cue ball to start choosing direct-pot advice.'
          : !canAdvise
            ? 'Live aiming needs a physical table position for every reported ball.'
            : suggestion
              ? `Aim ball ${suggestion.targetId} toward ${suggestion.pocket.label}. Aim through the ghost ball; the orange line estimates the object-ball path.`
              : 'No supported direct pot clears the current ball layout.';
  const isConnected = Boolean(surface.system.status.isConnected);
  const hasSharedReceiver = surface.nativeVideoElement !== undefined;

  useEffect(() => {
    if (!autoConnect || hasSharedReceiver) return undefined;
    const controller = new AbortController();
    void apiClient.getConfiguration(controller.signal).then((response) => {
      if (controller.signal.aborted || !response.success || !response.data) return;
      const apiOrigin = axiosClient.getInstance().defaults.baseURL;
      setFallbackSourceRevision(sourceConfigurationRevision(response.data.values));
      setFallbackStreamUrl(getConfiguredLiveStreamUrl(response.data.values?.api?.stream?.viewer_url, apiOrigin));
    }).catch(() => { if (!controller.signal.aborted) setFallbackStreamUrl(undefined); });
    return () => controller.abort();
  }, [autoConnect, hasSharedReceiver]);

  return <section className={`live-view ${className}`} aria-label="Live billiards coach">
    <header className="live-header">
      <div><p className="practice-eyebrow">Live camera · estimated positions</p><h1 className="live-title">Live coach</h1></div>
      <span className={`status-badge ${layoutTrusted ? 'status-badge-positive' : 'status-badge-muted'}`} role="status">
        {layoutTrusted ? 'Trusted layout' : balls.length ? 'Stale · last known' : 'Awaiting scan'}
      </span>
    </header>
    <section className="live-coach-controls" aria-label="Live shot coaching">
      <p role="status" aria-live="polite">{coachMessage}</p>
      <p>Use the marker labels to select balls; these labels are not recognized ball numbers. Green shows suggested aim, orange its intended object-ball path, and yellow the detected cue. Cyan shows current cue-ball travel; pink shows the first struck ball’s predicted direction.</p>
      <div className="practice-coordinate-entry">
        <label>Cue ball
          <select aria-label="Confirm cue ball" value={selectedCueId} onChange={(event) => setCueBallId(event.target.value)}>
            <option value="">Select a reported ball</option>
            {[...uniqueBalls].map(([id]) => <option key={id} value={id}>Ball {id}</option>)}
          </select>
        </label>
        <label>Target
          <select aria-label="Select target ball" value={targetId} onChange={(event) => setTargetId(event.target.value)}>
            <option value="">Suggest any target</option>
            {[...uniqueBalls].filter(([id]) => id !== selectedCueId).map(([id]) => <option key={id} value={id}>Ball {id}</option>)}
          </select>
        </label>
        <label>Pocket
          <select aria-label="Select pocket" value={pocketId} onChange={(event) => setPocketId(event.target.value)}>
            <option value="">Suggest any pocket</option>
            {POCKET_IDS.map((id) => <option key={id} value={id}>{id.replace('-', ' ')}</option>)}
          </select>
        </label>
      </div>
      <div className="live-alignment" role="status" aria-live="polite" data-testid="live-alignment" data-alignment={aim.state}>
        <strong>{aimMessage}</strong>
        {aim.offsetMm !== undefined && <span> Shaft offset {aim.offsetMm.toFixed(1)} mm.</span>}
        {manualTipUsed && <span> Direction uses your tip confirmation.</span>}
      </div>
      {canAdvise && projectedCue && !projectedCue.direction && <label className="live-tip-confirmation">
        <input type="checkbox" checked={confirmNearestTip} onChange={(event) => setTipConfirmationContext(event.target.checked ? manualTipContext : null)} />
        I confirm the cue tip is the end nearest my selected cue ball
      </label>}
      {measuredPath && <p data-testid="live-contact-summary">{measuredPath.target
        ? `Current aim hits ball ${measuredPath.target.id} first.${measuredPath.objectTravel?.hitBall ? ` Its path ends at contact with ball ${measuredPath.objectTravel.hitBall.id}; further collisions are not predicted.` : ' Its initial path is shown to the cushion boundary.'}`
        : 'Current aim reaches a cushion without hitting an object ball.'}</p>}
      <p className="live-ball-summary">Start with a level cue and a center-ball stroke, without English. Paths estimate initial direction up to the next contact, not speed or stopping position. Post-contact cue-ball roll, spin, throw, pocket capture, and rebounds are not predicted.</p>
      {suggestion && <p className="live-suggestion" data-testid="live-suggestion">
        Suggested direct pot: ball {suggestion.targetId} to {suggestion.pocket.label}; {Math.round(suggestion.cutAngleRadians * 180 / Math.PI)}° cut.
        {cueCurrent && perception?.cue?.signed_direction
          ? ' Current cue direction is signed and shown separately.'
          : manualTipUsed ? ' Current cue direction is confirmed by you and shown separately.'
            : cueCurrent ? ' Current cue line is visible; its direction is unknown.' : ' No current cue line; manual advice is ready.'}
      </p>}
      {cueCurrent && !suggestion && <p>Current cue detected; {perception?.cue?.signed_direction ? 'direction known.' : manualTipUsed ? 'direction confirmed by you.' : 'direction unknown.'} Cue age {Math.round(cueAge ?? 0)} ms.</p>}
      {imageUsable && balls.length > 0 && renderableCount < balls.length && <p role="note">{balls.length - renderableCount} reported ball(s) have no compatible display coordinates or source context; their markers are withheld.</p>}
      {!imageUsable && balls.length > 0 && <p role="note">Waiting for a live video image before drawing reported ball markers.</p>}
      <p className="live-ball-summary" role="status">
        {balls.length ? `${balls.length} ball${balls.length === 1 ? '' : 's'} · ${renderableCount} marker${renderableCount === 1 ? '' : 's'} drawn` : emptyMessage}
        {layoutTrusted && perception?.last_ball_observation_age_ms != null
          ? ` · scan ${Math.round(perception.last_ball_observation_age_ms / 1000)}s ago, quiet evidence current` : ''}
      </p>
    </section>
    <div data-testid="live-video-pane" className="video-pane" style={{ aspectRatio: imageUsable ? `${videoSize.width} / ${videoSize.height}` : '16 / 9' }}>
      {nativeVideo ? <div ref={imageSlotRef} className="native-video-image-slot" />
        : hasSharedReceiver && videoStore.delivery?.configured === false ? (
          <p role="status" className="stream-unavailable">{videoStore.delivery.error ?? 'Camera stream unavailable. Check Configuration.'}</p>
        ) : (
        <img ref={(element) => { imageRef.current = element; }} src={displayedFrame?.imageUrl ?? fallbackStreamUrl} alt="Video stream" className="live-video-image" onLoad={(event) => {
          const { naturalWidth, naturalHeight } = event.currentTarget;
          setVideoSize({ width: naturalWidth, height: naturalHeight }); surface.setVideoDimensions?.(naturalWidth, naturalHeight);
        }} />
      )}
      <canvas ref={canvasRef} className="live-overlay" data-testid="live-detection-overlay"
        data-freshness={layoutTrusted ? 'current' : 'stale'} data-frame-pairing="independent"
        data-cue-overlay={cueCurrent ? 'visible' : 'hidden'}
        data-current-aim={measuredPath ? 'visible' : 'hidden'}
        style={{ inset: 0, width: '100%', height: '100%', opacity: 1 }} aria-label="Ball and aiming overlays" />
      {!isConnected && <div className="connection-overlay"><div className="connection-overlay-message"><div className="connection-overlay-title">Connecting…</div><div className="connection-overlay-detail">The camera and producer state update independently.</div></div></div>}
    </div>
    <section className="ball-panel" aria-label="Latest producer ball layout">
      <div className="ball-panel-header"><h2 className="ball-panel-title">Latest table layout</h2><span role="status">{balls.length ? `${balls.length} balls ${layoutTrusted ? 'trusted' : 'last known'}` : emptyMessage}</span></div>
      <div className="ball-grid" role="list">{balls.map((ball, index) => <div key={`${asId(ball.id)}-${index}`} role="listitem" className={`ball-slot ${layoutTrusted ? 'ball-slot-observed' : 'ball-slot-stale'}`}>
        <span className="ball-slot-label">{ball.is_cue_ball || ball.type === 'cue' ? 'Cue ball' : `Ball ${ball.id}`}</span>
        <span>{!ballMatchesContext(ball, status) ? 'source context mismatch · marker withheld' : validPoint(ball.table_position) ? `${Math.round(ball.table_position.x)}, ${Math.round(ball.table_position.y)} mm from center` : validPoint(ball.position) ? 'image coordinates' : 'coordinates unavailable'}</span>
      </div>)}</div>
    </section>
  </section>;
});

LiveView.displayName = 'LiveView';
