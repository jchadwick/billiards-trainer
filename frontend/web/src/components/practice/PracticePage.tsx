import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { observer } from 'mobx-react-lite';
import { TrainingTabs } from './TrainingTabs';
import { PracticeCameraOverlay } from './PracticeCameraOverlay';
import type { PracticeOverlayBall, PracticeOverlayTarget } from './PracticeCameraOverlay';
import {
  PRACTICE_TABLE_DEFAULTS,
  standardPockets,
  suggestDirectPots,
  type PracticeBall,
  type PracticeTable,
  type PotCandidate,
} from '../video/shot-prediction';
import { advanceEvidenceAgeMs } from '../video/freshness';
import { centerToTable, tableProjectionIsCompatible } from '../video/live-coaching-geometry';
import type { BallData, GameStateData, ObservationStatus, TableProjection } from '../../types/api';
import { useRootStore } from '../../stores/context';
import {
  assessPracticePlacements,
  getPracticePlacementPolicy,
  practiceEvidenceIsEligible,
  practiceLayoutIsReady,
} from './practice-placement';

type PlacementMode = 'cue' | 'object' | 'edit';

interface PlacementTarget extends PracticeBall {
  label: string;
}

interface MotionEvidenceReceipt {
  eventSequence: number;
  at: number;
  motionAge: number | null;
}

const PLACEMENT_POLICY = getPracticePlacementPolicy();

function initialTargets(table: PracticeTable): PlacementTarget[] {
  return [
    { id: 'cue', label: 'Cue position', x: table.width * 0.17, y: table.height * 0.5, radius: table.ballRadius },
    { id: 'ball-1', label: 'Target 1', x: table.width * 0.53, y: table.height * 0.5, radius: table.ballRadius },
    { id: 'ball-2', label: 'Target 2', x: table.width * 0.72, y: table.height * 0.31, radius: table.ballRadius },
    { id: 'ball-3', label: 'Target 3', x: table.width * 0.78, y: table.height * 0.68, radius: table.ballRadius },
  ];
}

function asId(id: string | number): string { return String(id); }

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

/** A remembered physical center remains usable only in the same producer context. */
function ballMatchesContext(ball: BallData, status: ObservationStatus | undefined): boolean {
  const source = ball.source_observation;
  if (!source || !status) return true;
  const identity = status.observation_identity;
  const current: Record<string, unknown> = {
    engine_instance_id: identity?.engine_instance_id ?? status.engine_instance_id ?? status.vision_instance_id,
    session_id: identity?.session_id ?? status.source_session_id ?? status.source_process_id,
    configuration_revision: status.configuration_revision ?? identity?.configuration_revision,
    calibration_revision: status.table_calibration_revision ?? status.calibration_revision ?? identity?.calibration_revision,
    rectification_revision: status.rectification_revision ?? identity?.rectification_revision,
  };
  const reported: Record<string, unknown> = {
    engine_instance_id: source.engine_instance_id,
    session_id: source.session_id,
    configuration_revision: source.configuration_revision,
    calibration_revision: source.calibration_revision,
    rectification_revision: source.rectification_revision,
  };
  return Object.entries(reported).every(([key, sourceValue]) => sourceValue === undefined || sourceValue === null
    || (current[key] !== undefined && current[key] !== null && String(sourceValue) === String(current[key])));
}

function stateProducerKey(status: ObservationStatus | undefined): string {
  return JSON.stringify([
    status?.observation_identity?.engine_instance_id ?? status?.engine_instance_id ?? status?.vision_instance_id,
    status?.observation_identity?.session_id ?? status?.source_session_id ?? status?.source_process_id,
    status?.configuration_revision ?? status?.observation_identity?.configuration_revision,
    status?.camera_profile_id,
  ]);
}

function placementModeText(mode: PlacementMode): string {
  if (mode === 'cue') return 'Click the live table image or enter coordinates to place the cue target.';
  if (mode === 'object') return 'Click the live table image or enter coordinates to add an object target.';
  return 'Click the live table image or enter coordinates to move the selected target.';
}

function physicalBallIsOnTable(ball: PracticeBall, table: PracticeTable): boolean {
  return ball.x >= ball.radius && ball.x <= table.width - ball.radius
    && ball.y >= ball.radius && ball.y <= table.height - ball.radius;
}

function statusError(status: ObservationStatus | undefined): string | null {
  if (status?.freshness_status === 'source_silent') return 'The producer has stopped refreshing camera state.';
  return null;
}

export const PracticePage = observer(function PracticePage() {
  const store = useRootStore();
  const { videoStore } = store;
  const state: GameStateData | null = videoStore.latestGameState;
  const status = state?.observation_status;
  const perception = status?.perception;
  const delivery = videoStore.delivery;
  const image = store.nativeVideoElement;
  const imageAvailable = !!image && (image.parentElement?.id === 'shared-video-receiver-parking'
    || image.parentElement?.classList.contains('practice-camera-image-slot') === true);

  const [offlineTable, setOfflineTable] = useState<PracticeTable>({ ...PRACTICE_TABLE_DEFAULTS });
  const [targets, setTargets] = useState<PlacementTarget[]>(() => initialTargets(PRACTICE_TABLE_DEFAULTS as PracticeTable));
  const [placementMode, setPlacementMode] = useState<PlacementMode>('cue');
  const [placementTargetId, setPlacementTargetId] = useState('cue');
  const [placementX, setPlacementX] = useState(Math.round(PRACTICE_TABLE_DEFAULTS.width * 0.17));
  const [placementY, setPlacementY] = useState(Math.round(PRACTICE_TABLE_DEFAULTS.height * 0.5));
  const [targetId, setTargetId] = useState('');
  const [pocketId, setPocketId] = useState('');
  const [nextBallNumber, setNextBallNumber] = useState(4);
  const [cueConfirmed, setCueConfirmed] = useState(false);
  const [ageTick, setAgeTick] = useState(0);
  const motionReceiptRef = useRef<MotionEvidenceReceipt>({ eventSequence: -1, at: 0, motionAge: null });
  const sourceBindingRef = useRef({
    generation: delivery.sourceGeneration,
    producerKey: stateProducerKey(status),
    blockedKey: null as string | null,
    displayed: false,
  });

  const projectionCandidate = perception?.table_projection ?? status?.table_projection;
  const statusCalibrationRevision = status?.table_calibration_revision ?? status?.calibration_revision;
  const projectionRevisionMismatch = Boolean(projectionCandidate && (
    (statusCalibrationRevision !== undefined && statusCalibrationRevision !== null
      && String(statusCalibrationRevision) !== String(projectionCandidate.calibration_revision))
    || (perception?.decision_calibration_revision != null
      && String(perception.decision_calibration_revision) !== String(projectionCandidate.calibration_revision))
  ));
  const projectionCandidateCompatible = !projectionRevisionMismatch && tableProjectionIsCompatible(projectionCandidate, {
    calibrationRevision: statusCalibrationRevision as number | string | undefined,
    rectificationRevision: (perception?.decision_rectification_revision ?? status?.rectification_revision) as number | undefined,
    cameraProfileId: status?.camera_profile_id as string | undefined,
    imageSpace: status?.image_space as string | undefined,
  });
  const projection = projectionCandidateCompatible ? projectionCandidate as TableProjection : null;
  const decisionCompatible = knownValuesAgree(perception?.decision_calibration_revision, statusCalibrationRevision)
    && knownValuesAgree(perception?.decision_rectification_revision, status?.rectification_revision);

  const table: PracticeTable = projection
    ? { width: projection.table_width_mm, height: projection.table_height_mm, ballRadius: offlineTable.ballRadius }
    : offlineTable;
  const pockets = useMemo(() => standardPockets(table), [table.width, table.height]);

  useEffect(() => {
    if (!perception) {
      motionReceiptRef.current = { eventSequence: -1, at: 0, motionAge: null };
      return;
    }
    if (motionReceiptRef.current.eventSequence !== perception.event_sequence) {
      motionReceiptRef.current = {
        eventSequence: perception.event_sequence,
        at: performance.now(),
        motionAge: perception.latest_motion_check_age_ms ?? null,
      };
    }
  }, [perception?.event_sequence, perception?.latest_motion_check_age_ms, perception]);

  useEffect(() => {
    if (!perception) return undefined;
    const timer = window.setInterval(() => setAgeTick(performance.now()), 100);
    return () => window.clearInterval(timer);
  }, [perception?.event_sequence]);

  const motionAge = advanceEvidenceAgeMs(
    perception?.latest_motion_check_age_ms,
    motionReceiptRef.current.motionAge,
    motionReceiptRef.current.at,
    ageTick || motionReceiptRef.current.at,
  );
  const quietLayout = Boolean(perception?.layout_trusted
    && (perception.phase === 'pre_shot' || perception.phase === 'aiming')
    && motionAge !== null
    && motionAge <= perception.motion_evidence_max_age_ms);

  const producerKey = stateProducerKey(status);
  const binding = sourceBindingRef.current;
  if (binding.generation !== delivery.sourceGeneration) {
    if (binding.displayed) binding.blockedKey = binding.producerKey;
    binding.generation = delivery.sourceGeneration;
  }
  if (binding.blockedKey !== null && binding.blockedKey !== producerKey) binding.blockedKey = null;
  binding.producerKey = producerKey;
  binding.displayed ||= !!image && (image.naturalWidth > 0 || !!delivery.dimensions?.width);

  const expectedSourceRevision = delivery.sourceConfigurationRevision;
  const producerRevision = status?.configuration_revision ?? status?.observation_identity?.configuration_revision;
  const sourceRevisionCompatible = expectedSourceRevision === undefined
    || ((typeof producerRevision === 'number' || typeof producerRevision === 'string')
      && Number.isSafeInteger(Number(producerRevision))
      && Number(producerRevision) >= expectedSourceRevision);
  const sourceContextCompatible = binding.blockedKey === null
    && sourceRevisionCompatible
    && status?.freshness_status !== 'source_silent';

  const dimensions = delivery.dimensions;
  const cameraFeedReady = !!image
    && imageAvailable
    && image.naturalWidth > 0 && image.naturalHeight > 0
    && delivery.mode === 'native'
    && (delivery.state === 'loaded' || delivery.state === 'receiving')
    && delivery.configurationSynchronized
    && !!dimensions && dimensions.width > 0 && dimensions.height > 0;
  const imageAspectCompatible = !!projection && !!dimensions
    && Math.abs(dimensions.width / dimensions.height
      - projection.source_width_px / projection.source_height_px) < 0.01
    && (perception?.image_width_px == null || perception.image_height_px == null
      || Math.abs(perception.image_width_px / perception.image_height_px
        - projection.source_width_px / projection.source_height_px) < 0.01);
  const cameraProjectionReady = cameraFeedReady && sourceContextCompatible
    && decisionCompatible && !!projection && imageAspectCompatible;
  const overlayProjection = cameraProjectionReady ? projection : null;
  const stateConnectionReady = store.system.status.isConnected;
  const evidenceReady = practiceEvidenceIsEligible({
    cameraStreamReady: cameraProjectionReady,
    producerConnected: stateConnectionReady,
    sourceContextCompatible,
    projectionCompatible: !!projection && !!overlayProjection,
    decisionCompatible,
    layoutTrusted: Boolean(perception?.layout_trusted),
    phase: perception?.phase,
    motionAgeMs: motionAge,
    motionMaximumAgeMs: perception?.motion_evidence_max_age_ms,
  });

  // If a newly active calibration changes the physical plane, keep the user's
  // millimeter targets in place where possible and clamp only impossible edge
  // values to the new cushion-safe range.
  useEffect(() => {
    if (!projection) return;
    setTargets((current) => current.map((target) => ({
      ...target,
      x: Math.max(target.radius, Math.min(projection.table_width_mm - target.radius, target.x)),
      y: Math.max(target.radius, Math.min(projection.table_height_mm - target.radius, target.y)),
    })));
  }, [projection?.calibration_revision, projection?.table_width_mm, projection?.table_height_mm]);

  const compatibleBallRecords = (state?.balls ?? []).map((ball, sourceIndex) => ({ ball, sourceIndex }))
    .filter(({ ball }) => ballMatchesContext(ball, status)
      && validPoint(ball.table_position)
      && ball.position_kind !== 'extrapolated')
    .map(({ ball, sourceIndex }) => {
      const positive = centerToTable(ball.table_position!, table.width, table.height);
      const radius = typeof ball.table_radius_mm === 'number' && Number.isFinite(ball.table_radius_mm)
        && ball.table_radius_mm > 0 ? ball.table_radius_mm : table.ballRadius;
      return { ball, sourceIndex, center: { id: asId(ball.id), ...positive }, radius };
    })
    .filter(({ center, radius }) => physicalBallIsOnTable({ ...center, radius }, table));
  const eligibleCenters = evidenceReady ? compatibleBallRecords.map(({ center }) => center) : [];
  const assessment = assessPracticePlacements(targets, eligibleCenters, PLACEMENT_POLICY.toleranceMm);
  const matchesByTarget = new Map(assessment.matches.map((match) => [match.targetId, match]));
  const matchedSourceIndexes = new Set(assessment.matches.map((match) =>
    compatibleBallRecords[match.measuredIndex]?.sourceIndex).filter((value): value is number => value !== undefined));
  const assignedTargets = targets.map((target) => ({
    id: target.id,
    label: target.label,
    x: target.x,
    y: target.y,
    placed: evidenceReady && matchesByTarget.has(target.id),
  } satisfies PracticeOverlayTarget));

  const visibleBallRecords = cameraProjectionReady
    ? (state?.balls ?? []).map((ball, sourceIndex) => ({ ball, sourceIndex }))
      .filter(({ ball }) => ballMatchesContext(ball, status) && validPoint(ball.table_position))
    : [];
  const overlayBalls: PracticeOverlayBall[] = visibleBallRecords.map(({ ball, sourceIndex }) => ({
    x: ball.table_position!.x,
    y: ball.table_position!.y,
    radiusMm: typeof ball.table_radius_mm === 'number' && Number.isFinite(ball.table_radius_mm)
      && ball.table_radius_mm > 0 ? ball.table_radius_mm : table.ballRadius,
    matched: evidenceReady && matchedSourceIndexes.has(sourceIndex),
    extrapolated: ball.position_kind === 'extrapolated' || !evidenceReady,
  }));

  const ready = practiceLayoutIsReady(evidenceReady, targets.length, assessment);
  const mappedBalls: PracticeBall[] = assessment.matches.flatMap((match) => {
    const target = targets.find(({ id }) => id === match.targetId);
    const measured = compatibleBallRecords[match.measuredIndex];
    return target && measured
      ? [{ id: target.id, x: measured.center.x, y: measured.center.y, radius: measured.radius }]
      : [];
  });
  const cueBall = mappedBalls.find((ball) => ball.id === 'cue') ?? null;
  const objectBalls = mappedBalls.filter((ball) => ball.id !== 'cue');
  const assignedCenterIndexes = new Set(assessment.matches.map((match) => match.measuredIndex));
  const extraBallBlockers = evidenceReady
    ? compatibleBallRecords.flatMap((record, index) => assignedCenterIndexes.has(index) ? [] : [{
      id: `unassigned-${record.sourceIndex}`,
      x: record.center.x,
      y: record.center.y,
      radius: record.radius,
    }])
    : [];
  const candidates = ready && cueConfirmed && cueBall
    ? suggestDirectPots(cueBall, objectBalls, table, {
      ...(targetId ? { targetId } : {}),
      ...(pocketId ? { pocketId } : {}),
      additionalBlockers: extraBallBlockers,
    })
    : [];
  const selected: PotCandidate | undefined = candidates[0];

  let notice: string;
  if (!image || delivery.mode !== 'native' || !cameraFeedReady) {
    notice = delivery.state === 'error' || delivery.state === 'unconfigured' || delivery.state === 'stopped'
      ? 'Live camera video is unavailable. Coordinate targets remain editable, but placement is unverified.'
      : 'Waiting for the shared live camera image. Placement is unverified.';
  } else if (!projectionCandidate) {
    notice = 'The producer has not supplied a table projection. Calibrate the table before camera placement can be verified.';
  } else if (!projection) {
    notice = 'The table projection does not match the active camera or calibration context. Placement is unverified.';
  } else if (status?.freshness_status === 'source_silent') {
    notice = 'The producer camera state is silent. Last-known centers cannot count toward Ready.';
  } else if (!sourceContextCompatible || !decisionCompatible || !imageAspectCompatible) {
    notice = 'Camera source or measurement context changed. Waiting for matching producer state; placement is unverified.';
  } else if (!stateConnectionReady) {
    notice = 'The producer state connection is unavailable. Detected centers and Ready feedback are paused.';
  } else if (!perception) {
    notice = 'Producer perception state is unavailable. Detected centers cannot be trusted yet.';
  } else if (!quietLayout) {
    const reason = perception.trust_reason ? ` ${perception.trust_reason.replaceAll('_', ' ')}.` : '';
    notice = `Waiting for current quiet-layout evidence.${reason} Last-known centers do not count toward Ready.`;
  } else if (statusError(status)) {
    notice = `${statusError(status)} Placement remains unverified until physical centers are available.`;
  } else if (compatibleBallRecords.length === 0) {
    notice = state?.balls.length === 0
      ? 'The producer currently reports 0 balls with physical centers. No placement can be verified.'
      : 'No compatible physical ball centers are available. Placement is unverified.';
  } else if (ready) {
    notice = `All ${targets.length} target slots are placed within ±${PLACEMENT_POLICY.toleranceMm} mm.`;
  } else {
    notice = `Placed ${assessment.matches.length} of ${targets.length} target slots within ±${PLACEMENT_POLICY.toleranceMm} mm.`;
  }

  const cameraStatus = ready
    ? `All ${targets.length} target slots are matched within ±${PLACEMENT_POLICY.toleranceMm} mm.`
    : notice;
  const updateVideoDimensions = useCallback((width: number, height: number) => {
    store.setVideoDimensions(width, height);
  }, [store]);

  function updateTable(update: Partial<PracticeTable>) {
    if (projection && (update.width !== undefined || update.height !== undefined)) return;
    if (Object.values(update).some((value) => !Number.isFinite(value) || value <= 0)) return;
    const next = { ...offlineTable, ...update };
    if (next.width <= next.ballRadius * 2 || next.height <= next.ballRadius * 2) return;
    setOfflineTable(next);
    if (update.ballRadius !== undefined) {
      setTargets((current) => current.map((target) => ({ ...target, radius: next.ballRadius })));
    }
  }

  function placeAt(xValue: number, yValue: number) {
    const x = Math.max(table.ballRadius, Math.min(table.width - table.ballRadius, xValue));
    const y = Math.max(table.ballRadius, Math.min(table.height - table.ballRadius, yValue));
    setPlacementX(Math.round(x));
    setPlacementY(Math.round(y));
    setCueConfirmed(false);
    if (placementMode === 'object') {
      const id = `ball-${nextBallNumber}`;
      setTargets((current) => [...current, {
        id,
        label: `Target ${current.filter((target) => target.id !== 'cue').length + 1}`,
        x,
        y,
        radius: table.ballRadius,
      }]);
      setNextBallNumber((number) => number + 1);
      setPlacementTargetId(id);
      setPlacementMode('edit');
      setTargetId(id);
      return;
    }
    const selectedId = placementMode === 'cue' ? 'cue' : placementTargetId;
    setTargets((current) => current.map((target) => target.id === selectedId ? { ...target, x, y } : target));
  }

  function placeBallOnImage(x: number, y: number) {
    if (!overlayProjection) return;
    placeAt(x, y);
  }

  function removeTarget(id: string) {
    setTargets((current) => current.filter((target) => target.id !== id));
    if (placementTargetId === id) {
      setPlacementTargetId('cue');
      setPlacementMode('cue');
    }
    if (targetId === id) setTargetId('');
    setCueConfirmed(false);
  }

  function editTarget(target: PlacementTarget) {
    setPlacementTargetId(target.id);
    setPlacementX(Math.round(target.x));
    setPlacementY(Math.round(target.y));
    setPlacementMode(target.id === 'cue' ? 'cue' : 'edit');
  }

  function clearObjectTargets() {
    setTargets((current) => current.filter((target) => target.id === 'cue'));
    setPlacementTargetId('cue');
    setPlacementMode('cue');
    setTargetId('');
    setCueConfirmed(false);
  }

  const placementsById = matchesByTarget;
  const placements = targets.map((target) => {
    const match = placementsById.get(target.id);
    const distance = assessment.nearestDistanceByTarget.get(target.id);
    const label = !evidenceReady
      ? 'Unverified'
      : match ? `Placed ${match.distanceMm.toFixed(0)} mm from center`
        : distance !== null && distance !== undefined
          ? distance <= PLACEMENT_POLICY.toleranceMm
            ? 'Needs a separate detected ball; each center can fill one target only.'
            : `Place a ball closer; nearest detected center is ${distance.toFixed(0)} mm away.`
          : 'Waiting for a compatible detected ball center.';
    return { target, match, label };
  });

  return (
    <section className="practice-page" aria-labelledby="practice-title">
      <TrainingTabs active="practice" />
      <header className="practice-heading">
        <div>
          <p className="practice-eyebrow">Live camera · physical placement</p>
          <h1 id="practice-title">Set up your practice layout</h1>
          <p>Place target slots over the live table image. Detected ball centers are checked against the calibrated table plane.</p>
        </div>
        <span className={`practice-readiness-badge${ready ? ' is-ready' : ''}`} data-testid="practice-readiness-badge">
          {ready ? 'READY · LIVE PLACEMENT' : 'UNVERIFIED · CAMERA EVIDENCE REQUIRED'}
        </span>
      </header>

      <div className="practice-grid">
        <section className="practice-table-card" aria-label="Editable camera practice layout">
          <div className="practice-table-tools" role="group" aria-label="Practice target placement tool">
            <button type="button" className={placementMode === 'cue' ? 'selected' : ''} aria-pressed={placementMode === 'cue'} onClick={() => { setPlacementMode('cue'); setPlacementTargetId('cue'); }}>Place cue target</button>
            <button type="button" className={placementMode === 'object' ? 'selected' : ''} aria-pressed={placementMode === 'object'} onClick={() => setPlacementMode('object')}>Add object target</button>
            <button type="button" onClick={clearObjectTargets}>Clear object targets</button>
          </div>
          <p className="practice-instructions" id="practice-instructions">{placementModeText(placementMode)}</p>
          <form className="practice-coordinate-entry" aria-label="Place target by table coordinates" onSubmit={(event) => { event.preventDefault(); placeAt(placementX, placementY); }}>
            <label>Table X (mm)<input type="number" step="any" min={table.ballRadius} max={table.width - table.ballRadius} value={placementX} onChange={(event) => setPlacementX(Number(event.target.value))} /></label>
            <label>Table Y (mm)<input type="number" step="any" min={table.ballRadius} max={table.height - table.ballRadius} value={placementY} onChange={(event) => setPlacementY(Number(event.target.value))} /></label>
            <button type="submit">{placementMode === 'object' ? 'Add target at coordinates' : 'Place selected target'}</button>
          </form>

          <PracticeCameraOverlay
            image={image}
            imageAvailable={imageAvailable}
            mode={delivery.mode}
            dimensions={dimensions}
            projection={overlayProjection}
            tableRadiusMm={table.ballRadius}
            toleranceMm={PLACEMENT_POLICY.toleranceMm}
            targets={assignedTargets}
            balls={overlayBalls}
            canEditOnImage={!!overlayProjection}
            ready={ready}
            notice={notice}
            readinessMessage={cameraStatus}
            onPlaceAt={placeBallOnImage}
            onImageDimensions={updateVideoDimensions}
            shot={cueConfirmed ? selected ?? null : null}
          />

          <div className="practice-ball-list" aria-label="Editable target positions">
            {placements.map(({ target, label }) => <div className="practice-ball-row" key={target.id}>
              <button type="button" className={placementTargetId === target.id ? 'selected' : ''} aria-pressed={placementTargetId === target.id} onClick={() => editTarget(target)}>
                {target.label} · {Math.round(target.x)}, {Math.round(target.y)} mm · {label}
              </button>
              {target.id !== 'cue' && <button type="button" className="practice-remove-ball" aria-label={`Remove ${target.label}`} onClick={() => removeTarget(target.id)}>Remove</button>}
            </div>)}
            {targets.filter((target) => target.id !== 'cue').length === 0 && <p>No object targets. Choose “Add object target” and click the live table image.</p>}
            {evidenceReady && assessment.extraCenterCount > 0 && <p className="practice-extra-centers">{assessment.extraCenterCount} additional detected center{assessment.extraCenterCount === 1 ? '' : 's'} are not required by this exercise; unassigned centers still block predicted shot paths.</p>}
          </div>

          <details className="practice-dimensions">
            <summary>Table dimensions and placement tolerance</summary>
            <div>
              <label>Length (mm)<input type="number" step="any" min="100" disabled={!!projection} value={table.width} onChange={(event) => updateTable({ width: Number(event.target.value) })} /></label>
              <label>Width (mm)<input type="number" step="any" min="100" disabled={!!projection} value={table.height} onChange={(event) => updateTable({ height: Number(event.target.value) })} /></label>
              <label>Ball radius (mm)<input type="number" step="any" min="1" value={table.ballRadius} onChange={(event) => updateTable({ ballRadius: Number(event.target.value) })} /></label>
            </div>
            <small>{projection ? 'Length and width come from the active producer calibration.' : 'Offline dimensions are used for coordinate entry only; camera placement stays unverified without calibration.'} Center-distance tolerance is ±{PLACEMENT_POLICY.toleranceMm} mm; configure VITE_PRACTICE_PLACEMENT_TOLERANCE_MM (5–150 mm) to change it.</small>
          </details>
        </section>

        <aside className="practice-coach-card" aria-label="Placement feedback and shot coaching">
          <h2>Placement feedback</h2>
          <p className="practice-muted">A detected center fills at most one target. Extra detected balls are allowed and remain path blockers.</p>
          <div className={`practice-readiness${ready ? ' is-ready' : ''}`} role="status" aria-live="polite" data-testid="practice-ready-status">
            <strong>{ready ? 'Ready' : 'Not verified'}</strong>
            <span>{cameraStatus}</span>
          </div>

          <label className="practice-control">Cue position
            <span className="practice-cue-confirmation">{cueBall ? `Detected center placed ${matchesByTarget.get('cue')!.distanceMm.toFixed(0)} mm from target` : 'No verified cue-position match'}</span>
          </label>
          <button type="button" className="practice-confirm" disabled={!ready || !cueBall} aria-pressed={cueConfirmed} onClick={() => setCueConfirmed((confirmed) => !confirmed)}>
            {cueConfirmed ? 'Cue position confirmed · click to reset' : 'Confirm cue position'}
          </button>
          <label className="practice-control" htmlFor="practice-target">Object target
            <select id="practice-target" value={targetId} onChange={(event) => setTargetId(event.target.value)}>
              <option value="">Automatic suggestion</option>
              {targets.filter((target) => target.id !== 'cue').map((target) => <option key={target.id} value={target.id}>{target.label}</option>)}
            </select>
          </label>
          <label className="practice-control" htmlFor="practice-pocket">Pocket
            <select id="practice-pocket" value={pocketId} onChange={(event) => setPocketId(event.target.value)}>
              <option value="">Automatic suggestion</option>
              {pockets.map((pocket) => <option key={pocket.id} value={pocket.id}>{pocket.label}</option>)}
            </select>
          </label>

          {!ready ? <div className="practice-coach-result" role="status">
            <strong>Shot guidance is paused until placement is verified.</strong>
            <p>{notice}</p>
          </div> : !cueConfirmed ? <div className="practice-coach-result" role="status">
            <strong>Confirm the cue position to calculate a shot.</strong>
            <p>The suggested line uses the detected centers currently matched to your target slots.</p>
          </div> : selected ? <div className="practice-coach-result" role="status" aria-live="polite">
            <p className="practice-result-label">Suggested direct pot</p>
            <strong>{targets.find((target) => target.id === selected.targetId)?.label ?? 'Selected target'} → {selected.pocket.label}</strong>
            <p>{(selected.cutAngleRadians * 180 / Math.PI).toFixed(1)}° cut · {(selected.totalTravel / 1000).toFixed(2)} m combined travel</p>
            <p>Aim through the projected ghost-ball center. The green line estimates the cue path and the orange line estimates the object-ball path to the pocket.</p>
            <p className="practice-muted">Direct-pot geometry only. Spin, banks, combinations, and post-contact physics are not modeled.</p>
          </div> : <div className="practice-coach-result practice-no-shot" role="status" aria-live="polite">
            <strong>No supported direct pot for this selection.</strong>
            <p>Check the target positions or choose automatic target and pocket. Extra detected centers block paths but are not selected as targets.</p>
          </div>}
        </aside>
      </div>
    </section>
  );
});
