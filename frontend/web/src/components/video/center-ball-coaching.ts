import type { PerceptionState, TableProjection } from '../../types/api';
import { centerToTable, imageToTableMm, type Point2 } from './live-coaching-geometry';
import type { PotCandidate, PracticeBall, PracticeTable } from './shot-prediction';

export interface LiveAimPolicy {
  angleToleranceDegrees: number;
  centerToleranceRadius: number;
  maximumTipGapMm: number;
  minimumCueConfidence: number;
}

export const DEFAULT_LIVE_AIM_POLICY: Readonly<LiveAimPolicy> = Object.freeze({
  angleToleranceDegrees: 2,
  centerToleranceRadius: 0.25,
  maximumTipGapMm: 600,
  minimumCueConfidence: 0.5,
});

/** Product tolerances are adjustable; invalid build settings use safe defaults. */
export function getLiveAimPolicy(env: Record<string, unknown> = import.meta.env): LiveAimPolicy {
  function setting(name: string, fallback: number, minimum: number, maximum: number): number {
    const raw = env[name];
    const value = typeof raw === 'string' && raw.trim() !== '' ? Number(raw) : NaN;
    return Number.isFinite(value) && value >= minimum && value <= maximum ? value : fallback;
  }
  return {
    angleToleranceDegrees: setting('VITE_COACH_ANGLE_TOLERANCE_DEG', DEFAULT_LIVE_AIM_POLICY.angleToleranceDegrees, 0.1, 10),
    centerToleranceRadius: setting('VITE_COACH_CENTER_TOLERANCE_RADIUS', DEFAULT_LIVE_AIM_POLICY.centerToleranceRadius, 0.01, 1),
    maximumTipGapMm: setting('VITE_COACH_MAX_TIP_GAP_MM', DEFAULT_LIVE_AIM_POLICY.maximumTipGapMm, 10, 2000),
    minimumCueConfidence: setting('VITE_COACH_MIN_CUE_CONFIDENCE', DEFAULT_LIVE_AIM_POLICY.minimumCueConfidence, 0, 1),
  };
}

export interface TableCue {
  start: Point2;
  end: Point2;
  direction?: 'toward_start' | 'toward_end' | null;
  confidence: number;
}

/** Only call after the player explicitly confirms which end is their tip. */
export function confirmTipNearestCueBall(cue: TableCue, cueBall: PracticeBall): TableCue {
  if (cue.direction) return cue;
  const startDistance = Math.hypot(cueBall.x - cue.start.x, cueBall.y - cue.start.y);
  const endDistance = Math.hypot(cueBall.x - cue.end.x, cueBall.y - cue.end.y);
  if (Math.abs(startDistance - endDistance) <= GEOMETRY_EPSILON) return cue;
  return { ...cue, direction: startDistance < endDistance ? 'toward_start' : 'toward_end' };
}

/** Cue evidence and calibration may use different full-frame pixel resolutions. */
export function projectCueToTable(
  cue: NonNullable<PerceptionState['cue']>,
  projection: TableProjection,
  imageWidth: number,
  imageHeight: number,
): TableCue | null {
  if (![imageWidth, imageHeight].every((value) => Number.isFinite(value) && value > 0)) return null;
  const project = (point: [number, number]): Point2 | null => {
    const physical = imageToTableMm(projection, {
      x: point[0] * projection.source_width_px / imageWidth,
      y: point[1] * projection.source_height_px / imageHeight,
    });
    return physical ? centerToTable(physical, projection.table_width_mm, projection.table_height_mm) : null;
  };
  const start = project(cue.shaft_start_px);
  const end = project(cue.shaft_end_px);
  return start && end ? { start, end, direction: cue.signed_direction, confidence: cue.confidence } : null;
}

const GEOMETRY_EPSILON = 1e-7;
function finitePoint(point: Point2): boolean { return Number.isFinite(point.x) && Number.isFinite(point.y); }
function validBall(ball: PracticeBall, table: PracticeTable): boolean {
  return finitePoint(ball) && Number.isFinite(ball.radius) && ball.radius > 0
    && ball.x >= ball.radius && ball.x <= table.width - ball.radius
    && ball.y >= ball.radius && ball.y <= table.height - ball.radius;
}

export interface BallTravel {
  start: Point2;
  end: Point2;
  hitBall?: PracticeBall;
}

export interface CenterBallPrediction {
  cueTravel: BallTravel;
  target?: PracticeBall;
  objectTravel?: BallTravel;
}

/**
 * Trace a ball center until its first finite-radius collision. Cushions bound
 * the center by its radius, rather than extending it through the rail. Pockets
 * and rebounds are deliberately outside this initial-direction model.
 */
function traceBall(ball: PracticeBall, blockers: PracticeBall[], unit: Point2, table: PracticeTable): BallTravel {
  const railDistances = [
    unit.x > 0 ? (table.width - ball.radius - ball.x) / unit.x : unit.x < 0 ? (ball.radius - ball.x) / unit.x : Infinity,
    unit.y > 0 ? (table.height - ball.radius - ball.y) / unit.y : unit.y < 0 ? (ball.radius - ball.y) / unit.y : Infinity,
  ];
  let travel = Math.max(0, Math.min(...railDistances));
  let hitBall: PracticeBall | undefined;
  for (const other of blockers) {
    const dx = other.x - ball.x;
    const dy = other.y - ball.y;
    const along = dx * unit.x + dy * unit.y;
    const perpendicularSquared = Math.max(0, dx * dx + dy * dy - along * along);
    const separation = ball.radius + other.radius;
    if (along <= 0 || perpendicularSquared >= separation * separation - GEOMETRY_EPSILON) continue;
    const contactDistance = along - Math.sqrt(separation * separation - perpendicularSquared);
    if (contactDistance >= -GEOMETRY_EPSILON && contactDistance < travel) {
      travel = Math.max(0, contactDistance);
      hitBall = other;
    }
  }
  return { start: ball, end: { x: ball.x + unit.x * travel, y: ball.y + unit.y * travel }, hitBall };
}

/**
 * Predict incoming cue-ball travel and the first struck ball's initial path.
 * The object path stops at its next collision; later collisions, post-impact
 * cue-ball roll, spin, speed, throw, and stopping positions are not simulated.
 */
export function predictCenterBallContact(
  cueBall: PracticeBall,
  objects: PracticeBall[],
  direction: Point2,
  table: PracticeTable,
): CenterBallPrediction | null {
  const allBalls = [cueBall, ...objects];
  const length = Math.hypot(direction.x, direction.y);
  if (!Number.isFinite(table.width) || !Number.isFinite(table.height) || table.width <= 0 || table.height <= 0
    || !Number.isFinite(length) || length <= GEOMETRY_EPSILON
    || new Set(allBalls.map((ball) => ball.id)).size !== allBalls.length
    || allBalls.some((ball) => !validBall(ball, table))) return null;
  // An overlapping layout has no unambiguous initial collision order.
  if (allBalls.some((ball, index) => allBalls.slice(index + 1).some((other) =>
    Math.hypot(ball.x - other.x, ball.y - other.y) < ball.radius + other.radius - GEOMETRY_EPSILON))) return null;
  const unit = { x: direction.x / length, y: direction.y / length };
  const cueTravel = traceBall(cueBall, objects, unit, table);
  const target = cueTravel.hitBall;
  if (!target) return { cueTravel };
  const normal = { x: target.x - cueTravel.end.x, y: target.y - cueTravel.end.y };
  const normalLength = Math.hypot(normal.x, normal.y);
  const objectTravel = traceBall(target, objects.filter((ball) => ball.id !== target.id), {
    x: normal.x / normalLength, y: normal.y / normalLength,
  }, table);
  return { cueTravel, target, objectTravel };
}

export type AimState = 'unavailable' | 'direction_unknown' | 'away' | 'off_center' | 'adjust' | 'aligned' | 'ready';
export interface AimAssessment {
  state: AimState;
  offsetMm?: number;
  angleDegrees?: number;
  prediction?: CenterBallPrediction;
  wrongTarget?: boolean;
}

/** A parallel shaft elsewhere on the table must never create an aligned result. */
export function assessCenterBallAim(
  cue: TableCue | null,
  cueBall: PracticeBall | null,
  objects: PracticeBall[],
  table: PracticeTable | null,
  suggestion: PotCandidate | null,
  policy: LiveAimPolicy = DEFAULT_LIVE_AIM_POLICY,
): AimAssessment {
  if (!cue || !cueBall || !table || !finitePoint(cue.start) || !finitePoint(cue.end)
    || !Number.isFinite(cue.confidence) || cue.confidence < policy.minimumCueConfidence) return { state: 'unavailable' };
  if (cue.direction !== 'toward_start' && cue.direction !== 'toward_end') return { state: 'direction_unknown' };
  const tip = cue.direction === 'toward_end' ? cue.end : cue.start;
  const back = cue.direction === 'toward_end' ? cue.start : cue.end;
  const length = Math.hypot(tip.x - back.x, tip.y - back.y);
  if (!Number.isFinite(length) || length <= GEOMETRY_EPSILON) return { state: 'unavailable' };
  const unit = { x: (tip.x - back.x) / length, y: (tip.y - back.y) / length };
  const offset = { x: cueBall.x - tip.x, y: cueBall.y - tip.y };
  const forwardDistance = offset.x * unit.x + offset.y * unit.y;
  const offsetMm = Math.abs(offset.x * unit.y - offset.y * unit.x);
  if (forwardDistance < -cueBall.radius) return { state: 'away', offsetMm };
  if (forwardDistance > policy.maximumTipGapMm) return { state: 'unavailable', offsetMm };
  if (offsetMm > cueBall.radius * policy.centerToleranceRadius) return { state: 'off_center', offsetMm };
  const prediction = predictCenterBallContact(cueBall, objects, unit, table);
  if (!prediction) return { state: 'unavailable', offsetMm };
  if (!suggestion) return { state: 'ready', offsetMm, prediction };
  const desired = { x: suggestion.contactCenter.x - cueBall.x, y: suggestion.contactCenter.y - cueBall.y };
  const desiredLength = Math.hypot(desired.x, desired.y);
  if (desiredLength <= GEOMETRY_EPSILON) return { state: 'ready', offsetMm, prediction };
  const angleDegrees = Math.acos(Math.max(-1, Math.min(1,
    (unit.x * desired.x + unit.y * desired.y) / desiredLength))) * 180 / Math.PI;
  const wrongTarget = prediction.target?.id !== suggestion.targetId;
  return {
    state: !wrongTarget && angleDegrees <= policy.angleToleranceDegrees ? 'aligned' : 'adjust',
    offsetMm, angleDegrees, prediction, wrongTarget,
  };
}
