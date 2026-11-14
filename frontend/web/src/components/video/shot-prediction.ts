export interface ShotPoint {
  x: number;
  y: number;
  radius: number;
}

export interface ShotPrediction {
  cueBall: ShotPoint;
  cuePathEnd: { x: number; y: number };
  target?: ShotPoint;
  targetPathEnd?: { x: number; y: number };
}

export interface PracticeBall extends ShotPoint {
  id: string;
}

export interface PracticePocket {
  id: string;
  label: string;
  x: number;
  y: number;
}

export interface PracticeTable {
  width: number;
  height: number;
  ballRadius: number;
}

export interface PotCandidate {
  targetId: string;
  pocketId: string;
  cueBall: PracticeBall;
  target: PracticeBall;
  pocket: PracticePocket;
  /** Cue-ball center at impact (the ghost-ball center). */
  contactCenter: { x: number; y: number };
  /** Contact point on the object ball, useful for separating aim from ball center. */
  contactPoint: { x: number; y: number };
  cuePathEnd: { x: number; y: number };
  objectPathStart: { x: number; y: number };
  objectPathEnd: { x: number; y: number };
  cutAngleRadians: number;
  totalTravel: number;
}

export interface PotSuggestionOptions {
  /** Optional identity filters for the manual target and pocket controls. */
  targetId?: string;
  pocketId?: string;
  /** Live adapters can pass stale balls here so they block corridors without becoming targets. */
  additionalBlockers?: PracticeBall[];
  /** Maximum supported cut angle. Defaults to a right angle. */
  maximumCutAngleRadians?: number;
  /** Ranking priorities; defaults to the coaching policy of cut angle then travel. */
  ranking?: readonly ('cutAngle' | 'travel')[];
}

/**
 * Offline sandbox defaults mirror perception/contracts/configuration/v1/defaults.json:
 * calibrated table width/height and localization ball diameter. The same values
 * may be overridden by a future live configuration adapter.
 */
export const PRACTICE_TABLE_DEFAULTS: Readonly<PracticeTable> = Object.freeze({
  width: 2336.8,
  height: 1155.7,
  ballRadius: 57.15 / 2,
});

/** Six standard pocket mouths, expressed in the table's physical coordinate plane. */
export function standardPockets(table: Pick<PracticeTable, 'width' | 'height'>): PracticePocket[] {
  const midX = table.width / 2;
  return [
    { id: 'top-left', label: 'Top left', x: 0, y: 0 },
    { id: 'top-middle', label: 'Top middle', x: midX, y: 0 },
    { id: 'top-right', label: 'Top right', x: table.width, y: 0 },
    { id: 'bottom-left', label: 'Bottom left', x: 0, y: table.height },
    { id: 'bottom-middle', label: 'Bottom middle', x: midX, y: table.height },
    { id: 'bottom-right', label: 'Bottom right', x: table.width, y: table.height },
  ];
}

const DEFAULT_MAXIMUM_CUT_ANGLE_RADIANS = Math.PI / 2;
const DEFAULT_RANKING: readonly ('cutAngle' | 'travel')[] = ['cutAngle', 'travel'];

function distance(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

/** True when a third ball's physical radius intersects the open travel segment. */
function segmentIsClear(
  start: { x: number; y: number },
  end: { x: number; y: number },
  movingRadius: number,
  blockers: PracticeBall[],
): boolean {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= 0) return false;
  return blockers.every((ball) => {
    const projection = Math.max(0, Math.min(1,
      ((ball.x - start.x) * dx + (ball.y - start.y) * dy) / lengthSquared,
    ));
    const nearestX = start.x + projection * dx;
    const nearestY = start.y + projection * dy;
    const minimumSeparation = movingRadius + ball.radius;
    // Tangency is considered clear; any positive overlap blocks the predicted path.
    return (ball.x - nearestX) ** 2 + (ball.y - nearestY) ** 2
      >= minimumSeparation ** 2 - 1e-7;
  });
}

function isBallCenterInBounds(ball: PracticeBall, table: PracticeTable): boolean {
  return ball.x >= ball.radius && ball.x <= table.width - ball.radius
    && ball.y >= ball.radius && ball.y <= table.height - ball.radius
    && ball.radius > 0;
}

/**
 * Enumerate feasible direct pots in table-plane coordinates. Candidates are
 * ordered by cut angle, then total travel, then stable target/pocket IDs.
 * Invalid/out-of-bounds layouts and paths intersecting any third ball are rejected.
 */
export function suggestDirectPots(
  cueBall: PracticeBall | null,
  objectBalls: PracticeBall[],
  table: PracticeTable,
  options: PotSuggestionOptions = {},
): PotCandidate[] {
  if (!cueBall || !Number.isFinite(table.width) || !Number.isFinite(table.height)
    || table.width <= 0 || table.height <= 0 || table.ballRadius <= 0
    || !isBallCenterInBounds(cueBall, table)) return [];

  const pockets = standardPockets(table);
  // Config can narrow the coach's supported range, but can never opt into a backward hit.
  const configuredMaximumCut = Number.isFinite(options.maximumCutAngleRadians)
    ? Math.max(0, options.maximumCutAngleRadians as number)
    : DEFAULT_MAXIMUM_CUT_ANGLE_RADIANS;
  const maximumCutAngle = Math.min(configuredMaximumCut, DEFAULT_MAXIMUM_CUT_ANGLE_RADIANS);
  const candidates: PotCandidate[] = [];
  const blockers = [...objectBalls, ...(options.additionalBlockers ?? [])];

  for (const target of objectBalls) {
    if (target.id === cueBall.id || !isBallCenterInBounds(target, table)) continue;
    const pocketCandidates = options.pocketId
      ? pockets.filter((pocket) => pocket.id === options.pocketId)
      : pockets;
    if (options.targetId && target.id !== options.targetId) continue;

    for (const pocket of pocketCandidates) {
      const pocketDistance = distance(target, pocket);
      if (pocketDistance <= 0) continue;
      const objectDirection = { x: (pocket.x - target.x) / pocketDistance, y: (pocket.y - target.y) / pocketDistance };
      const contactCenter = {
        x: target.x - objectDirection.x * (cueBall.radius + target.radius),
        y: target.y - objectDirection.y * (cueBall.radius + target.radius),
      };
      if (!isBallCenterInBounds({ ...cueBall, ...contactCenter }, table)
        || distance(cueBall, target) < cueBall.radius + target.radius) continue;
      const cueTravel = distance(cueBall, contactCenter);
      if (cueTravel <= 0) continue;
      const cueDirection = { x: (contactCenter.x - cueBall.x) / cueTravel, y: (contactCenter.y - cueBall.y) / cueTravel };
      const forwardTransfer = cueDirection.x * objectDirection.x + cueDirection.y * objectDirection.y;
      // A tangent touch transfers no forward velocity, even if the configured
      // cut-angle limit permits ninety degrees.
      if (forwardTransfer <= 0) continue;
      const cutAngleRadians = Math.acos(Math.max(-1, Math.min(1, forwardTransfer)));
      if (cutAngleRadians > maximumCutAngle) continue;

      const contactPoint = {
        x: target.x - objectDirection.x * target.radius,
        y: target.y - objectDirection.y * target.radius,
      };
      const cueBlockers = blockers.filter((ball) => ball.id !== target.id);
      const objectBlockers = [cueBall, ...blockers.filter((ball) => ball.id !== target.id)];
      if (!segmentIsClear(cueBall, contactCenter, cueBall.radius, cueBlockers)
        || !segmentIsClear(target, pocket, target.radius, objectBlockers)) continue;

      candidates.push({
        targetId: target.id,
        pocketId: pocket.id,
        cueBall,
        target,
        pocket,
        contactCenter,
        contactPoint,
        cuePathEnd: contactCenter,
        objectPathStart: target,
        objectPathEnd: pocket,
        cutAngleRadians,
        totalTravel: cueTravel + pocketDistance,
      });
    }
  }

  const ranking = options.ranking ?? DEFAULT_RANKING;
  return candidates.sort((a, b) => {
    for (const priority of ranking) {
      const difference = priority === 'cutAngle'
        ? a.cutAngleRadians - b.cutAngleRadians
        : a.totalTravel - b.totalTravel;
      if (difference !== 0) return difference;
    }
    return a.targetId.localeCompare(b.targetId) || a.pocketId.localeCompare(b.pocketId);
  });
}

function distanceToEdge(
  origin: ShotPoint,
  direction: { x: number; y: number },
  width: number,
  height: number,
  table?: Array<{ x: number; y: number }>,
): number {
  if (table && table.length >= 3) {
    let nearest = Infinity;
    for (let index = 0; index < table.length; index += 1) {
      const start = table[index];
      const end = table[(index + 1) % table.length];
      const edge = { x: end.x - start.x, y: end.y - start.y };
      const offset = { x: start.x - origin.x, y: start.y - origin.y };
      const denominator = direction.x * edge.y - direction.y * edge.x;
      if (Math.abs(denominator) < 1e-8) continue;
      const along = (offset.x * edge.y - offset.y * edge.x) / denominator;
      const edgePosition = (offset.x * direction.y - offset.y * direction.x) / denominator;
      if (along >= 0 && edgePosition >= 0 && edgePosition <= 1) nearest = Math.min(nearest, along);
    }
    if (Number.isFinite(nearest)) return nearest;
  }
  const candidates = [
    direction.x > 0 ? (width - origin.x) / direction.x : direction.x < 0 ? -origin.x / direction.x : Infinity,
    direction.y > 0 ? (height - origin.y) / direction.y : direction.y < 0 ? -origin.y / direction.y : Infinity,
  ].filter((distance) => distance >= 0);
  return Math.min(...candidates);
}

/** Predict first contact with circular balls in the calibrated image plane. */
export function predictFirstContact(
  cueBall: ShotPoint,
  objectBalls: ShotPoint[],
  direction: { x: number; y: number },
  width: number,
  height: number,
  table?: Array<{ x: number; y: number }>,
): ShotPrediction | null {
  const length = Math.hypot(direction.x, direction.y);
  if (!length || !width || !height) return null;
  const unit = { x: direction.x / length, y: direction.y / length };
  const edgeDistance = distanceToEdge(cueBall, unit, width, height, table);
  let first: { ball: ShotPoint; distance: number } | null = null;
  for (const ball of objectBalls) {
    const offset = { x: ball.x - cueBall.x, y: ball.y - cueBall.y };
    const along = offset.x * unit.x + offset.y * unit.y;
    const perpendicularSquared = offset.x ** 2 + offset.y ** 2 - along ** 2;
    const combinedRadius = cueBall.radius + ball.radius;
    if (along <= 0 || perpendicularSquared > combinedRadius ** 2) continue;
    const distance = along - Math.sqrt(combinedRadius ** 2 - perpendicularSquared);
    if (distance >= 0 && distance <= edgeDistance && (!first || distance < first.distance)) {
      first = { ball, distance };
    }
  }
  const cuePathEnd = {
    x: cueBall.x + unit.x * (first?.distance ?? edgeDistance),
    y: cueBall.y + unit.y * (first?.distance ?? edgeDistance),
  };
  if (!first) return { cueBall, cuePathEnd };

  const cueContact = { x: cuePathEnd.x, y: cuePathEnd.y };
  const normalLength = Math.hypot(first.ball.x - cueContact.x, first.ball.y - cueContact.y);
  if (!normalLength) return { cueBall, cuePathEnd };
  const objectDirection = {
    x: (first.ball.x - cueContact.x) / normalLength,
    y: (first.ball.y - cueContact.y) / normalLength,
  };
  const objectEdgeDistance = distanceToEdge(first.ball, objectDirection, width, height, table);
  return {
    cueBall,
    cuePathEnd,
    target: first.ball,
    targetPathEnd: {
      x: first.ball.x + objectDirection.x * objectEdgeDistance,
      y: first.ball.y + objectDirection.y * objectEdgeDistance,
    },
  };
}
