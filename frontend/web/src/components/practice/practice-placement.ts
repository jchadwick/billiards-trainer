import type { Point2 } from '../video/live-coaching-geometry';

/** User-tunable physical center-distance tolerance for a placed ball. */
export interface PracticePlacementPolicy {
  toleranceMm: number;
}

export const DEFAULT_PRACTICE_PLACEMENT_POLICY: Readonly<PracticePlacementPolicy> = Object.freeze({
  // A useful setup tolerance for hand placement; it is intentionally much
  // larger than detector precision and remains configurable per installation.
  toleranceMm: 45,
});

export function getPracticePlacementPolicy(
  env: Record<string, unknown> = import.meta.env,
): PracticePlacementPolicy {
  const raw = env.VITE_PRACTICE_PLACEMENT_TOLERANCE_MM;
  const value = typeof raw === 'string' && raw.trim() !== '' ? Number(raw) : NaN;
  return {
    toleranceMm: Number.isFinite(value) && value >= 5 && value <= 150
      ? value
      : DEFAULT_PRACTICE_PLACEMENT_POLICY.toleranceMm,
  };
}

export interface PracticePlacementTarget extends Point2 {
  id: string;
}

/** Ball centers already normalized to the same physical table coordinate plane as targets. */
export interface PracticeMeasuredCenter extends Point2 {
  /** Used only to break exact geometric ties; no ball class or number is inferred. */
  sourceId?: string;
}

export interface PracticePlacementMatch {
  targetId: string;
  /** Index into the supplied measured center array. */
  measuredIndex: number;
  distanceMm: number;
}

export interface PracticePlacementAssessment {
  matches: PracticePlacementMatch[];
  unmatchedTargetIds: string[];
  nearestDistanceByTarget: ReadonlyMap<string, number | null>;
  /** Extra physical centers are reported but do not block a subset exercise. */
  extraCenterCount: number;
}

export interface PracticeEvidenceGates {
  cameraStreamReady: boolean;
  producerConnected: boolean;
  sourceContextCompatible: boolean;
  projectionCompatible: boolean;
  decisionCompatible: boolean;
  layoutTrusted: boolean;
  phase: 'recovering' | 'pre_shot' | 'aiming' | 'shot_underway' | undefined;
  motionAgeMs: number | null;
  motionMaximumAgeMs: number | undefined;
}

/** One gate for every piece of freshness/context evidence needed by Ready. */
export function practiceEvidenceIsEligible(gates: PracticeEvidenceGates): boolean {
  return gates.cameraStreamReady
    && gates.producerConnected
    && gates.sourceContextCompatible
    && gates.projectionCompatible
    && gates.decisionCompatible
    && gates.layoutTrusted
    && (gates.phase === 'pre_shot' || gates.phase === 'aiming')
    && gates.motionAgeMs !== null
    && Number.isFinite(gates.motionAgeMs)
    && gates.motionAgeMs >= 0
    && gates.motionMaximumAgeMs !== undefined
    && Number.isFinite(gates.motionMaximumAgeMs)
    && gates.motionMaximumAgeMs >= 0
    && gates.motionAgeMs <= gates.motionMaximumAgeMs;
}

/** Additional detections do not block exercises that define only a subset of balls. */
export function practiceLayoutIsReady(
  evidenceEligible: boolean,
  expectedTargetCount: number,
  assessment: PracticePlacementAssessment,
): boolean {
  return evidenceEligible
    && expectedTargetCount > 0
    && assessment.unmatchedTargetIds.length === 0
    && assessment.matches.length === expectedTargetCount;
}

interface RankedCenter extends PracticeMeasuredCenter {
  inputIndex: number;
}

/**
 * Match the largest possible number of placement slots with distinct physical
 * centers. Augmenting paths avoid greedy failures when target tolerance regions
 * overlap (one ball can never make two slots look placed). Candidate ordering
 * by distance and stable geometry makes equal layouts deterministic.
 */
export function assessPracticePlacements(
  targets: readonly PracticePlacementTarget[],
  measuredCenters: readonly PracticeMeasuredCenter[],
  toleranceMm: number,
): PracticePlacementAssessment {
  const safeTolerance = Number.isFinite(toleranceMm) && toleranceMm >= 0 ? toleranceMm : 0;
  const targetOrder = targets
    .map((target, inputIndex) => ({ ...target, inputIndex }))
    .sort((left, right) => left.id.localeCompare(right.id) || left.inputIndex - right.inputIndex);
  const centers = measuredCenters
    .map((center, inputIndex) => ({ ...center, inputIndex }))
    .sort((left, right) => left.x - right.x
      || left.y - right.y
      || (left.sourceId ?? '').localeCompare(right.sourceId ?? '')
      || left.inputIndex - right.inputIndex);

  const nearestDistanceByTarget = new Map<string, number | null>();
  const candidateCenters = new Map<number, Array<{ centerIndex: number; distanceMm: number }>>();
  targetOrder.forEach((target, targetIndex) => {
    const distances = centers.map((center, centerIndex) => ({
      centerIndex,
      distanceMm: Math.hypot(center.x - target.x, center.y - target.y),
    }));
    nearestDistanceByTarget.set(target.id, distances.length
      ? Math.min(...distances.map(({ distanceMm }) => distanceMm))
      : null);
    candidateCenters.set(targetIndex, distances
      .filter(({ distanceMm }) => distanceMm <= safeTolerance)
      .sort((left, right) => left.distanceMm - right.distanceMm || left.centerIndex - right.centerIndex));
  });

  const centerToTarget = new Map<number, number>();
  const assign = (targetIndex: number, visitedCenters: Set<number>): boolean => {
    for (const candidate of candidateCenters.get(targetIndex) ?? []) {
      if (visitedCenters.has(candidate.centerIndex)) continue;
      visitedCenters.add(candidate.centerIndex);
      const priorTarget = centerToTarget.get(candidate.centerIndex);
      if (priorTarget === undefined || assign(priorTarget, visitedCenters)) {
        centerToTarget.set(candidate.centerIndex, targetIndex);
        return true;
      }
    }
    return false;
  };

  // A maximum-cardinality bipartite matching is required when the physical
  // tolerance circles overlap; nearest-first greedy matching can strand slots.
  for (let targetIndex = 0; targetIndex < targetOrder.length; targetIndex += 1) {
    assign(targetIndex, new Set<number>());
  }

  const matches = [...centerToTarget.entries()].map(([centerIndex, targetIndex]) => {
    const target = targetOrder[targetIndex];
    const center = centers[centerIndex];
    return {
      targetId: target.id,
      measuredIndex: center.inputIndex,
      distanceMm: Math.hypot(center.x - target.x, center.y - target.y),
    };
  }).sort((left, right) => left.targetId.localeCompare(right.targetId));
  const matchedTargetIds = new Set(matches.map((match) => match.targetId));
  const unmatchedTargetIds = targets
    .filter((target) => !matchedTargetIds.has(target.id))
    .map((target) => target.id);

  return {
    matches,
    unmatchedTargetIds,
    nearestDistanceByTarget,
    extraCenterCount: Math.max(0, centers.length - matches.length),
  };
}
