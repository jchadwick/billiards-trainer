import { describe, expect, it } from 'vitest';
import {
  DEFAULT_PRACTICE_PLACEMENT_POLICY,
  assessPracticePlacements,
  getPracticePlacementPolicy,
  practiceEvidenceIsEligible,
  practiceLayoutIsReady,
} from './practice-placement';
import type { PracticeEvidenceGates } from './practice-placement';

const currentEvidence: PracticeEvidenceGates = {
  cameraStreamReady: true,
  producerConnected: true,
  sourceContextCompatible: true,
  projectionCompatible: true,
  decisionCompatible: true,
  layoutTrusted: true,
  phase: 'pre_shot',
  motionAgeMs: 250,
  motionMaximumAgeMs: 500,
};

describe('practice placement matching', () => {
  it('uses the configurable physical tolerance and a documented fallback', () => {
    expect(DEFAULT_PRACTICE_PLACEMENT_POLICY.toleranceMm).toBe(45);
    expect(getPracticePlacementPolicy({ VITE_PRACTICE_PLACEMENT_TOLERANCE_MM: '72' }).toleranceMm).toBe(72);
    expect(getPracticePlacementPolicy({ VITE_PRACTICE_PLACEMENT_TOLERANCE_MM: '4.9' }).toleranceMm).toBe(45);
    expect(getPracticePlacementPolicy({ VITE_PRACTICE_PLACEMENT_TOLERANCE_MM: 'NaN' }).toleranceMm).toBe(45);
  });

  it('never lets one center satisfy two overlapping target regions', () => {
    const result = assessPracticePlacements([
      { id: 'left', x: 0, y: 0 },
      { id: 'right', x: 10, y: 0 },
    ], [{ x: 5, y: 0, sourceId: 'detector-a' }], 10);

    expect(result.matches).toHaveLength(1);
    expect(result.unmatchedTargetIds).toEqual(['right']);
    expect(result.extraCenterCount).toBe(0);
  });

  it('uses an augmenting assignment to maximize placements when regions overlap', () => {
    const result = assessPracticePlacements([
      { id: 'left', x: 0, y: 0 },
      { id: 'right', x: 10, y: 0 },
    ], [
      { x: 5, y: 0, sourceId: 'shared-neighbor' },
      { x: -9, y: 0, sourceId: 'left-only' },
    ], 10);

    expect(result.matches).toHaveLength(2);
    expect(result.unmatchedTargetIds).toEqual([]);
    expect(result.matches.find((match) => match.targetId === 'right')?.measuredIndex).toBe(0);
    expect(result.extraCenterCount).toBe(0);
  });

  it('accepts the physical tolerance boundary and reports additional centers without requiring them', () => {
    const result = assessPracticePlacements(
      [{ id: 'cue-slot', x: 100, y: 100 }],
      [{ x: 145, y: 100 }, { x: 800, y: 300 }],
      45,
    );
    expect(result.matches[0]).toMatchObject({ targetId: 'cue-slot', distanceMm: 45 });
    expect(result.extraCenterCount).toBe(1);
  });

  it('rejects centers outside the physical tolerance', () => {
    const result = assessPracticePlacements([{ id: 'slot', x: 0, y: 0 }], [{ x: 46, y: 0 }], 45);
    expect(result.matches).toHaveLength(0);
    expect(result.unmatchedTargetIds).toEqual(['slot']);
  });
});

describe('practice evidence eligibility', () => {
  it('requires connected camera/state and matching current calibration context', () => {
    expect(practiceEvidenceIsEligible(currentEvidence)).toBe(true);
    for (const failure of [
      { cameraStreamReady: false },
      { producerConnected: false },
      { sourceContextCompatible: false },
      { projectionCompatible: false },
      { decisionCompatible: false },
      { layoutTrusted: false },
    ]) {
      expect(practiceEvidenceIsEligible({ ...currentEvidence, ...failure })).toBe(false);
    }
  });

  it('expires motion evidence, recovery, and shot motion without relying on a new ball scan', () => {
    expect(practiceEvidenceIsEligible({ ...currentEvidence, motionAgeMs: 500 })).toBe(true);
    expect(practiceEvidenceIsEligible({ ...currentEvidence, motionAgeMs: 501 })).toBe(false);
    expect(practiceEvidenceIsEligible({ ...currentEvidence, motionAgeMs: null })).toBe(false);
    expect(practiceEvidenceIsEligible({ ...currentEvidence, phase: 'recovering' })).toBe(false);
    expect(practiceEvidenceIsEligible({ ...currentEvidence, phase: 'shot_underway' })).toBe(false);
  });

  it('withdraws Ready as soon as evidence expires, even when a prior layout was fully matched', () => {
    const complete = assessPracticePlacements(
      [{ id: 'cue', x: 0, y: 0 }, { id: 'target', x: 200, y: 0 }],
      [{ x: 1, y: 0 }, { x: 201, y: 0 }, { x: 900, y: 300 }],
      45,
    );
    expect(practiceLayoutIsReady(practiceEvidenceIsEligible(currentEvidence), 2, complete)).toBe(true);
    expect(practiceLayoutIsReady(practiceEvidenceIsEligible({ ...currentEvidence, motionAgeMs: 501 }), 2, complete)).toBe(false);
    expect(practiceLayoutIsReady(practiceEvidenceIsEligible({ ...currentEvidence, sourceContextCompatible: false }), 2, complete)).toBe(false);
    expect(practiceLayoutIsReady(true, 3, complete)).toBe(false);
  });
});
