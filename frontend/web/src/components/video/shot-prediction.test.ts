import { describe, expect, it } from 'vitest';
import { PRACTICE_TABLE_DEFAULTS, predictFirstContact, standardPockets, suggestDirectPots } from './shot-prediction';

describe('predictFirstContact', () => {
  const cue = { x: 10, y: 50, radius: 2 };

  it('predicts a straight contact and the object ball path', () => {
    const prediction = predictFirstContact(cue, [{ x: 40, y: 50, radius: 2 }], { x: 1, y: 0 }, 100, 100);
    expect(prediction?.target).toEqual({ x: 40, y: 50, radius: 2 });
    expect(prediction?.cuePathEnd).toEqual({ x: 36, y: 50 });
    expect(prediction?.targetPathEnd?.x).toBe(100);
    expect(prediction?.targetPathEnd?.y).toBeCloseTo(50);
  });

  it('selects the nearest of multiple balls and predicts cut angles on either side', () => {
    const nearest = predictFirstContact(cue, [
      { x: 70, y: 50, radius: 2 }, { x: 40, y: 52, radius: 2 },
    ], { x: 1, y: 0 }, 100, 100);
    expect(nearest?.target?.x).toBe(40);
    expect(nearest?.targetPathEnd?.y).toBeGreaterThan(52);

    const right = predictFirstContact(cue, [{ x: 40, y: 48, radius: 2 }], { x: 1, y: 0 }, 100, 100);
    expect(right?.targetPathEnd?.y).toBeLessThan(48);
  });

  it('stops at the cushion for a miss and rejects an invalid direction', () => {
    expect(predictFirstContact(cue, [], { x: 1, y: 0 }, 100, 100)?.cuePathEnd)
      .toEqual({ x: 100, y: 50 });
    expect(predictFirstContact(cue, [], { x: 0, y: 0 }, 100, 100)).toBeNull();
  });

  it('uses the calibrated table boundary when supplied', () => {
    const table = [
      { x: 20, y: 20 }, { x: 80, y: 20 },
      { x: 80, y: 80 }, { x: 20, y: 80 },
    ];
    expect(predictFirstContact({ x: 30, y: 50, radius: 2 }, [], { x: 1, y: 0 }, 100, 100, table)?.cuePathEnd)
      .toEqual({ x: 80, y: 50 });
  });
});

describe('suggestDirectPots', () => {
  const table = { width: 1000, height: 500, ballRadius: 10 };
  const cue = { id: 'cue', x: 500, y: 400, radius: 10 };
  const target = { id: 'target', x: 500, y: 250, radius: 10 };

  it('finds a straight pot with physical ghost-ball and contact geometry', () => {
    const candidates = suggestDirectPots(cue, [target], table, { pocketId: 'top-middle' });
    expect(candidates).toHaveLength(1);
    expect(candidates[0].cutAngleRadians).toBeCloseTo(0);
    expect(candidates[0].contactCenter).toEqual({ x: 500, y: 270 });
    expect(candidates[0].contactPoint).toEqual({ x: 500, y: 260 });
    expect(candidates[0].objectPathEnd).toMatchObject({ x: 500, y: 0, id: 'top-middle' });
  });

  it('supports a cut pot and ranks the straighter option first', () => {
    const cut = suggestDirectPots({ ...cue, x: 200, y: 150 }, [target], table, { pocketId: 'bottom-middle' })[0];
    expect(cut.cutAngleRadians).toBeGreaterThan(0);
    expect(cut.cutAngleRadians).toBeLessThan(Math.PI / 2);
    const ranked = suggestDirectPots(cue, [target, { ...target, id: 'other', x: 550, y: 250 }], table);
    expect(ranked[0].cutAngleRadians).toBeLessThanOrEqual(ranked[ranked.length - 1].cutAngleRadians);
  });

  it('rejects cue-path and object-path blockers using combined ball radii', () => {
    const cueBlocked = suggestDirectPots(cue, [target, { id: 'block', x: 500, y: 320, radius: 10 }], table, { targetId: 'target', pocketId: 'top-middle' });
    expect(cueBlocked).toHaveLength(0);
    const objectBlocked = suggestDirectPots(cue, [target, { id: 'block', x: 500, y: 100, radius: 10 }], table, { targetId: 'target', pocketId: 'top-middle' });
    expect(objectBlocked).toHaveLength(0);
  });

  it('lets a later live adapter pass stale balls as blockers without targeting them', () => {
    const staleBall = { id: 'stale', x: 500, y: 100, radius: 10 };
    expect(suggestDirectPots(cue, [target], table, {
      targetId: 'target', pocketId: 'top-middle', additionalBlockers: [staleBall],
    })).toHaveLength(0);
  });

  it('returns no candidate for backward, out-of-bounds, and missing cue layouts', () => {
    expect(suggestDirectPots({ ...cue, y: 100 }, [target], table, { targetId: 'target', pocketId: 'top-middle' })).toHaveLength(0);
    expect(suggestDirectPots({ ...cue, y: 100 }, [target], table, {
      targetId: 'target', pocketId: 'top-middle', maximumCutAngleRadians: Math.PI,
    })).toHaveLength(0);
    expect(suggestDirectPots(cue, [{ ...target, x: 0 }], table)).toHaveLength(0);
    expect(suggestDirectPots(null, [target], table)).toHaveLength(0);
  });

  it('rejects a tangent touch with zero forward transfer even with a permissive cut limit', () => {
    expect(suggestDirectPots({ ...cue, x: 400, y: 270 }, [target], table, {
      targetId: 'target', pocketId: 'top-middle', maximumCutAngleRadians: Math.PI,
    })).toHaveLength(0);
  });

  it('uses deterministic identity ordering and supports manual target/pocket filters', () => {
    const a = { ...target, id: 'a' };
    const b = { ...target, id: 'b', x: 650 };
    const ranked = suggestDirectPots(cue, [b, a], table);
    const reversed = suggestDirectPots(cue, [a, b], table);
    expect(ranked.length).toBeGreaterThan(0);
    expect(ranked.map((candidate) => [candidate.targetId, candidate.pocketId]))
      .toEqual(reversed.map((candidate) => [candidate.targetId, candidate.pocketId]));
    const travelFirst = suggestDirectPots(cue, [a, b], table, { ranking: ['travel', 'cutAngle'] });
    expect(travelFirst).toHaveLength(ranked.length);
    expect(travelFirst.map((candidate) => [candidate.targetId, candidate.pocketId]))
      .toEqual(suggestDirectPots(cue, [a, b], table, { ranking: ['travel', 'cutAngle'] })
        .map((candidate) => [candidate.targetId, candidate.pocketId]));
    expect(suggestDirectPots(cue, [a, b], table, { targetId: 'b', pocketId: 'top-middle' })
      .every((candidate) => candidate.targetId === 'b' && candidate.pocketId === 'top-middle')).toBe(true);
  });

  it('derives six pockets and offline dimensions from the configured physical defaults', () => {
    expect(standardPockets(table)).toHaveLength(6);
    expect(PRACTICE_TABLE_DEFAULTS.ballRadius).toBe(28.575);
    expect(PRACTICE_TABLE_DEFAULTS.width).toBeGreaterThan(PRACTICE_TABLE_DEFAULTS.height);
  });
});
