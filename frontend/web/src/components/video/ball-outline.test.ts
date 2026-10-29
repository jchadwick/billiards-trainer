import { expect, it } from 'vitest';
import { observedBallOutline } from './ball-outline';
import type { BallData } from '../../types/api';
const frame = { frameId: '7', engineInstanceId: 'engine', sourceSessionId: 'session', configurationRevision: 1, calibrationRevision: 1, rectificationRevision: 0 };
const ball: Pick<BallData, 'outline' | 'source_observation' | 'is_observed'> = {
  is_observed: true, source_observation: { frame_id: '7', engine_instance_id: 'engine', session_id: 'session' },
  outline: { scale: [3, 3], points: [{ x: 10, y: 20 }, { x: 30, y: 25 }, { x: 22, y: 30 }, { x: 5, y: 24 }] },
};
it('preserves asymmetric source geometry through canonical and viewer scaling', () => {
  expect(observedBallOutline(ball, frame, { width: 1920, height: 1080 })).toEqual([
    { x: 15, y: 30 }, { x: 45, y: 37.5 }, { x: 33, y: 45 }, { x: 7.5, y: 36 },
  ]);
});
it('withholds stale frame, session and retained geometry', () => {
  for (const other of [{ ...frame, frameId: '8' }, { ...frame, sourceSessionId: 'other' }, null]) {
    expect(observedBallOutline(ball, other, { width: 1920, height: 1080 })).toBeNull();
  }
  expect(observedBallOutline({ ...ball, is_observed: false }, frame, { width: 1920, height: 1080 })).toBeNull();
});
it('leaves clipping to canvas and rejects malformed or absent contours', () => {
  const outside = { ...ball, outline: { scale: [1, 1] as [number, number], points: [{ x: -5, y: 2 }, { x: 5, y: 3 }, { x: 0, y: 8 }] } };
  expect(observedBallOutline(outside, frame, { width: 3840, height: 2160 })?.[0].x).toBe(-5);
  expect(observedBallOutline({ ...ball, outline: undefined }, frame, { width: 3840, height: 2160 })).toBeNull();
  expect(observedBallOutline({ ...ball, outline: { scale: [1, 1], points: [{ x: NaN, y: 1 }, { x: 2, y: 1 }, { x: 1, y: 2 }] } }, frame, { width: 3840, height: 2160 })).toBeNull();
});
