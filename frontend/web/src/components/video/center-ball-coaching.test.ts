import { describe, expect, it } from 'vitest';
import { assessCenterBallAim, confirmTipNearestCueBall, DEFAULT_LIVE_AIM_POLICY, getLiveAimPolicy, predictCenterBallContact, projectCueToTable, type TableCue } from './center-ball-coaching';
import { suggestDirectPots } from './shot-prediction';
import type { TableProjection } from '../../types/api';

const table = { width: 1000, height: 500, ballRadius: 10 };
const cueBall = { id: 'cue', x: 500, y: 400, radius: 10 };
const target = { id: 'target', x: 500, y: 250, radius: 10 };
const cue: TableCue = { start: { x: 500, y: 480 }, end: { x: 500, y: 420 }, direction: 'toward_end', confidence: 0.9 };
const suggestion = suggestDirectPots(cueBall, [target], table, { pocketId: 'top-middle' })[0];

describe('physical center-ball aim', () => {
  it('compares current aim with the desired contact point and supports reversed endpoint ordering', () => {
    const aim = assessCenterBallAim(cue, cueBall, [target], table, suggestion);
    expect(aim.state).toBe('aligned');
    expect(aim.offsetMm).toBeCloseTo(0);
    expect(aim.angleDegrees).toBeCloseTo(0);
    expect(aim.prediction?.target?.id).toBe('target');
    const reversed = assessCenterBallAim({ ...cue, start: cue.end, end: cue.start, direction: 'toward_start' }, cueBall, [target], table, suggestion);
    expect(reversed).toEqual(aim);
  });

  it('reports angular error when the shaft passes through center but differs from the suggested aim', () => {
    const radians = 4 * Math.PI / 180;
    const unit = { x: Math.sin(radians), y: -Math.cos(radians) };
    const angled = { ...cue, start: { x: cueBall.x - unit.x * 80, y: cueBall.y - unit.y * 80 }, end: { x: cueBall.x - unit.x * 20, y: cueBall.y - unit.y * 20 } };
    const aim = assessCenterBallAim(angled, cueBall, [target], table, suggestion);
    expect(aim.state).toBe('adjust');
    expect(aim.angleDegrees).toBeCloseTo(4);
    expect(aim.offsetMm).toBeCloseTo(0);
    expect(assessCenterBallAim(angled, cueBall, [target], table, suggestion, { ...DEFAULT_LIVE_AIM_POLICY, angleToleranceDegrees: 5 }).state).toBe('aligned');
  });

  it('suppresses prediction for an unrelated parallel shaft and a reversed/away cue', () => {
    const offCenter = assessCenterBallAim({ ...cue, start: { x: 510, y: 480 }, end: { x: 510, y: 420 } }, cueBall, [target], table, suggestion);
    expect(offCenter.state).toBe('off_center');
    expect(offCenter.offsetMm).toBeCloseTo(10);
    expect(offCenter.prediction).toBeUndefined();
    const away = assessCenterBallAim({ ...cue, direction: 'toward_start' }, cueBall, [target], table, suggestion);
    expect(away.state).toBe('away');
    expect(away.prediction).toBeUndefined();
  });

  it('requires a signed direction or an explicit nearest-tip confirmation', () => {
    const unsigned = { ...cue, direction: null };
    expect(assessCenterBallAim(unsigned, cueBall, [target], table, suggestion).state).toBe('direction_unknown');
    expect(assessCenterBallAim(confirmTipNearestCueBall(unsigned, cueBall), cueBall, [target], table, suggestion).state).toBe('aligned');
    expect(confirmTipNearestCueBall(cue, cueBall)).toEqual(cue);
    expect(confirmTipNearestCueBall({ ...unsigned, start: { x: 480, y: 400 }, end: { x: 520, y: 400 } }, cueBall).direction).toBeNull();
  });

  it('never calls an aim aligned when it hits a different ball or misses the selected target', () => {
    const blocker = { id: 'blocker', x: 500, y: 330, radius: 10 };
    const aim = assessCenterBallAim(cue, cueBall, [target, blocker], table, suggestion);
    expect(aim.state).toBe('adjust');
    expect(aim.wrongTarget).toBe(true);
    expect(aim.prediction?.target?.id).toBe('blocker');
    const miss = assessCenterBallAim(cue, cueBall, [{ ...target, x: 700 }], table, suggestion);
    expect(miss.state).toBe('adjust');
    expect(miss.prediction?.target).toBeUndefined();
  });

  it('rejects degenerate, nonfinite, low-confidence, and distant cue geometry', () => {
    for (const invalid of [
      { ...cue, start: cue.end },
      { ...cue, start: { x: NaN, y: 400 } },
      { ...cue, confidence: 0.1 },
      { ...cue, start: { x: 500, y: 1400 }, end: { x: 500, y: 1300 } },
    ]) {
      expect(assessCenterBallAim(invalid, cueBall, [target], table, suggestion).state).toBe('unavailable');
    }
    expect(assessCenterBallAim(null, cueBall, [target], table, suggestion).prediction).toBeUndefined();
    expect(assessCenterBallAim(cue, cueBall, [target], table, null).state).toBe('ready');
  });
});

describe('physical initial trajectories', () => {
  it('finds the nearest impact and clips the object path at its next ball contact', () => {
    const next = { id: 'next', x: 500, y: 100, radius: 10 };
    const prediction = predictCenterBallContact(cueBall, [next, target], { x: 0, y: -1 }, table)!;
    expect(prediction.cueTravel.end).toEqual({ x: 500, y: 270 });
    expect(prediction.target?.id).toBe('target');
    expect(prediction.objectTravel?.end).toEqual({ x: 500, y: 120 });
    expect(prediction.objectTravel?.hitBall?.id).toBe('next');
  });

  it('uses the collision normal for cuts on both sides', () => {
    const right = predictCenterBallContact(cueBall, [{ ...target, x: 510 }], { x: 0, y: -1 }, table)!;
    const left = predictCenterBallContact(cueBall, [{ ...target, x: 490 }], { x: 0, y: -1 }, table)!;
    expect(right.objectTravel!.end.x).toBeGreaterThan(510);
    expect(left.objectTravel!.end.x).toBeLessThan(490);
    expect(right.objectTravel!.end.y).toBeCloseTo(10);
    expect(left.objectTravel!.end.y).toBeCloseTo(10);
  });

  it('stops the ball center one radius before a cushion and does not treat tangency as a driven collision', () => {
    const miss = predictCenterBallContact(cueBall, [], { x: 1, y: 0 }, table)!;
    expect(miss.cueTravel.end).toEqual({ x: 990, y: 400 });
    const tangent = predictCenterBallContact(cueBall, [{ ...target, x: 520 }], { x: 0, y: -1 }, table)!;
    expect(tangent.target).toBeUndefined();
    const hit = predictCenterBallContact(cueBall, [target], { x: 0, y: -1 }, table)!;
    expect(hit.objectTravel?.end).toEqual({ x: 500, y: 10 });
  });

  it('rejects overlapping or invalid layouts and invalid directions', () => {
    expect(predictCenterBallContact(cueBall, [{ ...target, y: 395 }], { x: 0, y: -1 }, table)).toBeNull();
    expect(predictCenterBallContact(cueBall, [{ ...target, radius: -1 }], { x: 0, y: -1 }, table)).toBeNull();
    expect(predictCenterBallContact(cueBall, [{ ...target, x: 0 }], { x: 0, y: -1 }, table)).toBeNull();
    expect(predictCenterBallContact(cueBall, [target], { x: Infinity, y: 0 }, table)).toBeNull();
    expect(predictCenterBallContact(cueBall, [target], { x: 0, y: 0 }, table)).toBeNull();
  });
});

describe('live aim policy and coordinates', () => {
  it('validates configurable tolerances with defaults for absent or invalid values', () => {
    expect(getLiveAimPolicy({})).toEqual(DEFAULT_LIVE_AIM_POLICY);
    expect(getLiveAimPolicy({ VITE_COACH_ANGLE_TOLERANCE_DEG: 'NaN', VITE_COACH_CENTER_TOLERANCE_RADIUS: '-1', VITE_COACH_MAX_TIP_GAP_MM: ' ', VITE_COACH_MIN_CUE_CONFIDENCE: '2' })).toEqual(DEFAULT_LIVE_AIM_POLICY);
    expect(getLiveAimPolicy({ VITE_COACH_ANGLE_TOLERANCE_DEG: '1', VITE_COACH_CENTER_TOLERANCE_RADIUS: '0.1', VITE_COACH_MAX_TIP_GAP_MM: '300', VITE_COACH_MIN_CUE_CONFIDENCE: '0.8' })).toEqual({ angleToleranceDegrees: 1, centerToleranceRadius: 0.1, maximumTipGapMm: 300, minimumCueConfidence: 0.8 });
  });

  it('scales source pixels before homography and shifts center-origin table coordinates', () => {
    const projection: TableProjection = {
      direction: 'source_image_px_to_table_mm', image_space: 'raw', source_width_px: 2000, source_height_px: 1000,
      camera_profile_id: 'camera', calibration_revision: 1, rectification_revision: 1,
      table_width_mm: 1000, table_height_mm: 500, unit: 'millimeters', origin: 'playing_surface_center',
      x_axis: 'corner_0_to_1', y_axis: 'corner_0_to_3',
      pixel_to_table_mm: [[0.5, 0, -500], [0, 0.5, -250], [0, 0, 1]],
    };
    const projected = projectCueToTable({ shaft_start_px: [500, 480], shaft_end_px: [500, 420], axis_unit: [0, -1], confidence: 0.9, observation_age_ms: 0, signed_direction: 'toward_end' }, projection, 1000, 500);
    expect(projected).toEqual(cue);
    expect(assessCenterBallAim(projected, cueBall, [target], table, suggestion).state).toBe('aligned');
    expect(projectCueToTable({ shaft_start_px: [0, 0], shaft_end_px: [1, 1], axis_unit: [1, 1], confidence: 1, observation_age_ms: 0 }, projection, 0, 500)).toBeNull();
  });
});
