import { describe, expect, it } from 'vitest';
import {
  currentObservationAgeSeconds,
  DEFAULT_MAX_STATE_OBSERVATION_AGE_SECONDS,
  observationIsStale,
  positiveFreshnessLimit,
} from '../../components/video/freshness';

describe('state freshness helpers', () => {
  it('uses a finite positive configured state age and a documented fallback', () => {
    expect(positiveFreshnessLimit(0.25)).toBe(0.25);
    expect(positiveFreshnessLimit(Number.POSITIVE_INFINITY))
      .toBe(DEFAULT_MAX_STATE_OBSERVATION_AGE_SECONDS);
    expect(positiveFreshnessLimit(0)).toBe(DEFAULT_MAX_STATE_OBSERVATION_AGE_SECONDS);
  });

  it('advances age with a monotonic clock and never makes repeated state younger', () => {
    expect(currentObservationAgeSeconds(0.1, 0.1, 1000, 1300)).toBeCloseTo(0.4);
    expect(currentObservationAgeSeconds(0.6, 0.1, 1000, 1300)).toBe(0.6);
    expect(currentObservationAgeSeconds(undefined, null, 1000, 1300)).toBeNull();
  });

  it('expires source-silent and cached positions immediately', () => {
    expect(observationIsStale(0.2, 0.5, { freshness_status: 'source_silent' })).toBe(true);
    expect(observationIsStale(0.2, 0.5, { ball_positions_status: 'cached' })).toBe(true);
    expect(observationIsStale(0.2, 0.5, { ball_positions_status: 'observed' })).toBe(false);
  });
});
