import { describe, expect, it } from 'vitest';
import {
  canonicalToImagePoint,
  normalizeImageExtentToCanonical,
  normalizeImagePointToCanonical,
} from '../../utils/coordinates';

describe('live image coordinate handling', () => {
  it('places equivalent 1080p and 4K observations at the same image location', () => {
    const point1080 = normalizeImagePointToCanonical({ x: 960, y: 540, scale: [2, 2] });
    const point4k = normalizeImagePointToCanonical({ x: 1920, y: 1080, scale: [1, 1] });

    expect(point1080).toEqual(point4k);
    expect(canonicalToImagePoint(point1080!, { width: 1920, height: 1080 }))
      .toEqual({ x: 960, y: 540 });
    expect(canonicalToImagePoint(point4k!, { width: 3840, height: 2160 }))
      .toEqual({ x: 1920, y: 1080 });
  });

  it('keeps horizontal and vertical ball extents scaled independently', () => {
    expect(normalizeImageExtentToCanonical(10, [2, 3])).toEqual({ x: 20, y: 30 });
  });

  it('rejects missing or invalid scale metadata instead of guessing from the viewer frame', () => {
    expect(normalizeImagePointToCanonical({ x: 960, y: 540 })).toBeNull();
    expect(normalizeImagePointToCanonical({
      x: 960,
      y: 540,
      coordinate_space: '4k_canonical',
    })).toBeNull();
    expect(normalizeImagePointToCanonical({ x: 960, y: 540, scale: [0, 2] })).toBeNull();
    expect(normalizeImageExtentToCanonical(10, [Number.NaN, 2])).toBeNull();
  });
});
