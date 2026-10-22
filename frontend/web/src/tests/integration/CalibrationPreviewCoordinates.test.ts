import { describe, expect, it } from 'vitest';
import {
  fitImageInside,
  mapPointerToSourcePoint,
  sourcePointToCanvasPoint,
  type CanvasPointerMetrics,
} from '../../components/config/calibration/previewCoordinates';

function pointerMetrics(clientX: number, clientY: number): CanvasPointerMetrics {
  return {
    clientX,
    clientY,
    bounds: { left: 10, top: 20 },
    borderLeft: 2,
    borderTop: 2,
    clientWidth: 320,
    clientHeight: 180,
    canvasWidth: 640,
    canvasHeight: 360,
  };
}

describe('calibration preview coordinate mapping', () => {
  it('fits the full source frame and centers letterbox margins', () => {
    expect(fitImageInside(1000, 1000, 640, 360)).toEqual({
      x: 140,
      y: 0,
      width: 360,
      height: 360,
    });
  });

  it('accounts for CSS scaling, device canvas scaling, and the canvas border', () => {
    expect(mapPointerToSourcePoint(pointerMetrics(172, 112), 1920, 1080)).toEqual({
      x: 960,
      y: 540,
    });
  });

  it('rejects clicks in letterbox margins instead of inventing source coordinates', () => {
    const metrics = pointerMetrics(72, 112);
    expect(mapPointerToSourcePoint(metrics, 1000, 1000)).toBeNull();
  });

  it('maps the fitted image center to the exact source center', () => {
    expect(mapPointerToSourcePoint(pointerMetrics(172, 112), 1000, 1000)).toEqual({
      x: 500,
      y: 500,
    });
  });

  it('maps source points back into the same letterboxed canvas image', () => {
    expect(sourcePointToCanvasPoint(500, 500, 1000, 1000, 640, 360)).toEqual({
      x: 320,
      y: 180,
    });
  });

  it('rejects invalid dimensions, pointers outside the canvas, and the exclusive image edge', () => {
    expect(fitImageInside(0, 10, 640, 360)).toBeNull();
    expect(mapPointerToSourcePoint(pointerMetrics(9, 112), 1920, 1080)).toBeNull();
    expect(sourcePointToCanvasPoint(1000, 10, 1000, 1000, 640, 360)).toBeNull();
  });
});
