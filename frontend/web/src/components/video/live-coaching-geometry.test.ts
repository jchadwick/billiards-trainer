import { describe, expect, it } from 'vitest';
import { centerToTable, imageToTableMm, tableMmToImage, tableProjectionIsCompatible, tableToCenter } from './live-coaching-geometry';
import type { TableProjection } from '../../types/api';

const perspective: TableProjection = {
  direction: 'source_image_px_to_table_mm', image_space: 'raw',
  source_width_px: 1280, source_height_px: 720, camera_profile_id: 'cam-a',
  calibration_revision: 4, rectification_revision: 2,
  table_width_mm: 2000, table_height_mm: 1000,
  unit: 'millimeters', origin: 'playing_surface_center',
  x_axis: 'corner_0_to_1', y_axis: 'corner_0_to_3',
  pixel_to_table_mm: [[1.8, 0.1, -950], [0.05, 1.5, -510], [0.0001, 0.0002, 1]],
};

describe('live coaching projection', () => {
  it('inverts the same perspective mapping used for producer table positions', () => {
    const sourcePoint = { x: 811, y: 442 };
    const tablePoint = imageToTableMm(perspective, sourcePoint)!;
    const returned = tableMmToImage(perspective, tablePoint)!;
    expect(returned.x).toBeCloseTo(sourcePoint.x, 8);
    expect(returned.y).toBeCloseTo(sourcePoint.y, 8);
  });

  it('converts between perception center origin and coach positive origin consistently', () => {
    const centered = { x: -400, y: 125 };
    expect(tableToCenter(centerToTable(centered, 2000, 1000), 2000, 1000)).toEqual(centered);
  });

  it('accepts scaled video dimensions while rejecting known context mismatch', () => {
    expect(tableProjectionIsCompatible(perspective, {
      calibrationRevision: 4, rectificationRevision: 2, cameraProfileId: 'cam-a',
    })).toBe(true);
    expect(tableProjectionIsCompatible(perspective, { calibrationRevision: 5 })).toBe(false);
    expect(tableProjectionIsCompatible(perspective, { cameraProfileId: 'cam-b' })).toBe(false);
  });
});
