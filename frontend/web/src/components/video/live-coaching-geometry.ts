import type { TableProjection } from '../../types/api';

export interface Point2 { x: number; y: number }

function inverse3(matrix: TableProjection['pixel_to_table_mm']): number[][] | null {
  const [[a, b, c], [d, e, f], [g, h, i]] = matrix;
  const determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-12) return null;
  const inverse = [
    [e * i - f * h, c * h - b * i, b * f - c * e],
    [f * g - d * i, a * i - c * g, c * d - a * f],
    [d * h - e * g, b * g - a * h, a * e - b * d],
  ];
  return inverse.map((row) => row.map((value) => value / determinant));
}

function applyHomography(matrix: number[][], point: Point2): Point2 | null {
  const denominator = matrix[2][0] * point.x + matrix[2][1] * point.y + matrix[2][2];
  if (!Number.isFinite(denominator) || Math.abs(denominator) < 1e-12) return null;
  const x = (matrix[0][0] * point.x + matrix[0][1] * point.y + matrix[0][2]) / denominator;
  const y = (matrix[1][0] * point.x + matrix[1][1] * point.y + matrix[1][2]) / denominator;
  return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
}

/** Apply the producer's source-image-pixel to center-origin-millimeter map. */
export function imageToTableMm(projection: TableProjection, point: Point2): Point2 | null {
  return applyHomography(projection.pixel_to_table_mm, point);
}

/** Invert that same producer map so paths and markers land on the live image. */
export function tableMmToImage(projection: TableProjection, point: Point2): Point2 | null {
  const inverse = inverse3(projection.pixel_to_table_mm);
  return inverse ? applyHomography(inverse, point) : null;
}

/** Coach math uses positive-origin rectangles; perception reports table-center origin. */
export function centerToTable(point: Point2, width: number, height: number): Point2 {
  return { x: point.x + width / 2, y: point.y + height / 2 };
}

export function tableToCenter(point: Point2, width: number, height: number): Point2 {
  return { x: point.x - width / 2, y: point.y - height / 2 };
}

/**
 * Validate producer geometry against any active context the state contract
 * supplies. Resolution differences are allowed: source-frame coordinates are
 * scaled to the displayed full-frame image after projection.
 */
export function tableProjectionIsCompatible(
  projection: TableProjection | null | undefined,
  context: {
    calibrationRevision?: number | string;
    rectificationRevision?: number;
    cameraProfileId?: string;
    imageSpace?: string;
  },
): projection is TableProjection {
  if (!projection || projection.direction !== 'source_image_px_to_table_mm'
    || projection.unit !== 'millimeters' || projection.origin !== 'playing_surface_center'
    || projection.image_space !== 'raw' || projection.source_width_px <= 0
    || projection.source_height_px <= 0 || projection.table_width_mm <= 0
    || projection.table_height_mm <= 0
    || projection.pixel_to_table_mm.some((row) => row.some((value) => !Number.isFinite(value)))) return false;
  if (context.imageSpace && context.imageSpace !== projection.image_space) return false;
  if (context.cameraProfileId && context.cameraProfileId !== projection.camera_profile_id) return false;
  if (context.calibrationRevision !== undefined && context.calibrationRevision !== null
    && String(context.calibrationRevision) !== String(projection.calibration_revision)) return false;
  if (context.rectificationRevision !== undefined && context.rectificationRevision !== null
    && context.rectificationRevision !== projection.rectification_revision) return false;
  return imageToTableMm(projection, { x: 0, y: 0 }) !== null
    && tableMmToImage(projection, { x: 0, y: 0 }) !== null;
}
