/** Canonical image plane shared by the detector and the live overlay. */
export const CANONICAL_IMAGE_SIZE = Object.freeze({ width: 3840, height: 2160 });

type ScaleFactors = readonly [number, number];
type ImageSize = { width: number; height: number };
type ImagePoint = { x: number; y: number };

interface ScaledImagePoint {
  x: number;
  y: number;
  scale?: ScaleFactors;
  coordinate_space?: 'pixel' | '4k_canonical';
}

function validScale(scale: unknown): scale is ScaleFactors {
  return (
    Array.isArray(scale)
    && scale.length === 2
    && scale.every((factor) => typeof factor === 'number' && Number.isFinite(factor) && factor > 0)
  );
}

function validSize(size: ImageSize): boolean {
  return Number.isFinite(size.width) && size.width > 0
    && Number.isFinite(size.height) && size.height > 0;
}

/**
 * Normalize detector coordinates to the stable 4K plane. A wire point must
 * include its scale so frame size cannot be guessed from a later viewer frame.
 */
export function normalizeImagePointToCanonical(point: unknown): ImagePoint | null {
  if (!point || typeof point !== 'object' || Array.isArray(point)) return null;
  const wirePoint = point as ScaledImagePoint;
  if (
    wirePoint.coordinate_space !== undefined
    && wirePoint.coordinate_space !== 'pixel'
    && wirePoint.coordinate_space !== '4k_canonical'
  ) return null;
  if (wirePoint.coordinate_space === '4k_canonical' && wirePoint.scale === undefined) return null;
  if (
    typeof wirePoint.x !== 'number'
    || !Number.isFinite(wirePoint.x)
    || typeof wirePoint.y !== 'number'
    || !Number.isFinite(wirePoint.y)
    || !validScale(wirePoint.scale)
  ) return null;

  const normalized = {
    x: wirePoint.x * wirePoint.scale[0],
    y: wirePoint.y * wirePoint.scale[1],
  };
  return Number.isFinite(normalized.x) && Number.isFinite(normalized.y) ? normalized : null;
}

/** Normalize an image-space radius while preserving independent axis scales. */
export function normalizeImageExtentToCanonical(
  extent: number,
  scale: unknown,
): ImagePoint | null {
  if (!Number.isFinite(extent) || extent < 0 || !validScale(scale)) return null;
  const normalized = { x: extent * scale[0], y: extent * scale[1] };
  return Number.isFinite(normalized.x) && Number.isFinite(normalized.y) ? normalized : null;
}

/** Convert a canonical point to the pixels of the currently displayed image. */
export function canonicalToImagePoint(point: ImagePoint, imageSize: ImageSize): ImagePoint | null {
  if (!validSize(imageSize) || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return null;
  return {
    x: point.x * imageSize.width / CANONICAL_IMAGE_SIZE.width,
    y: point.y * imageSize.height / CANONICAL_IMAGE_SIZE.height,
  };
}
