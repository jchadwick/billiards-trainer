/** Geometry helpers for mapping the displayed Rust still frame to source pixels. */

export interface CanvasPoint {
  x: number;
  y: number;
}

export interface ImageRectangle {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface CanvasPointerMetrics {
  clientX: number;
  clientY: number;
  bounds: Pick<DOMRect, 'left' | 'top'>;
  borderLeft: number;
  borderTop: number;
  clientWidth: number;
  clientHeight: number;
  canvasWidth: number;
  canvasHeight: number;
}

/** Fit the full image inside the canvas while preserving aspect ratio. */
export function fitImageInside(
  sourceWidth: number,
  sourceHeight: number,
  canvasWidth: number,
  canvasHeight: number,
): ImageRectangle | null {
  if (
    !isPositiveFinite(sourceWidth) ||
    !isPositiveFinite(sourceHeight) ||
    !isPositiveFinite(canvasWidth) ||
    !isPositiveFinite(canvasHeight)
  ) {
    return null;
  }

  const scale = Math.min(canvasWidth / sourceWidth, canvasHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  return {
    x: (canvasWidth - width) / 2,
    y: (canvasHeight - height) / 2,
    width,
    height,
  };
}

/** Convert a browser pointer into intrinsic canvas coordinates. */
export function mapClientPointerToCanvasPoint(
  metrics: CanvasPointerMetrics,
): CanvasPoint | null {
  if (
    !Number.isFinite(metrics.clientX) ||
    !Number.isFinite(metrics.clientY) ||
    !Number.isFinite(metrics.bounds.left) ||
    !Number.isFinite(metrics.bounds.top) ||
    !Number.isFinite(metrics.borderLeft) ||
    !Number.isFinite(metrics.borderTop) ||
    !isPositiveFinite(metrics.clientWidth) ||
    !isPositiveFinite(metrics.clientHeight) ||
    !isPositiveFinite(metrics.canvasWidth) ||
    !isPositiveFinite(metrics.canvasHeight)
  ) {
    return null;
  }

  const cssX = metrics.clientX - metrics.bounds.left - metrics.borderLeft;
  const cssY = metrics.clientY - metrics.bounds.top - metrics.borderTop;
  if (cssX < 0 || cssY < 0 || cssX >= metrics.clientWidth || cssY >= metrics.clientHeight) {
    return null;
  }

  return {
    x: (cssX / metrics.clientWidth) * metrics.canvasWidth,
    y: (cssY / metrics.clientHeight) * metrics.canvasHeight,
  };
}

/** Map a pointer through canvas scaling and letterboxing to a source pixel. */
export function mapPointerToSourcePoint(
  metrics: CanvasPointerMetrics,
  sourceWidth: number,
  sourceHeight: number,
): CanvasPoint | null {
  if (!isPositiveFinite(sourceWidth) || !isPositiveFinite(sourceHeight)) return null;
  const canvasPoint = mapClientPointerToCanvasPoint(metrics);
  if (!canvasPoint) return null;
  const imageRect = fitImageInside(
    sourceWidth,
    sourceHeight,
    metrics.canvasWidth,
    metrics.canvasHeight,
  );
  if (!imageRect) return null;

  const relativeX = canvasPoint.x - imageRect.x;
  const relativeY = canvasPoint.y - imageRect.y;
  if (
    relativeX < 0 ||
    relativeY < 0 ||
    relativeX >= imageRect.width ||
    relativeY >= imageRect.height
  ) {
    return null;
  }

  return {
    x: (relativeX / imageRect.width) * sourceWidth,
    y: (relativeY / imageRect.height) * sourceHeight,
  };
}

/** Map source pixels into the image's letterboxed location on the canvas. */
export function sourcePointToCanvasPoint(
  sourceX: number,
  sourceY: number,
  sourceWidth: number,
  sourceHeight: number,
  canvasWidth: number,
  canvasHeight: number,
): CanvasPoint | null {
  if (
    !Number.isFinite(sourceX) ||
    !Number.isFinite(sourceY) ||
    sourceX < 0 ||
    sourceY < 0 ||
    !isPositiveFinite(sourceWidth) ||
    !isPositiveFinite(sourceHeight) ||
    sourceX >= sourceWidth ||
    sourceY >= sourceHeight
  ) {
    return null;
  }
  const imageRect = fitImageInside(sourceWidth, sourceHeight, canvasWidth, canvasHeight);
  if (!imageRect) return null;
  return {
    x: imageRect.x + (sourceX / sourceWidth) * imageRect.width,
    y: imageRect.y + (sourceY / sourceHeight) * imageRect.height,
  };
}

function isPositiveFinite(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}
