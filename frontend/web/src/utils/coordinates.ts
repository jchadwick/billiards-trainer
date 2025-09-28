/**
 * Coordinate transformation utilities for video streaming and overlays
 */

import type { Point2D, Size2D, ViewportTransform, CoordinateTransform, BoundingBox } from '../types/video';

/**
 * Creates coordinate transformation functions for video and canvas
 */
export function createCoordinateTransform(
  videoSize: Size2D,
  canvasSize: Size2D,
  transform: ViewportTransform = { x: 0, y: 0, scale: 1, rotation: 0 }
): CoordinateTransform {

  // Calculate scaling factors
  const scaleX = canvasSize.width / videoSize.width;
  const scaleY = canvasSize.height / videoSize.height;

  // Use uniform scaling to maintain aspect ratio
  const uniformScale = Math.min(scaleX, scaleY);

  // Calculate offset to center the video in canvas
  const offsetX = (canvasSize.width - videoSize.width * uniformScale) / 2;
  const offsetY = (canvasSize.height - videoSize.height * uniformScale) / 2;

  return {
    videoToCanvas: (point: Point2D): Point2D => {
      // Apply video to canvas transformation
      let x = point.x * uniformScale + offsetX;
      let y = point.y * uniformScale + offsetY;

      // Apply viewport transformation
      x = (x - transform.x) * transform.scale;
      y = (y - transform.y) * transform.scale;

      // Apply rotation if needed
      if (transform.rotation !== 0) {
        const centerX = canvasSize.width / 2;
        const centerY = canvasSize.height / 2;
        const cos = Math.cos(transform.rotation);
        const sin = Math.sin(transform.rotation);

        const dx = x - centerX;
        const dy = y - centerY;

        x = centerX + dx * cos - dy * sin;
        y = centerY + dx * sin + dy * cos;
      }

      return { x, y };
    },

    canvasToVideo: (point: Point2D): Point2D => {
      let x = point.x;
      let y = point.y;

      // Reverse rotation
      if (transform.rotation !== 0) {
        const centerX = canvasSize.width / 2;
        const centerY = canvasSize.height / 2;
        const cos = Math.cos(-transform.rotation);
        const sin = Math.sin(-transform.rotation);

        const dx = x - centerX;
        const dy = y - centerY;

        x = centerX + dx * cos - dy * sin;
        y = centerY + dx * sin + dy * cos;
      }

      // Reverse viewport transformation
      x = x / transform.scale + transform.x;
      y = y / transform.scale + transform.y;

      // Reverse canvas to video transformation
      x = (x - offsetX) / uniformScale;
      y = (y - offsetY) / uniformScale;

      return { x, y };
    },

    screenToCanvas: (point: Point2D): Point2D => {
      // This will be set by the canvas component based on canvas bounds
      return point;
    },

    canvasToScreen: (point: Point2D): Point2D => {
      // This will be set by the canvas component based on canvas bounds
      return point;
    },
  };
}

/**
 * Calculates the best fit scaling for a video within a container
 */
export function calculateBestFit(
  videoSize: Size2D,
  containerSize: Size2D,
  mode: 'contain' | 'cover' | 'fill' = 'contain'
): { scale: number; offset: Point2D; size: Size2D } {
  const scaleX = containerSize.width / videoSize.width;
  const scaleY = containerSize.height / videoSize.height;

  let scale: number;

  switch (mode) {
    case 'contain':
      scale = Math.min(scaleX, scaleY);
      break;
    case 'cover':
      scale = Math.max(scaleX, scaleY);
      break;
    case 'fill':
      // Non-uniform scaling - may distort
      return {
        scale: 1,
        offset: { x: 0, y: 0 },
        size: containerSize,
      };
    default:
      scale = Math.min(scaleX, scaleY);
  }

  const scaledSize = {
    width: videoSize.width * scale,
    height: videoSize.height * scale,
  };

  const offset = {
    x: (containerSize.width - scaledSize.width) / 2,
    y: (containerSize.height - scaledSize.height) / 2,
  };

  return { scale, offset, size: scaledSize };
}

/**
 * Checks if a point is within a bounding box
 */
export function isPointInBounds(point: Point2D, bounds: BoundingBox): boolean {
  return (
    point.x >= bounds.x &&
    point.x <= bounds.x + bounds.width &&
    point.y >= bounds.y &&
    point.y <= bounds.y + bounds.height
  );
}

/**
 * Clamps a point to within bounds
 */
export function clampToBounds(point: Point2D, bounds: BoundingBox): Point2D {
  return {
    x: Math.max(bounds.x, Math.min(bounds.x + bounds.width, point.x)),
    y: Math.max(bounds.y, Math.min(bounds.y + bounds.height, point.y)),
  };
}

/**
 * Calculates distance between two points
 */
export function distance(p1: Point2D, p2: Point2D): number {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculates angle between two points in radians
 */
export function angle(from: Point2D, to: Point2D): number {
  return Math.atan2(to.y - from.y, to.x - from.x);
}

/**
 * Converts angle from radians to degrees
 */
export function radiansToDegrees(radians: number): number {
  return radians * (180 / Math.PI);
}

/**
 * Converts angle from degrees to radians
 */
export function degreesToRadians(degrees: number): number {
  return degrees * (Math.PI / 180);
}

/**
 * Rotates a point around a center point
 */
export function rotatePoint(point: Point2D, center: Point2D, angle: number): Point2D {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  const dx = point.x - center.x;
  const dy = point.y - center.y;

  return {
    x: center.x + dx * cos - dy * sin,
    y: center.y + dx * sin + dy * cos,
  };
}

/**
 * Scales a point from a center point
 */
export function scalePoint(point: Point2D, center: Point2D, scale: number): Point2D {
  return {
    x: center.x + (point.x - center.x) * scale,
    y: center.y + (point.y - center.y) * scale,
  };
}

/**
 * Lerps between two points
 */
export function lerpPoint(from: Point2D, to: Point2D, t: number): Point2D {
  return {
    x: from.x + (to.x - from.x) * t,
    y: from.y + (to.y - from.y) * t,
  };
}

/**
 * Gets the bounds of an array of points
 */
export function getPointsBounds(points: Point2D[]): BoundingBox | null {
  if (points.length === 0) return null;

  let minX = points[0].x;
  let minY = points[0].y;
  let maxX = points[0].x;
  let maxY = points[0].y;

  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  };
}

/**
 * Constrains a transform to reasonable bounds
 */
export function constrainTransform(
  transform: ViewportTransform,
  canvasSize: Size2D,
  minScale = 0.1,
  maxScale = 10
): ViewportTransform {
  const scale = Math.max(minScale, Math.min(maxScale, transform.scale));

  // Constrain translation based on scale
  const maxTranslation = Math.max(canvasSize.width, canvasSize.height) * scale;

  return {
    x: Math.max(-maxTranslation, Math.min(maxTranslation, transform.x)),
    y: Math.max(-maxTranslation, Math.min(maxTranslation, transform.y)),
    scale,
    rotation: transform.rotation,
  };
}
