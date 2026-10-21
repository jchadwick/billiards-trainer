/** Renders measured ball paths and makes gaps explicit without interpolation. */

import type {
  CoordinateTransform,
  ObservedBallTrail,
  OverlayConfig,
} from '../../../types/video';

interface ObservedBallPathOverlayProps {
  trails: ObservedBallTrail[];
  transform: CoordinateTransform;
  config: OverlayConfig['trajectories'];
  ctx: CanvasRenderingContext2D;
}

export function renderObservedBallPathOverlay({
  trails,
  transform,
  config,
  ctx,
}: ObservedBallPathOverlayProps): void {
  if (!config.visible || trails.length === 0) return;

  ctx.save();
  ctx.globalAlpha = Math.min(1, config.opacity);
  ctx.lineWidth = Math.max(1, config.lineWidth);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  trails.forEach((trail) => {
    const points = trail.points;
    if (points.length === 0) return;

    // Draw only links between adjacent measurements. A gap starts a new path
    // segment and receives a conspicuous ring so the missing interval is clear.
    ctx.strokeStyle = '#35D0C4';
    ctx.setLineDash([]);
    for (let index = 1; index < points.length; index += 1) {
      const point = points[index];
      if (point.gapBefore) continue;
      const previous = transform.videoToCanvas(points[index - 1]);
      const current = transform.videoToCanvas(point);
      ctx.beginPath();
      ctx.moveTo(previous.x, previous.y);
      ctx.lineTo(current.x, current.y);
      ctx.stroke();
    }

    points.forEach((point, index) => {
      const position = transform.videoToCanvas(point);
      ctx.beginPath();
      ctx.fillStyle = '#35D0C4';
      ctx.arc(position.x, position.y, 2.5, 0, 2 * Math.PI);
      ctx.fill();

      if (index > 0 && point.gapBefore) {
        ctx.strokeStyle = '#F4B942';
        ctx.lineWidth = Math.max(1.5, config.lineWidth);
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.arc(position.x, position.y, 7, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });
  });

  ctx.restore();
}
