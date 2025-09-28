/**
 * Trajectory prediction overlay component
 */

import React from 'react';
import type { Trajectory, CoordinateTransform, OverlayConfig } from '../../../types/video';

interface TrajectoryOverlayProps {
  trajectories: Trajectory[];
  transform: CoordinateTransform;
  config: OverlayConfig['trajectories'];
  ctx: CanvasRenderingContext2D;
}

export function renderTrajectoryOverlay({ trajectories, transform, config, ctx }: TrajectoryOverlayProps): void {
  if (!config.visible || trajectories.length === 0) return;

  ctx.save();
  ctx.globalAlpha = config.opacity;

  trajectories.forEach(trajectory => {
    if (trajectory.points.length < 2) return;

    const lineWidth = config.lineWidth;
    const maxPoints = config.maxLength > 0 ? Math.min(trajectory.points.length, config.maxLength) : trajectory.points.length;
    const points = trajectory.points.slice(0, maxPoints);

    // Color based on trajectory type
    const typeColors = {
      primary: '#00FF00',
      reflection: '#0080FF',
      collision: '#FF8000',
    };

    const trajectoryColor = trajectory.color || typeColors[trajectory.type] || '#00FF00';

    // Draw trajectory line with gradient (fading based on time/distance)
    ctx.strokeStyle = trajectoryColor;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Create gradient for fading effect
    if (points.length > 2) {
      const startPoint = transform.videoToCanvas(points[0]);
      const endPoint = transform.videoToCanvas(points[points.length - 1]);

      const gradient = ctx.createLinearGradient(
        startPoint.x, startPoint.y,
        endPoint.x, endPoint.y
      );

      gradient.addColorStop(0, trajectoryColor);
      gradient.addColorStop(1, trajectoryColor.replace(')', ', 0.1)').replace('rgb', 'rgba'));

      ctx.strokeStyle = gradient;
    }

    // Draw main trajectory line
    ctx.beginPath();
    const startPoint = transform.videoToCanvas(points[0]);
    ctx.moveTo(startPoint.x, startPoint.y);

    // Use quadratic curves for smoother lines
    for (let i = 1; i < points.length; i++) {
      const currentPoint = transform.videoToCanvas(points[i]);

      if (i === points.length - 1) {
        // Last point - draw straight line
        ctx.lineTo(currentPoint.x, currentPoint.y);
      } else {
        // Intermediate points - use quadratic curve
        const nextPoint = transform.videoToCanvas(points[i + 1]);
        const controlX = (currentPoint.x + nextPoint.x) / 2;
        const controlY = (currentPoint.y + nextPoint.y) / 2;

        ctx.quadraticCurveTo(currentPoint.x, currentPoint.y, controlX, controlY);
      }
    }

    ctx.stroke();

    // Draw trajectory direction arrows
    if (points.length >= 3) {
      const arrowSpacing = Math.max(3, Math.floor(points.length / 5));

      for (let i = arrowSpacing; i < points.length - 1; i += arrowSpacing) {
        const point1 = transform.videoToCanvas(points[i - 1]);
        const point2 = transform.videoToCanvas(points[i]);

        const angle = Math.atan2(point2.y - point1.y, point2.x - point1.x);
        const arrowLength = 8;

        ctx.fillStyle = trajectoryColor;
        ctx.beginPath();
        ctx.moveTo(point2.x, point2.y);
        ctx.lineTo(
          point2.x - arrowLength * Math.cos(angle - Math.PI / 6),
          point2.y - arrowLength * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
          point2.x - arrowLength * Math.cos(angle + Math.PI / 6),
          point2.y - arrowLength * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();
      }
    }

    // Draw collision points
    trajectory.collisions.forEach((collision, index) => {
      const collisionPoint = transform.videoToCanvas(collision.position);

      // Different colors for different collision types
      const collisionColors = {
        ball: '#FFD700',
        rail: '#FF6B6B',
        pocket: '#FF0000',
      };

      const collisionColor = collisionColors[collision.type] || '#FFFF00';

      // Draw collision marker
      ctx.fillStyle = collisionColor;
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 2;

      if (collision.type === 'pocket') {
        // Draw X for pocket
        const size = 8;
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#FF0000';
        ctx.beginPath();
        ctx.moveTo(collisionPoint.x - size, collisionPoint.y - size);
        ctx.lineTo(collisionPoint.x + size, collisionPoint.y + size);
        ctx.moveTo(collisionPoint.x + size, collisionPoint.y - size);
        ctx.lineTo(collisionPoint.x - size, collisionPoint.y + size);
        ctx.stroke();
      } else {
        // Draw circle for ball/rail collision
        ctx.beginPath();
        ctx.arc(collisionPoint.x, collisionPoint.y, 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        // Draw collision angle indicator
        if (collision.angle !== undefined) {
          const angleLineLength = 15;
          const angleX = collisionPoint.x + Math.cos(collision.angle) * angleLineLength;
          const angleY = collisionPoint.y + Math.sin(collision.angle) * angleLineLength;

          ctx.strokeStyle = '#000000';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(collisionPoint.x, collisionPoint.y);
          ctx.lineTo(angleX, angleY);
          ctx.stroke();
        }
      }

      // Draw collision info
      if (collision.targetId) {
        ctx.fillStyle = '#000000';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(collision.targetId.slice(0, 4), collisionPoint.x, collisionPoint.y - 10);
      }

      // Draw collision index
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '8px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText((index + 1).toString(), collisionPoint.x, collisionPoint.y);
    });

    // Draw trajectory info
    if (points.length > 0) {
      const startPoint = transform.videoToCanvas(points[0]);

      // Draw trajectory type indicator
      ctx.fillStyle = '#000000';
      ctx.font = '10px Arial';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';

      const typeLabel = trajectory.type.charAt(0).toUpperCase() + trajectory.type.slice(1);
      ctx.fillText(typeLabel, startPoint.x + 5, startPoint.y - 5);

      // Draw probability if enabled
      if (config.showProbability) {
        const probabilityText = `${Math.round(trajectory.probability * 100)}%`;
        ctx.fillStyle = trajectoryColor;
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(probabilityText, startPoint.x + 5, startPoint.y + 5);
      }
    }

    // Draw trajectory end point marker
    if (points.length > 0) {
      const endPoint = transform.videoToCanvas(points[points.length - 1]);

      ctx.fillStyle = trajectoryColor;
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(endPoint.x, endPoint.y, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }

    // Draw uncertainty cone for predictions (optional advanced feature)
    if (trajectory.type === 'primary' && points.length > 5) {
      const startIdx = Math.floor(points.length * 0.7); // Start uncertainty cone at 70% of trajectory
      const startPoint = transform.videoToCanvas(points[startIdx]);
      const endPoint = transform.videoToCanvas(points[points.length - 1]);

      const coneWidth = 20 * (1 - trajectory.probability); // Wider cone for lower probability

      ctx.strokeStyle = trajectoryColor.replace(')', ', 0.3)').replace('rgb', 'rgba');
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);

      // Draw cone sides
      const angle = Math.atan2(endPoint.y - startPoint.y, endPoint.x - startPoint.x);
      const perpAngle = angle + Math.PI / 2;

      const cone1Start = {
        x: startPoint.x + Math.cos(perpAngle) * (coneWidth / 4),
        y: startPoint.y + Math.sin(perpAngle) * (coneWidth / 4),
      };
      const cone1End = {
        x: endPoint.x + Math.cos(perpAngle) * coneWidth,
        y: endPoint.y + Math.sin(perpAngle) * coneWidth,
      };

      const cone2Start = {
        x: startPoint.x - Math.cos(perpAngle) * (coneWidth / 4),
        y: startPoint.y - Math.sin(perpAngle) * (coneWidth / 4),
      };
      const cone2End = {
        x: endPoint.x - Math.cos(perpAngle) * coneWidth,
        y: endPoint.y - Math.sin(perpAngle) * coneWidth,
      };

      ctx.beginPath();
      ctx.moveTo(cone1Start.x, cone1Start.y);
      ctx.lineTo(cone1End.x, cone1End.y);
      ctx.moveTo(cone2Start.x, cone2Start.y);
      ctx.lineTo(cone2End.x, cone2End.y);
      ctx.stroke();

      ctx.setLineDash([]);
    }
  });

  ctx.restore();
}

// React wrapper component for testing/isolation
interface TrajectoryOverlayComponentProps {
  trajectories: Trajectory[];
  transform: CoordinateTransform;
  config: OverlayConfig['trajectories'];
}

export const TrajectoryOverlayComponent: React.FC<TrajectoryOverlayComponentProps> = ({
  trajectories,
  transform,
  config
}) => {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    renderTrajectoryOverlay({ trajectories, transform, config, ctx });
  }, [trajectories, transform, config]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ zIndex: 5 }}
    />
  );
};
