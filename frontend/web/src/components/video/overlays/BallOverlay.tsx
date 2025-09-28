/**
 * Ball detection overlay component
 */

import React from 'react';
import type { Ball, CoordinateTransform, OverlayConfig } from '../../../types/video';

interface BallOverlayProps {
  balls: Ball[];
  transform: CoordinateTransform;
  config: OverlayConfig['balls'];
  ctx: CanvasRenderingContext2D;
}

export function renderBallOverlay({ balls, transform, config, ctx }: BallOverlayProps): void {
  if (!config.visible || balls.length === 0) return;

  ctx.save();
  ctx.globalAlpha = config.opacity;

  balls.forEach(ball => {
    const center = transform.videoToCanvas(ball.position);
    const radius = config.radius * (center.x / 1000); // Scale with zoom

    // Draw ball circle
    ctx.beginPath();
    ctx.arc(center.x, center.y, radius, 0, 2 * Math.PI);

    // Color based on ball type
    const colors = {
      cue: '#FFFFFF',
      solid: '#FF6B6B',
      stripe: '#4ECDC4',
      eight: '#2C3E50',
    };

    const ballColor = ball.color || colors[ball.type] || '#CCCCCC';

    // Draw solid balls
    if (ball.type === 'solid' || ball.type === 'cue' || ball.type === 'eight') {
      ctx.fillStyle = ballColor;
      ctx.fill();
    }
    // Draw striped balls
    else if (ball.type === 'stripe') {
      // Base color (white)
      ctx.fillStyle = '#FFFFFF';
      ctx.fill();

      // Stripe pattern
      ctx.save();
      ctx.clip();
      ctx.strokeStyle = ballColor;
      ctx.lineWidth = radius / 3;

      for (let i = -radius; i <= radius; i += radius / 2) {
        ctx.beginPath();
        ctx.moveTo(center.x + i, center.y - radius);
        ctx.lineTo(center.x + i, center.y + radius);
        ctx.stroke();
      }
      ctx.restore();
    }

    // Draw outline
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw ball number if available and enabled
    if (config.showLabels && ball.number !== undefined) {
      ctx.fillStyle = ball.type === 'eight' ? '#FFFFFF' : '#000000';
      ctx.font = `${Math.max(12, radius / 2)}px Arial`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(ball.number.toString(), center.x, center.y);
    }

    // Draw ball ID if enabled
    if (config.showIds) {
      ctx.fillStyle = '#000000';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(ball.id.slice(0, 6), center.x, center.y + radius + 5);
    }

    // Draw velocity vector if enabled
    if (config.showVelocity && (ball.velocity.x !== 0 || ball.velocity.y !== 0)) {
      const velocityMagnitude = Math.sqrt(ball.velocity.x ** 2 + ball.velocity.y ** 2);
      const velocityScale = Math.min(velocityMagnitude * 100, 100); // Scale velocity display

      const velocityEnd = transform.videoToCanvas({
        x: ball.position.x + (ball.velocity.x / velocityMagnitude) * velocityScale,
        y: ball.position.y + (ball.velocity.y / velocityMagnitude) * velocityScale,
      });

      // Velocity vector
      ctx.strokeStyle = '#FF0000';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(center.x, center.y);
      ctx.lineTo(velocityEnd.x, velocityEnd.y);
      ctx.stroke();

      // Arrow head
      const angle = Math.atan2(velocityEnd.y - center.y, velocityEnd.x - center.x);
      const arrowLength = 10;

      ctx.fillStyle = '#FF0000';
      ctx.beginPath();
      ctx.moveTo(velocityEnd.x, velocityEnd.y);
      ctx.lineTo(
        velocityEnd.x - arrowLength * Math.cos(angle - Math.PI / 6),
        velocityEnd.y - arrowLength * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        velocityEnd.x - arrowLength * Math.cos(angle + Math.PI / 6),
        velocityEnd.y - arrowLength * Math.sin(angle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fill();

      // Velocity magnitude text
      const speedText = `${velocityMagnitude.toFixed(1)} px/s`;
      ctx.fillStyle = '#FF0000';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(speedText, velocityEnd.x, velocityEnd.y - 5);
    }

    // Draw confidence if enabled
    if (config.showConfidence) {
      ctx.fillStyle = '#000000';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      const confidenceText = `${Math.round(ball.confidence * 100)}%`;
      ctx.fillText(confidenceText, center.x, center.y - radius - 5);
    }

    // Draw selection highlight (if ball is somehow selected)
    if ((ball as any).selected) {
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.arc(center.x, center.y, radius + 5, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  });

  ctx.restore();
}

// React wrapper component for testing/isolation
interface BallOverlayComponentProps {
  balls: Ball[];
  transform: CoordinateTransform;
  config: OverlayConfig['balls'];
}

export const BallOverlayComponent: React.FC<BallOverlayComponentProps> = ({
  balls,
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
    renderBallOverlay({ balls, transform, config, ctx });
  }, [balls, transform, config]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ zIndex: 10 }}
    />
  );
};
