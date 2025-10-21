/**
 * Canvas overlay system for drawing detection overlays on top of video stream
 */

import React, { useRef, useEffect, useCallback, useState } from 'react';
import { observer } from 'mobx-react-lite';
import type { VideoStore } from '../../stores/VideoStore';
import type {
  Size2D,
  ViewportTransform,
  OverlayConfig,
  Point2D,
  CoordinateTransform,
  CanvasContext,
} from '../../types/video';
import { createCoordinateTransform } from '../../utils/coordinates';
import { renderCalibrationOverlay } from './overlays/CalibrationOverlay';

interface OverlayCanvasProps {
  videoStore: VideoStore;
  videoSize: Size2D;
  canvasSize: Size2D;
  transform: ViewportTransform;
  config: OverlayConfig;
  className?: string;
  onClick?: (point: Point2D, event: React.MouseEvent) => void;
  onDoubleClick?: (point: Point2D, event: React.MouseEvent) => void;
}

export const OverlayCanvas = observer<OverlayCanvasProps>(({
  videoStore,
  videoSize,
  canvasSize,
  transform,
  config,
  className = '',
  onClick,
  onDoubleClick,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const [coordinateTransform, setCoordinateTransform] = useState<CoordinateTransform | null>(null);

  // Create coordinate transformation functions
  useEffect(() => {
    if (videoSize.width > 0 && videoSize.height > 0 && canvasSize.width > 0 && canvasSize.height > 0) {
      const transform2D = createCoordinateTransform(videoSize, canvasSize, transform);
      setCoordinateTransform(transform2D);
    }
  }, [videoSize, canvasSize, transform]);

  // Setup high-DPI canvas
  const setupCanvas = useCallback((canvas: HTMLCanvasElement): CanvasContext | null => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    const devicePixelRatio = window.devicePixelRatio || 1;

    // Set actual canvas size in memory (scaled for high-DPI)
    canvas.width = canvasSize.width * devicePixelRatio;
    canvas.height = canvasSize.height * devicePixelRatio;

    // Scale canvas back down using CSS
    canvas.style.width = `${canvasSize.width}px`;
    canvas.style.height = `${canvasSize.height}px`;

    // Scale drawing context so everything draws at high-DPI
    ctx.scale(devicePixelRatio, devicePixelRatio);

    // Set drawing defaults
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    return {
      canvas,
      ctx,
      size: canvasSize,
      devicePixelRatio,
    };
  }, [canvasSize]);

  // Clear canvas
  const clearCanvas = useCallback((ctx: CanvasRenderingContext2D) => {
    ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);
  }, [canvasSize]);

  // Draw ball overlay
  const drawBalls = useCallback((ctx: CanvasRenderingContext2D, transform: CoordinateTransform) => {
    if (!config.balls.visible) return;

    const balls = videoStore.currentBalls;

    balls.forEach(ball => {
      const center = transform.videoToCanvas(ball.position);

      // Apply opacity
      ctx.globalAlpha = config.balls.opacity;

      // Draw ball circle
      ctx.beginPath();
      ctx.arc(center.x, center.y, ball.radius * (transform as any).scale || config.balls.radius, 0, 2 * Math.PI);

      // Color based on ball type
      const colors = {
        cue: '#FFFFFF',
        solid: '#FF6B6B',
        stripe: '#4ECDC4',
        eight: '#2C3E50',
      };
      ctx.fillStyle = ball.color || colors[ball.type] || '#CCCCCC';
      ctx.fill();

      // Draw outline
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw ball number if available and enabled
      if (config.balls.showLabels && ball.number !== undefined) {
        ctx.fillStyle = ball.type === 'eight' ? '#FFFFFF' : '#000000';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(ball.number.toString(), center.x, center.y);
      }

      // Draw ball ID if enabled
      if (config.balls.showIds) {
        ctx.fillStyle = '#000000';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(ball.id.slice(0, 6), center.x, center.y + ball.radius + 5);
      }

      // Draw velocity vector if enabled
      if (config.balls.showVelocity && (ball.velocity.x !== 0 || ball.velocity.y !== 0)) {
        const velocityEnd = transform.videoToCanvas({
          x: ball.position.x + ball.velocity.x * 50,
          y: ball.position.y + ball.velocity.y * 50,
        });

        ctx.strokeStyle = '#FF0000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(center.x, center.y);
        ctx.lineTo(velocityEnd.x, velocityEnd.y);
        ctx.stroke();

        // Arrow head
        const angle = Math.atan2(velocityEnd.y - center.y, velocityEnd.x - center.x);
        const arrowLength = 8;
        ctx.beginPath();
        ctx.moveTo(velocityEnd.x, velocityEnd.y);
        ctx.lineTo(
          velocityEnd.x - arrowLength * Math.cos(angle - Math.PI / 6),
          velocityEnd.y - arrowLength * Math.sin(angle - Math.PI / 6)
        );
        ctx.moveTo(velocityEnd.x, velocityEnd.y);
        ctx.lineTo(
          velocityEnd.x - arrowLength * Math.cos(angle + Math.PI / 6),
          velocityEnd.y - arrowLength * Math.sin(angle + Math.PI / 6)
        );
        ctx.stroke();
      }

      // Draw confidence if enabled
      if (config.balls.showConfidence) {
        ctx.fillStyle = '#000000';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        const confidenceText = `${Math.round(ball.confidence * 100)}%`;
        ctx.fillText(confidenceText, center.x, center.y - ball.radius - 5);
      }
    });

    ctx.globalAlpha = 1.0;
  }, [config.balls, videoStore]);

  // Draw trajectory overlays
  const drawTrajectories = useCallback((ctx: CanvasRenderingContext2D, transform: CoordinateTransform) => {
    if (!config.trajectories.visible) return;

    const trajectories = videoStore.currentTrajectories;

    trajectories.forEach(trajectory => {
      if (trajectory.points.length < 2) return;

      ctx.globalAlpha = config.trajectories.opacity;
      ctx.strokeStyle = trajectory.color || '#00FF00';
      ctx.lineWidth = config.trajectories.lineWidth;

      // Draw trajectory line
      ctx.beginPath();
      const startPoint = transform.videoToCanvas(trajectory.points[0]);
      ctx.moveTo(startPoint.x, startPoint.y);

      trajectory.points.slice(1).forEach(point => {
        const canvasPoint = transform.videoToCanvas(point);
        ctx.lineTo(canvasPoint.x, canvasPoint.y);
      });

      ctx.stroke();

      // Draw collision points
      trajectory.collisions.forEach(collision => {
        const collisionPoint = transform.videoToCanvas(collision.position);

        ctx.fillStyle = collision.type === 'pocket' ? '#FF0000' : '#FFFF00';
        ctx.beginPath();
        ctx.arc(collisionPoint.x, collisionPoint.y, 4, 0, 2 * Math.PI);
        ctx.fill();

        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Draw probability if enabled
      if (config.trajectories.showProbability && trajectory.points.length > 0) {
        const startPoint = transform.videoToCanvas(trajectory.points[0]);
        ctx.fillStyle = '#000000';
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        const probabilityText = `${Math.round(trajectory.probability * 100)}%`;
        ctx.fillText(probabilityText, startPoint.x + 5, startPoint.y - 20);
      }
    });

    ctx.globalAlpha = 1.0;
  }, [config.trajectories, videoStore]);

  // Draw table overlay
  const drawTable = useCallback((ctx: CanvasRenderingContext2D, transform: CoordinateTransform) => {
    if (!config.table.visible) return;

    const table = videoStore.currentTable;
    if (!table) return;

    ctx.globalAlpha = config.table.opacity;
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = config.table.lineWidth;

    // Draw table corners
    if (table.corners.length >= 4) {
      ctx.beginPath();
      const startCorner = transform.videoToCanvas(table.corners[0]);
      ctx.moveTo(startCorner.x, startCorner.y);

      table.corners.slice(1).forEach(corner => {
        const canvasCorner = transform.videoToCanvas(corner);
        ctx.lineTo(canvasCorner.x, canvasCorner.y);
      });

      ctx.closePath();
      ctx.stroke();
    }

    // Draw pockets if enabled
    if (config.table.showPockets) {
      table.pockets.forEach(pocket => {
        const pocketPoint = transform.videoToCanvas(pocket);

        ctx.fillStyle = '#000000';
        ctx.beginPath();
        ctx.arc(pocketPoint.x, pocketPoint.y, 8, 0, 2 * Math.PI);
        ctx.fill();

        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    }

    // Draw rails if enabled
    if (config.table.showRails) {
      table.rails.forEach(rail => {
        if (rail.length >= 2) {
          ctx.strokeStyle = '#FFFF00';
          ctx.lineWidth = 2;
          ctx.beginPath();

          const startPoint = transform.videoToCanvas(rail[0]);
          ctx.moveTo(startPoint.x, startPoint.y);

          rail.slice(1).forEach(point => {
            const canvasPoint = transform.videoToCanvas(point);
            ctx.lineTo(canvasPoint.x, canvasPoint.y);
          });

          ctx.stroke();
        }
      });
    }

    ctx.globalAlpha = 1.0;
  }, [config.table, videoStore]);

  // Draw calibration overlay
  const drawCalibration = useCallback((ctx: CanvasRenderingContext2D, transform: CoordinateTransform) => {
    if (!config.calibration.visible) return;

    const calibrationData = videoStore.calibrationData;
    if (!calibrationData || !calibrationData.isValid) return;

    renderCalibrationOverlay({
      calibrationData,
      transform,
      config: config.calibration,
      ctx,
    });
  }, [config.calibration, videoStore]);

  // Draw cue stick overlay
  const drawCue = useCallback((ctx: CanvasRenderingContext2D, transform: CoordinateTransform) => {
    if (!config.cue.visible) return;

    const cue = videoStore.currentCue;
    if (!cue || !cue.detected) return;

    ctx.globalAlpha = config.cue.opacity;
    ctx.strokeStyle = '#FF8800';
    ctx.lineWidth = config.cue.lineWidth;

    // Draw cue stick
    const tipPoint = transform.videoToCanvas(cue.tipPosition);
    const tailPoint = transform.videoToCanvas(cue.tailPosition);

    ctx.beginPath();
    ctx.moveTo(tipPoint.x, tipPoint.y);
    ctx.lineTo(tailPoint.x, tailPoint.y);
    ctx.stroke();

    // Draw tip indicator
    ctx.fillStyle = '#FF0000';
    ctx.beginPath();
    ctx.arc(tipPoint.x, tipPoint.y, 4, 0, 2 * Math.PI);
    ctx.fill();

    // Draw angle information if enabled
    if (config.cue.showAngle) {
      ctx.fillStyle = '#000000';
      ctx.font = '12px Arial';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      const angleText = `${Math.round(cue.angle * 180 / Math.PI)}°`;
      ctx.fillText(angleText, tipPoint.x + 10, tipPoint.y - 10);
    }

    // Draw guide lines if enabled
    if (config.cue.showGuideLines) {
      const extendedTip = {
        x: cue.tipPosition.x + Math.cos(cue.angle) * 200,
        y: cue.tipPosition.y + Math.sin(cue.angle) * 200,
      };
      const extendedPoint = transform.videoToCanvas(extendedTip);

      ctx.strokeStyle = '#FF8800';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(tipPoint.x, tipPoint.y);
      ctx.lineTo(extendedPoint.x, extendedPoint.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.globalAlpha = 1.0;
  }, [config.cue, videoStore]);

  // Draw grid overlay
  const drawGrid = useCallback((ctx: CanvasRenderingContext2D) => {
    if (!config.grid.visible) return;

    ctx.globalAlpha = config.grid.opacity;
    ctx.strokeStyle = '#808080';
    ctx.lineWidth = 1;

    const spacing = config.grid.spacing;

    // Vertical lines
    for (let x = 0; x < canvasSize.width; x += spacing) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvasSize.height);
      ctx.stroke();
    }

    // Horizontal lines
    for (let y = 0; y < canvasSize.height; y += spacing) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvasSize.width, y);
      ctx.stroke();
    }

    ctx.globalAlpha = 1.0;
  }, [config.grid, canvasSize]);

  // Main render function - stored in ref to avoid animation loop recreation
  const renderOverlaysRef = useRef<() => void>();

  // Update the render function when dependencies change
  useEffect(() => {
    renderOverlaysRef.current = () => {
      if (!canvasRef.current || !coordinateTransform) return;

      const canvasContext = setupCanvas(canvasRef.current);
      if (!canvasContext) return;

      const { ctx } = canvasContext;

      // Clear canvas
      clearCanvas(ctx);

      // Draw overlays in order
      drawGrid(ctx);
      drawCalibration(ctx, coordinateTransform);
      drawTable(ctx, coordinateTransform);
      drawTrajectories(ctx, coordinateTransform);
      drawBalls(ctx, coordinateTransform);
      drawCue(ctx, coordinateTransform);
    };
  }, [coordinateTransform, setupCanvas, clearCanvas, drawGrid, drawCalibration, drawTable, drawTrajectories, drawBalls, drawCue]);

  // Animation loop - runs once on mount
  useEffect(() => {
    const animate = () => {
      if (renderOverlaysRef.current) {
        renderOverlaysRef.current();
      }
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []); // Empty dependency array - runs once

  // Handle canvas clicks
  const handleCanvasClick = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!coordinateTransform || !onClick) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const canvasPoint: Point2D = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };

    const videoPoint = coordinateTransform.canvasToVideo(canvasPoint);
    onClick(videoPoint, event);
  }, [coordinateTransform, onClick]);

  const handleCanvasDoubleClick = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!coordinateTransform || !onDoubleClick) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const canvasPoint: Point2D = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };

    const videoPoint = coordinateTransform.canvasToVideo(canvasPoint);
    onDoubleClick(videoPoint, event);
  }, [coordinateTransform, onDoubleClick]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-auto ${className}`}
      style={{
        width: canvasSize.width,
        height: canvasSize.height,
      }}
      onClick={handleCanvasClick}
      onDoubleClick={handleCanvasDoubleClick}
    />
  );
});

OverlayCanvas.displayName = 'OverlayCanvas';
