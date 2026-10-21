/**
 * Simple live view component showing video stream with ball detection overlay
 */

import React, { useEffect, useRef, useState } from 'react';
import { observer } from 'mobx-react-lite';
import { apiClient } from '../../api/client';
import { getConfiguredLiveStreamUrl } from './live-stream-url';
import { canonicalToImagePoint, normalizeImageExtentToCanonical, normalizeImagePointToCanonical } from '../../utils/coordinates';
import type { GameType } from '../../types/api';
import { useRootStore } from '../../stores/context';

interface Ball {
  id: string | number;
  position: { x: number; y: number; scale: [number, number] };
  radius: number;
  radius_scale?: [number, number];
  radius_extent?: { x: number; y: number; scale: [number, number] };
  number?: number | null;
  type?: 'cue' | 'solid' | 'stripe' | 'eight' | 'unknown';
  is_cue_ball?: boolean;
  is_observed?: boolean;
  position_kind?: 'measured' | 'filtered' | 'extrapolated' | 'last_observed';
  confidence: number;
  confidence_kind?: 'detector_observation' | 'unavailable';
}

interface Cue {
  tip_position?: { x: number; y: number; scale: [number, number] };
  butt_position?: { x: number; y: number; scale: [number, number] };
  angle?: number;
  confidence?: number;
  detected?: boolean;
  is_visible?: boolean;
}

interface LiveViewProps {
  className?: string;
  autoConnect?: boolean;
}

const GAME_TYPE_LABELS: Record<GameType, string> = {
  practice: 'Practice',
  '8ball': '8-ball',
  '9ball': '9-ball',
  straight: 'Straight Pool',
};

export const LiveView: React.FC<LiveViewProps> = observer(({ className = '', autoConnect = true }) => {
  const { system, videoStore } = useRootStore();
  const videoRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });
  const [viewerStreamUrl, setViewerStreamUrl] = useState<string | undefined>();
  const [viewerConfigLoaded, setViewerConfigLoaded] = useState(false);
  const latestState = videoStore.latestGameState;
  const balls: Ball[] = latestState?.balls ?? [];
  const cue: Cue | null = latestState?.cue ?? null;
  const gameType = latestState?.game_type ?? null;
  const reportedAge = latestState?.observation_status?.frame_publication_age_seconds;
  const visionAgeMs = typeof reportedAge === 'number' && Number.isFinite(reportedAge) && reportedAge >= 0
    ? Math.round(reportedAge * 1000)
    : null;
  const isConnected = autoConnect && system.status.isConnected;

  // The detector may subscribe to a loopback camera URI while browsers need a
  // LAN-reachable hostname. Keep the viewer URL as runtime application config.
  useEffect(() => {
    if (!autoConnect) {
      setViewerConfigLoaded(false);
      return;
    }

    let cancelled = false;
    void apiClient.getConfiguration().then((response) => {
      if (cancelled) return;
      const values = response.data?.values;
      const streamUrl = values?.api?.stream?.viewer_url;
      setViewerStreamUrl(getConfiguredLiveStreamUrl(streamUrl));
      setViewerConfigLoaded(true);
    }).catch(() => {
      if (cancelled) return;
      setViewerStreamUrl(undefined);
      setViewerConfigLoaded(true);
    });

    return () => { cancelled = true; };
  }, [autoConnect]);

  const gameTypeLabel = gameType
    ? GAME_TYPE_LABELS[gameType as GameType] ?? gameType
    : 'Unknown';

  const getBallLabel = (ball: Ball): string => {
    if (typeof ball.number === 'number') return `Ball ${ball.number}`;
    if (ball.is_cue_ball || ball.type === 'cue') return 'Cue ball';
    if (ball.type === 'eight') return '8-ball';
    return 'Ball';
  };

  const getBallFreshnessLabel = (ball: Ball): string => {
    if (ball.position_kind === 'extrapolated') return 'Predicted track';
    if (ball.position_kind === 'filtered') return 'Filtered track';
    return 'Last observed';
  };

  const observedBalls = balls.filter((ball) => ball.is_observed !== false);
  const nonObservedBalls = balls.filter((ball) => ball.is_observed === false);
  const cueVisible = Boolean(cue && (cue.is_visible ?? cue.detected));

  // Track video size when image loads
  useEffect(() => {
    const img = videoRef.current;
    if (!img) return;

    const handleLoad = () => {
      setVideoSize({
        width: img.naturalWidth,
        height: img.naturalHeight
      });
    };

    img.addEventListener('load', handleLoad);
    return () => img.removeEventListener('load', handleLoad);
  }, [autoConnect]);

  // Draw balls on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video || !videoSize.width || !videoSize.height) return;

    // Match canvas size to video display size
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate scale factor from video coordinates to canvas coordinates
    const scaleX = rect.width / videoSize.width;
    const scaleY = rect.height / videoSize.height;

    // Draw each ball
    balls.forEach(ball => {
      const canonicalPosition = normalizeImagePointToCanonical(ball.position);
      const canonicalRadius = ball.radius_extent
        ? normalizeImagePointToCanonical(ball.radius_extent)
        : normalizeImageExtentToCanonical(ball.radius, ball.radius_scale ?? ball.position.scale);
      if (!canonicalPosition || !canonicalRadius) return;
      const imagePosition = canonicalToImagePoint(canonicalPosition, videoSize);
      const imageRadius = canonicalToImagePoint(canonicalRadius, videoSize);
      if (!imagePosition || !imageRadius) return;

      // The active image size is applied after canonical normalization so the
      // same physical detection aligns over 1080p and 4K source frames.
      const x = imagePosition.x * scaleX;
      const y = imagePosition.y * scaleY;
      const radiusX = imageRadius.x * scaleX;
      const radiusY = imageRadius.y * scaleY;
      const radius = Math.sqrt(radiusX * radiusY);
      const isObserved = ball.is_observed !== false;

      // Draw ball circle
      ctx.beginPath();
      ctx.ellipse(x, y, radiusX, radiusY, 0, 0, 2 * Math.PI);

      // Set ball color based on type
      const colors = {
        cue: 'rgba(255, 255, 255, 0.7)',
        solid: 'rgba(255, 107, 107, 0.7)',
        stripe: 'rgba(78, 205, 196, 0.7)',
        eight: 'rgba(44, 62, 80, 0.7)',
        unknown: 'rgba(200, 200, 200, 0.45)',
      };
      const ballType = ball.is_cue_ball ? 'cue' : ball.type ?? 'unknown';
      ctx.globalAlpha = isObserved ? 1 : 0.45;
      ctx.fillStyle = colors[ballType] || colors.unknown;
      ctx.fill();

      // Draw outline
      ctx.strokeStyle = isObserved ? '#000000' : '#fbbf24';
      ctx.lineWidth = 2;
      ctx.setLineDash(isObserved ? [] : [5, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw ball number if available
      if (typeof ball.number === 'number') {
        ctx.fillStyle = ball.type === 'eight' ? '#FFFFFF' : '#000000';
        ctx.font = `${Math.max(14, radius * 0.8)}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(ball.number.toString(), x, y);
      }
      ctx.globalAlpha = 1;
    });

    // Cue geometry is preserved in the state message. Emphasize its measured
    // axis and endpoints so the stick detection is easy to distinguish from
    // the ball overlays. Do not derive image length from the cue's physical
    // length (meters).
    if (cue && cueVisible && cue.tip_position && cue.butt_position) {
      const tip = normalizeImagePointToCanonical(cue.tip_position);
      const butt = normalizeImagePointToCanonical(cue.butt_position);
      const tipImage = tip ? canonicalToImagePoint(tip, videoSize) : null;
      const buttImage = butt ? canonicalToImagePoint(butt, videoSize) : null;
      if (tipImage && buttImage) {
        const tipX = tipImage.x * scaleX;
        const tipY = tipImage.y * scaleY;
        const buttX = buttImage.x * scaleX;
        const buttY = buttImage.y * scaleY;
        ctx.save();
        ctx.globalAlpha = 1;
        ctx.lineCap = 'round';

        // A dark under-stroke keeps the cue axis visible over both the felt
        // and the real cue. The bright center stroke identifies detector output.
        ctx.beginPath();
        ctx.moveTo(buttX, buttY);
        ctx.lineTo(tipX, tipY);
        ctx.strokeStyle = 'rgba(15, 23, 42, 0.95)';
        ctx.lineWidth = 12;
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(buttX, buttY);
        ctx.lineTo(tipX, tipY);
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 6;
        ctx.stroke();

        // Mark the measured tip and butt differently; the tip is the endpoint
        // directed toward the table center by the backend detector.
        ctx.beginPath();
        ctx.arc(tipX, tipY, 9, 0, 2 * Math.PI);
        ctx.fillStyle = '#0f172a';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(tipX, tipY, 6, 0, 2 * Math.PI);
        ctx.fillStyle = '#fb7185';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(tipX, tipY, 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = '#fff';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(buttX, buttY, 7, 0, 2 * Math.PI);
        ctx.fillStyle = '#0f172a';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(buttX, buttY, 4, 0, 2 * Math.PI);
        ctx.fillStyle = '#38bdf8';
        ctx.fill();

        const confidenceLabel =
          typeof cue.confidence === 'number'
            ? `CUE STICK · ${Math.round(cue.confidence * 100)}%`
            : 'CUE STICK';
        ctx.font = '700 13px system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        const labelPaddingX = 9;
        const labelHeight = 26;
        const labelWidth = ctx.measureText(confidenceLabel).width + labelPaddingX * 2;
        const midpointX = (tipX + buttX) / 2;
        const midpointY = (tipY + buttY) / 2;
        const labelX = Math.max(
          4,
          Math.min(canvas.width - labelWidth - 4, midpointX - labelWidth / 2),
        );
        const labelY = Math.max(
          labelHeight / 2 + 4,
          Math.min(
            canvas.height - labelHeight / 2 - 4,
            midpointY - 20,
          ),
        );
        ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
        ctx.fillRect(labelX, labelY - labelHeight / 2, labelWidth, labelHeight);
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 2;
        ctx.strokeRect(labelX, labelY - labelHeight / 2, labelWidth, labelHeight);
        ctx.fillStyle = '#fef08a';
        ctx.fillText(confidenceLabel, labelX + labelPaddingX, labelY);

        ctx.restore();
      }
    }
  }, [balls, cue, cueVisible, videoSize]);

  return (
    <div className={`flex flex-col h-screen ${className}`}>
      {/* Header */}
      <div className="flex-none bg-gray-800 text-white p-4">
        <h1 className="text-xl font-bold">Live View</h1>
        <div className="text-sm mt-1">
          <span className={`inline-block w-2 h-2 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
          {isConnected ? 'Connected' : 'Disconnected'} | {observedBalls.length} balls observed
          {nonObservedBalls.length > 0 ? `, ${nonObservedBalls.length} stale` : ''}
          {' | Cue stick: '}{cueVisible ? `visible${typeof cue?.confidence === 'number' ? ` (${Math.round(cue.confidence * 100)}%)` : ''}` : 'not detected'}
          {visionAgeMs !== null ? ` | Vision: ${visionAgeMs} ms old` : ''}
          {' | Mode: '}{gameTypeLabel}
        </div>
      </div>

      {/* Video container */}
      <div className="flex-1 relative bg-black flex items-center justify-center">
        {/* MJPEG video stream */}
        <img
          ref={videoRef}
          src={autoConnect && viewerConfigLoaded
            ? (viewerStreamUrl ?? apiClient.getVideoStreamUrl())
            : undefined}
          alt="Video stream"
          className="max-w-full max-h-full object-contain"
          style={{ display: 'block' }}
        />

        {/* Ball detection overlay canvas */}
        <canvas
          ref={canvasRef}
          className="absolute pointer-events-none"
          style={{
            // The image is centered in the taller video pane. Center the overlay
            // canvas the same way so image coordinates don't start at the pane's
            // top-left letterbox edge.
            left: '50%',
            top: '50%',
            transform: 'translate(-50%, -50%)',
            width: videoRef.current?.clientWidth || 0,
            height: videoRef.current?.clientHeight || 0,
          }}
        />

        {/* Status overlay when not connected */}
        {!isConnected && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75">
            <div className="text-white text-center">
              <div className="text-xl mb-2">Connecting...</div>
              <div className="text-sm">Waiting for video stream and ball detection</div>
            </div>
          </div>
        )}
      </div>

      {/* Ball list */}
      <div className="flex-none bg-gray-100 p-4 border-t">
        <h2 className="font-semibold mb-2">Detected Balls</h2>
        <div className="flex gap-2 flex-wrap">
          {observedBalls.length === 0 ? (
            <div className="text-gray-500 text-sm">No balls detected</div>
          ) : (
            observedBalls.map(ball => (
              <div
                key={ball.id}
                className="px-3 py-1 bg-white border rounded shadow-sm text-sm"
              >
                {getBallLabel(ball)}
                <span className="text-gray-500 ml-2">
                  {ball.confidence_kind === 'unavailable'
                    ? 'n/a'
                    : `${Math.round(ball.confidence * 100)}%`}
                </span>
              </div>
            ))
          )}
          {nonObservedBalls.map(ball => (
            <div
              key={`stale-${ball.id}`}
              className="px-3 py-1 bg-amber-50 border border-dashed border-amber-400 rounded shadow-sm text-sm text-amber-900"
            >
              {getBallFreshnessLabel(ball)}
              <span className="text-amber-700 ml-2">
                {ball.confidence_kind === 'unavailable'
                  ? 'n/a'
                  : `${Math.round(ball.confidence * 100)}%`}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
});

LiveView.displayName = 'LiveView';
