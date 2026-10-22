/**
 * Simple live view component showing video stream with ball detection overlay
 */

import React, { useEffect, useRef, useState } from 'react';
import { observer } from 'mobx-react-lite';
import { apiClient } from '../../api/client';
import { getConfiguredLiveStreamUrl } from './live-stream-url';
import {
  currentObservationAgeSeconds,
  DEFAULT_MAX_STATE_OBSERVATION_AGE_SECONDS,
  observationIsStale,
  positiveFreshnessLimit,
} from './freshness';
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

// A standard pool set contains the cue ball and fifteen object balls. Keep the
// live layout stable even while only part of the set is visible to the camera.
const BALL_SLOT_COUNT = 16;

export const LiveView: React.FC<LiveViewProps> = observer(({ className = '', autoConnect = true }) => {
  const { system, videoStore } = useRootStore();
  const videoRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });
  const [viewerStreamUrl, setViewerStreamUrl] = useState<string | undefined>();
  const [viewerConfigLoaded, setViewerConfigLoaded] = useState(false);
  const [maximumObservationAgeSeconds, setMaximumObservationAgeSeconds] = useState(
    DEFAULT_MAX_STATE_OBSERVATION_AGE_SECONDS,
  );
  const [ageTickMonotonicMs, setAgeTickMonotonicMs] = useState(0);
  const receivedAtMonotonicMsRef = useRef(0);
  const ageAtLastUniqueObservationSecondsRef = useRef<number | null>(null);
  const lastUniqueObservationKeyRef = useRef<string | number | null>(null);
  const latestState = videoStore.latestGameState;
  const balls: Ball[] = latestState?.balls ?? [];
  const cue: Cue | null = latestState?.cue ?? null;
  const gameType = latestState?.game_type ?? null;
  const observationStatus = latestState?.observation_status;
  const reportedAge = observationStatus?.frame_publication_age_seconds;
  const observationIdentity = observationStatus?.observation_identity;
  const sourceIdentityParts = [
    observationIdentity?.engine_instance_id
      ?? observationStatus?.engine_instance_id
      ?? observationStatus?.vision_instance_id,
    observationIdentity?.session_id
      ?? observationStatus?.source_session_id
      ?? observationStatus?.source_process_id,
    observationIdentity?.frame_id
      ?? observationStatus?.source_frame_id
      ?? observationStatus?.source_frame_number
      ?? observationStatus?.source_timestamp
      ?? latestState?.frame_number,
    observationIdentity?.snapshot_revision ?? observationStatus?.snapshot_revision,
  ];
  const uniqueObservationKey = sourceIdentityParts.some(
    (part) => part !== undefined && part !== null && part !== '',
  )
    ? sourceIdentityParts.map((part) => part ?? '').join('\u0000')
    : latestState ? '__anonymous_observation__' : null;
  const currentAgeSeconds = currentObservationAgeSeconds(
    reportedAge,
    ageAtLastUniqueObservationSecondsRef.current,
    receivedAtMonotonicMsRef.current,
    ageTickMonotonicMs,
  );
  const visionAgeMs = currentAgeSeconds === null
    ? null
    : Math.round(currentAgeSeconds * 1000);
  const freshnessStatusIsStale = observationIsStale(
    null,
    maximumObservationAgeSeconds,
    observationStatus,
  );
  const isFreshnessUnknown = latestState !== null
    && currentAgeSeconds === null
    && !freshnessStatusIsStale;
  const isObservationStale = latestState !== null && (
    isFreshnessUnknown
    || observationIsStale(
      currentAgeSeconds,
      maximumObservationAgeSeconds,
      observationStatus,
    )
  );
  const isConnected = autoConnect && system.status.isConnected;
  const hasReportedAge = typeof reportedAge === 'number' && Number.isFinite(reportedAge);

  // Age keeps advancing without WebSocket messages. The origin changes only
  // for a new source observation, so cached/repeated publications cannot reset
  // the client freshness clock and make old positions look current.
  useEffect(() => {
    if (uniqueObservationKey === null) {
      lastUniqueObservationKeyRef.current = null;
      return;
    }
    if (
      typeof reportedAge !== 'number'
      || !Number.isFinite(reportedAge)
      || reportedAge < 0
    ) return;
    if (lastUniqueObservationKeyRef.current === uniqueObservationKey) return;
    lastUniqueObservationKeyRef.current = uniqueObservationKey;
    receivedAtMonotonicMsRef.current = performance.now();
    ageAtLastUniqueObservationSecondsRef.current = reportedAge;
    setAgeTickMonotonicMs(receivedAtMonotonicMsRef.current);
  }, [uniqueObservationKey, reportedAge]);

  useEffect(() => {
    if (!hasReportedAge) return;
    const interval = window.setInterval(
      () => setAgeTickMonotonicMs(performance.now()),
      100,
    );
    return () => window.clearInterval(interval);
  }, [hasReportedAge]);

  const allBallPositionsAreStale = isObservationStale;
  const observedBalls = allBallPositionsAreStale
    ? []
    : balls.filter((ball) => ball.is_observed !== false);
  const nonObservedBalls = allBallPositionsAreStale
    ? balls
    : balls.filter((ball) => ball.is_observed === false);
  const cueVisible = !allBallPositionsAreStale
    && Boolean(cue && (cue.is_visible ?? cue.detected));
  const displayedBalls = [
    ...observedBalls.map((ball) => ({ ball, entireStateIsStale: false })),
    ...nonObservedBalls.map((ball) => ({ ball, entireStateIsStale: allBallPositionsAreStale })),
  ].slice(0, BALL_SLOT_COUNT);
  const videoAspectRatio = videoSize.width > 0 && videoSize.height > 0
    ? `${videoSize.width} / ${videoSize.height}`
    : '16 / 9';

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
      const configuredMaximumAge = values?.perception?.freshness
        ?.maximum_observation_age_seconds;
      setMaximumObservationAgeSeconds(
        positiveFreshnessLimit(configuredMaximumAge),
      );
      setViewerStreamUrl(getConfiguredLiveStreamUrl(streamUrl));
      setViewerConfigLoaded(true);
    }).catch(() => {
      if (cancelled) return;
      setViewerStreamUrl(undefined);
      setMaximumObservationAgeSeconds(DEFAULT_MAX_STATE_OBSERVATION_AGE_SECONDS);
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

  const getBallFreshnessLabel = (
    ball: Ball,
    entireStateIsStale: boolean,
  ): string => {
    if (entireStateIsStale) {
      return isFreshnessUnknown
        ? 'Last known state (freshness unknown)'
        : 'Last known state (stale)';
    }
    if (ball.position_kind === 'extrapolated') return 'Predicted track';
    if (ball.position_kind === 'filtered') return 'Filtered track';
    return 'Last observed';
  };

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
    if (isObservationStale) return;

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
  }, [balls, cue, cueVisible, isObservationStale, videoSize]);

  return (
    <div className={`flex flex-col w-full ${className}`}>
      {/* Header */}
      <div className="flex-none bg-gray-800 text-white p-4">
        <h1 className="text-xl font-bold">Live View</h1>
        <div className="text-sm mt-1">
          <span className={`inline-block w-2 h-2 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
          {isConnected ? 'Connected' : 'Disconnected'} | {observedBalls.length} balls observed
          {nonObservedBalls.length > 0 ? `, ${nonObservedBalls.length} stale` : ''}
          {' | Cue stick: '}{cueVisible ? `visible${typeof cue?.confidence === 'number' ? ` (${Math.round(cue.confidence * 100)}%)` : ''}` : 'not detected'}
          {visionAgeMs !== null
            ? ` | State age: ${visionAgeMs} ms${isObservationStale ? ' · STALE' : ''}`
            : latestState
              ? ` | State age unavailable${isFreshnessUnknown ? ' · positions not current-qualified' : ''}`
              : ''}
          {' | Mode: '}{gameTypeLabel}
        </div>
      </div>

      {/* Video container */}
      <div
        data-testid="live-video-pane"
        className="relative w-full flex-none overflow-hidden bg-black"
        style={{ aspectRatio: videoAspectRatio }}
      >
        {/* MJPEG video stream */}
        {autoConnect && viewerConfigLoaded && !viewerStreamUrl ? (
          <p role="status" className="px-4 text-center text-white">
            Direct camera stream unavailable. Configure a browser-reachable
            <code className="mx-1">api.stream.viewer_url</code> to display video.
          </p>
        ) : (
          <img
            ref={videoRef}
            src={autoConnect && viewerConfigLoaded ? viewerStreamUrl : undefined}
            alt="Video stream"
            className="absolute inset-0 block h-full w-full object-contain"
          />
        )}

        {/* Ball detection overlay canvas */}
        <canvas
          ref={canvasRef}
          className="absolute pointer-events-none"
          data-freshness={isObservationStale
            ? (isFreshnessUnknown ? 'unknown' : 'stale')
            : 'current'}
          style={{
            // The pane follows the source image aspect ratio, so the source and
            // overlay share the same origin and dimensions without letterbox
            // offsets or crop-dependent coordinate transforms.
            inset: 0,
            width: '100%',
            height: '100%',
            opacity: isObservationStale ? 0 : 1,
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
      <div className="flex-none border-t bg-gray-100 px-4 py-3">
        <div className="mb-2 flex items-center gap-3">
          <h2 className="font-semibold">Detected Balls</h2>
          <span
            role="status"
            aria-live="polite"
            className="rounded-full bg-white px-2 py-0.5 text-sm text-gray-600"
          >
            {observedBalls.length} {observedBalls.length === 1 ? 'ball' : 'balls'} detected
          </span>
        </div>
        <div
          role="grid"
          aria-label="Detected ball slots"
          className="grid grid-cols-4 gap-2"
        >
          {Array.from({ length: BALL_SLOT_COUNT }, (_, index) => {
            const entry = displayedBalls[index];
            if (!entry) {
              return (
                <div
                  key={`empty-${index}`}
                  role="gridcell"
                  aria-label={`Empty ball slot ${index + 1}`}
                  className="min-h-12 rounded border border-dashed border-gray-300 bg-gray-50"
                />
              );
            }

            const { ball, entireStateIsStale } = entry;
            const isObserved = ball.is_observed !== false && !entireStateIsStale;
            const label = isObserved
              ? getBallLabel(ball)
              : getBallFreshnessLabel(ball, entireStateIsStale);
            const confidence = ball.confidence_kind === 'unavailable'
              ? 'n/a'
              : `${Math.round(ball.confidence * 100)}%`;

            return (
              <div
                key={`${isObserved ? 'observed' : 'stale'}-${ball.id}`}
                role="gridcell"
                title={`${label} · ${confidence}`}
                className={`flex min-h-12 min-w-0 flex-col items-center justify-center rounded border px-2 py-1 text-center text-xs shadow-sm ${
                  isObserved
                    ? 'border-gray-300 bg-white text-gray-900'
                    : 'border-dashed border-amber-400 bg-amber-50 text-amber-900'
                }`}
              >
                <span className="w-full truncate font-medium">{label}</span>
                <span className={isObserved ? 'text-gray-500' : 'text-amber-700'}>
                  {confidence}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
});

LiveView.displayName = 'LiveView';
