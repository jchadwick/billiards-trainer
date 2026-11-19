import { useEffect, useRef, useState } from 'react';
import { Check } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import client from '../../api/axios-client';
import { useRootStore } from '../../stores/context';

type Point = [number, number];
type PlacementObservation = {
  width_px: number;
  height_px: number;
  target_center_px: Point;
  target_radius_px: Point;
  ball_center_px: Point | null;
  in_target: boolean;
  observation_available: boolean;
};

const PLACEMENT_ENDPOINT = '/api/v1/setup/calibration/ball-placement';
const POLL_INTERVAL_MS = 600;
const REQUEST_TIMEOUT_MS = 1800;
const OBSERVATION_LIMIT_MS = 2000;

export const LiveBallPlacement = observer(function LiveBallPlacement({
  slot, target, paused,
}: { slot: number; target: Point; paused: boolean }) {
  const store = useRootStore();
  const image = store.nativeVideoElement;
  const { configurationSynchronized, dimensions, mode, sourceGeneration, state } = store.videoStore.delivery;
  const imageSlot = useRef<HTMLDivElement>(null);
  const [x, y] = target;
  const live = !!image && mode === 'native' && state === 'loaded' && configurationSynchronized;
  const [status, setStatus] = useState<{
    observation: PlacementObservation;
    current: boolean;
    sourceGeneration: number;
  }>();
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    const slot = imageSlot.current;
    if (!image || !slot || mode !== 'native') return;
    const parking = image.parentElement;
    slot.appendChild(image);
    return () => {
      if (image.parentElement === slot) parking?.appendChild(image);
    };
  }, [image, mode]);

  useEffect(() => {
    setStatus(undefined);
    setUnavailable(false);
    if (!live || paused) return;

    const controller = new AbortController();
    let pollTimer: number | undefined;
    let expiryTimer: number | undefined;
    async function poll() {
      try {
        const { data } = await client.get<PlacementObservation>(PLACEMENT_ENDPOINT, {
          params: { x_mm: x, y_mm: y },
          signal: controller.signal,
          timeout: REQUEST_TIMEOUT_MS,
        });
        if (controller.signal.aborted) return;
        setStatus({ observation: data, current: true, sourceGeneration });
        setUnavailable(false);
        window.clearTimeout(expiryTimer);
        expiryTimer = window.setTimeout(() => {
          setStatus({ observation: data, current: false, sourceGeneration });
          setUnavailable(true);
        }, OBSERVATION_LIMIT_MS);
      } catch {
        if (!controller.signal.aborted) setUnavailable(true);
      } finally {
        if (!controller.signal.aborted) pollTimer = window.setTimeout(poll, POLL_INTERVAL_MS);
      }
    }
    void poll();

    return () => {
      controller.abort();
      window.clearTimeout(pollTimer);
      window.clearTimeout(expiryTimer);
    };
  }, [live, paused, sourceGeneration, x, y]);

  const observation = status?.sourceGeneration === sourceGeneration ? status.observation : undefined;
  // Coordinates scale with the preview. Refuse to overlay a stream whose
  // aspect ratio differs from the producer's calibrated image geometry.
  const overlay = observation && dimensions && observation.width_px > 0 && observation.height_px > 0
    && Math.abs(observation.width_px / observation.height_px - dimensions.width / dimensions.height) < 0.01
    ? observation : undefined;
  const seen = live && !paused && status?.current && !!overlay?.observation_available;
  const inTarget = seen && overlay.in_target;
  const ball = seen ? overlay.ball_center_px : null;
  const message = paused ? 'Capturing the ball…'
    : !live ? state === 'error' || state === 'unconfigured'
      ? 'Live camera video is unavailable. You can still place the ball using the cushion measurements.'
      : 'Waiting for live camera video…'
    : inTarget ? 'Ball in target'
    : unavailable || (observation && (!overlay || !observation.observation_available)) ? 'Live placement feedback is unavailable. You can still capture and review the ball.'
    : ball ? 'Move the ball into the target.'
    : overlay ? 'Place one ball inside the target and keep it still.'
    : 'Locating the placement target…';

  return (
    <figure className="w-full max-w-3xl space-y-2">
      <div
        className="relative overflow-hidden rounded-md bg-black"
        style={{ aspectRatio: dimensions ? `${dimensions.width} / ${dimensions.height}` : '16 / 9' }}
        role="img" aria-label={`Live camera showing the target for position ${slot + 1}`}
      >
        <div className="native-video-image-slot" ref={imageSlot} />
        {overlay && (
          <svg
            className="pointer-events-none absolute inset-0 size-full"
            viewBox={`0 0 ${overlay.width_px} ${overlay.height_px}`}
            aria-hidden="true"
          >
            <ellipse cx={overlay.target_center_px[0]} cy={overlay.target_center_px[1]}
              rx={overlay.target_radius_px[0]} ry={overlay.target_radius_px[1]}
              fill={inTarget ? '#34d399' : '#fbbf24'} fillOpacity=".24" stroke="#141414" strokeWidth="6"
              vectorEffect="non-scaling-stroke" />
            <ellipse cx={overlay.target_center_px[0]} cy={overlay.target_center_px[1]}
              rx={overlay.target_radius_px[0]} ry={overlay.target_radius_px[1]}
              fill="none" stroke={inTarget ? '#34d399' : '#fbbf24'} strokeWidth="3"
              vectorEffect="non-scaling-stroke" />
            {ball && !inTarget && (
              <line x1={ball[0]} y1={ball[1]} x2={overlay.target_center_px[0]} y2={overlay.target_center_px[1]}
                stroke="white" strokeWidth="2" strokeDasharray="6 4" vectorEffect="non-scaling-stroke" />
            )}
          </svg>
        )}
        {inTarget && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center">
            <span className="rounded-full bg-emerald-400 p-3 text-emerald-950 shadow-lg">
              <Check className="size-16 sm:size-20" strokeWidth={3} aria-hidden="true" />
            </span>
          </div>
        )}
      </div>
      <figcaption role="status" className={inTarget ? 'font-semibold text-emerald-400' : undefined}>
        {message}
      </figcaption>
    </figure>
  );
});
