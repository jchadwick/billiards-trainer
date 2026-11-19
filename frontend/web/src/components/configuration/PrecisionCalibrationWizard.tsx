import { useEffect, useState } from 'react';
import type { MouseEvent } from 'react';
import axios from 'axios';
import { Check } from 'lucide-react';
import client from '../../api/axios-client';
import { Button } from '../ui/button';
import { LiveBallPlacement } from './LiveBallPlacement';

type Mode = 'optics' | 'ball-centers';
type Point = [number, number];
type Frame = {
  descriptor: {
    width_px: number;
    height_px: number;
    engine_instance_id: string;
    capture_session_id: string;
    source_frame_id: string;
    configuration_revision: number;
    calibration_revision: number;
    rectification_revision: number;
  };
  jpeg_base64: string;
  capture_token: string;
};
type BallMark = { image_center_px: Point; radius_px: number };
type ArchivedCapture = {
  slot: number;
  thumbnail_base64: string;
  table_center_mm?: Point;
};
type CaptureArchive = { captures: ArchivedCapture[]; table_size_mm?: Point };
type FitReport = {
  rms_px?: number;
  holdout_rms_px?: number;
  views?: number;
  fit_report?: { rms_mm: number; leave_one_out_rms_mm: number };
};
type CaptureReport = FitReport & { report?: FitReport };

const BOARD = { columns: 9, rows: 6, square_mm: 50.8 };
const BOARD_VIEWS = [
  ['Center', 'Hold the board flat in the middle of the image.'],
  ['Upper left', 'Move the board to the upper-left area and keep it mostly flat.'],
  ['Right side, tilted', 'Move the board right and raise its right edge toward the camera.'],
  ['Lower left, tilted', 'Move the board to the lower left and raise its left edge.'],
  ['Lower right, tilted', 'Move the board to the lower right and raise its bottom edge.'],
  ['Upper right, tilted', 'Move the board to the upper right and raise its top edge.'],
  ['Top center, tilted', 'Move the board to the top center, raise its bottom edge, and lift it slightly closer to the camera.'],
  ['Center left, rotated', 'Move the board left of center, rotate it clockwise, and raise its top edge.'],
] as const;
const PDF_ENDPOINT = '/api/v1/setup/calibration/checkerboard.pdf';
const BALL_SAMPLE_LIMITS = { minimum: 12, maximum: 24 };
const MILLIMETERS_PER_INCH = 25.4;
const FRAME_ENDPOINT = '/api/v1/setup/calibration/frame';
const CALIBRATION_ENDPOINT = '/api/v1/setup/calibration';
// Prioritize both ends of the camera image, with interior references to
// constrain the fit. Later positions fill gaps when a user wants more samples.
const BALL_TARGETS: ReadonlyArray<Point> = [
  [-0.44, -0.35], [-0.44, 0], [-0.44, 0.35], [0.44, -0.35],
  [0.44, 0], [0.44, 0.35], [-0.27, -0.22], [0.27, -0.22],
  [-0.27, 0.22], [0.27, 0.22], [0, -0.35], [0, 0.35],
  [-0.35, -0.43], [0.35, -0.43], [-0.35, 0.43], [0.35, 0.43],
  [-0.42, -0.18], [0.42, -0.18], [-0.42, 0.18], [0.42, 0.18],
  [-0.18, 0], [0.18, 0], [0, 0], [0, 0.18],
];
const FIELD = 'rounded-md border border-input bg-background px-2 py-1 text-foreground';

function imageSource(frame: Frame) {
  return `data:image/jpeg;base64,${frame.jpeg_base64}`;
}

function failureMessage(error: unknown) {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.error ?? error.response?.data?.message;
    if (typeof detail === 'string') return detail;
  }
  return error instanceof Error ? error.message : String(error);
}

async function captureFrame(): Promise<Frame> {
  const { data: { descriptor, jpeg_base64, capture_token } } = await client.get<Frame>(FRAME_ENDPOINT);
  if (!capture_token) throw new Error('This camera frame is missing its calibration token. Try capturing again.');
  return { descriptor, jpeg_base64, capture_token };
}

async function fitCalibration(mode: Mode, request: unknown) {
  const { data } = await client.getInstance().post<CaptureReport>(`${CALIBRATION_ENDPOINT}/${mode}`, request, {
    timeout: 180_000,
  });
  return data.report ?? data;
}

function useCaptureArchive(mode: Mode) {
  const endpoint = `${CALIBRATION_ENDPOINT}/captures/${mode}`;
  const [archive, setArchive] = useState<CaptureArchive>();
  const [loadError, setLoadError] = useState('');
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    client.get<CaptureArchive>(endpoint, { signal: controller.signal }).then(({ data }) => {
      if (!data || !Array.isArray(data.captures)) {
        throw new Error('The app did not receive a saved capture list from the server.');
      }
      if (!controller.signal.aborted) setArchive(data);
    }).catch((failure: unknown) => {
      if (!controller.signal.aborted) setLoadError(failureMessage(failure));
    });
    return () => controller.abort();
  }, [endpoint, attempt]);

  async function save(slot: number, frame: Frame & Partial<BallMark> & { table_center_mm?: Point }) {
    const { data } = await client.getInstance().put<ArchivedCapture>(`${endpoint}/${slot}`, frame);
    if (data?.slot !== slot || typeof data.thumbnail_base64 !== 'string') {
      throw new Error('The app did not receive a saved capture from the server.');
    }
    setArchive((current) => current && ({
      ...current,
      captures: [...current.captures.filter((item) => item.slot !== slot), data]
        .sort((a, b) => a.slot - b.slot),
    }));
  }

  async function remove(slot?: number) {
    await client.getInstance().delete(slot === undefined ? endpoint : `${endpoint}/${slot}`);
    setArchive((current) => current && ({
      ...current, captures: slot === undefined ? [] : current.captures.filter((item) => item.slot !== slot),
    }));
  }

  function retry() {
    setLoadError('');
    setAttempt((previous) => previous + 1);
  }

  return { archive, loadError, retry, save, remove };
}

function ArchiveStatus({ error, retry }: { error: string; retry: () => void }) {
  return error ? (
    <div className="space-y-2">
      <p role="alert">Could not load saved captures. {error}</p>
      <Button onClick={retry}>Retry loading captures</Button>
    </div>
  ) : <p role="status">Loading saved captures…</p>;
}

function ClearCaptures({ busy, onClear }: { busy: boolean; onClear: () => Promise<void> }) {
  const [confirming, setConfirming] = useState(false);
  if (!confirming) return <Button variant="outline" disabled={busy} onClick={() => setConfirming(true)}>Clear captures</Button>;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <p>Delete all saved photos for this calibration?</p>
      <Button variant="destructive" disabled={busy} onClick={() => void onClear().finally(() => setConfirming(false))}>
        Delete all captures
      </Button>
      <Button variant="outline" disabled={busy} onClick={() => setConfirming(false)}>Keep captures</Button>
    </div>
  );
}

function thumbnailSource(capture: ArchivedCapture) {
  return `data:image/jpeg;base64,${capture.thumbnail_base64}`;
}

type SaveCallback = (message: string) => void;

export function PrecisionCalibrationWizard({ mode, onSaved }: { mode: Mode; onSaved: SaveCallback }) {
  return mode === 'optics' ? <OpticsWizard onSaved={onSaved} /> : <BallCenterWizard onSaved={onSaved} />;
}

function OpticsWizard({ onSaved }: { onSaved: SaveCallback }) {
  const saved = useCaptureArchive('optics');
  const [capturing, setCapturing] = useState<number>();
  const [fitting, setFitting] = useState(false);
  const [error, setError] = useState('');
  const captures = saved.archive?.captures ?? [];
  const busy = capturing !== undefined || fitting;
  const apiOrigin = new URL(client.getInstance().defaults?.baseURL || window.location.origin, window.location.origin);
  const pdfUrl = new URL(PDF_ENDPOINT, apiOrigin);

  async function capture(index: number) {
    setCapturing(index);
    setError('');
    try {
      await saved.save(index, await captureFrame());
    } catch (failure) {
      setError(`${BOARD_VIEWS[index][0]}: ${failureMessage(failure)}`);
    } finally {
      setCapturing(undefined);
    }
  }

  async function fit() {
    if (captures.length !== BOARD_VIEWS.length) return;
    setFitting(true);
    setError('');
    try {
      const report = await fitCalibration('optics', { ...BOARD, slots: captures.map(({ slot }) => slot) });
      const quality = typeof report.holdout_rms_px === 'number'
        ? ` Held-out error was ${report.holdout_rms_px.toFixed(2)} px.` : '';
      onSaved(`Checkerboard calibration saved.${quality}`);
    } catch (failure) {
      setError(failureMessage(failure));
    } finally {
      setFitting(false);
    }
  }

  async function clear() {
    setFitting(true);
    setError('');
    try {
      await saved.remove();
    } catch (failure) {
      setError(failureMessage(failure));
    } finally {
      setFitting(false);
    }
  }

  return (
    <section className="space-y-3 border-t border-border pt-3" aria-label="Checkerboard calibration wizard">
      <Button asChild><a href={pdfUrl.href} download>Download printable checkerboard PDF</a></Button>
      <p>Print the four US Letter pages at actual size (100%). Assemble pages 1 and 2 above pages 3 and 4. Line up the overlapping squares and tape the pages to a flat, rigid board. Check that a square measures 2 inches (50.8 mm) before capturing.</p>
      <p>Capture each view below. Keep the entire checkerboard sharp and visible, including its outer edges.</p>
      {!saved.archive ? <ArchiveStatus error={saved.loadError} retry={saved.retry} /> : (
        <p role="status">{captures.length} of {BOARD_VIEWS.length} views saved.</p>
      )}
      <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4" aria-label="Required checkerboard views">
        {BOARD_VIEWS.map(([name, instruction], index) => {
          const frame = captures.find(({ slot }) => slot === index);
          return (
            <li key={name} className="space-y-2 rounded-md border border-border p-3">
              <h4 className="font-semibold">{index + 1}. {name}</h4>
              <p>{instruction}</p>
              {frame ? (
                <div className="relative grid place-items-center">
                  <img src={thumbnailSource(frame)} alt={`${name} capture`} className="aspect-video w-full rounded bg-background object-contain" />
                  <span className="absolute rounded-full bg-emerald-500 p-2 text-black" role="img" aria-label="Captured">
                    <Check aria-hidden="true" />
                  </span>
                </div>
              ) : <div className="grid aspect-video place-items-center rounded bg-background text-muted-foreground">No capture yet</div>}
              <Button disabled={!saved.archive || busy} onClick={() => void capture(index)} aria-label={`${frame ? 'Recapture' : 'Capture'} ${name}`}>
                {capturing === index ? 'Capturing…' : frame ? 'Recapture' : 'Capture'}
              </Button>
            </li>
          );
        })}
      </ol>
      <div className="flex flex-wrap gap-2">
        <Button variant="default" disabled={busy || captures.length !== BOARD_VIEWS.length} onClick={() => void fit()}>
          {fitting ? 'Fitting calibration…' : 'Fit and activate checkerboard calibration'}
        </Button>
        {captures.length > 0 && <ClearCaptures busy={busy} onClear={clear} />}
      </div>
      {error && <p role="alert">{error}</p>}
    </section>
  );
}

function BallCenterWizard({ onSaved }: { onSaved: SaveCallback }) {
  const { minimum, maximum } = BALL_SAMPLE_LIMITS;
  const saved = useCaptureArchive('ball-centers');
  const [frame, setFrame] = useState<Frame>();
  const [selectedSlot, setSelectedSlot] = useState<number>();
  const [center, setCenter] = useState<Point>();
  const [radius, setRadius] = useState('');
  const [zoom, setZoom] = useState(false);
  const [marking, setMarking] = useState(false);
  const [candidateMessage, setCandidateMessage] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  const samples = saved.archive?.captures ?? [];
  const size = saved.archive?.table_size_mm;
  const positions = size ? BALL_TARGETS.map((_, slot) => {
    const placement = ballPlacement(size, slot);
    const sample = samples.find((item) => item.slot === slot);
    const covered = samples.some(({ table_center_mm: measured }) =>
      measured?.[0] === placement.center[0] && measured[1] === placement.center[1]);
    return { ...placement, slot, sample, covered };
  }) : [];
  const suggestion = positions.find(({ sample, covered }) => !sample && !covered)?.slot;
  const slot = selectedSlot ?? suggestion ?? -1;
  const placement = positions[slot];
  const validRadius = Number.isFinite(Number(radius)) && Number(radius) > 0;

  function discardFrame() {
    setFrame(undefined);
    setCenter(undefined);
    setRadius('');
    setZoom(false);
    setMarking(false);
    setCandidateMessage('');
  }

  async function run(action: string, execute: () => Promise<void>) {
    setBusy(action);
    setError('');
    try {
      await execute();
    } catch (failure) {
      setError(failureMessage(failure));
    } finally {
      setBusy('');
    }
  }

  async function capture() {
    if (!placement || placement.sample || placement.covered) return;
    const measured = placement.center;
    await run('capture', async () => {
      const next = await captureFrame();
      discardFrame();
      setFrame(next);
      try {
        const { data } = await client.getInstance().post<{ candidate: BallMark | null }>(
          `${CALIBRATION_ENDPOINT}/ball-candidate`,
          { ...next, table_center_mm: measured },
        );
        if (data.candidate) {
          setCenter(data.candidate.image_center_px);
          setRadius(String(data.candidate.radius_px));
          setZoom(true);
          setCandidateMessage('Does the circle outline the ball you placed?');
        } else {
          setCandidateMessage('No ball was found near the measured position. Mark it below or retake the photo.');
          setMarking(true);
        }
      } catch (failure) {
        setCandidateMessage(`Automatic detection failed. ${failureMessage(failure)} You can mark the ball below.`);
        setMarking(true);
      }
    });
  }

  async function addSample() {
    if (!frame || !placement || !center || !validRadius) return;
    await run('save', async () => {
      await saved.save(slot, {
        ...frame, image_center_px: center, table_center_mm: placement.center, radius_px: Number(radius),
      });
      discardFrame();
      setSelectedSlot(undefined);
    });
  }

  async function fit() {
    if (samples.length < minimum) return;
    await run('fit', async () => {
      const report = await fitCalibration('ball-centers', { slots: samples.map((sample) => sample.slot) });
      const quality = typeof report.fit_report?.leave_one_out_rms_mm === 'number'
        ? ` Leave-one-out error was ${report.fit_report.leave_one_out_rms_mm.toFixed(2)} mm.` : '';
      onSaved(`Ball-center calibration saved.${quality}`);
    });
  }

  function correct() {
    setCenter(undefined);
    setRadius('');
    setZoom(false);
    setMarking(true);
    setCandidateMessage('');
  }

  return (
    <section className="space-y-4 border-t border-border pt-3" aria-label="Ball-center calibration wizard">
      <p>The camera sees the top of a ball above the cloth. These samples teach the app where the ball actually sits on the table, so shot guidance can use its true position.</p>
      {!saved.archive && <ArchiveStatus error={saved.loadError} retry={saved.retry} />}
      {saved.archive && <p role="status">{samples.length} of {minimum} required positions saved (up to {maximum}).</p>}
      {saved.archive && !size && <p role="alert">Table dimensions are unavailable. Reopen the wizard after table calibration is ready.</p>}

      {size && (
        <div className="space-y-2">
          <h4 className="font-semibold">Choose a position</h4>
          <p>Pick any numbered spot. Checked positions already have a saved sample.</p>
          <div role="group" aria-label="Ball placement positions"
            className="relative min-h-52 w-full max-w-3xl rounded-md border border-emerald-900 bg-emerald-950"
            style={{ aspectRatio: size[0] / size[1] }}>
            {positions.map(({ slot: index, center, sample, covered }) => {
              const [x, y] = sample?.table_center_mm ?? center;
              return (
              <button key={index} type="button" disabled={!!busy || !!frame || (!sample && covered)}
                aria-label={`Position ${index + 1}${sample ? ', saved' : covered ? ', covered by a saved sample' : ''}`}
                aria-pressed={slot === index}
                onClick={() => { setSelectedSlot(index); setError(''); }}
                className={`absolute grid size-6 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border text-xs font-semibold
                  disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white sm:size-8
                  ${slot === index ? 'border-white bg-amber-300 text-black' :
                  sample || covered ? 'border-emerald-200 bg-emerald-300 text-black' : 'border-emerald-500 bg-emerald-900 text-white hover:bg-emerald-700'}`}
                style={{ left: `${(0.5 + x / size[0]) * 100}%`, top: `${(0.5 + y / size[1]) * 100}%` }}>
                {sample || covered ? <Check className="size-4" aria-hidden="true" /> : index + 1}
              </button>
              );
            })}
          </div>
        </div>
      )}
      {placement?.sample && (
        <div className="space-y-2 rounded-md border border-border p-3">
          <h4 className="font-semibold">Position {slot + 1} saved</h4>
          <img src={thumbnailSource(placement.sample)} alt={`Saved capture for position ${slot + 1}`} className="w-full max-w-xs rounded" />
          {placement.sample.table_center_mm && (
            <p>Saved sample: {placement.sample.table_center_mm.join(', ')} mm from table center.</p>
          )}
          <Button size="sm" disabled={!!busy} onClick={() => void run('remove', () => saved.remove(slot))}>
            Remove saved capture
          </Button>
        </div>
      )}
      {placement && !placement.sample && !placement.covered && (
        <div className="space-y-3 rounded-md border border-border p-3">
          <h4 className="font-semibold">{frame ? 'Review the ball' : `Place the ball at position ${slot + 1}`}</h4>
          <dl aria-label="Cushion distances" className="grid max-w-3xl gap-3 sm:grid-cols-2">
            {placement.distances.map(({ cushion, millimeters }) => (
              <div key={cushion} className="rounded-md border border-amber-400/50 bg-amber-400/10 p-3 sm:p-4">
                <dt className="text-sm font-semibold uppercase tracking-wide text-amber-300">From {cushion} cushion</dt>
                <dd className="mt-1 flex flex-wrap items-baseline gap-x-4 gap-y-1 font-bold leading-none tabular-nums">
                  <span className="whitespace-nowrap text-3xl sm:text-4xl">{millimeters} <span className="text-lg">mm</span></span>
                  <span className="whitespace-nowrap text-3xl sm:text-4xl">{(millimeters / MILLIMETERS_PER_INCH).toFixed(2)} <span className="text-lg">in</span></span>
                </dd>
              </div>
            ))}
          </dl>
          {!frame && <>
            <LiveBallPlacement key={slot} slot={slot} paused={!!busy} target={placement.center} />
            <ol className="list-decimal space-y-1 pl-5">
              <li>Clear the table. Measure straight inward, perpendicular to each cushion nose (where the cushion meets the playing area).</li>
              <li>Mark where the two measurements meet on the cloth. A thin paper ring with a center hole works well.</li>
              <li>Center the ball over the mark. The green check confirms the ball is in the target area.</li>
            </ol>
          </>}
          {frame && (
            <div className="space-y-3">
              {candidateMessage && <p role="status">{candidateMessage}</p>}
              {marking && <p>{center ? 'Click the edge of the ball or enter the radius below.' : 'Click the center of the ball or enter its image coordinates below.'}</p>}
              <BallPreview frame={frame} center={center} radius={validRadius ? Number(radius) : undefined}
                zoom={zoom} onPoint={marking && !busy ? (point) => {
                  if (!center) {
                    setCenter(point);
                    setZoom(true);
                  } else {
                    setRadius(String(Math.hypot(point[0] - center[0], point[1] - center[1])));
                  }
                } : undefined} />
              {marking && (
                <fieldset className="flex flex-wrap gap-3" disabled={!!busy}>
                  {(['Image center X (px)', 'Image center Y (px)'] as const).map((title, axis) => {
                    const limit = (axis === 0 ? frame.descriptor.width_px : frame.descriptor.height_px) - 1;
                    return <label key={title} className="grid gap-1">{title}
                      <input className={FIELD} type="number" min="0" max={limit} step="any"
                        value={center ? Math.round(center[axis] * 100) / 100 : ''}
                        onChange={(event) => {
                          const value = Math.max(0, Math.min(limit, Number(event.target.value)));
                          const [oldX, oldY] = center ?? [(frame.descriptor.width_px - 1) / 2, (frame.descriptor.height_px - 1) / 2];
                          setCenter(axis === 0 ? [value, oldY] : [oldX, value]);
                          setRadius('');
                        }} />
                    </label>;
                  })}
                  <label className="grid gap-1">Image radius (px)
                    <input className={FIELD} type="number" min="0" step="any" value={radius} onChange={(event) => setRadius(event.target.value)} />
                  </label>
                </fieldset>
              )}
              <div className="flex flex-wrap gap-2">
                <Button variant="default" disabled={!!busy || !center || !validRadius} onClick={() => void addSample()}>
                  {busy === 'save' ? 'Saving…' : 'Confirm and save position'}
                </Button>
                {center && <Button disabled={!!busy} onClick={() => setZoom(!zoom)}>{zoom ? 'Show full frame' : 'Zoom in on ball'}</Button>}
                {center && <Button disabled={!!busy} onClick={correct}>{marking ? 'Reset ball marking' : 'Mark ball manually'}</Button>}
                <Button disabled={!!busy} onClick={discardFrame}>Back to placement</Button>
              </div>
            </div>
          )}
          <Button variant={frame ? 'outline' : 'default'} disabled={!!busy} onClick={() => void capture()}>
            {busy === 'capture' ? 'Finding the ball…' : frame ? 'Retake and find ball' : 'Ball is on the mark — capture'}
          </Button>
        </div>
      )}

      {samples.length > 0 && (
        <div className="space-y-2">
          <h4 className="font-semibold">Saved positions</h4>
          <ol className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-6" aria-label="Saved ball positions">
            {samples.map((sample) => (
              <li key={sample.slot} className="space-y-1 rounded-md border border-border p-2">
                <img src={thumbnailSource(sample)} alt={`Ball position ${sample.slot + 1}`} className="aspect-video w-full rounded bg-background object-contain" />
                <p className="text-sm">{sample.slot + 1}. {sample.table_center_mm?.join(', ')} mm</p>
                <Button size="sm" disabled={!!busy || !!frame} onClick={() => void run('remove', () => saved.remove(sample.slot))}>
                  Remove position {sample.slot + 1}
                </Button>
              </li>
            ))}
          </ol>
        </div>
      )}
      <div className="flex flex-wrap gap-2">
        <Button variant="default" disabled={!!busy || !!frame || samples.length < minimum} onClick={() => void fit()}>
          {busy === 'fit' ? 'Fitting calibration…' : 'Fit and activate ball-center calibration'}
        </Button>
        {samples.length > 0 && <ClearCaptures busy={!!busy} onClear={() => run('clear', async () => {
          await saved.remove();
          discardFrame();
          setSelectedSlot(undefined);
        })} />}
      </div>
      {error && <p role="alert">{error}</p>}
    </section>
  );
}

function ballPlacement([width, height]: Point, slot: number) {
  const [x, y] = BALL_TARGETS[slot];
  const xDistance = Math.round((0.5 - Math.abs(x)) * width);
  const yDistance = Math.round((0.5 - Math.abs(y)) * height);
  const center: Point = [
    (x <= 0 ? 1 : -1) * (xDistance - width / 2),
    (y <= 0 ? 1 : -1) * (yDistance - height / 2),
  ];
  return {
    center,
    distances: [
      { cushion: x <= 0 ? 'left' : 'right', millimeters: xDistance },
      { cushion: y <= 0 ? 'top' : 'bottom', millimeters: yDistance },
    ],
  };
}

function BallPreview({ frame, center, radius, zoom, onPoint }: {
  frame: Frame; center?: Point; radius?: number; zoom: boolean; onPoint?: (point: Point) => void;
}) {
  const { width_px: width, height_px: height } = frame.descriptor;
  const side = Math.min(width, height, Math.max((radius ?? 0) * 6, Math.min(width, height) / 8));
  const crop = zoom && center;
  const left = crop ? Math.max(0, Math.min(width - side, crop[0] - side / 2)) : 0;
  const top = crop ? Math.max(0, Math.min(height - side, crop[1] - side / 2)) : 0;
  const viewWidth = crop ? side : width;
  const viewHeight = crop ? side : height;
  const lineWidth = Math.min(viewWidth, viewHeight) / 150;

  function pick(event: MouseEvent<SVGSVGElement>) {
    if (!onPoint) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) return;
    onPoint([
      Math.max(0, Math.min(width - 1, left + ((event.clientX - bounds.left) / bounds.width) * viewWidth)),
      Math.max(0, Math.min(height - 1, top + ((event.clientY - bounds.top) / bounds.height) * viewHeight)),
    ]);
  }

  return (
    <svg className={`block h-auto w-full rounded-md ${crop ? 'max-w-xs' : 'max-w-3xl'} ${onPoint ? 'cursor-crosshair' : ''}`}
      viewBox={`${left} ${top} ${viewWidth} ${viewHeight}`} aria-label={crop ? 'Zoomed ball frame' : 'Frozen ball frame'}
      role="img" onClick={pick}>
      <image href={imageSource(frame)} width={width} height={height} />
      {center && <>
        <circle cx={center[0]} cy={center[1]} r={lineWidth * 1.5} fill="#a3ff4b" />
        {radius && <circle cx={center[0]} cy={center[1]} r={radius} fill="none" stroke="#a3ff4b" strokeWidth={lineWidth} />}
      </>}
    </svg>
  );
}
