import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import type { KeyboardEvent, MouseEvent } from 'react';
import axios from 'axios';
import client from '../../api/axios-client';
import { useRootStore } from '../../stores/context';
import { Button } from '../ui/button';
import { getSetupStatus } from '../../api/setup';
import type { SetupStatus } from '../../api/setup';
import { PrecisionCalibrationWizard } from './PrecisionCalibrationWizard';

const AdvancedConfiguration = lazy(async () => ({
  default: (await import('./ConfigurationPage')).ConfigurationPage,
}));

type Camera = { display_name: string | null; capture_selector: string };
type Point = { x_px: number; y_px: number };
type FrozenFrame = {
  descriptor: { width_px: number; height_px: number };
  jpeg_base64: string;
  expected_store_revision: number | null;
};

const CORNERS = ['top left', 'top right', 'bottom right', 'bottom left'];
const TABLE_SIZES = [
  { label: '39 × 78 inches', short: 39, long: 78 },
  { label: '44 × 88 inches', short: 44, long: 88 },
  { label: '50 × 100 inches', short: 50, long: 100 },
];
const MM_PER_INCH = 25.4;
const STATUS_INTERVAL_MS = 2000;
const FIELD_STYLE = 'rounded-md border border-input bg-background px-2 py-1 text-foreground';

function messageFor(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const message = error.response?.data?.error ?? error.response?.data?.message;
    if (typeof message === 'string') return message;
  }
  return error instanceof Error ? error.message : String(error);
}

function TableCalibration({ onSaved }: { onSaved: () => void }) {
  const [frame, setFrame] = useState<FrozenFrame>();
  const [corners, setCorners] = useState<Point[]>([]);
  const [cursor, setCursor] = useState<Point>();
  const [preset, setPreset] = useState('');
  const [rotated, setRotated] = useState(false);
  const [width, setWidth] = useState('');
  const [height, setHeight] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);

  async function freeze() {
    setBusy(true);
    setMessage('');
    try {
      const response = await client.get<FrozenFrame>('/api/v1/setup/calibration/frame');
      const next = response.data;
      setFrame(next);
      setCorners([]);
      setCursor({
        x_px: (next.descriptor.width_px - 1) / 2,
        y_px: (next.descriptor.height_px - 1) / 2,
      });
    } catch (error) {
      setMessage(messageFor(error));
    } finally {
      setBusy(false);
    }
  }

  function pick(point: Point) {
    setCorners((points) => points.length < CORNERS.length ? [...points, point] : points);
    setCursor(point);
  }

  function pickByMouse(event: MouseEvent<SVGSVGElement>) {
    if (!frame || busy) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    pick({
      x_px: ((event.clientX - bounds.left) / bounds.width) * (frame.descriptor.width_px - 1),
      y_px: ((event.clientY - bounds.top) / bounds.height) * (frame.descriptor.height_px - 1),
    });
  }

  function pickByKeyboard(event: KeyboardEvent<SVGSVGElement>) {
    if (!frame || !cursor || busy) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      pick(cursor);
      return;
    }
    const directions: Record<string, [number, number]> = {
      ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1],
    };
    const direction = directions[event.key];
    if (!direction) return;
    event.preventDefault();
    const step = event.shiftKey ? 10 : 1;
    const { width_px, height_px } = frame.descriptor;
    setCursor({
      x_px: Math.max(0, Math.min(width_px - 1, cursor.x_px + direction[0] * step)),
      y_px: Math.max(0, Math.min(height_px - 1, cursor.y_px + direction[1] * step)),
    });
  }

  const selected = preset === '' || preset === 'custom' ? undefined : TABLE_SIZES[Number(preset)];
  const widthMm = selected ? (rotated ? selected.short : selected.long) * MM_PER_INCH : Number(width);
  const heightMm = selected ? (rotated ? selected.long : selected.short) * MM_PER_INCH : Number(height);
  const validDimensions = !!preset && Number.isFinite(widthMm) && Number.isFinite(heightMm)
    && widthMm > 0 && heightMm > 0;

  async function save() {
    if (!frame || corners.length !== CORNERS.length || !validDimensions) return;
    setBusy(true);
    setMessage('');
    try {
      await client.getInstance().post('/api/v1/setup/calibration', {
        schema_version: 1,
        descriptor: frame.descriptor,
        expected_store_revision: frame.expected_store_revision,
        table_width_mm: widthMm,
        table_height_mm: heightMm,
        corners_px: corners,
      });
      setFrame(undefined);
      onSaved();
    } catch (error) {
      const issue = messageFor(error);
      if (issue === 'table_visibility_reference_quality') {
        setMessage('The table was difficult to verify in this frame. Clear the playing surface, check the lighting, and freeze a new frame.');
      } else if (axios.isAxiosError(error) && error.response?.status === 409) {
        setMessage('The calibration frame is out of date. Freeze a new frame and mark the corners again.');
      } else {
        setMessage(issue);
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="space-y-3 rounded-lg border border-border bg-card p-4" aria-labelledby="table-calibration-title">
      <h2 id="table-calibration-title" className="text-lg font-semibold">Calibrate the table</h2>
      <p>Clear the playing surface and make sure the camera can see all four corners.</p>
      <Button onClick={() => void freeze()} disabled={busy}>{frame ? 'Freeze a new frame' : 'Freeze camera frame'}</Button>
      {frame && (
        <>
          <p id="corner-instructions">
            {corners.length < CORNERS.length ? `Mark the ${CORNERS[corners.length]} corner of the playing surface.` : 'All four corners marked.'}
            {' '}With the keyboard, use the arrow keys to move and Enter to mark a point. Hold Shift to move faster.
          </p>
          <svg
            className="block h-auto w-full max-w-4xl cursor-crosshair"
            viewBox={`0 0 ${frame.descriptor.width_px} ${frame.descriptor.height_px}`}
            tabIndex={0}
            aria-label="Frozen camera frame for marking table corners"
            aria-describedby="corner-instructions"
            onClick={pickByMouse}
            onKeyDown={pickByKeyboard}
          >
            <image href={`data:image/jpeg;base64,${frame.jpeg_base64}`} width="100%" height="100%" />
            {corners.length > 1 && (
              <polyline
                points={(corners.length === CORNERS.length ? [...corners, corners[0]] : corners)
                  .map((point) => `${point.x_px},${point.y_px}`).join(' ')}
                fill="none"
                stroke="lime"
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
            )}
            {corners.map((point, index) => (
              <circle key={index} cx={point.x_px} cy={point.y_px} r="6" fill="lime" stroke="black" strokeWidth="2" />
            ))}
            {cursor && corners.length < CORNERS.length && (
              <circle cx={cursor.x_px} cy={cursor.y_px} r="8" fill="none" stroke="white" strokeWidth="2" />
            )}
          </svg>
          <Button variant="outline" disabled={busy || !corners.length} onClick={() => setCorners((points) => points.slice(0, -1))}>Undo last corner</Button>
          <div className="space-y-2">
            <label>Playing surface size{' '}
              <select className={FIELD_STYLE} value={preset} onChange={(event) => setPreset(event.target.value)}>
                <option value="">Choose measured dimensions</option>
                {TABLE_SIZES.map((size, index) => <option key={size.label} value={index}>{size.label}</option>)}
                <option value="custom">Enter measurements in millimeters</option>
              </select>
            </label>
            {selected && (
              <label className="block"><input type="checkbox" checked={rotated} onChange={(event) => setRotated(event.target.checked)} /> The long side runs from top to bottom in the image</label>
            )}
            {preset === 'custom' && (
              <div className="flex flex-wrap gap-3">
                <label>Width, left to right (mm) <input className={FIELD_STYLE} type="number" min="1" step="any" value={width} onChange={(event) => setWidth(event.target.value)} /></label>
                <label>Height, top to bottom (mm) <input className={FIELD_STYLE} type="number" min="1" step="any" value={height} onChange={(event) => setHeight(event.target.value)} /></label>
              </div>
            )}
          </div>
          <Button disabled={busy || corners.length !== CORNERS.length || !validDimensions} onClick={() => void save()}>Save calibration</Button>
        </>
      )}
      {message && <p role="status">{message}</p>}
    </section>
  );
}

const CALIBRATION_LABELS = {
  ready: 'Compatible',
  missing: 'Missing',
  invalid: 'Invalid',
  blocked: 'Blocked',
  unverified: 'Not verified',
} as const;

function CalibrationAssets({ status }: { status?: SetupStatus }) {
  const [editing, setEditing] = useState(false);
  const [wizard, setWizard] = useState<'optics' | 'ball-centers'>();
  const [optics, setOptics] = useState('');
  const [centers, setCenters] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const checks = status?.precision_calibration;

  async function savePaths() {
    setBusy(true);
    setMessage('');
    try {
      const paths = {
        ...(optics.trim() && { optics_profile_path: optics.trim() }),
        ...(centers.trim() && { ball_center_calibration_path: centers.trim() }),
      };
      const response = await client.getInstance().put<{
        success: boolean; validation_errors?: { message: string }[];
      }>('/api/v1/config', { perception: { rust_live: { ball_localization: paths } } });
      if (!response.data.success) {
        setMessage(response.data.validation_errors?.map(({ message }) => message).join(' ') || 'Calibration paths were rejected.');
        return;
      }
      setMessage('Paths saved. Restart the service to apply them. If the table calibration becomes inactive, recalibrate the table and fit ball centers against the new table revision.');
      setEditing(false);
      setOptics('');
      setCenters('');
    } catch (error) {
      setMessage(messageFor(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      {([
        ['optics', 'Camera optics', checks?.optics, 'Fit an optics profile from checkerboard captures for this camera and capture resolution.'],
        ['ball-centers', 'Ball-center calibration', checks?.ball_centers, 'Fit ball centers from independently measured ball positions using the optics profile and the current table calibration.'],
      ] as const).map(([mode, name, check, instructions]) => (
        <div key={name} className="space-y-1 rounded-md border border-border p-3">
          <h3 className="font-semibold">{name}</h3>
          <p className={check?.state === 'ready' ? 'text-emerald-400' : 'text-amber-400'}>
            {check ? CALIBRATION_LABELS[check.state] : 'Status unavailable'}
          </p>
          {check?.detail && <p>{check.detail}</p>}
          {check?.state !== 'ready' && <p>{instructions}</p>}
          {status?.state === 'running' && (mode === 'optics' || (status.calibrated && checks?.optics.state === 'ready')) && (
            <>
              <Button aria-expanded={wizard === mode} onClick={() => setWizard(wizard === mode ? undefined : mode)}>
                {wizard === mode ? 'Close wizard' : mode === 'optics' ? 'Calibrate camera optics' : 'Calibrate ball centers'}
              </Button>
              {wizard === mode && (
                <PrecisionCalibrationWizard key={`${mode}-${status.configuration_revision}`} mode={mode}
                  onSaved={(result) => { setWizard((open) => open === mode ? undefined : open); setMessage(result); }} />
              )}
            </>
          )}
        </div>
      ))}
      <Button variant="outline" aria-expanded={editing} onClick={() => setEditing((open) => !open)}>
        {editing ? 'Hide calibration file settings' : 'Configure calibration files'}
      </Button>
      {editing && (
        <form className="space-y-2" onSubmit={(event) => { event.preventDefault(); void savePaths(); }}>
          <p>Enter paths to calibration JSON files on the server. Leave either field blank to keep its saved path.</p>
          <label className="block">Camera optics profile path <input className={FIELD_STYLE} value={optics} onChange={(event) => setOptics(event.target.value)} /></label>
          <label className="block">Ball-center calibration path <input className={FIELD_STYLE} value={centers} onChange={(event) => setCenters(event.target.value)} /></label>
          <Button type="submit" disabled={busy || (!optics.trim() && !centers.trim())}>Save calibration paths</Button>
        </form>
      )}
      {message && <p role="status">{message}</p>}
    </>
  );
}

export function SetupPage() {
  const rootStore = useRootStore();
  const [status, setStatus] = useState<SetupStatus>();
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [source, setSource] = useState('');
  const [reconfigureCamera, setReconfigureCamera] = useState(false);
  const [networkUrl, setNetworkUrl] = useState('');
  const [advanced, setAdvanced] = useState(false);
  const [recalibrate, setRecalibrate] = useState(false);
  const [calibrationSaved, setCalibrationSaved] = useState(false);
  const [busy, setBusy] = useState(false);
  const [searching, setSearching] = useState(false);
  const [cameraMessage, setCameraMessage] = useState('');
  const [error, setError] = useState('');
  const sourceChange = useRef(0);
  const discoveredInitially = useRef(false);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    async function refresh() {
      const change = sourceChange.current;
      try {
        const next = await getSetupStatus(controller.signal);
        if (!controller.signal.aborted && change === sourceChange.current) setStatus(next);
      } catch (failure) {
        if (!controller.signal.aborted && change === sourceChange.current) setError(messageFor(failure));
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(refresh, STATUS_INTERVAL_MS);
      }
    }
    void refresh();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  const configuredSource = status?.configured_source;
  const showCameraChooser = !!status && status.state !== 'disabled' && (!configuredSource || reconfigureCamera);
  const cameraName = configuredSource?.startsWith('gstreamer-device-v1:')
    ? cameras.find((camera) => camera.capture_selector === configuredSource)?.display_name || 'Local camera'
    : 'Network camera (MJPEG)';
  useEffect(() => {
    if (showCameraChooser && !discoveredInitially.current) {
      discoveredInitially.current = true;
      void discover();
    }
  }, [showCameraChooser]);

  async function discover() {
    setSearching(true);
    setCameraMessage('');
    try {
      const response = await client.get<{ devices: Array<Omit<Camera, 'capture_selector'> & { capture_selector: string | null }> }>('/api/v1/setup/cameras');
      const devices = response.data.devices.filter((camera): camera is Camera => !!camera.capture_selector);
      const counts = new Map<string, number>();
      for (const device of devices) counts.set(device.capture_selector, (counts.get(device.capture_selector) ?? 0) + 1);
      const unique = devices.filter((camera) => counts.get(camera.capture_selector) === 1);
      setCameras(unique);
      if (!unique.length) setCameraMessage('No local cameras with a stable identity were found. Reconnect your camera and search again, or choose a network camera.');
    } catch (failure) {
      setCameraMessage(messageFor(failure));
    } finally {
      setSearching(false);
    }
  }

  async function selectCamera(nextSource: string) {
    if (status?.configuration_revision == null || !nextSource) return;
    sourceChange.current += 1;
    setBusy(true);
    setError('');
    try {
      const response = await client.getInstance().put<SetupStatus>('/api/v1/setup/source', {
        source: nextSource,
        expected_revision: status.configuration_revision,
      });
      sourceChange.current += 1;
      setStatus(response.data);
      setReconfigureCamera(false);
      setSource('');
      setNetworkUrl('');
      setRecalibrate(false);
      setCalibrationSaved(false);
      rootStore.notifyVideoConfigurationApplied();
    } catch (failure) {
      setError(messageFor(failure));
    } finally {
      setBusy(false);
    }
  }

  const nextSource = source === 'network' ? networkUrl.trim() : source;
  const calibrationSourceChange = sourceChange.current;
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Camera and table setup</h1>
      <section className="space-y-3 rounded-lg border border-border bg-card p-4" aria-labelledby="camera-setup-title">
        <h2 id="camera-setup-title" className="text-lg font-semibold">{configuredSource ? 'Camera' : 'Choose a camera'}</h2>
        {!status && <p role="status">Checking camera setup…</p>}
        {status?.state === 'disabled' && <p>Live capture is disabled in this gateway.</p>}
        {status?.state === 'starting' && <p role="status">Starting camera…</p>}
        {status?.state === 'running' && <p role="status">Camera is running.</p>}
        {status?.error && <p role="alert">{status.error}</p>}
        {error && <p role="alert">{error}</p>}
        {configuredSource && <p>Saved source: {cameraName}</p>}
        {configuredSource && status?.state !== 'disabled' && (
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" aria-expanded={reconfigureCamera} disabled={busy} onClick={() => setReconfigureCamera((open) => !open)}>
              {reconfigureCamera ? 'Cancel reconfiguration' : 'Reconfigure camera'}
            </Button>
            {status?.state === 'error' && (
              <Button variant="outline" disabled={busy} onClick={() => void selectCamera(configuredSource)}>Retry saved camera</Button>
            )}
          </div>
        )}
        {showCameraChooser && <>
          <Button variant="outline" disabled={searching} onClick={() => void discover()}>
            {searching ? 'Searching…' : 'Refresh local cameras'}
          </Button>
          {cameraMessage && <p role="status">{cameraMessage}</p>}
          <form className="space-y-2" onSubmit={(event) => { event.preventDefault(); void selectCamera(nextSource); }}>
            <label>Camera{' '}
              <select className={FIELD_STYLE} value={source} onChange={(event) => setSource(event.target.value)}>
                <option value="">Choose a source</option>
                {cameras.map((camera) => <option value={camera.capture_selector} key={camera.capture_selector}>{camera.display_name || 'Local camera'}</option>)}
                <option value="network">Network camera (MJPEG)</option>
              </select>
            </label>
            {source === 'network' && <label className="block">Camera URL <input className={FIELD_STYLE} type="url" value={networkUrl} onChange={(event) => setNetworkUrl(event.target.value)} required /></label>}
            <div className="flex gap-2">
              <Button type="submit" disabled={!status || status.state === 'disabled' || busy || !nextSource}>Use this camera</Button>
            </div>
          </form>
        </>}
      </section>
      <section className="space-y-3 rounded-lg border border-border bg-card p-4" aria-label="Calibration checklist">
        <h2 className="text-lg font-semibold">Calibration checklist</h2>
        <div className="space-y-1 rounded-md border border-border p-3">
          <h3 className="font-semibold">Table geometry</h3>
          <p className={status?.calibrated ? 'text-emerald-400' : 'text-amber-400'}>
            {status?.calibrated ? 'Active' : status?.state === 'running' ? 'Missing or inactive' : 'Waiting for camera'}
          </p>
          {status?.state === 'running' && status.calibrated && !recalibrate && (
            <Button variant="outline" onClick={() => { setCalibrationSaved(false); setRecalibrate(true); }}>Recalibrate table</Button>
          )}
        </div>
        {status?.state === 'running' && (!status.calibrated || recalibrate) && (
          <TableCalibration key={status.configuration_revision} onSaved={() => {
            if (calibrationSourceChange !== sourceChange.current) return;
            sourceChange.current += 1;
            setCalibrationSaved(true);
            setRecalibrate(false);
          }} />
        )}
        {calibrationSaved && !status?.calibrated && status?.state !== 'error' && (
          <p role="status">Calibration saved. Waiting for the camera to apply it.</p>
        )}
        <CalibrationAssets status={status} />
      </section>
      <Button variant="outline" aria-expanded={advanced} onClick={() => setAdvanced((open) => !open)}>
        {advanced ? 'Hide advanced configuration' : 'Show advanced configuration'}
      </Button>
      {advanced && <Suspense fallback={<p role="status">Loading advanced configuration…</p>}><AdvancedConfiguration /></Suspense>}
    </div>
  );
}
