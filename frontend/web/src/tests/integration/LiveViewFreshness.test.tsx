import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LiveView } from '../../components/video/LiveView';
import type { GameStateData, TableProjection } from '../../types/api';
import type { DisplayMjpegFrame } from '../../components/video/use-frame-aware-mjpeg';

vi.mock('../../stores/context', () => ({ useRootStore: vi.fn() }));
import { useRootStore } from '../../stores/context';

const projection: TableProjection = {
  direction: 'source_image_px_to_table_mm', image_space: 'raw',
  source_width_px: 1000, source_height_px: 500, camera_profile_id: 'camera-1',
  calibration_revision: 1, rectification_revision: 1,
  table_width_mm: 1000, table_height_mm: 500,
  unit: 'millimeters', origin: 'playing_surface_center',
  x_axis: 'corner_0_to_1', y_axis: 'corner_0_to_3',
  pixel_to_table_mm: [[1, 0, -500], [0, 1, -250], [0, 0, 1]],
};

function makeState(options: { trusted?: boolean; balls?: GameStateData['balls']; cue?: boolean; age?: number; calibration?: number } = {}): GameStateData {
  const balls = options.balls ?? [
    { id: 'cue-id', is_cue_ball: true, position: { x: 500, y: 400, scale: [3.84, 4.32] }, table_position: { x: 0, y: 150 }, table_radius_mm: 10, radius: 10, confidence: 0.9 },
    { id: 'target-id', position: { x: 500, y: 250, scale: [3.84, 4.32] }, table_position: { x: 0, y: 0 }, table_radius_mm: 10, radius: 10, confidence: 0.9 },
  ];
  return {
    balls,
    observation_status: {
      engine_instance_id: 'engine-1', source_session_id: 'session-1', configuration_revision: 1,
      calibration_revision: options.calibration ?? 1,
      rectification_revision: 1,
      camera_profile_id: 'camera-1',
      perception: {
        event_sequence: 4,
        phase: 'aiming', layout_revision: 3,
        layout_trusted: options.trusted ?? true,
        empty_layout_verified: balls.length === 0,
        guidance_valid: true, trust_reason: options.trusted === false ? 'motion_suspected' : 'trusted',
        last_ball_observation_age_ms: options.age ?? 8_000,
        latest_motion_check_age_ms: 10,
        cue_evidence_age_ms: 5,
        motion_evidence_max_age_ms: 250,
        cue_evidence_max_age_ms: 150,
        layout_verification_max_age_ms: 1,
        image_width_px: 1000, image_height_px: 500,
        table_projection: projection,
        cue: options.cue ? {
          shaft_start_px: [420, 420], shaft_end_px: [480, 405], axis_unit: [0.98, -0.19],
          confidence: 0.8, observation_age_ms: 5, signed_direction: null,
        } : null,
      },
    },
  } as unknown as GameStateData;
}

function installStore(state: GameStateData) {
  const store = {
    system: { status: { isConnected: true } },
    videoStore: {
      latestGameState: state, latestFrame: null as DisplayMjpegFrame | null,
      delivery: { mode: null as string | null, sourceGeneration: 0, sourceConfigurationRevision: undefined as number | undefined },
    },
    nativeVideoElement: null,
    setVideoDimensions: vi.fn(),
  };
  vi.mocked(useRootStore).mockReturnValue(store as never);
  return store;
}

const canvasContext = {
  clearRect: vi.fn(), beginPath: vi.fn(), ellipse: vi.fn(), fill: vi.fn(), stroke: vi.fn(),
  setLineDash: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(), arc: vi.fn(), save: vi.fn(), restore: vi.fn(),
  translate: vi.fn(), fillText: vi.fn(), strokeText: vi.fn(),
} as unknown as CanvasRenderingContext2D;

function loadVideoImage() {
  const image = screen.getByRole('img', { name: 'Video stream' });
  Object.defineProperty(image, 'naturalWidth', { configurable: true, value: 1000 });
  Object.defineProperty(image, 'naturalHeight', { configurable: true, value: 500 });
  fireEvent.load(image);
}

beforeEach(() => { vi.clearAllMocks(); });

afterEach(() => {
  cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); vi.mocked(useRootStore).mockReset();
});

describe('live producer layout and coach', () => {
  function installStraightShot(signed = true) {
    const state = makeState({ cue: true });
    Object.assign(state.observation_status!.perception!.cue!, {
      shaft_start_px: [500, 480], shaft_end_px: [500, 420], axis_unit: [0, -1],
      signed_direction: signed ? 'toward_end' : null,
    });
    const store = installStore(state);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext);
    const view = render(<LiveView autoConnect={false} />);
    loadVideoImage();
    fireEvent.change(screen.getByLabelText('Confirm cue ball'), { target: { value: 'cue-id' } });
    fireEvent.change(screen.getByLabelText('Select pocket'), { target: { value: 'top-middle' } });
    return { store, ...view };
  }

  it('draws current cue travel, contact, and the struck object path with an aligned assessment', () => {
    installStraightShot();
    expect(screen.getByTestId('live-alignment').getAttribute('data-alignment')).toBe('aligned');
    expect(screen.getByTestId('live-alignment').textContent).toContain('within 2°');
    expect(screen.getByTestId('live-contact-summary').textContent).toContain('hits ball target-id first');
    expect(canvasContext.lineTo).toHaveBeenCalledWith(500, 270);
    expect(canvasContext.lineTo).toHaveBeenCalledWith(500, 10);
    expect(canvasContext.arc).toHaveBeenCalledWith(500, 270, 6, 0, Math.PI * 2);
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-current-aim')).toBe('visible');
  });

  it('renders the diagnostics camera without coach controls and keeps unpaired ghost centers', () => {
    installStore(makeState({ trusted: false }));
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext);
    vi.spyOn(HTMLImageElement.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 1000, 500));

    render(<LiveView autoConnect={false} presentation="diagnostics" />);
    loadVideoImage();

    expect(screen.getByRole('region', { name: 'Diagnostic camera' })).toBeTruthy();
    expect(screen.getByTestId('live-video-pane')).toBeTruthy();
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-frame-pairing')).toBe('independent');
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-freshness')).toBe('stale');
    expect(screen.queryByRole('region', { name: 'Live shot coaching' })).toBeNull();
    expect(screen.queryByLabelText('Confirm cue ball')).toBeNull();
    expect(screen.queryByRole('list', { name: 'Latest producer ball layout' })).toBeNull();
    expect(canvasContext.ellipse).toHaveBeenCalledTimes(2);
    expect(canvasContext.setLineDash).toHaveBeenCalledWith([6, 5]);
  });

  it('supports explicit manual direction confirmation and revokes it for a new layout', () => {
    const { store, rerender } = installStraightShot(false);
    expect(screen.getByTestId('live-alignment').getAttribute('data-alignment')).toBe('direction_unknown');
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-current-aim')).toBe('hidden');
    fireEvent.click(screen.getByRole('checkbox', { name: /I confirm the cue tip/ }));
    expect(screen.getByTestId('live-alignment').getAttribute('data-alignment')).toBe('aligned');
    expect(screen.getByTestId('live-alignment').textContent).toContain('your tip confirmation');
    store.videoStore.latestGameState.observation_status!.perception!.layout_revision += 1;
    vi.mocked(canvasContext.lineTo).mockClear();
    rerender(<LiveView autoConnect={false} className="new-layout" />);
    expect((screen.getByRole('checkbox', { name: /I confirm the cue tip/ }) as HTMLInputElement).checked).toBe(false);
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-current-aim')).toBe('hidden');
    expect(canvasContext.lineTo).not.toHaveBeenCalledWith(500, 10);
  });

  it('draws the object path only to its next ball contact, even without a clear pot suggestion', () => {
    const { store, rerender } = installStraightShot();
    store.videoStore.latestGameState.balls = [...store.videoStore.latestGameState.balls, {
      id: 'next-ball', position: { x: 500, y: 100, scale: [3.84, 4.32] },
      table_position: { x: 0, y: -150 }, table_radius_mm: 10, radius: 10, confidence: 0.9,
    }];
    vi.mocked(canvasContext.lineTo).mockClear();
    rerender(<LiveView autoConnect={false} className="second-contact" />);
    fireEvent.change(screen.getByLabelText('Select target ball'), { target: { value: 'target-id' } });
    expect(screen.queryByTestId('live-suggestion')).toBeNull();
    expect(canvasContext.lineTo).toHaveBeenCalledWith(500, 120);
    expect(canvasContext.lineTo).not.toHaveBeenCalledWith(500, 10);
    expect(screen.getByTestId('live-contact-summary').textContent).toContain('contact with ball next-ball');
  });

  it('does not retain manual direction confirmation after cue expiry', () => {
    vi.useFakeTimers();
    const { store, rerender } = installStraightShot(false);
    fireEvent.click(screen.getByRole('checkbox', { name: /I confirm the cue tip/ }));
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-current-aim')).toBe('visible');
    act(() => { vi.advanceTimersByTime(200); });
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-current-aim')).toBe('hidden');
    const perception = store.videoStore.latestGameState.observation_status!.perception!;
    perception.event_sequence += 1;
    perception.cue!.source_frame_id = 'new-cue-observation';
    perception.cue_evidence_age_ms = 0;
    rerender(<LiveView autoConnect={false} className="fresh-cue" />);
    act(() => { vi.advanceTimersByTime(100); });
    expect((screen.getByRole('checkbox', { name: /I confirm the cue tip/ }) as HTMLInputElement).checked).toBe(false);
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-current-aim')).toBe('hidden');
  });

  it.each(['parallel shaft', 'away direction', 'unknown direction'])('withholds current trajectories for %s while keeping desired guidance', (failure) => {
    const { store, rerender } = installStraightShot();
    const cue = store.videoStore.latestGameState.observation_status!.perception!.cue!;
    if (failure === 'parallel shaft') {
      cue.shaft_start_px = [520, 480]; cue.shaft_end_px = [520, 420];
    } else cue.signed_direction = failure === 'away direction' ? 'toward_start' : null;
    vi.mocked(canvasContext.lineTo).mockClear();
    rerender(<LiveView autoConnect={false} className={failure} />);
    expect(screen.getByTestId('live-suggestion')).toBeTruthy();
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-current-aim')).toBe('hidden');
    expect(canvasContext.lineTo).not.toHaveBeenCalledWith(500, 10);
  });

  it('expires measured trajectories independently while retaining trusted desired guidance', () => {
    vi.useFakeTimers();
    installStraightShot();
    vi.mocked(canvasContext.lineTo).mockClear();
    act(() => { vi.advanceTimersByTime(200); });
    expect(screen.getByTestId('live-suggestion')).toBeTruthy();
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-freshness')).toBe('current');
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-current-aim')).toBe('hidden');
    // The 100ms tick may still have drawn a fresh path. After expiry, a rerender
    // must never restore it from the retained producer layout.
    vi.mocked(canvasContext.lineTo).mockClear();
    act(() => { vi.advanceTimersByTime(20); });
    expect(canvasContext.lineTo).not.toHaveBeenCalledWith(500, 10);
    expect(screen.queryByTestId('live-contact-summary')).toBeNull();
  });

  it('withholds measured paths during motion and after incompatible calibration', () => {
    const { store, rerender } = installStraightShot();
    store.videoStore.latestGameState.observation_status!.perception!.layout_trusted = false;
    vi.mocked(canvasContext.lineTo).mockClear();
    rerender(<LiveView autoConnect={false} className="moving" />);
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-current-aim')).toBe('hidden');
    expect(canvasContext.lineTo).not.toHaveBeenCalledWith(500, 10);
    store.videoStore.latestGameState.observation_status!.perception!.layout_trusted = true;
    store.videoStore.latestGameState.observation_status!.calibration_revision = 2;
    rerender(<LiveView autoConnect={false} className="calibration-changed" />);
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-current-aim')).toBe('hidden');
    expect(screen.getByText(/projection does not match/)).toBeTruthy();
  });

  it('draws center markers without frame IDs and does not expire a stationary layout by scan age', () => {
    installStore(makeState({ age: 8_000 }));
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext);
    vi.spyOn(HTMLImageElement.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 1000, 500));
    render(<LiveView autoConnect={false} />);
    const image = screen.getByRole('img', { name: 'Video stream' });
    Object.defineProperty(image, 'naturalWidth', { configurable: true, value: 1000 });
    Object.defineProperty(image, 'naturalHeight', { configurable: true, value: 500 });
    fireEvent.load(image);
    fireEvent.change(screen.getByLabelText('Confirm cue ball'), { target: { value: 'cue-id' } });
    expect(canvasContext.ellipse).toHaveBeenCalled();
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-frame-pairing')).toBe('independent');
    expect(screen.getByText(/Suggested direct pot/)).toBeTruthy();
    expect(screen.getByText(/scan 8s ago, quiet evidence current/)).toBeTruthy();
  });

  it('keeps producer ghosts when quiet evidence and cue expire during delivery silence', () => {
    vi.useFakeTimers();
    const store = installStore(makeState({ cue: true }));
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext);
    const { rerender } = render(<LiveView autoConnect={false} />);
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-cue-overlay')).toBe('visible');
    act(() => { vi.advanceTimersByTime(100); });
    // Re-delivering the same producer event does not restart either evidence clock.
    vi.mocked(useRootStore).mockReturnValue(store as never);
    rerender(<LiveView autoConnect={false} className="same-publication" />);
    act(() => { vi.advanceTimersByTime(200); });
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-freshness')).toBe('stale');
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-cue-overlay')).toBe('hidden');
    expect(screen.getByText(/2 balls ·/)).toBeTruthy();
    expect(screen.getByText(/Live advice paused/)).toBeTruthy();
  });

  it('clears a negative cue immediately and explains a clean empty table', () => {
    const store = installStore(makeState({ cue: true }));
    const { rerender } = render(<LiveView autoConnect={false} />);
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-cue-overlay')).toBe('visible');
    store.videoStore.latestGameState = makeState({ balls: [], cue: false });
    vi.mocked(useRootStore).mockReturnValue(store as never);
    rerender(<LiveView autoConnect={false} className="updated" />);
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-cue-overlay')).toBe('hidden');
    expect(screen.getAllByText(/Clean scan confirmed an empty table/).length).toBeGreaterThan(0);
  });

  it('supports manual cue, target, and pocket selection with the shared direct-pot geometry', () => {
    installStore(makeState({ cue: true }));
    render(<LiveView autoConnect={false} />);
    fireEvent.change(screen.getByLabelText('Confirm cue ball'), { target: { value: 'cue-id' } });
    fireEvent.change(screen.getByLabelText('Select target ball'), { target: { value: 'target-id' } });
    fireEvent.change(screen.getByLabelText('Select pocket'), { target: { value: 'top-middle' } });
    expect(screen.getByTestId('live-suggestion').textContent).toContain('target-id to Top middle');
    expect(screen.getByTestId('live-suggestion').textContent).toContain('Current cue line is visible; its direction is unknown.');
  });

  it('stops old markers and advice when known calibration context disagrees', () => {
    const state = makeState({ calibration: 2 });
    state.balls.forEach((ball) => { ball.source_observation = { calibration_revision: 1 }; });
    installStore(state);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext);
    render(<LiveView autoConnect={false} />);
    loadVideoImage();
    expect(canvasContext.ellipse).not.toHaveBeenCalled();
    expect(screen.getByText(/projection does not match/)).toBeTruthy();
    expect(screen.queryByTestId('live-suggestion')).toBeNull();
  });

  it('aligns markers and aiming paths with contained video and labels selectable IDs', () => {
    const state = makeState();
    state.balls[0].table_position = { x: -300, y: 150 };
    installStore(state);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext);
    vi.spyOn(HTMLImageElement.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 1000, 400));
    render(<LiveView autoConnect={false} />);
    loadVideoImage();
    fireEvent.change(screen.getByLabelText('Confirm cue ball'), { target: { value: 'cue-id' } });
    fireEvent.change(screen.getByLabelText('Select pocket'), { target: { value: 'top-middle' } });
    expect(canvasContext.translate).toHaveBeenCalledWith(100, 0);
    // Content is 800x400, centered within the 1000x400 image element.
    expect(canvasContext.ellipse).toHaveBeenCalledWith(160, 320, 8, 8, 0, 0, 2 * Math.PI);
    expect(canvasContext.moveTo).toHaveBeenCalledWith(160, 320);
    expect(canvasContext.fillText).toHaveBeenCalledWith('cue-id', 160, 312);
    expect(screen.getByRole('option', { name: 'Ball cue-id' })).toBeTruthy();
  });

  it.each(['singular projection', 'missing table coordinate'])('keeps compatible image markers with %s but disables advice', (failure) => {
    const state = makeState();
    if (failure === 'singular projection') {
      state.observation_status!.perception!.table_projection = { ...projection, pixel_to_table_mm: [[0, 0, 0], [0, 0, 0], [0, 0, 0]] };
    } else {
      delete state.balls[0].table_position;
    }
    installStore(state);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext);
    vi.spyOn(HTMLImageElement.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 1000, 500));
    render(<LiveView autoConnect={false} />);
    loadVideoImage();
    fireEvent.change(screen.getByLabelText('Confirm cue ball'), { target: { value: 'cue-id' } });
    expect(canvasContext.ellipse).toHaveBeenCalledWith(500, 400, 10, 10, 0, 0, 2 * Math.PI);
    expect(screen.getByText(/2 balls · 2 markers drawn/)).toBeTruthy();
    expect(screen.queryByTestId('live-suggestion')).toBeNull();
  });

  it.each(['session', 'calibration'])('withholds spatial output for a known displayed %s mismatch', (mismatch) => {
    const store = installStore(makeState({ cue: true }));
    store.videoStore.latestFrame = {
      imageUrl: 'blob:current', frameId: 'newer-than-scan', engineInstanceId: 'engine-1',
      sourceSessionId: mismatch === 'session' ? 'session-2' : 'session-1',
      configurationRevision: 1, calibrationRevision: mismatch === 'calibration' ? 2 : 1, rectificationRevision: 1,
    };
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext);
    render(<LiveView autoConnect={false} />);
    loadVideoImage();
    fireEvent.change(screen.getByLabelText('Confirm cue ball'), { target: { value: 'cue-id' } });
    expect(canvasContext.ellipse).not.toHaveBeenCalled();
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-cue-overlay')).toBe('hidden');
    expect(screen.queryByTestId('live-suggestion')).toBeNull();
  });

  it('allows newer frame numbers within the same display context', () => {
    const store = installStore(makeState({ cue: true }));
    store.videoStore.latestFrame = {
      imageUrl: 'blob:current', frameId: 'newer-than-scan', engineInstanceId: 'engine-1',
      sourceSessionId: 'session-1', configurationRevision: 1, calibrationRevision: 1, rectificationRevision: 1,
    };
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext);
    render(<LiveView autoConnect={false} />);
    loadVideoImage();
    fireEvent.change(screen.getByLabelText('Confirm cue ball'), { target: { value: 'cue-id' } });
    expect(canvasContext.ellipse).toHaveBeenCalled();
    expect(screen.getByTestId('live-suggestion')).toBeTruthy();
  });

  it('disables advice and live cue when the decision revision contradicts the scene', () => {
    const state = makeState({ cue: true });
    state.observation_status!.perception!.decision_calibration_revision = 2;
    installStore(state);
    render(<LiveView autoConnect={false} />);
    fireEvent.change(screen.getByLabelText('Confirm cue ball'), { target: { value: 'cue-id' } });
    expect(screen.queryByTestId('live-suggestion')).toBeNull();
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-cue-overlay')).toBe('hidden');
  });

  it('invalidates old spatial context when the native source generation changes', () => {
    const store = installStore(makeState({ cue: true }));
    store.videoStore.delivery.mode = 'native';
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext);
    const { rerender } = render(<LiveView autoConnect={false} />);
    loadVideoImage();
    expect(canvasContext.ellipse).toHaveBeenCalled();
    vi.mocked(canvasContext.ellipse).mockClear();
    store.videoStore.delivery.sourceGeneration += 1;
    rerender(<LiveView autoConnect={false} className="new-native-source" />);
    expect(canvasContext.ellipse).not.toHaveBeenCalled();
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-cue-overlay')).toBe('hidden');
    expect(screen.queryByTestId('live-suggestion')).toBeNull();
    store.videoStore.latestGameState = makeState({ cue: true });
    Object.assign(store.videoStore.latestGameState.observation_status!, { source_session_id: 'session-2', configuration_revision: 2 });
    store.videoStore.latestGameState.observation_status!.perception!.event_sequence += 1;
    rerender(<LiveView autoConnect={false} className="new-source-state" />);
    expect(canvasContext.ellipse).toHaveBeenCalled();
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-cue-overlay')).toBe('visible');
  });

  it('keeps source binding across route remounts using the revision that selected the feed', () => {
    const store = installStore(makeState({ cue: true }));
    store.videoStore.delivery.mode = 'native';
    store.videoStore.delivery.sourceGeneration = 2;
    store.videoStore.delivery.sourceConfigurationRevision = 2;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext);
    const { rerender } = render(<LiveView autoConnect={false} />);
    loadVideoImage();
    expect(canvasContext.ellipse).not.toHaveBeenCalled();
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-cue-overlay')).toBe('hidden');
    store.videoStore.latestGameState = makeState({ cue: true });
    store.videoStore.latestGameState.observation_status!.configuration_revision = 2;
    store.videoStore.latestGameState.observation_status!.perception!.event_sequence += 1;
    rerender(<LiveView autoConnect={false} className="current-source-configuration" />);
    expect(canvasContext.ellipse).toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText('Confirm cue ball'), { target: { value: 'cue-id' } });
    expect(screen.getByTestId('live-suggestion')).toBeTruthy();
  });

  it('allows later same-source revisions but rejects a producer older than source selection', () => {
    const state = makeState({ cue: true });
    state.observation_status!.configuration_revision = 8;
    const store = installStore(state);
    store.videoStore.delivery.sourceConfigurationRevision = 7;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext);
    const { rerender } = render(<LiveView autoConnect={false} />);
    loadVideoImage();
    fireEvent.change(screen.getByLabelText('Confirm cue ball'), { target: { value: 'cue-id' } });
    expect(canvasContext.ellipse).toHaveBeenCalled();
    expect(screen.getByTestId('live-suggestion')).toBeTruthy();
    vi.mocked(canvasContext.ellipse).mockClear();
    store.videoStore.latestGameState = makeState({ cue: true });
    store.videoStore.latestGameState.observation_status!.configuration_revision = 6;
    rerender(<LiveView autoConnect={false} className="older-than-current-source" />);
    expect(canvasContext.ellipse).not.toHaveBeenCalled();
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-cue-overlay')).toBe('hidden');
    expect(screen.queryByTestId('live-suggestion')).toBeNull();
  });

  it('withholds a stale ball whose measurement context belongs to another calibration', () => {
    const state = makeState();
    state.balls[0].source_observation = { calibration_revision: 0 };
    installStore(state);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext);
    vi.spyOn(HTMLImageElement.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 1000, 500));
    render(<LiveView autoConnect={false} />);
    const image = screen.getByRole('img', { name: 'Video stream' });
    Object.defineProperty(image, 'naturalWidth', { configurable: true, value: 1000 });
    Object.defineProperty(image, 'naturalHeight', { configurable: true, value: 500 });
    fireEvent.load(image);
    expect(canvasContext.ellipse).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/source context mismatch · marker withheld/)).toBeTruthy();
    expect(screen.queryByTestId('live-suggestion')).toBeNull();
  });

  it('shows stale ghosts during motion and recomputes after a clean reacquisition without resetting IDs', () => {
    const store = installStore(makeState());
    const { rerender } = render(<LiveView autoConnect={false} />);
    fireEvent.change(screen.getByLabelText('Confirm cue ball'), { target: { value: 'cue-id' } });
    fireEvent.change(screen.getByLabelText('Select target ball'), { target: { value: 'target-id' } });
    expect(screen.getByTestId('live-suggestion')).toBeTruthy();
    store.videoStore.latestGameState = makeState({ trusted: false });
    vi.mocked(useRootStore).mockReturnValue(store as never);
    rerender(<LiveView autoConnect={false} className="motion" />);
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-freshness')).toBe('stale');
    expect(screen.queryByTestId('live-suggestion')).toBeNull();
    store.videoStore.latestGameState = makeState({ age: 9_000 });
    vi.mocked(useRootStore).mockReturnValue(store as never);
    rerender(<LiveView autoConnect={false} className="reacquired" />);
    expect((screen.getByLabelText('Confirm cue ball') as HTMLSelectElement).value).not.toBe('');
    expect(screen.getByTestId('live-suggestion')).toBeTruthy();
  });
});
