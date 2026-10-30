import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../../api/client';
import { LiveView } from '../../components/video/LiveView';

vi.mock('../../stores/context', () => ({ useRootStore: vi.fn() }));
import { useRootStore } from '../../stores/context';

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.mocked(useRootStore).mockReset();
});

describe('live view freshness', () => {
  it('displays server-retained balls beyond the local cache horizon without making them coaching-eligible', () => {
    vi.mocked(useRootStore).mockReturnValue({
      system: { status: { isConnected: true } },
      videoStore: {
        latestGameState: {
          balls: [{
            id: '3',
            number: 3,
            position: { x: 400, y: 300, scale: [1, 1] },
            radius: 20,
            confidence: 0.9,
            is_observed: false,
            position_kind: 'last_observed',
          }],
          observation_status: {
            observation_identity: {
              engine_instance_id: 'engine-retained',
              session_id: 'camera-session-retained',
              frame_id: 'frame-gap',
            },
            engine_instance_id: 'engine-retained',
            source_session_id: 'camera-session-retained',
            calibration_revision: 'calibration-retained',
            frame_publication_age_seconds: 2,
            ball_positions_status: 'cached',
            position_retention: {
              retained: true,
              retained_age_ms: 1800,
              unavailable_age_ms: 100,
              grace_ms: 500,
              current_status: 'unavailable',
              original_status: 'observed',
              source_identity: {
                engine_instance_id: 'engine-retained',
                source_session_id: 'camera-session-retained',
                calibration_revision: 'calibration-retained',
              },
            },
            perception: {
              event_sequence: 4,
              image_width_px: 1280,
              image_height_px: 720,
              phase: 'aiming',
              layout_revision: 2,
              layout_trusted: true,
              guidance_valid: true,
              trust_reason: 'trusted',
              last_ball_observation_age_ms: 1800,
              latest_motion_check_age_ms: 20,
              cue_evidence_age_ms: 30,
              motion_evidence_max_age_ms: 200,
              cue_evidence_max_age_ms: 250,
              layout_verification_max_age_ms: 1500,
              cue: {
                shaft_start_px: [300, 500],
                shaft_end_px: [800, 450],
                axis_unit: [0.995, -0.099],
                confidence: 0.88,
                observation_age_ms: 30,
                signed_direction: null,
              },
            },
          },
        },
      },
    } as never);

    render(<LiveView autoConnect={false} />);

    expect(screen.getByRole('status').textContent).toBe('1 ball last known');
    expect(screen.getByRole('gridcell', { name: /Last known layout · 1800 ms ago/ })).toBeTruthy();
    expect(screen.getByTestId('live-detection-overlay')
      .getAttribute('data-cue-overlay-mode')).toBe('detection-only');
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-freshness')).toBe('stale');
  });

  it('shows retained verified layouts and uses cue evidence only while its own validity holds', () => {
    vi.mocked(useRootStore).mockReturnValue({
      system: { status: { isConnected: true } },
      videoStore: {
        latestGameState: {
          balls: [{
            id: 'cue-ball',
            position: { x: 400, y: 300, scale: [1, 1] },
            radius: 20,
            confidence: 0.9,
            is_observed: false,
          }],
          observation_status: {
            frame_publication_age_seconds: 0.8,
            perception: {
              event_sequence: 7,
              image_width_px: 1920,
              image_height_px: 1080,
              phase: 'aiming',
              layout_revision: 3,
              layout_trusted: true,
              guidance_valid: true,
              trust_reason: 'trusted',
              last_ball_observation_age_ms: 800,
              latest_motion_check_age_ms: 20,
              cue_evidence_age_ms: 40,
              motion_evidence_max_age_ms: 200,
              cue_evidence_max_age_ms: 250,
              layout_verification_max_age_ms: 1500,
              cue: {
                shaft_start_px: [300, 500],
                shaft_end_px: [800, 450],
                axis_unit: [0.995, -0.099],
                confidence: 0.88,
                observation_age_ms: 40,
                signed_direction: null,
              },
            },
          },
        },
      },
    } as never);

    render(<LiveView autoConnect={false} />);
    const metadata = document.querySelector('.live-meta')?.getAttribute('aria-label') ?? '';
    expect(metadata).toContain('aiming');
    expect(metadata).toContain('verified layout retained');
    expect(metadata).toContain('State age: 800 ms · STALE');
    expect(metadata).toContain('0 balls freshly observed, 1 retained');
    expect(metadata).toContain('Cue stick: visible');
    expect(screen.getByRole('status').textContent).toBe('1 ball in verified layout');
    expect(screen.getByTitle(/Verified layout retained/)).toBeTruthy();
  });

  it('draws fresh cue geometry and marks retained untrusted ball positions as stale', () => {
    const context = {
      clearRect: vi.fn(),
      beginPath: vi.fn(),
      ellipse: vi.fn(),
      fill: vi.fn(),
      stroke: vi.fn(),
      setLineDash: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      arc: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      measureText: vi.fn(() => ({ width: 100 })),
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
      fillText: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);
    vi.spyOn(HTMLImageElement.prototype, 'getBoundingClientRect').mockReturnValue(
      new DOMRect(0, 0, 640, 360),
    );
    vi.mocked(useRootStore).mockReturnValue({
      system: { status: { isConnected: true } },
      videoStore: {
        latestGameState: {
          balls: [{
            id: 'cue-ball',
            position: { x: 400, y: 300, scale: [1, 1] },
            radius: 20,
            confidence: 0.9,
            is_observed: true,
          }],
          observation_status: {
            frame_publication_age_seconds: 1.2,
            perception: {
              event_sequence: 8,
              image_width_px: 1280,
              image_height_px: 720,
              phase: 'pre_shot',
              layout_revision: 4,
              layout_trusted: false,
              guidance_valid: false,
              trust_reason: 'motion_suspected',
              last_ball_observation_age_ms: 900,
              latest_motion_check_age_ms: 30,
              cue_evidence_age_ms: 30,
              motion_evidence_max_age_ms: 200,
              cue_evidence_max_age_ms: 250,
              layout_verification_max_age_ms: 1500,
              cue: {
                shaft_start_px: [300, 500],
                shaft_end_px: [800, 450],
                axis_unit: [0.995, -0.099],
                confidence: 0.88,
                observation_age_ms: 30,
                signed_direction: null,
              },
            },
          },
        },
      },
    } as never);

    render(<LiveView autoConnect={false} />);
    const image = screen.getByRole('img', { name: 'Video stream' });
    Object.defineProperty(image, 'naturalWidth', { configurable: true, value: 1280 });
    Object.defineProperty(image, 'naturalHeight', { configurable: true, value: 720 });
    fireEvent.load(image);

    const overlay = screen.getByTestId('live-detection-overlay');
    expect(overlay.getAttribute('data-cue-overlay')).toBe('visible');
    expect(overlay.getAttribute('data-cue-overlay-mode')).toBe('detection-only');
    expect(overlay.style.opacity).toBe('1');
    expect(context.ellipse).toHaveBeenCalledTimes(1);
    expect(context.moveTo).toHaveBeenCalledTimes(2);
    expect(context.lineTo).toHaveBeenCalledTimes(2);
    expect(screen.getByLabelText(/Cue detected/)).toBeTruthy();
  });

  it('never marks legacy cue geometry as coaching-eligible without perception evidence', () => {
    vi.mocked(useRootStore).mockReturnValue({
      system: { status: { isConnected: true } },
      videoStore: {
        latestGameState: {
          balls: [],
          cue: {
            tip_position: { x: 700, y: 400, scale: [1, 1] },
            butt_position: { x: 200, y: 500, scale: [1, 1] },
            detected: true,
            is_visible: true,
            confidence: 0.8,
          },
          observation_status: { frame_publication_age_seconds: 0.1 },
        },
      },
    } as never);

    render(<LiveView autoConnect={false} />);

    expect(screen.getByLabelText('Cue detected at 80 percent confidence')).toBeTruthy();
    expect(screen.getByTestId('live-detection-overlay')
      .getAttribute('data-cue-overlay-mode')).toBe('detection-only');
  });

  it('keeps the last measured layout visible through brief phase gaps, then expires it', () => {
    vi.useFakeTimers();
    const context = {
      clearRect: vi.fn(),
      beginPath: vi.fn(),
      ellipse: vi.fn(),
      fill: vi.fn(),
      stroke: vi.fn(),
      setLineDash: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      arc: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      measureText: vi.fn(() => ({ width: 100 })),
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
      fillText: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);
    vi.spyOn(HTMLImageElement.prototype, 'getBoundingClientRect').mockReturnValue(
      new DOMRect(0, 0, 640, 360),
    );

    const measuredBall = {
      id: '7',
      number: 7,
      position: { x: 400, y: 300, scale: [1, 1] as [number, number] },
      radius: 20,
      confidence: 0.9,
      is_observed: true,
    };
    const measuredState = {
      frame_number: 20,
      balls: [measuredBall],
      observation_status: {
        observation_identity: {
          engine_instance_id: 'engine-a',
          session_id: 'camera-session-a',
          frame_id: 'frame-20',
          snapshot_revision: 20,
        },
        calibration_revision: 'calibration-a',
        frame_publication_age_seconds: 0.1,
        ball_positions_status: 'observed',
        perception: {
          event_sequence: 20,
          image_width_px: 1280,
          image_height_px: 720,
          phase: 'pre_shot',
          layout_revision: 20,
          layout_trusted: true,
          guidance_valid: false,
          trust_reason: 'trusted',
          last_ball_observation_age_ms: 100,
          latest_motion_check_age_ms: 20,
          cue_evidence_age_ms: null,
          motion_evidence_max_age_ms: 200,
          cue_evidence_max_age_ms: 250,
          layout_verification_max_age_ms: 1500,
          cue: null,
        },
      },
    };
    const videoStore = { latestGameState: measuredState };
    vi.mocked(useRootStore).mockReturnValue({
      system: { status: { isConnected: true } },
      videoStore,
    } as never);

    const view = render(<LiveView autoConnect={false} />);
    const image = screen.getByRole('img', { name: 'Video stream' });
    Object.defineProperty(image, 'naturalWidth', { configurable: true, value: 1280 });
    Object.defineProperty(image, 'naturalHeight', { configurable: true, value: 720 });
    fireEvent.load(image);
    expect(context.ellipse).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Ball 7')).toBeTruthy();

    videoStore.latestGameState = {
      frame_number: 21,
      balls: [],
      observation_status: {
        observation_identity: {
          engine_instance_id: 'engine-a',
          session_id: 'camera-session-a',
          frame_id: 'frame-21',
          snapshot_revision: 21,
        },
        calibration_revision: 'calibration-a',
        frame_publication_age_seconds: 0.2,
        freshness_status: 'source_silent',
        ball_positions_status: 'unavailable',
        perception: {
          event_sequence: 21,
          image_width_px: 1280,
          image_height_px: 720,
          phase: 'recovering',
          layout_revision: 20,
          layout_trusted: false,
          guidance_valid: false,
          trust_reason: 'motion_suspected',
          last_ball_observation_age_ms: 300,
          latest_motion_check_age_ms: 30,
          cue_evidence_age_ms: null,
          motion_evidence_max_age_ms: 200,
          cue_evidence_max_age_ms: 250,
          layout_verification_max_age_ms: 1500,
          cue: null,
        },
      },
    };
    view.rerender(<LiveView autoConnect={false} className="phase-change" />);

    expect(screen.getByRole('status').textContent).toBe('1 ball last known');
    expect(screen.getByLabelText(/1 ball; 0 balls freshly observed, 1 last known/)).toBeTruthy();
    expect(screen.getByLabelText('Cue not detected')).toBeTruthy();
    expect(screen.getByTitle(/Last known layout/)).toBeTruthy();
    expect(screen.getByRole('group', { name: /0 balls freshly observed, 1 last known/ })).toBeTruthy();
    expect(context.ellipse).toHaveBeenCalledTimes(2);

    act(() => { vi.advanceTimersByTime(1600); });
    expect(screen.queryByTitle(/Last known layout/)).toBeNull();
    expect(screen.getByRole('status').textContent).toBe('0 balls detected');
    expect(context.ellipse).toHaveBeenCalledTimes(2);
  });

  it('keeps a transient observed-empty payload, then clears only a verified empty layout', () => {
    const measuredBall = {
      id: '7',
      number: 7,
      position: { x: 400, y: 300, scale: [1, 1] as [number, number] },
      radius: 20,
      confidence: 0.9,
      is_observed: true,
    };
    const makeState = (frame: number, balls: typeof measuredBall[], emptyLayoutVerified?: boolean) => ({
      frame_number: frame,
      balls,
      observation_status: {
        observation_identity: {
          engine_instance_id: 'engine-empty-test',
          session_id: 'camera-session-empty-test',
          frame_id: `frame-${frame}`,
          snapshot_revision: frame,
        },
        calibration_revision: 'calibration-empty-test',
        frame_publication_age_seconds: 0.1,
        ball_positions_status: 'observed',
        perception: {
          event_sequence: frame,
          image_width_px: 1280,
          image_height_px: 720,
          phase: emptyLayoutVerified ? 'pre_shot' : balls.length === 0 ? 'recovering' : 'pre_shot',
          layout_revision: frame,
          layout_trusted: emptyLayoutVerified === true || balls.length > 0,
          guidance_valid: false,
          trust_reason: emptyLayoutVerified ? 'trusted' : balls.length === 0 ? 'empty_unverified' : 'trusted',
          ...(emptyLayoutVerified === undefined ? {} : { empty_layout_verified: emptyLayoutVerified }),
          last_ball_observation_age_ms: 100,
          latest_motion_check_age_ms: 10,
          cue_evidence_age_ms: null,
          motion_evidence_max_age_ms: 200,
          cue_evidence_max_age_ms: 250,
          layout_verification_max_age_ms: 1500,
          cue: null,
        },
      },
    });
    const videoStore = { latestGameState: makeState(10, [measuredBall]) };
    vi.mocked(useRootStore).mockReturnValue({
      system: { status: { isConnected: true } },
      videoStore,
    } as never);

    const view = render(<LiveView autoConnect={false} />);
    expect(screen.getByText('Ball 7')).toBeTruthy();

    videoStore.latestGameState = makeState(11, []);
    view.rerender(<LiveView autoConnect={false} className="transient-empty" />);
    expect(screen.getByRole('status').textContent).toBe('1 ball last known');
    expect(screen.getByTitle(/Last known layout/)).toBeTruthy();

    videoStore.latestGameState = makeState(12, [], true);
    view.rerender(<LiveView autoConnect={false} className="verified-empty" />);
    expect(screen.getByRole('status').textContent).toBe('0 balls in verified layout');
    expect(screen.queryByTitle(/Last known layout/)).toBeNull();
  });

  it('ages a source observation without new messages and hides stale locations', async () => {
    vi.useFakeTimers();
    vi.spyOn(apiClient, 'getConfiguration').mockResolvedValue({
      success: true,
      data: {
        values: {
          api: {
            stream: { viewer_url: 'http://camera.test/stream' },
          },
          perception: { freshness: { maximum_observation_age_seconds: 0.4 } },
        },
      },
      timestamp: new Date().toISOString(),
    } as never);
    vi.mocked(useRootStore).mockReturnValue({
      system: { status: { isConnected: true } },
      videoStore: {
        latestGameState: {
          frame_number: 12,
          timestamp: '2026-10-21T12:00:00Z',
          game_type: 'practice',
          balls: [{
            id: '6',
            number: 6,
            position: { x: 100, y: 100, scale: [1, 1] },
            radius: 10,
            confidence: 0.95,
            is_observed: true,
          }],
          cue: null,
          observation_status: {
            source_frame_number: 12,
            frame_publication_age_seconds: 0.1,
            ball_positions_status: 'observed',
          },
        },
      },
    } as never);

    render(<LiveView />);
    await act(async () => { await Promise.resolve(); });
    expect(screen.getByRole('group', { name: /State age: 100 ms/ }).getAttribute('aria-label'))
      .not.toContain('STALE');
    expect(screen.getByText('Ball 6')).toBeTruthy();
    expect(screen.getByRole('status').textContent).toBe('1 ball detected');
    expect(screen.getByRole('grid', { name: 'Detected ball slots' })
      .querySelectorAll('[role="gridcell"]')).toHaveLength(16);
    expect(screen.getByTestId('live-video-pane').style.aspectRatio).toBe('16 / 9');
    expect(screen.getByRole('img', { name: 'Video stream' }).getAttribute('src'))
      .toBe('http://camera.test/stream');

    // A loaded source frame determines the pane ratio. The overlay shares the
    // pane's full bounds, so markers stay aligned without an offset transform.
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null as never);
    const image = screen.getByRole('img', { name: 'Video stream' });
    Object.defineProperty(image, 'naturalWidth', { configurable: true, value: 640 });
    Object.defineProperty(image, 'naturalHeight', { configurable: true, value: 480 });
    fireEvent.load(image);
    expect(screen.getByTestId('live-video-pane').style.aspectRatio).toBe('640 / 480');
    expect(document.querySelector('canvas')?.style.width).toBe('100%');
    expect(document.querySelector('canvas')?.style.height).toBe('100%');

    await act(async () => { vi.advanceTimersByTime(300); });

    expect(screen.getByRole('group', { name: /State age: 400 ms · STALE/ })).toBeTruthy();
    expect(screen.queryByText('Ball 6')).toBeNull();
    expect(screen.getByRole('status').textContent).toBe('0 balls detected');
    expect(screen.getByText('Last known state (stale)')).toBeTruthy();
    expect(document.querySelector('canvas')?.getAttribute('data-freshness')).toBe('stale');
  });

  it('uses the configured observation window instead of the UI default', async () => {
    vi.useFakeTimers();
    vi.spyOn(apiClient, 'getConfiguration').mockResolvedValue({
      success: true,
      data: {
        values: {
          api: { stream: { viewer_url: 'http://camera.test/stream' } },
          perception: { freshness: { maximum_observation_age_seconds: 9.0 } },
        },
      },
      timestamp: new Date().toISOString(),
    } as never);
    vi.mocked(useRootStore).mockReturnValue({
      system: { status: { isConnected: true } },
      videoStore: {
        latestGameState: {
          frame_number: 13,
          timestamp: '2026-10-21T12:00:00Z',
          game_type: 'practice',
          balls: [{
            id: '6',
            number: 6,
            position: { x: 100, y: 100, scale: [1, 1] },
            radius: 10,
            confidence: 0.95,
            is_observed: true,
          }],
          cue: null,
          observation_status: {
            source_frame_number: 13,
            frame_publication_age_seconds: 0.1,
            ball_positions_status: 'observed',
          },
        },
      },
    } as never);

    render(<LiveView />);
    await act(async () => { await Promise.resolve(); });
    await act(async () => { vi.advanceTimersByTime(1_000); });

    expect(screen.getByRole('group', { name: /State age: 1100 ms/ }).getAttribute('aria-label'))
      .not.toContain('STALE');
    expect(screen.getByText('Ball 6')).toBeTruthy();
    expect(screen.getByRole('status').textContent).toBe('1 ball detected');
  });

  it('shows exact-frame detections when table visibility is unknown but withholds guidance', async () => {
    const context = {
      clearRect: vi.fn(), beginPath: vi.fn(), ellipse: vi.fn(), fill: vi.fn(), stroke: vi.fn(),
      setLineDash: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(), arc: vi.fn(), save: vi.fn(),
      restore: vi.fn(), measureText: vi.fn(() => ({ width: 80 })), fillRect: vi.fn(),
      strokeRect: vi.fn(), fillText: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);
    vi.spyOn(HTMLImageElement.prototype, 'getBoundingClientRect').mockReturnValue(
      new DOMRect(0, 0, 640, 360),
    );
    vi.spyOn(apiClient, 'getConfiguration').mockResolvedValue({
      success: true,
      data: { values: { api: { stream: {
        viewer_url: 'http://camera.test/identified-stream',
        frame_pairing: {
          maximum_frame_bytes: 1024,
          maximum_part_header_bytes: 2048,
          buffered_frame_count: 4,
        },
      } } } },
      timestamp: new Date().toISOString(),
    } as never);
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:identified-frame');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

    const header = new TextEncoder().encode(
      '--table\r\nContent-Type: image/jpeg\r\nContent-Length: 5\r\n'
      + 'X-Frame-Id: frame-41\r\nX-Engine-Instance-Id: engine-a\r\n'
      + 'X-Source-Session-Id: session-a\r\nX-Configuration-Revision: 4\r\n'
      + 'X-Calibration-Revision: 2\r\nX-Rectification-Revision: 3\r\n\r\n',
    );
    const jpeg = new Uint8Array([0xff, 0xd8, 65, 0xff, 0xd9]);
    const suffix = new TextEncoder().encode('\r\n--table--\r\n');
    const bytes = new Uint8Array(header.length + jpeg.length + suffix.length);
    bytes.set(header);
    bytes.set(jpeg, header.length);
    bytes.set(suffix, header.length + jpeg.length);
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(bytes, {
      status: 200,
      headers: { 'content-type': 'multipart/x-mixed-replace; boundary=table' },
    }));

    vi.mocked(useRootStore).mockReturnValue({
      system: { status: { isConnected: true } },
      videoStore: {
        latestGameState: {
          balls: [{
            id: 'cue-ball',
            position: { x: 640, y: 360, scale: [1, 1] },
            radius: 24,
            radius_extent: { x: 12, y: 24, scale: [1, 1] },
            confidence: 0.95,
            is_observed: true,
          }],
          observation_status: {
            observation_identity: {
              engine_instance_id: 'engine-a', session_id: 'session-a', frame_id: 'frame-41',
            },
            engine_instance_id: 'engine-a',
            source_session_id: 'session-a',
            source_frame_id: 'frame-41',
            configuration_revision: 4,
            calibration_revision: 2,
            rectification_revision: 3,
            frame_publication_age_seconds: 0.01,
            ball_positions_status: 'observed',
            perception: {
              event_sequence: 5,
              image_width_px: 1280,
              image_height_px: 720,
              phase: 'aiming',
              layout_revision: 2,
              layout_trusted: true,
              guidance_valid: true,
              trust_reason: 'trusted',
              last_ball_observation_age_ms: 10,
              latest_motion_check_age_ms: 10,
              cue_evidence_age_ms: 10,
              motion_evidence_max_age_ms: 200,
              cue_evidence_max_age_ms: 250,
              layout_verification_max_age_ms: 1500,
              decision_frame_id: 'frame-42',
              decision_calibration_revision: 2,
              decision_rectification_revision: 3,
              table_visibility: {
                status: 'unknown', reason: 'insufficient_evidence',
                observation_frame_id: 'frame-42', observation_age_ms: 10,
              },
              cue: {
                shaft_start_px: [300, 500], shaft_end_px: [800, 450],
                axis_unit: [0.995, -0.099], confidence: 0.9,
                observation_age_ms: 10, signed_direction: 'toward_end',
                source_frame_id: 'frame-41',
              },
            },
          },
        },
      },
    } as never);

    render(<LiveView />);
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    const image = screen.getByRole('img', { name: 'Video stream' });
    expect(image.getAttribute('src')).toBe('blob:identified-frame');
    Object.defineProperty(image, 'naturalWidth', { configurable: true, value: 1280 });
    Object.defineProperty(image, 'naturalHeight', { configurable: true, value: 720 });
    fireEvent.load(image);

    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-frame-pairing'))
      .toBe('exact');
    expect(screen.getByTestId('live-detection-overlay').getAttribute('data-cue-overlay-mode'))
      .toBe('detection-only');
    expect(context.ellipse).toHaveBeenCalled();
    expect(context.ellipse).toHaveBeenCalledWith(
      expect.any(Number),
      expect.any(Number),
      4,
      4,
      0,
      0,
      2 * Math.PI,
    );
    expect(context.lineTo).toHaveBeenCalled();
  });
});
