import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createRootRoute, createRoute, createRouter, createMemoryHistory, RouterProvider } from '@tanstack/react-router';
import { PracticePage } from './PracticePage';
import { PRACTICE_TABLE_DEFAULTS } from '../video/shot-prediction';

const { practiceStore } = vi.hoisted(() => ({
  practiceStore: {
    system: { status: { isConnected: false } },
    videoStore: {
      latestGameState: null,
      delivery: {
        state: 'unconfigured',
        mode: null,
        configured: false,
        configurationSynchronized: false,
        sourceGeneration: 0,
        sourceConfigurationRevision: undefined,
        dimensions: null,
        error: 'Camera unavailable in the synthetic test fixture.',
      },
    },
    nativeVideoElement: null,
    setVideoDimensions: vi.fn(),
  },
}));

vi.mock('../../stores/context', () => ({ useRootStore: () => practiceStore }));

async function renderPractice() {
  const root = createRootRoute();
  const practice = createRoute({ getParentRoute: () => root, path: '/practice', component: PracticePage });
  const coach = createRoute({ getParentRoute: () => root, path: '/coach', component: () => <p>Live coach destination</p> });
  const router = createRouter({ rootRoute: root, routeTree: root.addChildren([practice, coach]), history: createMemoryHistory({ initialEntries: ['/practice'] }) });
  await router.load();
  render(<RouterProvider router={router} />);
}

beforeEach(() => { vi.spyOn(window, 'scrollTo').mockImplementation(() => {}); });
afterEach(() => {
  cleanup();
  document.getElementById('shared-video-receiver-parking')?.remove();
  Object.assign(practiceStore.system.status, { isConnected: false });
  Object.assign(practiceStore.videoStore, { latestGameState: null });
  Object.assign(practiceStore.videoStore.delivery, {
    state: 'unconfigured', mode: null, configured: false, configurationSynchronized: false,
    sourceGeneration: 0, sourceConfigurationRevision: undefined, dimensions: null,
    error: 'Camera unavailable in the synthetic test fixture.',
  });
  Object.assign(practiceStore, { nativeVideoElement: null });
  vi.restoreAllMocks();
});

function configureSyntheticCameraState(ballCount: number) {
  const tableWidth = PRACTICE_TABLE_DEFAULTS.width;
  const tableHeight = PRACTICE_TABLE_DEFAULTS.height;
  const imageWidth = 2336;
  const imageHeight = 1156;
  const calibrationRevision = 3;
  const rectificationRevision = 1;
  const cameraProfileId = 'synthetic-camera-fixture';
  const projection = {
    direction: 'source_image_px_to_table_mm',
    image_space: 'raw',
    source_width_px: imageWidth,
    source_height_px: imageHeight,
    camera_profile_id: cameraProfileId,
    calibration_revision: calibrationRevision,
    rectification_revision: rectificationRevision,
    table_width_mm: tableWidth,
    table_height_mm: tableHeight,
    unit: 'millimeters',
    origin: 'playing_surface_center',
    x_axis: 'corner_0_to_1',
    y_axis: 'corner_0_to_3',
    pixel_to_table_mm: [
      [tableWidth / imageWidth, 0, -tableWidth / 2],
      [0, tableHeight / imageHeight, -tableHeight / 2],
      [0, 0, 1],
    ],
  };
  const targets = [
    { id: 'cue', x: tableWidth * 0.17, y: tableHeight * 0.5 },
    { id: 'track-1', x: tableWidth * 0.53, y: tableHeight * 0.5 },
    { id: 'track-2', x: tableWidth * 0.72, y: tableHeight * 0.31 },
    { id: 'track-3', x: tableWidth * 0.78, y: tableHeight * 0.68 },
  ];
  const balls = targets.slice(0, ballCount).map((target, index) => ({
    id: `track-${index + 1}`,
    confidence: 0.99,
    position: { x: 0, y: 0, scale: [1, 1] },
    radius: PRACTICE_TABLE_DEFAULTS.ballRadius,
    position_kind: 'last_observed',
    table_position: { x: target.x - tableWidth / 2, y: target.y - tableHeight / 2 },
    table_radius_mm: PRACTICE_TABLE_DEFAULTS.ballRadius,
    source_observation: {
      engine_instance_id: 'synthetic-engine', session_id: 'synthetic-session',
      configuration_revision: 1, calibration_revision: calibrationRevision,
      rectification_revision: rectificationRevision,
    },
  }));
  const image = document.createElement('img');
  Object.defineProperty(image, 'naturalWidth', { configurable: true, value: imageWidth });
  Object.defineProperty(image, 'naturalHeight', { configurable: true, value: imageHeight });
  const parking = document.createElement('div');
  parking.id = 'shared-video-receiver-parking';
  parking.appendChild(image);
  document.body.appendChild(parking);

  Object.assign(practiceStore.system.status, { isConnected: true });
  Object.assign(practiceStore, { nativeVideoElement: image });
  Object.assign(practiceStore.videoStore.delivery, {
    state: 'loaded', mode: 'native', configured: true, configurationSynchronized: true,
    sourceGeneration: 1, sourceConfigurationRevision: 1,
    dimensions: { width: imageWidth, height: imageHeight }, error: null,
  });
  Object.assign(practiceStore.videoStore, {
    latestGameState: {
      balls,
      observation_status: {
        observation_identity: {
          engine_instance_id: 'synthetic-engine', session_id: 'synthetic-session', frame_id: 'synthetic-frame',
          configuration_revision: 1, calibration_revision: calibrationRevision,
          rectification_revision: rectificationRevision,
        },
        configuration_revision: 1,
        calibration_revision: calibrationRevision,
        table_calibration_revision: calibrationRevision,
        rectification_revision: rectificationRevision,
        camera_profile_id: cameraProfileId,
        image_space: 'raw',
        perception: {
          event_sequence: 1,
          image_width_px: imageWidth,
          image_height_px: imageHeight,
          phase: 'pre_shot',
          layout_revision: 1,
          layout_trusted: true,
          guidance_valid: true,
          trust_reason: 'quiet_layout',
          latest_motion_check_age_ms: 10,
          motion_evidence_max_age_ms: 500,
          cue_evidence_max_age_ms: 500,
          layout_verification_max_age_ms: 500,
          decision_calibration_revision: calibrationRevision,
          decision_rectification_revision: rectificationRevision,
          table_projection: projection,
        },
      },
    },
  });
}

describe('PracticePage', () => {
  it('renders an honest camera-unavailable state instead of a simulated physical table', async () => {
    await renderPractice();
    expect(screen.getByRole('link', { name: 'Live Coach' }).getAttribute('href')).toBe('/coach');
    expect(screen.getByRole('link', { name: 'Practice layout' }).getAttribute('aria-current')).toBe('page');
    expect(screen.getByTestId('practice-camera-unavailable').textContent).toContain('Live camera video is unavailable');
    expect(screen.getByTestId('practice-camera-state').textContent).toContain('Unverified');
    expect(screen.getByTestId('practice-ready-status').textContent).toContain('Not verified');
    expect(screen.getByRole('button', { name: 'Confirm cue position' }).hasAttribute('disabled')).toBe(true);
    expect(screen.queryByTestId('practice-table')).toBeNull();
    expect(screen.queryByText('SIMULATION · NOT LIVE DETECTION')).toBeNull();
    fireEvent.click(screen.getByRole('link', { name: 'Live Coach' }));
    expect(await screen.findByText('Live coach destination')).toBeTruthy();
  });

  it('keeps coordinate editing, target selection, removal, and physical dimension controls', async () => {
    await renderPractice();
    fireEvent.click(screen.getByRole('button', { name: 'Add object target' }));
    fireEvent.change(screen.getByLabelText('Table X (mm)'), { target: { value: '980' } });
    fireEvent.change(screen.getByLabelText('Table Y (mm)'), { target: { value: '340' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add target at coordinates' }));
    expect(screen.getByRole('option', { name: 'Target 4' })).toBeTruthy();
    expect(screen.getByRole('button', { name: /Target 4 · 980, 340 mm/ })).toBeTruthy();

    fireEvent.change(screen.getByLabelText('Object target'), { target: { value: 'ball-4' } });
    fireEvent.change(screen.getByLabelText('Pocket'), { target: { value: 'top-middle' } });
    expect((screen.getByLabelText('Object target') as HTMLSelectElement).value).toBe('ball-4');
    expect((screen.getByLabelText('Pocket') as HTMLSelectElement).value).toBe('top-middle');

    fireEvent.click(screen.getByRole('button', { name: 'Remove Target 4' }));
    expect(screen.queryByRole('option', { name: 'Target 4' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Place cue target' }));
    fireEvent.change(screen.getByLabelText('Table X (mm)'), { target: { value: '900' } });
    fireEvent.change(screen.getByLabelText('Table Y (mm)'), { target: { value: '200' } });
    fireEvent.click(screen.getByRole('button', { name: 'Place selected target' }));
    expect(screen.getByRole('button', { name: /Cue position · 900, 200 mm/ })).toBeTruthy();

    for (const label of ['Length (mm)', 'Width (mm)', 'Ball radius (mm)']) {
      expect((screen.getByLabelText(label) as HTMLInputElement).checkValidity()).toBe(true);
    }
    fireEvent.change(screen.getByLabelText('Ball radius (mm)'), { target: { value: '30' } });
    expect((screen.getByLabelText('Ball radius (mm)') as HTMLInputElement).value).toBe('30');
    expect(screen.getByTestId('practice-camera-state').textContent).toContain('Unverified');
  });

  it('shows zero detected balls as unverified over the calibrated camera image', async () => {
    configureSyntheticCameraState(0);
    await renderPractice();

    await waitFor(() => expect(screen.getByTestId('practice-camera-state').textContent)
      .toContain('0 balls with physical centers'));
    expect(screen.getByTestId('practice-camera-state').textContent).toContain('Unverified');
    expect(screen.getByTestId('practice-camera-overlay')).toBeTruthy();
    expect(screen.getByTestId('practice-target-cue')).toBeTruthy();
    expect(screen.queryByTestId('practice-detected-center')).toBeNull();
    expect(screen.getByRole('button', { name: 'Confirm cue position' }).hasAttribute('disabled')).toBe(true);
  });

  it('shows Ready only for a complete synthetic layout with current quiet evidence', async () => {
    configureSyntheticCameraState(4);
    await renderPractice();

    await waitFor(() => expect(screen.getByTestId('practice-ready-status').textContent).toContain('Ready'));
    expect(screen.getByTestId('practice-readiness-badge').textContent).toContain('READY · LIVE PLACEMENT');
    expect(screen.getByTestId('practice-target-cue').querySelector('.practice-target-marker')?.classList.contains('is-placed')).toBe(true);
    expect(screen.getAllByTestId('practice-detected-center')).toHaveLength(4);
    expect(screen.getByRole('button', { name: 'Confirm cue position' }).hasAttribute('disabled')).toBe(false);
  });

  it('names the target slot and measured distance that still needs adjustment', async () => {
    configureSyntheticCameraState(4);
    const syntheticState = practiceStore.videoStore.latestGameState as {
      balls: Array<{ table_position: { x: number; y: number } }>;
    };
    syntheticState.balls[3].table_position.x += 100;
    await renderPractice();

    await waitFor(() => expect(screen.getByTestId('practice-ready-status').textContent).toContain('Placed 3 of 4'));
    const targetRow = screen.getAllByRole('button', { name: /Target 3/ })
      .find((button) => button.textContent?.includes('nearest detected center'));
    expect(targetRow?.textContent).toContain('nearest detected center is 100 mm away');
    expect(screen.getByRole('button', { name: 'Confirm cue position' }).hasAttribute('disabled')).toBe(true);
  });
});
