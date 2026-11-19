import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { PrecisionCalibrationWizard } from './PrecisionCalibrationWizard';

const { api, video } = vi.hoisted(() => ({
  api: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
  video: {
    nativeVideoElement: null as HTMLImageElement | null,
    videoStore: { delivery: {
      configurationSynchronized: true, dimensions: { width: 101, height: 101 },
      mode: 'native', sourceGeneration: 1, state: 'loaded',
    } },
  },
}));
vi.mock('../../stores/context', () => ({ useRootStore: () => video }));
vi.mock('../../api/axios-client', () => ({
  default: { get: api.get, getInstance: () => api },
}));

const ROOT = '/api/v1/setup/calibration';

function cameraFrame(id: number) {
  return {
    descriptor: {
      width_px: 101, height_px: 101, engine_instance_id: 'engine',
      capture_session_id: 'session', source_frame_id: `frame-${id}`,
      configuration_revision: 5, calibration_revision: 3, rectification_revision: 0,
    },
    jpeg_base64: 'AA==',
    capture_token: `token-${id}`,
  };
}

function archiveServer() {
  type Capture = { slot: number; thumbnail_base64: string; table_center_mm?: number[] };
  const archives = { optics: new Map<number, Capture>(), 'ball-centers': new Map<number, Capture>() };
  let nextFrame = 1;
  api.get.mockImplementation(async (path: string) => {
    if (path === `${ROOT}/frame`) return { data: cameraFrame(nextFrame++) };
    if (path === `${ROOT}/ball-placement`) return { data: {
      width_px: 101, height_px: 101, target_center_px: [40, 50],
      target_radius_px: [10, 10], ball_center_px: [40, 50], in_target: true, observation_available: true,
    } };
    const mode = path.split('/').at(-1) as keyof typeof archives;
    return { data: { captures: [...archives[mode].values()], table_size_mm: [2000, 1000] } };
  });
  api.put.mockImplementation(async (path: string, frame: { table_center_mm?: number[] }) => {
    const [mode, slot] = path.split('/').slice(-2) as [keyof typeof archives, string];
    const data = { slot: Number(slot), thumbnail_base64: 'AQ==', table_center_mm: frame.table_center_mm };
    archives[mode].set(data.slot, data);
    return { data };
  });
  api.delete.mockImplementation(async (path: string) => {
    const parts = path.split('/');
    const last = parts.at(-1)!;
    if (last in archives) archives[last as keyof typeof archives].clear();
    else archives[parts.at(-2) as keyof typeof archives].delete(Number(last));
  });
  return archives;
}

function mountCamera() {
  const parking = document.createElement('div');
  document.body.appendChild(parking);
  const image = document.createElement('img');
  image.src = 'http://camera.test/stream';
  parking.appendChild(image);
  video.nativeVideoElement = image;
  return { image, parking };
}

afterEach(() => {
  cleanup();
  video.nativeVideoElement?.parentElement?.remove();
  video.nativeVideoElement = null;
  vi.resetAllMocks();
});

it('restores checkerboard thumbnails and refits from all eight saved slots after remount', async () => {
  const onSaved = vi.fn();
  const archives = archiveServer();
  api.post.mockResolvedValue({ data: { holdout_rms_px: 0.25 } });
  const first = render(<PrecisionCalibrationWizard mode="optics" onSaved={onSaved} />);
  const download = screen.getByRole('link', { name: /download printable checkerboard/i });
  expect(new URL(download.getAttribute('href')!).pathname).toBe(`${ROOT}/checkerboard.pdf`);
  const slots = within(screen.getByRole('list', { name: 'Required checkerboard views' })).getAllByRole('listitem');
  expect(slots).toHaveLength(8);
  const fit = screen.getByRole<HTMLButtonElement>('button', { name: /fit and activate/i });
  expect(fit.disabled).toBe(true);
  await screen.findByText('0 of 8 views saved.');

  for (const slot of slots) {
    fireEvent.click(within(slot).getByRole('button', { name: /capture/i }));
    await within(slot).findByRole('img', { name: 'Captured' });
  }
  fireEvent.click(within(slots[0]).getByRole('button', { name: /recapture/i }));
  await waitFor(() => expect(api.put).toHaveBeenLastCalledWith(`${ROOT}/captures/optics/0`, cameraFrame(9)));
  first.unmount();

  render(<PrecisionCalibrationWizard mode="optics" onSaved={onSaved} />);
  expect((await screen.findAllByRole('img', { name: 'Captured' }))).toHaveLength(8);
  expect(screen.getByRole('img', { name: 'Center capture' }).getAttribute('src')).toBe('data:image/jpeg;base64,AQ==');
  fireEvent.click(screen.getByRole('button', { name: /fit and activate/i }));
  await waitFor(() => expect(onSaved).toHaveBeenCalledWith(expect.stringContaining('0.25 px')));
  expect(api.post).toHaveBeenCalledWith(`${ROOT}/optics`, {
    columns: 9, rows: 6, square_mm: 50.8, slots: [0, 1, 2, 3, 4, 5, 6, 7],
  }, expect.any(Object));
  expect(archives.optics.size).toBe(8);
});

it('guides ball placement, confirms a cropped detection, and reloads the saved sample', async () => {
  const archives = archiveServer();
  const { image, parking } = mountCamera();
  const mark = { image_center_px: [40, 50], radius_px: 10 };
  api.post.mockResolvedValue({ data: { candidate: mark } });
  const first = render(<PrecisionCalibrationWizard mode="ball-centers" onSaved={vi.fn()} />);
  const live = await screen.findByRole('img', { name: /target for position 1$/i });
  await screen.findByText('Ball in target');
  expect(live.contains(image)).toBe(true);
  expect(api.get).toHaveBeenCalledWith(`${ROOT}/ball-placement`, expect.objectContaining({ params: { x_mm: -880, y_mm: -350 } }));
  expect(screen.getByText('From left cushion').nextElementSibling?.textContent).toMatch(/120 mm\s*4\.72 in/);
  expect(screen.getByText('From top cushion').nextElementSibling?.textContent).toMatch(/150 mm\s*5\.91 in/);
  fireEvent.click(screen.getByRole('button', { name: 'Ball is on the mark — capture' }));

  await screen.findByText('Does the circle outline the ball you placed?');
  expect(screen.getByRole('img', { name: 'Zoomed ball frame' }).getAttribute('viewBox')).toBe('10 20 60 60');
  expect(api.post).toHaveBeenCalledWith(`${ROOT}/ball-candidate`, { ...cameraFrame(1), table_center_mm: [-880, -350] });
  expect(archives['ball-centers'].size).toBe(0);
  fireEvent.click(screen.getByRole('button', { name: 'Confirm and save position' }));
  expect(await screen.findByRole('img', { name: 'Ball position 1' })).toBeTruthy();
  expect(api.put).toHaveBeenCalledWith(`${ROOT}/captures/ball-centers/0`, {
    ...cameraFrame(1), ...mark, table_center_mm: [-880, -350],
  });
  first.unmount();
  expect(parking.contains(image)).toBe(true);

  render(<PrecisionCalibrationWizard mode="ball-centers" onSaved={vi.fn()} />);
  expect(await screen.findByRole('img', { name: /target for position 2$/i })).toBeTruthy();
  expect(screen.getByRole('img', { name: 'Ball position 1' }).getAttribute('src')).toBe('data:image/jpeg;base64,AQ==');
  fireEvent.click(screen.getByRole('button', { name: 'Remove position 1' }));
  await waitFor(() => expect(screen.queryByRole('img', { name: 'Ball position 1' })).toBeNull());
  expect(screen.getByRole('img', { name: /target for position 1$/i })).toBeTruthy();
});

it('clears the live check when detection reports the ball moved', async () => {
  archiveServer();
  mountCamera();
  const get = api.get.getMockImplementation()!;
  let inTarget = true;
  api.get.mockImplementation((path, config) => {
    if (path !== `${ROOT}/ball-placement`) return get(path, config);
    return Promise.resolve({ data: {
      width_px: 101, height_px: 101, target_center_px: [40, 50],
      target_radius_px: [10, 10], ball_center_px: inTarget ? [40, 50] : [20, 20], in_target: inTarget, observation_available: true,
    } });
  });
  render(<PrecisionCalibrationWizard mode="ball-centers" onSaved={vi.fn()} />);
  await screen.findByText('Ball in target');
  inTarget = false;
  await screen.findByText('Move the ball into the target.');
  expect(screen.queryByText('Ball in target')).toBeNull();
});

it('shows every target, lets the operator choose an edge position, and preserves a legacy saved placement', async () => {
  const archives = archiveServer();
  archives['ball-centers'].set(0, { slot: 0, thumbnail_base64: 'AQ==', table_center_mm: [-300, -150] });
  render(<PrecisionCalibrationWizard mode="ball-centers" onSaved={vi.fn()} />);
  const map = await screen.findByRole('group', { name: 'Ball placement positions' });
  expect(within(map).getAllByRole('button')).toHaveLength(24);
  const saved = within(map).getByRole('button', { name: 'Position 1, saved' });
  expect(saved.style.left).toBe('35%');
  fireEvent.click(saved);
  expect(screen.getByRole('img', { name: 'Saved capture for position 1' })).toBeTruthy();

  fireEvent.click(within(map).getByRole('button', { name: 'Position 4' }));
  expect(screen.getByText('From right cushion').nextElementSibling?.textContent).toMatch(/120 mm\s*4\.72 in/);
  expect(screen.getByText('From top cushion').nextElementSibling?.textContent).toMatch(/150 mm\s*5\.91 in/);
  expect(screen.getByRole('img', { name: 'Live camera showing the target for position 4' })).toBeTruthy();
});

it('can mark a missed ball manually and fit from the saved sample set', async () => {
  const archives = archiveServer();
  for (let slot = 0; slot < 11; slot++) {
    archives['ball-centers'].set(slot, { slot, thumbnail_base64: 'AQ==', table_center_mm: [-slot * 50, -45] });
  }
  api.post.mockResolvedValueOnce({ data: { candidate: null } })
    .mockResolvedValueOnce({ data: { fit_report: { leave_one_out_rms_mm: 1.25 } } });
  const onSaved = vi.fn();
  render(<PrecisionCalibrationWizard mode="ball-centers" onSaved={onSaved} />);
  await screen.findByRole('img', { name: /target for position 12$/i });
  fireEvent.click(screen.getByRole('button', { name: 'Ball is on the mark — capture' }));
  expect(await screen.findByText(/no ball was found/i)).toBeTruthy();
  const add = screen.getByRole<HTMLButtonElement>('button', { name: 'Confirm and save position' });
  expect(add.disabled).toBe(true);
  fireEvent.change(screen.getByLabelText('Image center X (px)'), { target: { value: '40' } });
  fireEvent.change(screen.getByLabelText('Image center Y (px)'), { target: { value: '50' } });
  fireEvent.change(screen.getByLabelText('Image radius (px)'), { target: { value: '10' } });
  fireEvent.click(add);
  await screen.findByRole('img', { name: 'Ball position 12' });

  fireEvent.click(screen.getByRole('button', { name: /fit and activate/i }));
  await waitFor(() => expect(onSaved).toHaveBeenCalledWith(expect.stringContaining('1.25 mm')));
  expect(api.post).toHaveBeenLastCalledWith(`${ROOT}/ball-centers`, { slots: Array.from({ length: 12 }, (_, i) => i) }, expect.any(Object));
});
