import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { SetupPage } from './SetupPage';

const api = vi.hoisted(() => ({ get: vi.fn(), put: vi.fn(), post: vi.fn(), notify: vi.fn() }));
vi.mock('../../api/axios-client', () => ({
  default: { get: api.get, getInstance: () => ({ put: api.put, post: api.post }) },
}));
vi.mock('../../stores/context', () => ({
  useRootStore: () => ({ notifyVideoConfigurationApplied: api.notify }),
}));
vi.mock('@tanstack/react-router', () => ({
  Link: ({ to, children }: { to: string; children: string }) => <a href={to}>{children}</a>,
}));

it('requires a source choice and binds calibration to the exact frozen frame and revision', async () => {
  const source = 'gstreamer-device-v1:test-camera';
  const descriptor = { width_px: 101, height_px: 101, source_frame_id: 'exact-frame' };
  api.get.mockImplementation(async (path: string) => {
    if (path.endsWith('/cameras')) {
      return { data: { devices: [{ display_name: 'Overhead camera', capture_selector: source }] } };
    }
    if (path.endsWith('/frame')) {
      return { data: { descriptor, jpeg_base64: 'AA==', expected_store_revision: 6 } };
    }
    return { data: { state: 'idle', configuration_revision: 7 } };
  });
  api.put.mockResolvedValue({ data: { state: 'running', configuration_revision: 8 } });
  api.post.mockResolvedValue({ data: { status: 'applied' } });

  render(<SetupPage />);
  expect(api.put).not.toHaveBeenCalled();
  expect(screen.queryByRole('button', { name: 'Freeze camera frame' })).toBeNull();
  await screen.findByRole('option', { name: 'Overhead camera' });
  fireEvent.change(screen.getByLabelText('Camera'), { target: { value: source } });
  fireEvent.click(screen.getByRole('button', { name: 'Use this camera' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Freeze camera frame' }));
  expect(api.put).toHaveBeenCalledWith('/api/v1/setup/source', { source, expected_revision: 7 });
  expect(api.notify).toHaveBeenCalledTimes(1);

  const frame = await screen.findByLabelText('Frozen camera frame for marking table corners');
  vi.spyOn(frame, 'getBoundingClientRect').mockReturnValue(new DOMRect(10, 20, 200, 200));
  const points = [[30, 60], [190, 60], [190, 180], [30, 180]];
  for (const [clientX, clientY] of points) fireEvent.click(frame, { clientX, clientY });
  fireEvent.change(screen.getByLabelText('Playing surface size'), { target: { value: 'custom' } });
  fireEvent.change(screen.getByLabelText('Width, left to right (mm)'), { target: { value: '2000' } });
  fireEvent.change(screen.getByLabelText('Height, top to bottom (mm)'), { target: { value: '1000' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save calibration' }));

  expect(api.post).toHaveBeenCalledWith('/api/v1/setup/calibration', {
    schema_version: 1,
    descriptor,
    expected_store_revision: 6,
    table_width_mm: 2000,
    table_height_mm: 1000,
    corners_px: [{ x_px: 10, y_px: 20 }, { x_px: 90, y_px: 20 }, { x_px: 90, y_px: 80 }, { x_px: 10, y_px: 80 }],
  });
  expect(await screen.findByText('Calibration saved. Waiting for the camera to apply it.')).toBeTruthy();
  expect(screen.queryByRole('link', { name: 'Open Live Diagnostics' })).toBeNull();
});

it('shows the saved camera and opens the chooser only for reconfiguration', async () => {
  api.get.mockImplementation(async (path: string) => ({
    data: path.endsWith('/cameras')
      ? { devices: [] }
      : { state: 'running', configuration_revision: 7, configured_source: 'http://camera.local/stream', calibrated: true },
  }));

  render(<SetupPage />);
  expect(await screen.findByText('Saved source: Network camera (MJPEG)')).toBeTruthy();
  expect(screen.queryByRole('combobox')).toBeNull();
  expect(api.get.mock.calls.some(([path]) => path.endsWith('/cameras'))).toBe(false);

  fireEvent.click(screen.getByRole('button', { name: 'Reconfigure camera' }));
  expect(screen.getByRole('combobox', { name: 'Camera' })).toBeTruthy();
  await screen.findByText(/No local cameras/);
  fireEvent.click(screen.getByRole('button', { name: 'Cancel reconfiguration' }));
  expect(screen.queryByRole('combobox')).toBeNull();
});
