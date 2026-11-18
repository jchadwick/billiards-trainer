import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { PracticeCameraOverlay } from './PracticeCameraOverlay';
import type { TableProjection } from '../../types/api';
import type { PotCandidate } from '../video/shot-prediction';

const projection: TableProjection = {
  direction: 'source_image_px_to_table_mm',
  image_space: 'raw',
  source_width_px: 1000,
  source_height_px: 500,
  camera_profile_id: 'synthetic-camera-fixture',
  calibration_revision: 3,
  rectification_revision: 1,
  table_width_mm: 1000,
  table_height_mm: 500,
  unit: 'millimeters',
  origin: 'playing_surface_center',
  x_axis: 'corner_0_to_1',
  y_axis: 'corner_0_to_3',
  pixel_to_table_mm: [[1, 0, -500], [0, 1, -250], [0, 0, 1]],
};

const shot: PotCandidate = {
  targetId: 'ball-1',
  pocketId: 'right-pocket',
  cueBall: { id: 'cue', x: 200, y: 250, radius: 10 },
  target: { id: 'ball-1', x: 650, y: 250, radius: 10 },
  pocket: { id: 'right-pocket', label: 'Right pocket', x: 1000, y: 250 },
  contactCenter: { x: 500, y: 250 },
  contactPoint: { x: 640, y: 250 },
  cuePathEnd: { x: 500, y: 250 },
  objectPathStart: { x: 660, y: 250 },
  objectPathEnd: { x: 1000, y: 250 },
  cutAngleRadians: 0,
  totalTravel: 800,
};

function makeImage(): HTMLImageElement {
  const image = document.createElement('img');
  Object.defineProperty(image, 'naturalWidth', { configurable: true, value: 1000 });
  Object.defineProperty(image, 'naturalHeight', { configurable: true, value: 500 });
  return image;
}

afterEach(() => {
  cleanup();
  document.getElementById('shared-video-receiver-parking')?.remove();
});

describe('PracticeCameraOverlay', () => {
  it('projects physical target and measured-center ellipses over the shared camera image', () => {
    const parking = document.createElement('div');
    parking.id = 'shared-video-receiver-parking';
    const image = makeImage();
    parking.appendChild(image);
    document.body.appendChild(parking);

    const view = render(<PracticeCameraOverlay
      image={image}
      imageAvailable
      mode="native"
      dimensions={{ width: 1000, height: 500 }}
      projection={projection}
      tableRadiusMm={10}
      toleranceMm={45}
      targets={[{ id: 'cue', label: 'Cue position', x: 500, y: 250, placed: true }]}
      balls={[{ x: 0, y: 0, radiusMm: 10, matched: true, extrapolated: false }]}
      canEditOnImage
      ready
      notice="Synthetic projection fixture only."
      readinessMessage="All target slots are placed."
      shot={shot}
      onPlaceAt={vi.fn()}
      onImageDimensions={vi.fn()}
    />);

    const target = screen.getByTestId('practice-target-cue').querySelector('.practice-target-marker');
    expect(target?.getAttribute('cx')).toBe('500');
    expect(target?.getAttribute('cy')).toBe('250');
    expect(target?.getAttribute('rx')).toBe('10');
    expect(screen.getByTestId('practice-target-cue').querySelector('.practice-target-tolerance')?.getAttribute('rx')).toBe('45');
    expect(screen.getAllByTestId('practice-detected-center')).toHaveLength(1);
    expect(document.querySelector('.practice-shot-cue-line')?.getAttribute('x1')).toBe('200');
    expect(document.querySelector('.practice-shot-object-line')?.getAttribute('x2')).toBe('1000');
    expect(screen.getByTestId('practice-camera-state').textContent).toContain('Ready');

    view.unmount();
    expect(image.parentElement).toBe(parking);
  });

  it('shows an explicit unverified camera state without drawing a substitute table', () => {
    render(<PracticeCameraOverlay
      image={null}
      imageAvailable={false}
      mode={null}
      dimensions={null}
      projection={null}
      tableRadiusMm={10}
      toleranceMm={45}
      targets={[{ id: 'cue', label: 'Cue position', x: 500, y: 250, placed: false }]}
      balls={[]}
      canEditOnImage={false}
      ready={false}
      notice="No current camera or calibration is available."
      readinessMessage="No current camera or calibration is available."
      shot={null}
      onPlaceAt={vi.fn()}
      onImageDimensions={vi.fn()}
    />);

    expect(screen.getByTestId('practice-camera-unavailable').textContent).toContain('Live camera video is unavailable');
    expect(screen.getByTestId('practice-camera-state').textContent).toContain('Unverified');
    expect(screen.queryByTestId('practice-camera-overlay')).toBeNull();
    expect(screen.queryByTestId('practice-table')).toBeNull();
  });

  it('does not take the shared image away from another visible route', () => {
    const otherRoute = document.createElement('div');
    otherRoute.className = 'native-video-image-slot';
    const image = makeImage();
    otherRoute.appendChild(image);
    document.body.appendChild(otherRoute);

    render(<PracticeCameraOverlay
      image={image}
      imageAvailable={false}
      mode="native"
      dimensions={{ width: 1000, height: 500 }}
      projection={projection}
      tableRadiusMm={10}
      toleranceMm={45}
      targets={[]}
      balls={[]}
      canEditOnImage={false}
      ready={false}
      notice="Camera is already displayed in another view."
      readinessMessage="Camera is already displayed in another view."
      shot={null}
      onPlaceAt={vi.fn()}
      onImageDimensions={vi.fn()}
    />);

    expect(image.parentElement).toBe(otherRoute);
    expect(screen.getByTestId('practice-camera-state').textContent).toContain('another view');
  });
});
