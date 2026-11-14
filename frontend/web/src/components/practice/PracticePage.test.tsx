import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createRootRoute, createRoute, createRouter, createMemoryHistory, RouterProvider } from '@tanstack/react-router';
import { PracticePage } from './PracticePage';

async function renderPractice() {
  const root = createRootRoute();
  const practice = createRoute({ getParentRoute: () => root, path: '/practice', component: PracticePage });
  const coach = createRoute({ getParentRoute: () => root, path: '/coach', component: () => <p>Live coach destination</p> });
  const router = createRouter({ routeTree: root.addChildren([practice, coach]), history: createMemoryHistory({ initialEntries: ['/practice'] }) });
  await router.load();
  render(<RouterProvider router={router} />);
}

beforeEach(() => { vi.spyOn(window, 'scrollTo').mockImplementation(() => {}); });
afterEach(() => { vi.restoreAllMocks(); });

describe('PracticePage', () => {
  it('labels the table as simulated and requires explicit cue-ball confirmation', async () => {
    await renderPractice();
    expect(screen.getByRole('link', { name: 'Live Coach' }).getAttribute('href')).toBe('/coach');
    expect(screen.getByRole('link', { name: 'Practice layout' }).getAttribute('aria-current')).toBe('page');
    expect(screen.getByText('SIMULATION · NOT LIVE DETECTION')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Confirm cue ball' })).toBeTruthy();
    expect(screen.getByRole('status').textContent).toContain('Confirm the cue ball');
    fireEvent.click(screen.getByRole('button', { name: 'Confirm cue ball' }));
    expect(screen.getByRole('button', { name: /Cue ball confirmed/ }).getAttribute('aria-pressed')).toBe('true');
    expect(screen.getByRole('status').textContent).toContain('Suggested direct pot');
    fireEvent.click(screen.getByRole('link', { name: 'Live Coach' }));
    expect(await screen.findByText('Live coach destination')).toBeTruthy();
  });

  it('allows static object-ball editing, target selection, and pocket selection', async () => {
    await renderPractice();
    fireEvent.click(screen.getByRole('button', { name: 'Add object ball' }));
    const table = screen.getByTestId('practice-table');
    // Simulate a scaled SVG with border/letterbox offsets in screen space.
    Object.defineProperty(table, 'getScreenCTM', { value: () => ({ inverse: () => ({ a: 2, b: 0, c: 0, d: 2, e: -220, f: -160 }) }) });
    fireEvent.click(table, { clientX: 600, clientY: 250 });
    expect(screen.getByRole('option', { name: /Ball 4/ })).toBeTruthy();
    const placed = table.querySelectorAll('.practice-ball')[3];
    expect(placed.getAttribute('cx')).toBe('980');
    expect(placed.getAttribute('cy')).toBe('340');
    fireEvent.change(screen.getByLabelText('Target ball'), { target: { value: 'ball-4' } });
    fireEvent.change(screen.getByLabelText('Pocket'), { target: { value: 'top-middle' } });
    expect((screen.getByLabelText('Target ball') as HTMLSelectElement).value).toBe('ball-4');
    expect((screen.getByLabelText('Pocket') as HTMLSelectElement).value).toBe('top-middle');
    fireEvent.click(screen.getByRole('button', { name: 'Remove ball 4' }));
    expect(screen.queryByRole('option', { name: /Ball 4/ })).toBeNull();
    fireEvent.change(screen.getByLabelText('Table X (mm)'), { target: { value: '900' } });
    fireEvent.change(screen.getByLabelText('Table Y (mm)'), { target: { value: '200' } });
    expect((screen.getByLabelText('Table X (mm)') as HTMLInputElement).checkValidity()).toBe(true);
    expect((screen.getByLabelText('Table Y (mm)') as HTMLInputElement).checkValidity()).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'Place selected ball' }));
    expect(screen.getByRole('option', { name: 'Ball 4' })).toBeTruthy();
    for (const label of ['Length (mm)', 'Width (mm)', 'Ball radius (mm)']) {
      expect((screen.getByLabelText(label) as HTMLInputElement).checkValidity()).toBe(true);
    }
    fireEvent.change(screen.getByLabelText('Ball radius (mm)'), { target: { value: '30' } });
    expect(table.querySelector('.practice-cue-ball')?.getAttribute('r')).toBe('30');
    expect(table.querySelector('.practice-ball')?.getAttribute('r')).toBe('30');
  });
});
