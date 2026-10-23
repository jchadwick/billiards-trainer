import { Outlet, createRootRoute } from '@tanstack/react-router';

/** Keep the active browser surface deliberately small: the live table view. */
export const Route = createRootRoute({
  component: () => <Outlet />,
  notFoundComponent: () => (
    <main className="not-found">
      <p>This build currently provides the live table view at <a className="not-found-link" href="/live">/live</a>.</p>
    </main>
  ),
});
