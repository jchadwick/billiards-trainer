import { Outlet, createRootRoute } from '@tanstack/react-router';

/** Keep the active browser surface deliberately small: the live table view. */
export const Route = createRootRoute({
  component: () => <Outlet />,
  notFoundComponent: () => (
    <main className="flex min-h-screen items-center justify-center bg-gray-950 p-6 text-gray-100">
      <p>This build currently provides the live table view at <a className="underline" href="/live">/live</a>.</p>
    </main>
  ),
});
