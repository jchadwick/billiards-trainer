import { createRootRoute, Link } from "@tanstack/react-router";
import { AppLayout } from "../components/AppLayout";
export const Route = createRootRoute({
  component: AppLayout,
  notFoundComponent: () => (
    <p>
      Page unavailable. <Link to="/live">Open Live Diagnostics</Link>
    </p>
  ),
});
