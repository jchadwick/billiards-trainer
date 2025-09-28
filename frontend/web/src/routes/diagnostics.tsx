import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/diagnostics")({
  component: RouteComponent,
});

function RouteComponent() {
  return (
    <div>
      <h1>Diagnostics</h1>
      <p>View system diagnostics and troubleshooting information.</p>
    </div>
  );
}
