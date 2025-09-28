import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/configuration")({
  component: RouteComponent,
});

function RouteComponent() {
  return (
    <div>
      <h1>Configuration</h1>
      <p>Manage system settings and preferences.</p>
    </div>
  );
}
