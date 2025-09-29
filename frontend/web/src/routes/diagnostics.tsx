import { createFileRoute } from "@tanstack/react-router";
import { DiagnosticsSystem } from "../components/diagnostics/DiagnosticsSystem";

export const Route = createFileRoute("/diagnostics")({
  component: RouteComponent,
});

function RouteComponent() {
  return <DiagnosticsSystem />;
}
