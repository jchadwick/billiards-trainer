import { createFileRoute } from "@tanstack/react-router";
import { DashboardLayout } from "../components/monitoring/DashboardLayout";

export const Route = createFileRoute("/diagnostics")({
  component: RouteComponent,
});

function RouteComponent() {
  return <DashboardLayout />;
}
