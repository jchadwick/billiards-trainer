import { createFileRoute } from "@tanstack/react-router";
import { WebSocketEventMonitor } from "../components/diagnostics/WebSocketEventMonitor";

export const Route = createFileRoute("/websocket-events")({
  component: RouteComponent,
});

function RouteComponent() {
  return <WebSocketEventMonitor />;
}
