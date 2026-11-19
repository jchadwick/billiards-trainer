import { Link, Outlet } from "@tanstack/react-router";
import { observer } from "mobx-react-lite";
import {
  Activity,
  Settings2,
  HeartPulse,
  ListTree,
  Radio,
  Camera,
  Target,
} from "lucide-react";
import { useRootStore } from "../stores/context";
import axiosClient from "../api/axios-client";
import { VideoReceiverHost } from "./video/VideoReceiverHost";
export const AppLayout = observer(() => {
  const { system, videoStore, diagnosticsStore } = useRootStore();
  const state = videoStore.latestGameState;
  const diagnosticSnapshot = diagnosticsStore.snapshot;
  const diagnosticsFresh = diagnosticSnapshot !== null
    && !diagnosticsStore.isStale
    && diagnosticsStore.connection === "connected";
  const warningStages = diagnosticSnapshot?.stages.filter((stage) =>
    ["degraded", "stalled", "failed"].includes(stage.status),
  ).length ?? 0;
  const allStagesHealthy = diagnosticsFresh
    && diagnosticSnapshot.stages.every((stage) => ["healthy", "idle"].includes(stage.status));
  const diagnosticsLabel = !diagnosticSnapshot
    ? diagnosticsStore.connection === "connected" ? "Stages unknown" : "Diagnostics unknown"
    : !diagnosticsFresh
      ? "Last known stages · stale"
      : warningStages > 0
        ? `${warningStages} stage${warningStages === 1 ? "" : "s"} need attention`
        : diagnosticSnapshot.stages.every((stage) => ["healthy", "idle"].includes(stage.status))
          ? "Stages healthy"
          : "Stages assessed";
  return (
    <div className="application dark">
      <VideoReceiverHost />
      <header className="app-header">
        <Link to="/live" className="header-brand">
          <span className="brand-symbol" aria-hidden="true">
            <i />
            <i />
            <i />
            <i />
          </span>
          Billiards Trainer
        </Link>
        <div className="header-signals">
          <span className={allStagesHealthy ? "text-emerald-300" : warningStages > 0 || diagnosticSnapshot && !diagnosticsFresh ? "text-amber-300" : "text-muted-foreground"} title={diagnosticsStore.protocolError ?? undefined}>
            <HeartPulse size={14} />
            {diagnosticsLabel}
          </span>
          <span className={videoStore.delivery.state === "receiving" || videoStore.delivery.state === "loaded" ? "text-emerald-300" : "text-muted-foreground"} title={videoStore.delivery.error ?? undefined}>
            <Camera size={14} />
            {videoStore.delivery.state === "receiving" ? "Video receiving" : videoStore.delivery.state === "loaded" ? "Video loaded · freshness unavailable" : videoStore.delivery.state === "stalled" ? "Video stalled" : videoStore.delivery.state === "error" ? "Video error" : videoStore.delivery.state === "stopped" ? "Video stopped" : videoStore.delivery.state === "unconfigured" ? "Video unavailable" : "Video connecting"}
          </span>
          {!videoStore.delivery.configurationSynchronized && videoStore.delivery.configured && <span className="text-amber-300">Video configuration status unknown</span>}
          <span
            className={
              system.status.isConnected
                ? "text-emerald-300"
                : "text-muted-foreground"
            }
          >
            <Radio size={14} />
            {system.status.isConnected ? "Receiving" : "Disconnected"}
          </span>
          <span>
            <Activity size={14} />
            {state?.observation_status?.perception?.phase ?? "Phase unknown"}
          </span>
        </div>
        <span className="backend-address">
          {new URL(axiosClient.getWebSocketUrl("/api/v1/websocket/ws")).host}
        </span>
      </header>
      <div className="nav-wrap">
        <nav className="primary-nav" aria-label="Main navigation">
          <Link to="/practice" activeProps={{ className: "active" }}>
            <Target size={16} />
            Practice
          </Link>
          <Link to="/live" activeProps={{ className: "active" }}>
            <Activity size={16} />
            Live Diagnostics
          </Link>
          <Link to="/configuration" activeProps={{ className: "active" }}>
            <Settings2 size={16} />
            Configuration
          </Link>
          <Link to="/health" activeProps={{ className: "active" }}>
            <HeartPulse size={16} />
            System Health
          </Link>
          <Link to="/events" activeProps={{ className: "active" }}>
            <ListTree size={16} />
            Events &amp; Debugging
          </Link>
        </nav>
        <span className="nav-context">
          Operational workspace{" "}
          <span className="font-mono">
            /{" "}
            {state?.observation_status?.engine_instance_id ??
              "Unknown producer"}
          </span>
        </span>
      </div>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
});
