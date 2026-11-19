import { createFileRoute } from "@tanstack/react-router";
import { HealthPage } from "../components/health/HealthPage";
export const Route = createFileRoute("/health")({ component: HealthPage });
