import { createFileRoute } from "@tanstack/react-router";
import { SetupPage } from "../components/configuration/SetupPage";
export const Route = createFileRoute("/configuration")({
  component: SetupPage,
});
