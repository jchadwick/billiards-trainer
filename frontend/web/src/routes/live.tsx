import { createFileRoute } from "@tanstack/react-router";
import { Diagnostics } from "../components/Diagnostics";
export const Route = createFileRoute("/live")({ component: Diagnostics });
