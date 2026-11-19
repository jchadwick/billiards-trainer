import { createFileRoute } from "@tanstack/react-router";
import { EventsPage } from "../components/events/EventsPage";
export const Route = createFileRoute("/events")({ component: EventsPage });
