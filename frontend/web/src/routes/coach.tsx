import { createFileRoute } from '@tanstack/react-router';
import { CoachPage } from '../components/practice/CoachPage';

export const Route = createFileRoute('/coach')({ component: CoachPage });
