import { Link } from '@tanstack/react-router';
import '../video/live-aim.css';

/** These routes share the app-owned camera receiver and WebSocket connection. */
export function TrainingTabs({ active }: { active: 'coach' | 'practice' }) {
  return <nav className="training-tabs" aria-label="Training views">
    <Link to="/coach" aria-current={active === 'coach' ? 'page' : undefined}>Live Coach</Link>
    <Link to="/practice" aria-current={active === 'practice' ? 'page' : undefined}>Practice layout</Link>
  </nav>;
}
