import { LiveView } from '../video/LiveView';
import { TrainingTabs } from './TrainingTabs';

export function CoachPage() {
  return <section aria-label="Live table training">
    <TrainingTabs active="coach" />
    <p className="coach-introduction">Place a cue ball and an object ball for a straight pot. Confirm the cue-ball marker below, choose your target and pocket, then bring a level cue behind the cue ball. Compare your current aim with the suggested line.</p>
    <LiveView />
  </section>;
}
