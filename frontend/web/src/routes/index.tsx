import { createFileRoute, redirect } from '@tanstack/react-router';

import { getSetupStatus } from '../api/setup';

export const Route = createFileRoute('/')({
  beforeLoad: async ({ abortController }) => {
    const setup = await getSetupStatus(abortController.signal).catch(() => undefined);
    throw redirect({ to: setup?.state === 'idle' || setup?.state === 'error' ||
      setup?.state === 'starting' || (setup?.state === 'running' && setup.calibrated === false) ? '/configuration' : '/live' });
  },
});
