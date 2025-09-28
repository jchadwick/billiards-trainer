import { createFileRoute } from '@tanstack/react-router'
import { LiveView } from '../components/video/LiveView'

export const Route = createFileRoute('/live')({
  component: LiveViewPage,
})

function LiveViewPage() {
  return (
    <div className="h-screen">
      <LiveView />
    </div>
  )
}
