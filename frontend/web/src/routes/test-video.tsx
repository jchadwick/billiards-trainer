import { createFileRoute } from '@tanstack/react-router'
import { VideoStreamTest } from '../components/video/VideoStreamTest'

export const Route = createFileRoute('/test-video')({
  component: TestVideoPage,
})

function TestVideoPage() {
  return (
    <div className="h-screen">
      <VideoStreamTest useMockData={true} />
    </div>
  )
}
