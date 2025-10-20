"""Stream module for video frame consumption."""

from .video_consumer import VideoConsumer, VideoModuleNotAvailableError

__all__ = ["VideoConsumer", "VideoModuleNotAvailableError"]
