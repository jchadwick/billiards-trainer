export interface RustFrameTimestamp {
  ticks: string;
  unit: "nanoseconds";
  clock: "sensor" | "monotonic" | "unix_epoch";
  domain_id: string;
  stage:
    | "exposure"
    | "camera_read_complete"
    | "shared_memory_publication"
    | "frame_arrival"
    | "inference_started"
    | "inference_complete"
    | "batch_publication";
}

export interface RustFramePreviewDescriptor {
  schema_version: 1;
  engine_instance_id: string;
  capture_session_id: string;
  source_frame_id: string;
  width_px: number;
  height_px: number;
  pixel_format: "rgb8";
  image_space: "raw" | "rectified";
  camera_profile_id: string;
  configuration_revision: number;
  calibration_revision: number;
  rectification_revision: number;
  source_timestamp: RustFrameTimestamp | null;
  source_timestamp_semantics:
    | { kind: "read_completion_only" }
    | ({ kind: "source_timestamp" } & Omit<RustFrameTimestamp, "ticks">);
  host_read_completed: RustFrameTimestamp & {
    semantics: "read_completion_only";
  };
}

export interface RustFramePreview {
  image: Blob;
  descriptor: RustFramePreviewDescriptor;
}

export interface PerceptionRuntimeStatus {
  provider: "python" | "rust" | "unavailable";
  mode: "live" | "replay" | "unavailable";
  preview: string;
  live_camera: boolean;
}
