"""Supervise the explicitly replay-only Rust perception executable.

This adapter owns the Rust process and its unauthenticated loopback scene
ingress for one FastAPI lifespan. It intentionally supports only finite
recorded PNG/JPEG input; it does not open a camera, fall back to Python, or
claim live readiness.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import tempfile
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from backend.config import load_schema_reference
from backend.perception.scene_ingress import SceneIngressConfig, SceneSnapshotIngress
from backend.perception.scene_state_worker import SceneSnapshotStateWorker

if TYPE_CHECKING:
    from backend.perception.scene_state_adapter import SceneSnapshotStatePublisher

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_TAIL_CAPACITY_BYTES = 64 * 1024
_RUST_CHILD_ENVIRONMENT_KEYS = ("TMPDIR",)
_OPENVINO_CHILD_ENVIRONMENT_KEYS = ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")


def _count_replay_images(input_path: Path) -> int:
    """Count exactly the supported corpus files before scheduling repetitions."""
    if input_path.is_file():
        return int(input_path.suffix.lower() in _IMAGE_SUFFIXES)
    if not input_path.is_dir():
        return 0
    return sum(
        1
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


class ConfigurationReader(Protocol):
    """Small interface shared by the validated config store and tests."""

    def get(self, key: str, default: Any = None) -> Any: ...


class RustReplayConfigurationError(ValueError):
    """A replay configuration cannot safely launch the Rust provider."""


class RustReplayStartupError(RuntimeError):
    """The supervised process failed before publishing a usable observation."""


@dataclass(frozen=True)
class RustReplaySettings:
    """Validated launch settings for one finite recorded-input run."""

    executable_path: Path
    onnx_path: Path
    source_checkpoint_path: Path
    input_path: Path
    provider: str
    image_size: int
    confidence: float
    iou: float
    warmup_runs: int
    replay_rate_fps: float
    public_state_delivery_schema_version: int
    replay_repeat_count: int
    maximum_replay_frames: int
    source_image_space: str
    ball_class_id: int
    cue_class_id: int | None
    camera_profile_id: str
    association_gate_px: float
    association_prediction_horizon_ms: int
    lost_after_ms: int
    maximum_track_age_ms: int
    maximum_active_tracks: int
    recent_observation_capacity: int
    configuration_revision: int
    rectification_revision: int
    calibration_bundle_path: Path | None
    applied_calibration_revision: int | None
    applied_rectification_revision: int | None
    applied_camera_profile_id: str | None
    bind_address: str
    port: int
    maximum_message_bytes: int
    read_timeout_seconds: float
    connect_timeout_ms: int
    write_timeout_ms: int
    reconnect_initial_delay_ms: int
    reconnect_maximum_delay_ms: int
    source_silence_timeout_seconds: float
    startup_timeout_seconds: float
    stop_timeout_seconds: float
    maximum_detection_batch_output_bytes: int
    output_limit_poll_interval_seconds: float
    project_root: Path = _DEFAULT_PROJECT_ROOT

    @classmethod
    def from_configuration(
        cls,
        configuration: ConfigurationReader,
        *,
        project_root: Path = _DEFAULT_PROJECT_ROOT,
    ) -> RustReplaySettings:
        """Load every setting from validated configuration, never CLI defaults."""
        perception = load_schema_reference().get("perception")
        defaults = (
            perception.get("rust_replay") if isinstance(perception, dict) else None
        )
        if not isinstance(defaults, dict):
            raise RustReplayConfigurationError(
                "bundled configuration must define perception.rust_replay"
            )

        def value(key: str) -> Any:
            return configuration.get(f"perception.rust_replay.{key}", defaults[key])

        def path_value(key: str, *, optional: bool = False) -> Path | None:
            raw = value(key)
            if optional and raw in (None, ""):
                return None
            if not isinstance(raw, str) or not raw.strip():
                raise RustReplayConfigurationError(
                    f"perception.rust_replay.{key} must be a non-empty path"
                )
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = project_root / candidate
            return candidate.resolve()

        raw_calibration_revision = value("applied_calibration_revision")
        raw_applied_rectification_revision = value("applied_rectification_revision")
        raw_applied_camera_profile_id = value("applied_camera_profile_id")
        settings = cls(
            executable_path=path_value("executable_path"),  # type: ignore[arg-type]
            onnx_path=path_value("onnx_path"),  # type: ignore[arg-type]
            source_checkpoint_path=path_value("source_checkpoint_path"),  # type: ignore[arg-type]
            input_path=path_value("input_path"),  # type: ignore[arg-type]
            provider=value("provider"),
            image_size=value("image_size"),
            confidence=value("confidence"),
            iou=value("iou"),
            warmup_runs=value("warmup_runs"),
            replay_rate_fps=value("replay_rate_fps"),
            public_state_delivery_schema_version=value(
                "public_state_delivery_schema_version"
            ),
            replay_repeat_count=value("replay_repeat_count"),
            maximum_replay_frames=value("maximum_replay_frames"),
            source_image_space=value("source_image_space"),
            ball_class_id=value("ball_class_id"),
            cue_class_id=value("cue_class_id"),
            camera_profile_id=value("camera_profile_id"),
            association_gate_px=value("association_gate_px"),
            association_prediction_horizon_ms=value(
                "association_prediction_horizon_ms"
            ),
            lost_after_ms=value("lost_after_ms"),
            maximum_track_age_ms=value("maximum_track_age_ms"),
            maximum_active_tracks=value("maximum_active_tracks"),
            recent_observation_capacity=value("recent_observation_capacity"),
            configuration_revision=value("configuration_revision"),
            rectification_revision=value("rectification_revision"),
            calibration_bundle_path=path_value(
                "calibration_bundle_path", optional=True
            ),
            applied_calibration_revision=raw_calibration_revision,
            applied_rectification_revision=raw_applied_rectification_revision,
            applied_camera_profile_id=(
                raw_applied_camera_profile_id
                if raw_applied_camera_profile_id not in (None, "")
                else None
            ),
            bind_address=value("bind_address"),
            port=value("port"),
            maximum_message_bytes=value("maximum_message_bytes"),
            read_timeout_seconds=value("read_timeout_seconds"),
            connect_timeout_ms=value("connect_timeout_ms"),
            write_timeout_ms=value("write_timeout_ms"),
            reconnect_initial_delay_ms=value("reconnect_initial_delay_ms"),
            reconnect_maximum_delay_ms=value("reconnect_maximum_delay_ms"),
            source_silence_timeout_seconds=value("source_silence_timeout_seconds"),
            startup_timeout_seconds=value("startup_timeout_seconds"),
            stop_timeout_seconds=value("stop_timeout_seconds"),
            maximum_detection_batch_output_bytes=value(
                "maximum_detection_batch_output_bytes"
            ),
            output_limit_poll_interval_seconds=value(
                "output_limit_poll_interval_seconds"
            ),
            project_root=project_root.resolve(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Reject unsafe, incomplete, or unsupported replay configurations."""
        if self.provider not in {"cpu", "auto", "openvino"}:
            raise RustReplayConfigurationError(
                "perception.rust_replay.provider must be 'cpu', 'auto', or 'openvino'"
            )
        if self.source_image_space not in {"raw", "rectified"}:
            raise RustReplayConfigurationError(
                "perception.rust_replay.source_image_space must be 'raw' or 'rectified'"
            )
        if not self.executable_path.is_file() or not os.access(
            self.executable_path, os.X_OK
        ):
            raise RustReplayConfigurationError(
                "configured Rust inference executable must exist and be executable"
            )
        if not self.onnx_path.is_file():
            raise RustReplayConfigurationError("configured ONNX model does not exist")
        if not self.source_checkpoint_path.is_file():
            raise RustReplayConfigurationError(
                "configured source checkpoint does not exist"
            )
        manifest_path = Path(f"{self.onnx_path}.manifest.json")
        if not manifest_path.is_file():
            raise RustReplayConfigurationError(
                "the ONNX export manifest is required beside the model"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            class_names = manifest["model"]["class_names"]
            if isinstance(class_names, list) and all(
                isinstance(item, dict) and isinstance(item.get("id"), int)
                for item in class_names
            ):
                class_ids = {item["id"] for item in class_names}
            elif isinstance(class_names, list):
                class_ids = set(range(len(class_names)))
            elif isinstance(class_names, dict):
                class_ids = {int(key) for key in class_names}
            else:
                raise TypeError("class_names is not a list or map")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RustReplayConfigurationError(
                "the ONNX export manifest has no readable class map"
            ) from exc
        selected_ids = [self.ball_class_id]
        if self.cue_class_id is not None:
            selected_ids.append(self.cue_class_id)
        if any(
            not isinstance(class_id, int) or isinstance(class_id, bool)
            for class_id in selected_ids
        ):
            raise RustReplayConfigurationError("tracking class IDs must be integers")
        if any(class_id not in class_ids for class_id in selected_ids):
            raise RustReplayConfigurationError(
                "configured tracking class IDs must exist in the ONNX manifest"
            )
        if self.cue_class_id == self.ball_class_id:
            raise RustReplayConfigurationError(
                "ball and cue class IDs must be different"
            )
        if not self.input_path.exists():
            raise RustReplayConfigurationError("configured replay input does not exist")
        if (
            self.input_path.is_file()
            and self.input_path.suffix.lower() not in _IMAGE_SUFFIXES
        ):
            raise RustReplayConfigurationError(
                "Rust recorded replay accepts only PNG/JPEG files or directories"
            )
        replay_image_count = _count_replay_images(self.input_path)
        if replay_image_count == 0:
            raise RustReplayConfigurationError(
                "configured replay directory contains no PNG/JPEG images"
            )
        if (
            not isinstance(self.image_size, int)
            or isinstance(self.image_size, bool)
            or self.image_size <= 0
            or self.image_size % 32 != 0
        ):
            raise RustReplayConfigurationError(
                "image_size must be a positive multiple of 32"
            )
        for label, number, minimum, maximum in (
            ("confidence", self.confidence, 0.0, 1.0),
            ("iou", self.iou, 0.0, 1.0),
        ):
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or not minimum <= number <= maximum
            ):
                raise RustReplayConfigurationError(
                    f"{label} must be finite and between zero and one"
                )
        self._require_positive_finite("replay_rate_fps", self.replay_rate_fps)
        if (
            type(self.public_state_delivery_schema_version) is not int
            or self.public_state_delivery_schema_version != 1
        ):
            raise RustReplayConfigurationError(
                "public_state_delivery_schema_version must select supported v1"
            )
        self._require_positive_finite("association_gate_px", self.association_gate_px)
        for name in (
            "warmup_runs",
            "replay_repeat_count",
            "maximum_replay_frames",
            "association_prediction_horizon_ms",
            "lost_after_ms",
            "maximum_track_age_ms",
            "maximum_active_tracks",
            "recent_observation_capacity",
            "maximum_message_bytes",
            "connect_timeout_ms",
            "write_timeout_ms",
            "reconnect_initial_delay_ms",
            "reconnect_maximum_delay_ms",
            "maximum_detection_batch_output_bytes",
        ):
            number = getattr(self, name)
            lower_bound = 0 if name == "warmup_runs" else 1
            if (
                not isinstance(number, int)
                or isinstance(number, bool)
                or number < lower_bound
            ):
                raise RustReplayConfigurationError(
                    f"{name} must be an integer >= {lower_bound}"
                )
        if self.maximum_message_bytes > 0xFFFFFFFF:
            raise RustReplayConfigurationError(
                "maximum_message_bytes must fit the scene transport frame length"
            )
        if self.maximum_replay_frames > 0xFFFFFFFF:
            raise RustReplayConfigurationError(
                "maximum_replay_frames must fit the Rust replay frame identity"
            )
        scheduled_frames = replay_image_count * self.replay_repeat_count
        if scheduled_frames > self.maximum_replay_frames:
            raise RustReplayConfigurationError(
                "replay input image count multiplied by replay_repeat_count "
                "exceeds maximum_replay_frames"
            )
        if self.reconnect_maximum_delay_ms < self.reconnect_initial_delay_ms:
            raise RustReplayConfigurationError(
                "maximum reconnect delay cannot be shorter than its initial delay"
            )
        for name in (
            "read_timeout_seconds",
            "source_silence_timeout_seconds",
            "startup_timeout_seconds",
            "stop_timeout_seconds",
            "output_limit_poll_interval_seconds",
        ):
            self._require_positive_finite(name, getattr(self, name))
        if (
            not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or self.port < 0
            or self.port > 65535
        ):
            raise RustReplayConfigurationError("scene ingress port is out of range")
        for name in ("configuration_revision", "rectification_revision"):
            revision = getattr(self, name)
            if (
                not isinstance(revision, int)
                or isinstance(revision, bool)
                or revision < 0
            ):
                raise RustReplayConfigurationError(
                    f"{name} must be a non-negative integer"
                )
        if not self.camera_profile_id.strip():
            raise RustReplayConfigurationError("camera_profile_id must not be empty")
        if self.calibration_bundle_path is None:
            if any(
                value is not None
                for value in (
                    self.applied_calibration_revision,
                    self.applied_rectification_revision,
                    self.applied_camera_profile_id,
                )
            ):
                raise RustReplayConfigurationError(
                    "calibration application metadata requires a calibration bundle"
                )
        else:
            if not self.calibration_bundle_path.is_file():
                raise RustReplayConfigurationError(
                    "configured calibration bundle does not exist"
                )
            if (
                any(
                    not isinstance(value, int) or isinstance(value, bool) or value < 0
                    for value in (
                        self.applied_calibration_revision,
                        self.applied_rectification_revision,
                    )
                )
                or self.applied_calibration_revision is None
                or self.applied_rectification_revision is None
                or not self.applied_camera_profile_id
            ):
                raise RustReplayConfigurationError(
                    "calibrated replay requires camera profile and applied revisions"
                )
            if self.applied_camera_profile_id != self.camera_profile_id:
                raise RustReplayConfigurationError(
                    "tracking and calibration camera profiles must match"
                )
            if self.applied_rectification_revision != self.rectification_revision:
                raise RustReplayConfigurationError(
                    "tracking and applied rectification revisions must match"
                )

        # SceneIngressConfig validates loopback-only binding and enough message
        # capacity for the identity-scoped hello/ready handshake.
        SceneIngressConfig(
            bind_address=self.bind_address,
            port=self.port,
            maximum_message_bytes=self.maximum_message_bytes,
            read_timeout_seconds=self.read_timeout_seconds,
        )

    @staticmethod
    def _require_positive_finite(name: str, value: Any) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise RustReplayConfigurationError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class RustReplayStatus:
    """Bounded public summary of the selected replay provider lifecycle."""

    mode: str
    process_started: bool
    process_running: bool
    first_observation_received: bool
    process_return_code: int | None
    live_camera: bool
    preview: str
    predictions: str
    ingress_accepted_snapshots: int
    ingress_queue_replacements: int
    state_snapshots_published: int
    process_status: str
    source_health: str
    last_observation_age_seconds: float | None
    source_silence_timeout_seconds: float
    stderr_tail: str


class RustReplaySupervisor:
    """Run the finite Rust replay producer and its Python publication gateway."""

    def __init__(
        self,
        settings: RustReplaySettings,
        publisher: SceneSnapshotStatePublisher,
    ) -> None:
        settings.validate()
        self._settings = settings
        self._publisher = publisher
        ingress_settings = SceneIngressConfig(
            bind_address=settings.bind_address,
            port=settings.port,
            maximum_message_bytes=settings.maximum_message_bytes,
            read_timeout_seconds=settings.read_timeout_seconds,
            accept_public_state_delivery=True,
        )
        self._ingress = SceneSnapshotIngress(ingress_settings)
        self._worker = SceneSnapshotStateWorker(
            self._ingress,
            publisher,
            source_silence_timeout_seconds=settings.source_silence_timeout_seconds,
        )
        self._process: asyncio.subprocess.Process | None = None
        self._process_wait_task: asyncio.Task[int] | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._output_limit_task: asyncio.Task[None] | None = None
        self._stdout_tail = bytearray()
        self._stderr_tail = bytearray()
        self._output_directory: tempfile.TemporaryDirectory[str] | None = None
        self._batch_output_path: Path | None = None
        self._failure: str | None = None
        self._failure_event = asyncio.Event()
        self._started = False
        self._closed = False
        self._first_observation_received = False

    @property
    def status(self) -> RustReplayStatus:
        """Return process, readiness and bounded ingress counters."""
        process = self._process
        process_return_code = process.returncode if process is not None else None
        if process is None:
            process_status = "not_started"
        elif self._failure is not None:
            process_status = "failed"
        elif process_return_code is None:
            process_status = "running"
        elif process_return_code == 0:
            process_status = "completed"
        else:
            process_status = "failed"
        worker_stats = self._worker.stats
        source_health = (
            "process_failed"
            if process_status == "failed"
            else worker_stats.source_health
        )
        return RustReplayStatus(
            mode="recorded_replay",
            process_started=process is not None,
            process_running=process is not None and process.returncode is None,
            first_observation_received=self._first_observation_received,
            process_return_code=process_return_code,
            live_camera=False,
            preview="unavailable",
            predictions="unavailable",
            ingress_accepted_snapshots=self._ingress.stats.accepted_snapshots,
            ingress_queue_replacements=self._ingress.stats.queue_replacements,
            state_snapshots_published=self._worker.stats.published,
            process_status=process_status,
            source_health=source_health,
            last_observation_age_seconds=(worker_stats.last_observation_age_seconds),
            source_silence_timeout_seconds=(
                worker_stats.source_silence_timeout_seconds
            ),
            stderr_tail=self._stderr_tail.decode("utf-8", errors="replace"),
        )

    async def start(self) -> RustReplayStatus:
        """Bind ingress, start publisher/process and await one usable location."""
        if self._started:
            return self.status
        if self._closed:
            raise RuntimeError("a closed Rust replay supervisor cannot restart")
        self._started = True
        await self._worker.start()
        address = self._ingress.address
        self._output_directory = tempfile.TemporaryDirectory(
            prefix="billiards-rust-replay-"
        )
        self._batch_output_path = Path(self._output_directory.name) / "batches.jsonl"
        engine_instance_id = str(uuid.uuid4())
        source_session_id = str(uuid.uuid4())
        arguments = self._build_arguments(
            address,
            self._batch_output_path,
            engine_instance_id=engine_instance_id,
            source_session_id=source_session_id,
        )
        # The executable, model/input files, and working directory are all
        # passed explicitly. Keep the child environment minimal, but preserve
        # the dynamic-library search path when OpenVINO is explicitly selected:
        # the native OpenVINO C API binding loads the vendor runtime at process
        # startup on Linux or macOS.
        # Do not forward unrelated application, developer, or credential
        # environment values.
        environment = {
            key: os.environ[key]
            for key in _RUST_CHILD_ENVIRONMENT_KEYS
            if key in os.environ
        }
        if self._settings.provider == "openvino":
            environment.update(
                {
                    key: os.environ[key]
                    for key in _OPENVINO_CHILD_ENVIRONMENT_KEYS
                    if key in os.environ
                }
            )

        try:
            logger.info(
                "Starting Rust recorded replay with executable %s and input %s",
                self._settings.executable_path,
                self._settings.input_path,
            )
            self._process = await asyncio.create_subprocess_exec(
                str(self._settings.executable_path),
                *arguments,
                cwd=self._settings.project_root,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._process_wait_task = asyncio.create_task(
                self._wait_for_process(), name="rust-replay-process-wait"
            )
            assert self._process.stdout is not None
            assert self._process.stderr is not None
            self._stdout_task = asyncio.create_task(
                self._drain_output(self._process.stdout, self._stdout_tail),
                name="rust-replay-stdout-drain",
            )
            self._stderr_task = asyncio.create_task(
                self._drain_output(self._process.stderr, self._stderr_tail),
                name="rust-replay-stderr-drain",
            )
            self._output_limit_task = asyncio.create_task(
                self._watch_batch_output_limit(),
                name="rust-replay-batch-output-limit",
            )
            await self._await_first_observation()
            self._first_observation_received = True
            return self.status
        except Exception:
            await self._stop()
            raise

    async def close(self) -> None:
        """Stop and reap the child before closing its loopback ingress."""
        if self._closed:
            return
        self._closed = True
        await self._stop()

    def _build_arguments(
        self,
        address: tuple[str, int],
        batch_output_path: Path,
        *,
        engine_instance_id: str,
        source_session_id: str,
    ) -> list[str]:
        """Build a shell-free command from explicit, validated settings."""
        settings = self._settings
        arguments = [
            "--onnx",
            str(settings.onnx_path),
            "--source-checkpoint",
            str(settings.source_checkpoint_path),
            "--input",
            str(settings.input_path),
            "--provider",
            settings.provider,
            "--image-size",
            str(settings.image_size),
            "--confidence",
            str(settings.confidence),
            "--iou",
            str(settings.iou),
            "--warmup-runs",
            str(settings.warmup_runs),
            "--replay-rate-fps",
            str(settings.replay_rate_fps),
            "--replay-repeat-count",
            str(settings.replay_repeat_count),
            "--maximum-replay-frames",
            str(settings.maximum_replay_frames),
            "--source-image-space",
            settings.source_image_space,
            "--detection-batches-jsonl",
            str(batch_output_path),
            "--engine-instance-id",
            engine_instance_id,
            "--source-session-id",
            source_session_id,
            "--scene-snapshots-tcp",
            f"{address[0]}:{address[1]}",
            "--public-state-delivery-schema-version",
            str(settings.public_state_delivery_schema_version),
            "--scene-snapshots-max-message-bytes",
            str(settings.maximum_message_bytes),
            "--scene-snapshots-connect-timeout-ms",
            str(settings.connect_timeout_ms),
            "--scene-snapshots-write-timeout-ms",
            str(settings.write_timeout_ms),
            "--scene-snapshots-reconnect-initial-delay-ms",
            str(settings.reconnect_initial_delay_ms),
            "--scene-snapshots-reconnect-maximum-delay-ms",
            str(settings.reconnect_maximum_delay_ms),
            "--tracking-ball-class-id",
            str(settings.ball_class_id),
            "--tracking-camera-profile-id",
            settings.camera_profile_id,
            "--tracking-association-gate-px",
            str(settings.association_gate_px),
            "--tracking-association-prediction-horizon-ms",
            str(settings.association_prediction_horizon_ms),
            "--tracking-lost-after-ms",
            str(settings.lost_after_ms),
            "--tracking-maximum-track-age-ms",
            str(settings.maximum_track_age_ms),
            "--tracking-maximum-active-tracks",
            str(settings.maximum_active_tracks),
            "--tracking-recent-observation-capacity",
            str(settings.recent_observation_capacity),
            "--tracking-configuration-revision",
            str(settings.configuration_revision),
            "--tracking-rectification-revision",
            str(settings.rectification_revision),
        ]
        if settings.cue_class_id is not None:
            arguments.extend(("--tracking-cue-class-id", str(settings.cue_class_id)))
        if settings.calibration_bundle_path is not None:
            assert settings.applied_camera_profile_id is not None
            assert settings.applied_calibration_revision is not None
            assert settings.applied_rectification_revision is not None
            arguments.extend(
                (
                    "--calibration-bundle",
                    str(settings.calibration_bundle_path),
                    "--applied-camera-profile-id",
                    settings.applied_camera_profile_id,
                    "--applied-calibration-revision",
                    str(settings.applied_calibration_revision),
                    "--applied-rectification-revision",
                    str(settings.applied_rectification_revision),
                )
            )
        return arguments

    async def _await_first_observation(self) -> None:
        """Do not report ready on bind or process start alone."""
        deadline = (
            asyncio.get_running_loop().time() + self._settings.startup_timeout_seconds
        )
        observation_task = asyncio.create_task(
            self._worker.wait_for_first_observation(),
            name="rust-replay-first-observation",
        )
        failure_task = asyncio.create_task(self._failure_event.wait())
        waiters: set[asyncio.Task[Any]] = {observation_task, failure_task}
        if self._process_wait_task is not None:
            waiters.add(self._process_wait_task)
        try:
            done, _ = await asyncio.wait(
                waiters,
                timeout=self._settings.startup_timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # A successful scene does not excuse a failed required artifact.
            # Check the failure first when both conditions complete together.
            if self._failure is not None:
                raise RustReplayStartupError(self._failure)
            if failure_task in done:
                raise RustReplayStartupError(self._failure or "replay output failed")
            if observation_task in done:
                return
            if self._process_wait_task in done:
                return_code = self._process_wait_task.result()
                if return_code != 0:
                    stderr_tail = self._stderr_tail.decode(
                        "utf-8", errors="replace"
                    ).strip()
                    details = f": {stderr_tail}" if stderr_tail else ""
                    raise RustReplayStartupError(
                        "Rust replay exited with status "
                        f"{return_code} before a usable observation{details}"
                    )
                # A short recorded input can exit just after sending its final
                # scene; let the independent publisher drain before declaring
                # the stream empty.
                try:
                    await asyncio.wait_for(
                        observation_task,
                        timeout=max(
                            0.0,
                            deadline - asyncio.get_running_loop().time(),
                        ),
                    )
                    return
                except asyncio.TimeoutError:
                    pass
            raise RustReplayStartupError(
                "Rust replay did not publish a usable location before the readiness deadline"
            )
        finally:
            for task in (observation_task, failure_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(observation_task, failure_task, return_exceptions=True)

    async def _drain_output(
        self, stream: asyncio.StreamReader, destination: bytearray
    ) -> None:
        """Drain continuously to avoid pipe backpressure while retaining a tail."""
        while chunk := await stream.read(8192):
            destination.extend(chunk)
            excess = len(destination) - _OUTPUT_TAIL_CAPACITY_BYTES
            if excess > 0:
                del destination[:excess]

    async def _watch_batch_output_limit(self) -> None:
        """Stop a replay before its required batch artifact consumes unbounded disk."""
        while self._process is not None and self._process.returncode is None:
            await asyncio.sleep(self._settings.output_limit_poll_interval_seconds)
            if self._check_batch_output_limit():
                return

    async def _wait_for_process(self) -> int:
        """Check the final artifact size even if the child exits between polls."""
        process = self._process
        if process is None:
            raise RuntimeError("Rust replay process was not started")
        return_code = await process.wait()
        self._check_batch_output_limit()
        return return_code

    def _check_batch_output_limit(self) -> bool:
        """Record and stop on an oversized artifact; return whether it failed."""
        path = self._batch_output_path
        if path is None:
            return False
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        except OSError:
            self._failure = "Rust replay detection-batch output could not be inspected"
            self._failure_event.set()
            logger.exception("%s", self._failure)
            self._terminate_process()
            return True
        if size <= self._settings.maximum_detection_batch_output_bytes:
            return False

        self._failure = (
            "Rust replay detection-batch output exceeded the configured byte limit"
        )
        self._failure_event.set()
        logger.error("%s", self._failure)
        self._terminate_process()
        return True

    def _terminate_process(self) -> None:
        """Terminate the child if it is still alive, tolerating an exit race."""
        process = self._process
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()

    async def _stop(self) -> None:
        """Idempotent cleanup used for startup failures and normal shutdown."""
        process = self._process
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=self._settings.stop_timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        tasks = [
            task
            for task in (
                self._process_wait_task,
                self._stdout_task,
                self._stderr_task,
                self._output_limit_task,
            )
            if task is not None
        ]
        for task in tasks:
            if task is not self._process_wait_task and not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._worker.close()
        if self._output_directory is not None:
            self._output_directory.cleanup()
            self._output_directory = None
