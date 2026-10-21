//! Opt-in deterministic transport microbenchmark.
//!
//! This experiment compares a bounded in-process handoff with transferring
//! identical, setup-generated RGB8 payloads over a supervised child process's
//! stdout pipe. Payload generation and SHA-256 verification happen before the
//! measured interval. It does not exercise a camera, detector, production `CaptureSource`,
//! capture mailbox, or a client-facing scene transport.

use std::io::{self, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

const READY_MAGIC: &[u8; 8] = b"BTXPORT1";
const FRAME_MAGIC: &[u8; 4] = b"FRM1";
const WARMUP_DONE: &[u8; 8] = b"WARMEND1";
const MEASUREMENT_DONE: &[u8; 8] = b"MEASDON1";
const PROTOCOL_VERSION: u16 = 1;
const READY_RECORD_BYTES: usize = 116;
const FRAME_HEADER_BYTES: usize = 20;
const CONTROL_WARMUP: u8 = b'W';
const CONTROL_MEASUREMENT: u8 = b'M';
const PHASE_WARMUP: u8 = 1;
const PHASE_MEASUREMENT: u8 = 2;

/// Explicit bounds and workload supplied by the user for one audit invocation.
#[derive(Debug, Clone, Copy)]
pub struct BenchmarkConfig {
    pub width_px: u32,
    pub height_px: u32,
    pub warmup_frames: u64,
    pub measured_frames: u64,
    pub repetitions: u32,
    pub sampling_limit: usize,
    pub seed: u64,
    pub max_payload_bytes: usize,
    /// Maximum generated frames per arm over all repetitions.
    pub max_total_frames: u64,
    /// Maximum timing/hash samples retained over both arms and all repetitions.
    pub max_total_samples: usize,
}

impl BenchmarkConfig {
    pub fn validate(self) -> Result<ValidatedConfig> {
        ensure!(self.width_px > 0, "--width-px must be positive");
        ensure!(self.height_px > 0, "--height-px must be positive");
        ensure!(
            self.measured_frames > 0,
            "--measured-frames must be positive"
        );
        ensure!(self.repetitions > 0, "--repetitions must be positive");
        ensure!(self.sampling_limit > 0, "--sampling-limit must be positive");
        ensure!(
            self.max_payload_bytes > 0,
            "--max-payload-bytes must be positive"
        );
        ensure!(
            self.max_total_frames > 0,
            "--max-total-frames must be positive"
        );
        ensure!(
            self.max_total_samples > 0,
            "--max-total-samples must be positive"
        );

        let payload_bytes_u64 = u64::from(self.width_px)
            .checked_mul(u64::from(self.height_px))
            .and_then(|pixels| pixels.checked_mul(3))
            .context("RGB8 payload byte count overflow")?;
        let payload_bytes = usize::try_from(payload_bytes_u64)
            .context("RGB8 payload byte count does not fit this process address space")?;
        let application_frame_buffer_bytes_per_arm = payload_bytes
            .checked_mul(2)
            .context("two-frame application buffer byte count overflow")?;
        ensure!(
            payload_bytes <= self.max_payload_bytes,
            "RGB8 payload is {payload_bytes} bytes, above --max-payload-bytes ({})",
            self.max_payload_bytes
        );
        ensure!(
            payload_bytes <= u32::MAX as usize,
            "RGB8 payload exceeds the protocol's u32 frame-length field"
        );

        let frames_per_repetition = self
            .warmup_frames
            .checked_add(self.measured_frames)
            .context("warmup plus measured frame count overflow")?;
        let frames_per_arm = frames_per_repetition
            .checked_mul(u64::from(self.repetitions))
            .context("total frame count overflow")?;
        ensure!(
            frames_per_arm <= self.max_total_frames,
            "workload requires {frames_per_arm} frames per arm, above --max-total-frames ({})",
            self.max_total_frames
        );
        let measured_frames_usize = usize::try_from(self.measured_frames)
            .context("measured frame count does not fit this process address space")?;
        ensure!(
            self.sampling_limit <= measured_frames_usize,
            "--sampling-limit cannot exceed --measured-frames"
        );
        let retained_samples_per_arm = self
            .sampling_limit
            .checked_mul(self.repetitions as usize)
            .context("sample count overflow")?;
        let retained_samples = retained_samples_per_arm
            .checked_mul(2)
            .context("sample count overflow")?;
        ensure!(
            retained_samples <= self.max_total_samples,
            "configured sampling retains up to {retained_samples} samples, above --max-total-samples ({})",
            self.max_total_samples
        );

        Ok(ValidatedConfig {
            requested: self,
            payload_bytes,
            application_frame_buffer_bytes_per_arm,
            frames_per_arm,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ValidatedConfig {
    requested: BenchmarkConfig,
    payload_bytes: usize,
    application_frame_buffer_bytes_per_arm: usize,
    frames_per_arm: u64,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkReport {
    pub schema_version: u8,
    pub experiment: &'static str,
    pub scope: &'static str,
    pub metadata: RuntimeMetadata,
    pub workload: WorkloadMetadata,
    pub measurements_not_taken: MeasurementLimitations,
    pub repetitions: Vec<RepetitionResult>,
}

#[derive(Debug, Serialize)]
pub struct RuntimeMetadata {
    pub crate_version: &'static str,
    pub rustc_version: Option<&'static str>,
    pub rustc_version_status: &'static str,
    pub build_profile: &'static str,
    pub target_os: &'static str,
    pub target_arch: &'static str,
    pub target_family: &'static str,
    pub logical_cpu_count: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct WorkloadMetadata {
    pub pixel_format: &'static str,
    pub width_px: u32,
    pub height_px: u32,
    pub payload_bytes_per_frame: usize,
    pub warmup_frames_per_repetition: u64,
    pub measured_frames_per_repetition: u64,
    pub repetitions: u32,
    pub seed: u64,
    pub max_payload_bytes: usize,
    pub application_frame_buffer_count_per_arm: usize,
    pub application_frame_buffer_bytes_per_arm: usize,
    pub max_total_frames_per_arm: u64,
    pub max_total_samples: usize,
    pub sampling_limit_per_arm_per_repetition: usize,
    pub frame_records_per_arm_over_all_repetitions: u64,
    pub payload_generator: &'static str,
    pub payload_content_model: &'static str,
    pub payload_preparation_and_sha256: &'static str,
    pub timed_operations: &'static str,
    pub sampling_method: &'static str,
    pub producer_consumer_model: &'static str,
}

#[derive(Debug, Serialize)]
pub struct MeasurementLimitations {
    pub camera_source_to_parent_latency: UnavailableMeasurement,
    pub detector_accuracy_and_latency: UnavailableMeasurement,
    pub process_cpu_time: UnavailableMeasurement,
    pub resident_memory: UnavailableMeasurement,
    pub operating_system_pipe_capacity: UnavailableMeasurement,
    pub capture_mailbox_and_client_delivery: UnavailableMeasurement,
}

#[derive(Debug, Serialize)]
pub struct UnavailableMeasurement {
    pub status: &'static str,
    pub reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct RepetitionResult {
    pub repetition: u32,
    pub in_process: ArmResult,
    pub supervised_stdout_pipe: ArmResult,
}

#[derive(Debug, Serialize)]
pub struct ArmResult {
    pub arm: &'static str,
    pub data_movement_model: &'static str,
    pub measured_frames: u64,
    pub measured_payload_bytes: u64,
    pub elapsed_ns: u64,
    pub frames_per_second: f64,
    /// Logical frame payload volume divided by elapsed time. This is not
    /// physical memory bandwidth, especially for ownership-only handoff.
    pub logical_payload_mib_per_second: f64,
    pub frame_payload_sha256: String,
    pub sampled_frame_completion: SampledFrameCompletion,
    pub startup_to_ready_ns: MeasurementValue<u64>,
}

#[derive(Debug, Serialize)]
pub struct SampledFrameCompletion {
    pub clock: &'static str,
    pub basis: &'static str,
    pub sampled_frames: usize,
    pub sampling_limit: usize,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub samples: Vec<TimingSample>,
}

#[derive(Debug, Serialize)]
pub struct TimingSample {
    pub frame_index: u64,
    pub completion_ns: u64,
    pub frame_payload_sha256: String,
}

#[derive(Debug, Serialize)]
pub struct MeasurementValue<T> {
    pub status: &'static str,
    pub value: Option<T>,
    pub unit: &'static str,
    pub reason: Option<&'static str>,
}

impl<T> MeasurementValue<T> {
    fn available(value: T, unit: &'static str) -> Self {
        Self {
            status: "available",
            value: Some(value),
            unit,
            reason: None,
        }
    }
}

#[derive(Debug, Clone)]
struct ReadyRecord {
    width_px: u32,
    height_px: u32,
    payload_bytes: u64,
    warmup_frames: u64,
    measured_frames: u64,
    seed: u64,
    payload_sha256: String,
}

#[derive(Debug, Clone, Copy)]
struct FrameRecord {
    phase: u8,
    index: u64,
    payload_bytes: u32,
}

#[derive(Debug)]
struct Reservoir {
    capacity: usize,
    observed: u64,
    state: u64,
    samples: Vec<TimingSample>,
}

impl Reservoir {
    fn new(capacity: usize, seed: u64) -> Result<Self> {
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(capacity)
            .context("could not reserve bounded latency sample buffer")?;
        Ok(Self {
            capacity,
            observed: 0,
            state: seed ^ 0xA076_1D64_78BD_642F,
            samples,
        })
    }

    fn observe(&mut self, frame_index: u64, completion_ns: u64, frame_payload_sha256: &str) {
        self.observed += 1;
        if self.samples.len() < self.capacity {
            self.samples.push(TimingSample {
                frame_index,
                completion_ns,
                frame_payload_sha256: frame_payload_sha256.to_owned(),
            });
            return;
        }

        let selected = next_random(&mut self.state) % self.observed;
        if selected < self.capacity as u64 {
            self.samples[selected as usize] = TimingSample {
                frame_index,
                completion_ns,
                frame_payload_sha256: frame_payload_sha256.to_owned(),
            };
        }
    }

    fn finish(mut self, sampling_limit: usize) -> SampledFrameCompletion {
        self.samples
            .sort_unstable_by_key(|sample| sample.completion_ns);
        let p50_index = nearest_rank_index(self.samples.len(), 50);
        let p95_index = nearest_rank_index(self.samples.len(), 95);
        let p99_index = nearest_rank_index(self.samples.len(), 99);
        SampledFrameCompletion {
            clock: "parent_process_monotonic",
            basis: "parent-side frame receipt and identity validation; excludes buffer recycling, payload generation, and SHA-256",
            sampled_frames: self.samples.len(),
            sampling_limit,
            p50_ns: self.samples[p50_index].completion_ns,
            p95_ns: self.samples[p95_index].completion_ns,
            p99_ns: self.samples[p99_index].completion_ns,
            samples: self.samples,
        }
    }
}

pub fn run_benchmark(config: BenchmarkConfig) -> Result<BenchmarkReport> {
    let config = config.validate()?;
    let mut repetitions = Vec::new();
    repetitions
        .try_reserve_exact(config.requested.repetitions as usize)
        .context("could not reserve repetition result buffer")?;

    for repetition in 0..config.requested.repetitions {
        let (in_process, supervised_stdout_pipe) = if repetition % 2 == 0 {
            let in_process = run_in_process(config)?;
            let mut helper = helper_command(config);
            let supervised_stdout_pipe = run_stdout_pipe(config, &mut helper)?;
            (in_process, supervised_stdout_pipe)
        } else {
            let mut helper = helper_command(config);
            let supervised_stdout_pipe = run_stdout_pipe(config, &mut helper)?;
            let in_process = run_in_process(config)?;
            (in_process, supervised_stdout_pipe)
        };
        ensure!(
            in_process.frame_payload_sha256 == supervised_stdout_pipe.frame_payload_sha256,
            "payload SHA-256 mismatch between in-process and stdout-pipe arms in repetition {}",
            repetition + 1
        );
        repetitions.push(RepetitionResult {
            repetition: repetition + 1,
            in_process,
            supervised_stdout_pipe,
        });
    }

    Ok(BenchmarkReport {
        schema_version: 3,
        experiment: "prepared_rgb8_in_process_vs_supervised_stdout_pipe",
        scope: "isolated transport microbenchmark; not a camera, detector, CaptureSource mailbox, or client-delivery benchmark",
        metadata: runtime_metadata(),
        workload: WorkloadMetadata {
            pixel_format: "RGB8",
            width_px: config.requested.width_px,
            height_px: config.requested.height_px,
            payload_bytes_per_frame: config.payload_bytes,
            warmup_frames_per_repetition: config.requested.warmup_frames,
            measured_frames_per_repetition: config.requested.measured_frames,
            repetitions: config.requested.repetitions,
            seed: config.requested.seed,
            max_payload_bytes: config.requested.max_payload_bytes,
            application_frame_buffer_count_per_arm: 2,
            application_frame_buffer_bytes_per_arm: config
                .application_frame_buffer_bytes_per_arm,
            max_total_frames_per_arm: config.requested.max_total_frames,
            max_total_samples: config.requested.max_total_samples,
            sampling_limit_per_arm_per_repetition: config.requested.sampling_limit,
            frame_records_per_arm_over_all_repetitions: config.frames_per_arm,
            payload_generator: "deterministic_xorshift64star_v1",
            payload_content_model: "one deterministic RGB8 payload template repeated for every frame record in each arm",
            payload_preparation_and_sha256: "generated and SHA-256 checked during arm setup/readiness; excluded from measured frame loops",
            timed_operations: "in-process Vec ownership handoff versus helper frame serialization and parent pipe reads into a reusable buffer; payload generation and SHA-256 are excluded",
            sampling_method: "reservoir_sampling_algorithm_r",
            producer_consumer_model: "in_process_producer_thread_to_capacity_one_sync_channel_to_consumer; helper_process_writer_to_parent_consumer_over_OS_stdout_pipe; two_owned_RGB8_buffers_per_arm",
        },
        measurements_not_taken: MeasurementLimitations {
            camera_source_to_parent_latency: unavailable(
                "No camera source or comparable camera clock is used by this experiment.",
            ),
            detector_accuracy_and_latency: unavailable(
                "No detector or image interpretation is invoked by this experiment.",
            ),
            process_cpu_time: unavailable(
                "CPU time is not sampled by this portable microbenchmark.",
            ),
            resident_memory: unavailable(
                "Per-process RSS is not sampled by this portable microbenchmark.",
            ),
            operating_system_pipe_capacity: unavailable(
                "The pipe buffer is bounded by the operating system but its active capacity is not queried by this portable microbenchmark.",
            ),
            capture_mailbox_and_client_delivery: unavailable(
                "The production capture mailbox and client delivery path are not exercised.",
            ),
        },
        repetitions,
    })
}

fn unavailable(reason: &'static str) -> UnavailableMeasurement {
    UnavailableMeasurement {
        status: "unavailable",
        reason,
    }
}

fn runtime_metadata() -> RuntimeMetadata {
    RuntimeMetadata {
        crate_version: env!("CARGO_PKG_VERSION"),
        rustc_version: option_env!("RUSTC_VERSION"),
        rustc_version_status: if option_env!("RUSTC_VERSION").is_some() {
            "embedded_at_build_time"
        } else {
            "unavailable_not_embedded_at_build_time"
        },
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        target_os: std::env::consts::OS,
        target_arch: std::env::consts::ARCH,
        target_family: std::env::consts::FAMILY,
        logical_cpu_count: std::thread::available_parallelism()
            .ok()
            .map(std::num::NonZeroUsize::get),
    }
}

fn run_in_process(config: ValidatedConfig) -> Result<ArmResult> {
    let expected_frame_payload_sha256 = frame_payload_sha256(config)?;
    let (frame_sender, frame_receiver) = mpsc::sync_channel(1);
    let (free_buffer_sender, free_buffer_receiver) = mpsc::sync_channel(2);
    free_buffer_sender
        .send(allocate_payload(config.payload_bytes)?)
        .map_err(|_| anyhow::anyhow!("could not initialize in-process frame buffers"))?;
    free_buffer_sender
        .send(allocate_payload(config.payload_bytes)?)
        .map_err(|_| anyhow::anyhow!("could not initialize in-process frame buffers"))?;
    let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
    let (control_sender, control_receiver) = mpsc::sync_channel(1);
    let producer_config = config;
    let startup_start = Instant::now();
    let producer = thread::Builder::new()
        .name("capture-benchmark-in-process-producer".to_owned())
        .spawn(move || {
            run_in_process_producer(
                producer_config,
                frame_sender,
                free_buffer_receiver,
                ready_sender,
                control_receiver,
            )
        })
        .context("could not start in-process benchmark producer")?;
    let (producer_frame_payload_sha256, first_buffer, second_buffer) = ready_receiver
        .recv()
        .map_err(|_| anyhow::anyhow!("in-process producer exited before readiness"))?;
    ensure!(
        producer_frame_payload_sha256 == expected_frame_payload_sha256
            && sha256_hex(&first_buffer) == expected_frame_payload_sha256
            && sha256_hex(&second_buffer) == expected_frame_payload_sha256,
        "in-process producer readiness payload SHA-256 does not match the requested seed and geometry"
    );
    free_buffer_sender
        .send(first_buffer)
        .and_then(|()| free_buffer_sender.send(second_buffer))
        .map_err(|_| anyhow::anyhow!("could not recycle prepared in-process buffers"))?;
    let startup_to_ready_ns = duration_ns(startup_start.elapsed())?;

    let measurement_result = (|| -> Result<(u64, Reservoir)> {
        control_sender
            .send(CONTROL_WARMUP)
            .map_err(|_| anyhow::anyhow!("in-process producer stopped before warmup"))?;
        consume_in_process_warmup(&frame_receiver, &free_buffer_sender, config)?;

        let mut reservoir = Reservoir::new(config.requested.sampling_limit, config.requested.seed)?;
        let start = Instant::now();
        control_sender
            .send(CONTROL_MEASUREMENT)
            .map_err(|_| anyhow::anyhow!("in-process producer stopped before measurement"))?;
        for frame_index in 0..config.requested.measured_frames {
            let frame_start = Instant::now();
            let frame = frame_receiver
                .recv()
                .context("in-process producer stopped before measured frame")?;
            let (absolute_index, payload) = receive_pipeline_frame(
                frame,
                PHASE_MEASUREMENT,
                config.requested.warmup_frames + frame_index,
            )?;
            let elapsed = duration_ns(frame_start.elapsed())?;
            reservoir.observe(
                absolute_index - config.requested.warmup_frames,
                elapsed,
                &expected_frame_payload_sha256,
            );
            if frame_index + 1 < config.requested.measured_frames {
                free_buffer_sender.send(payload).map_err(|_| {
                    anyhow::anyhow!("in-process producer stopped while recycling a frame")
                })?;
            }
        }
        match frame_receiver
            .recv()
            .context("in-process producer stopped before completion")?
        {
            PipelineMessage::MeasurementDone => {}
            other => anyhow::bail!("unexpected in-process message at measurement end: {other:?}"),
        }
        let elapsed_ns = duration_ns(start.elapsed())?;
        Ok((elapsed_ns, reservoir))
    })();

    // Close channels before joining on failure so the producer cannot remain
    // blocked waiting for a recycled frame or phase-control byte.
    drop(control_sender);
    drop(frame_receiver);
    drop(free_buffer_sender);
    let producer_result = producer
        .join()
        .map_err(|_| anyhow::anyhow!("in-process benchmark producer panicked"))?;
    producer_result?;
    let (elapsed_ns, reservoir) = measurement_result?;
    arm_result(
        "in_process_bounded_handoff",
        config,
        elapsed_ns,
        expected_frame_payload_sha256,
        reservoir,
        MeasurementValue::available(startup_to_ready_ns, "nanoseconds"),
    )
}

#[derive(Debug)]
enum PipelineMessage {
    Frame {
        phase: u8,
        index: u64,
        payload: Vec<u8>,
    },
    WarmupDone,
    MeasurementDone,
}

fn run_in_process_producer(
    config: ValidatedConfig,
    frame_sender: SyncSender<PipelineMessage>,
    free_buffer_receiver: Receiver<Vec<u8>>,
    ready_sender: SyncSender<(String, Vec<u8>, Vec<u8>)>,
    control_receiver: Receiver<u8>,
) -> Result<()> {
    let mut first = free_buffer_receiver
        .recv()
        .context("benchmark consumer stopped before first template buffer")?;
    fill_payload(&mut first, config.requested.seed, 0);
    let payload_sha256 = sha256_hex(&first);
    let mut second = free_buffer_receiver
        .recv()
        .context("benchmark consumer stopped before second template buffer")?;
    fill_payload(&mut second, config.requested.seed, 0);
    ready_sender
        .send((payload_sha256, first, second))
        .map_err(|_| anyhow::anyhow!("benchmark consumer exited before producer readiness"))?;
    receive_pipeline_control(&control_receiver, CONTROL_WARMUP, "warmup")?;
    for frame_index in 0..config.requested.warmup_frames {
        let payload = free_buffer_receiver
            .recv()
            .context("benchmark consumer stopped before warmup frame")?;
        frame_sender
            .send(PipelineMessage::Frame {
                phase: PHASE_WARMUP,
                index: frame_index,
                payload,
            })
            .map_err(|_| anyhow::anyhow!("benchmark consumer stopped during warmup"))?;
    }
    frame_sender
        .send(PipelineMessage::WarmupDone)
        .map_err(|_| anyhow::anyhow!("benchmark consumer stopped before warmup completion"))?;
    receive_pipeline_control(&control_receiver, CONTROL_MEASUREMENT, "measurement")?;
    for frame_index in 0..config.requested.measured_frames {
        let absolute_index = config.requested.warmup_frames + frame_index;
        let payload = free_buffer_receiver
            .recv()
            .context("benchmark consumer stopped before measured frame")?;
        frame_sender
            .send(PipelineMessage::Frame {
                phase: PHASE_MEASUREMENT,
                index: absolute_index,
                payload,
            })
            .map_err(|_| anyhow::anyhow!("benchmark consumer stopped during measurement"))?;
    }
    frame_sender
        .send(PipelineMessage::MeasurementDone)
        .map_err(|_| anyhow::anyhow!("benchmark consumer stopped before measurement completion"))?;
    Ok(())
}

fn consume_in_process_warmup(
    frame_receiver: &Receiver<PipelineMessage>,
    free_buffer_sender: &SyncSender<Vec<u8>>,
    config: ValidatedConfig,
) -> Result<()> {
    for frame_index in 0..config.requested.warmup_frames {
        let frame = frame_receiver
            .recv()
            .context("in-process producer stopped before warmup frame")?;
        let (received_index, payload) = receive_pipeline_frame(frame, PHASE_WARMUP, frame_index)?;
        debug_assert_eq!(received_index, frame_index);
        free_buffer_sender
            .send(payload)
            .map_err(|_| anyhow::anyhow!("in-process producer stopped during warmup"))?;
    }
    match frame_receiver
        .recv()
        .context("in-process producer stopped before warmup completion")?
    {
        PipelineMessage::WarmupDone => Ok(()),
        other => anyhow::bail!("unexpected in-process message at warmup end: {other:?}"),
    }
}

fn receive_pipeline_control(
    receiver: &Receiver<u8>,
    expected: u8,
    description: &str,
) -> Result<()> {
    let actual = receiver
        .recv()
        .with_context(|| format!("waiting for parent {description} control"))?;
    ensure!(
        actual == expected,
        "invalid parent control while waiting for {description}"
    );
    Ok(())
}

fn receive_pipeline_frame(
    message: PipelineMessage,
    expected_phase: u8,
    expected_index: u64,
) -> Result<(u64, Vec<u8>)> {
    match message {
        PipelineMessage::Frame {
            phase,
            index,
            payload,
        } => {
            ensure!(
                phase == expected_phase && index == expected_index,
                "in-process frame identity mismatch: got phase {phase} index {index}, expected phase {expected_phase} index {expected_index}"
            );
            Ok((index, payload))
        }
        other => {
            anyhow::bail!("unexpected in-process message for frame {expected_index}: {other:?}")
        }
    }
}

fn run_stdout_pipe(config: ValidatedConfig, command: &mut Command) -> Result<ArmResult> {
    let expected_frame_payload_sha256 = frame_payload_sha256(config)?;
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let startup_start = Instant::now();
    let mut child = ChildGuard::spawn(command)?;
    let ready = read_ready(child.stdout_mut())?;
    let startup_to_ready_ns = duration_ns(startup_start.elapsed())?;
    validate_ready(ready, config, &expected_frame_payload_sha256)?;

    let mut payload = allocate_payload(config.payload_bytes)?;
    send_control(child.stdin_mut(), CONTROL_WARMUP)?;
    for frame_index in 0..config.requested.warmup_frames {
        read_frame(child.stdout_mut(), &mut payload, PHASE_WARMUP, frame_index)?;
    }
    read_marker(child.stdout_mut(), WARMUP_DONE, "warmup completion")?;

    let mut reservoir = Reservoir::new(config.requested.sampling_limit, config.requested.seed)?;
    let start = Instant::now();
    send_control(child.stdin_mut(), CONTROL_MEASUREMENT)?;
    for frame_index in 0..config.requested.measured_frames {
        let frame_start = Instant::now();
        read_frame(
            child.stdout_mut(),
            &mut payload,
            PHASE_MEASUREMENT,
            config.requested.warmup_frames + frame_index,
        )?;
        let elapsed = duration_ns(frame_start.elapsed())?;
        reservoir.observe(frame_index, elapsed, &expected_frame_payload_sha256);
    }
    read_marker(
        child.stdout_mut(),
        MEASUREMENT_DONE,
        "measurement completion",
    )?;
    let elapsed_ns = duration_ns(start.elapsed())?;
    child.close_stdin();
    let status = child.wait()?;
    ensure!(
        status.success(),
        "capture transport helper exited with {status}"
    );

    arm_result(
        "supervised_child_stdout_pipe",
        config,
        elapsed_ns,
        expected_frame_payload_sha256,
        reservoir,
        MeasurementValue::available(startup_to_ready_ns, "nanoseconds"),
    )
}

fn arm_result(
    arm: &'static str,
    config: ValidatedConfig,
    elapsed_ns: u64,
    frame_payload_sha256: String,
    reservoir: Reservoir,
    startup_to_ready_ns: MeasurementValue<u64>,
) -> Result<ArmResult> {
    let measured_payload_bytes = u64::try_from(config.payload_bytes)
        .context("payload byte count does not fit in u64")?
        .checked_mul(config.requested.measured_frames)
        .context("measured payload byte count overflow")?;
    let elapsed_seconds = elapsed_ns as f64 / 1_000_000_000.0;
    ensure!(
        elapsed_seconds > 0.0,
        "measured duration rounded to zero nanoseconds"
    );
    Ok(ArmResult {
        arm,
        data_movement_model: if arm == "in_process_bounded_handoff" {
            "frame Vec ownership moves through a bounded in-process sync_channel; payload bytes are not copied by application code"
        } else {
            "helper writes each payload to an OS stdout pipe; parent reads into a reusable frame buffer; platform copy behavior is not inferred"
        },
        measured_frames: config.requested.measured_frames,
        measured_payload_bytes,
        elapsed_ns,
        frames_per_second: config.requested.measured_frames as f64 / elapsed_seconds,
        logical_payload_mib_per_second: (measured_payload_bytes as f64 / (1024.0 * 1024.0))
            / elapsed_seconds,
        frame_payload_sha256,
        sampled_frame_completion: reservoir.finish(config.requested.sampling_limit),
        startup_to_ready_ns,
    })
}

fn helper_command(config: ValidatedConfig) -> Command {
    let executable =
        std::env::current_exe().expect("the running benchmark executable path must be available");
    let mut command = Command::new(executable);
    command
        .arg("--internal-child")
        .arg("--width-px")
        .arg(config.requested.width_px.to_string())
        .arg("--height-px")
        .arg(config.requested.height_px.to_string())
        .arg("--warmup-frames")
        .arg(config.requested.warmup_frames.to_string())
        .arg("--measured-frames")
        .arg(config.requested.measured_frames.to_string())
        .arg("--repetitions")
        .arg(config.requested.repetitions.to_string())
        .arg("--sampling-limit")
        .arg(config.requested.sampling_limit.to_string())
        .arg("--seed")
        .arg(config.requested.seed.to_string())
        .arg("--max-payload-bytes")
        .arg(config.requested.max_payload_bytes.to_string())
        .arg("--max-total-frames")
        .arg(config.requested.max_total_frames.to_string())
        .arg("--max-total-samples")
        .arg(config.requested.max_total_samples.to_string());
    command
}

/// Child-side protocol implementation, selected only by the benchmark binary's
/// private `--internal-child` invocation.
pub fn run_helper_child(config: BenchmarkConfig) -> Result<()> {
    let config = config.validate()?;
    let mut payload = allocate_payload(config.payload_bytes)?;
    fill_payload(&mut payload, config.requested.seed, 0);
    let ready = ReadyRecord {
        width_px: config.requested.width_px,
        height_px: config.requested.height_px,
        payload_bytes: config.payload_bytes as u64,
        warmup_frames: config.requested.warmup_frames,
        measured_frames: config.requested.measured_frames,
        seed: config.requested.seed,
        payload_sha256: sha256_hex(&payload),
    };
    write_ready(&mut io::stdout().lock(), ready)?;

    let mut input = io::stdin().lock();
    read_control(&mut input, CONTROL_WARMUP, "warmup")?;
    let mut output = io::stdout().lock();
    for frame_index in 0..config.requested.warmup_frames {
        write_frame(&mut output, PHASE_WARMUP, frame_index, &payload)?;
    }
    output
        .write_all(WARMUP_DONE)
        .and_then(|()| output.flush())
        .context("could not write warmup completion marker")?;
    drop(output);

    read_control(&mut input, CONTROL_MEASUREMENT, "measurement")?;
    let mut output = io::stdout().lock();
    for frame_index in 0..config.requested.measured_frames {
        let absolute_index = config.requested.warmup_frames + frame_index;
        write_frame(&mut output, PHASE_MEASUREMENT, absolute_index, &payload)?;
    }
    output
        .write_all(MEASUREMENT_DONE)
        .and_then(|()| output.flush())
        .context("could not write measurement completion marker")?;
    Ok(())
}

fn write_ready<W: Write>(writer: &mut W, ready: ReadyRecord) -> Result<()> {
    writer.write_all(READY_MAGIC)?;
    writer.write_all(&PROTOCOL_VERSION.to_be_bytes())?;
    writer.write_all(&0_u16.to_be_bytes())?;
    writer.write_all(&ready.width_px.to_be_bytes())?;
    writer.write_all(&ready.height_px.to_be_bytes())?;
    writer.write_all(&ready.payload_bytes.to_be_bytes())?;
    writer.write_all(&ready.warmup_frames.to_be_bytes())?;
    writer.write_all(&ready.measured_frames.to_be_bytes())?;
    writer.write_all(&ready.seed.to_be_bytes())?;
    writer.write_all(ready.payload_sha256.as_bytes())?;
    writer.flush().context("could not flush helper readiness")?;
    Ok(())
}

fn read_ready<R: Read>(reader: &mut R) -> Result<ReadyRecord> {
    let mut bytes = [0_u8; READY_RECORD_BYTES];
    reader
        .read_exact(&mut bytes)
        .context("could not read complete helper readiness record")?;
    ensure!(
        &bytes[0..8] == READY_MAGIC,
        "invalid helper readiness magic"
    );
    ensure!(
        u16::from_be_bytes([bytes[8], bytes[9]]) == PROTOCOL_VERSION,
        "unsupported helper protocol version"
    );
    ensure!(
        bytes[10..12] == [0, 0],
        "helper readiness reserved bytes must be zero"
    );
    Ok(ReadyRecord {
        width_px: u32::from_be_bytes(bytes[12..16].try_into().expect("fixed slice")),
        height_px: u32::from_be_bytes(bytes[16..20].try_into().expect("fixed slice")),
        payload_bytes: u64::from_be_bytes(bytes[20..28].try_into().expect("fixed slice")),
        warmup_frames: u64::from_be_bytes(bytes[28..36].try_into().expect("fixed slice")),
        measured_frames: u64::from_be_bytes(bytes[36..44].try_into().expect("fixed slice")),
        seed: u64::from_be_bytes(bytes[44..52].try_into().expect("fixed slice")),
        payload_sha256: std::str::from_utf8(&bytes[52..116])
            .context("helper readiness payload SHA-256 is not ASCII")?
            .to_owned(),
    })
}

fn validate_ready(
    ready: ReadyRecord,
    config: ValidatedConfig,
    expected_frame_payload_sha256: &str,
) -> Result<()> {
    ensure!(
        ready.width_px == config.requested.width_px
            && ready.height_px == config.requested.height_px
            && ready.payload_bytes == config.payload_bytes as u64
            && ready.warmup_frames == config.requested.warmup_frames
            && ready.measured_frames == config.requested.measured_frames
            && ready.seed == config.requested.seed,
        "helper readiness does not match the requested workload"
    );
    ensure!(
        ready.payload_sha256 == expected_frame_payload_sha256,
        "helper readiness payload SHA-256 does not match the requested seed and geometry"
    );
    Ok(())
}

fn read_frame<R: Read>(
    reader: &mut R,
    payload: &mut [u8],
    expected_phase: u8,
    expected_index: u64,
) -> Result<FrameRecord> {
    let mut header_bytes = [0_u8; FRAME_HEADER_BYTES];
    reader
        .read_exact(&mut header_bytes)
        .with_context(|| format!("could not read frame {expected_index} header"))?;
    ensure!(
        &header_bytes[0..4] == FRAME_MAGIC,
        "invalid frame magic at frame {expected_index}"
    );
    ensure!(
        header_bytes[5..8] == [0, 0, 0],
        "frame reserved bytes must be zero"
    );
    let record = FrameRecord {
        phase: header_bytes[4],
        index: u64::from_be_bytes(header_bytes[8..16].try_into().expect("fixed slice")),
        payload_bytes: u32::from_be_bytes(header_bytes[16..20].try_into().expect("fixed slice")),
    };
    ensure!(
        record.phase == expected_phase && record.index == expected_index,
        "unexpected frame identity: got phase {} index {}, expected phase {} index {}",
        record.phase,
        record.index,
        expected_phase,
        expected_index
    );
    ensure!(
        record.payload_bytes as usize == payload.len(),
        "frame payload length {} does not match configured length {}",
        record.payload_bytes,
        payload.len()
    );
    reader
        .read_exact(payload)
        .with_context(|| format!("could not read frame {expected_index} payload"))?;
    Ok(record)
}

fn write_frame<W: Write>(writer: &mut W, phase: u8, index: u64, payload: &[u8]) -> Result<()> {
    let payload_bytes = u32::try_from(payload.len()).context("frame is larger than u32")?;
    writer.write_all(FRAME_MAGIC)?;
    writer.write_all(&[phase, 0, 0, 0])?;
    writer.write_all(&index.to_be_bytes())?;
    writer.write_all(&payload_bytes.to_be_bytes())?;
    writer.write_all(payload)?;
    writer.flush().context("could not flush child frame")?;
    Ok(())
}

fn read_marker<R: Read>(reader: &mut R, expected: &[u8; 8], description: &str) -> Result<()> {
    let mut actual = [0_u8; 8];
    reader
        .read_exact(&mut actual)
        .with_context(|| format!("could not read {description} marker"))?;
    ensure!(
        &actual == expected,
        "invalid {description} marker: {:?}",
        actual
    );
    Ok(())
}

fn send_control(writer: &mut ChildStdin, command: u8) -> Result<()> {
    writer
        .write_all(&[command])
        .and_then(|()| writer.flush())
        .context("could not send phase control to helper")
}

fn read_control<R: Read>(reader: &mut R, expected: u8, description: &str) -> Result<()> {
    let mut actual = [0_u8; 1];
    reader
        .read_exact(&mut actual)
        .with_context(|| format!("waiting for parent {description} control"))?;
    ensure!(
        actual[0] == expected,
        "invalid parent control while waiting for {description}"
    );
    Ok(())
}

fn allocate_payload(payload_bytes: usize) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(payload_bytes)
        .context("could not allocate configured single-frame RGB8 buffer")?;
    payload.resize(payload_bytes, 0);
    Ok(payload)
}

fn frame_payload_sha256(config: ValidatedConfig) -> Result<String> {
    let mut payload = allocate_payload(config.payload_bytes)?;
    fill_payload(&mut payload, config.requested.seed, 0);
    Ok(sha256_hex(&payload))
}

fn fill_payload(payload: &mut [u8], seed: u64, frame_index: u64) {
    let mut state = seed ^ frame_index.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    if state == 0 {
        state = 0xA076_1D64_78BD_642F;
    }
    for byte in payload {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        *byte = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8;
    }
}

fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn sha256_hex(bytes: &[u8]) -> String {
    sha256_digest_hex(Sha256::digest(bytes))
}

fn sha256_digest_hex(digest: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = digest.as_ref();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn duration_ns(duration: std::time::Duration) -> Result<u64> {
    u64::try_from(duration.as_nanos()).context("monotonic duration exceeds u64 nanoseconds")
}

fn nearest_rank_index(sample_count: usize, percentile: usize) -> usize {
    debug_assert!(sample_count > 0);
    (sample_count * percentile).div_ceil(100).max(1) - 1
}

struct ChildGuard {
    child: Child,
    reaped: bool,
}

impl ChildGuard {
    fn spawn(command: &mut Command) -> Result<Self> {
        let child = command
            .spawn()
            .context("could not start supervised pipe helper")?;
        Ok(Self {
            child,
            reaped: false,
        })
    }

    fn stdin_mut(&mut self) -> &mut ChildStdin {
        self.child
            .stdin
            .as_mut()
            .expect("stdin was configured as piped")
    }

    fn stdout_mut(&mut self) -> &mut ChildStdout {
        self.child
            .stdout
            .as_mut()
            .expect("stdout was configured as piped")
    }

    fn close_stdin(&mut self) {
        self.child.stdin.take();
    }

    fn wait(&mut self) -> Result<ExitStatus> {
        let status = self.child.wait().context("could not reap pipe helper")?;
        self.reaped = true;
        Ok(status)
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if self.reaped {
            return;
        }
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
        self.reaped = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn test_config() -> BenchmarkConfig {
        BenchmarkConfig {
            width_px: 4,
            height_px: 3,
            warmup_frames: 2,
            measured_frames: 5,
            repetitions: 1,
            sampling_limit: 3,
            seed: 19,
            max_payload_bytes: 256,
            max_total_frames: 10,
            max_total_samples: 6,
        }
    }

    #[test]
    fn deterministic_payloads_match_for_equal_inputs_and_vary_by_frame() {
        let mut first = [0_u8; 36];
        let mut second = [0_u8; 36];
        fill_payload(&mut first, 33, 7);
        fill_payload(&mut second, 33, 7);
        assert_eq!(first, second);
        fill_payload(&mut second, 33, 8);
        assert_ne!(first, second);
    }

    #[test]
    fn validates_geometry_overflow_and_explicit_workload_limits() {
        let mut config = test_config();
        config.max_payload_bytes = 35;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("above --max-payload-bytes"));

        config = test_config();
        config.max_total_frames = 6;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("--max-total-frames"));

        config = test_config();
        config.max_total_samples = 5;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("--max-total-samples"));

        config = test_config();
        config.warmup_frames = 0;
        assert!(config.validate().is_ok());

        config = test_config();
        config.width_px = u32::MAX;
        config.height_px = u32::MAX;
        assert!(config.validate().is_err());
    }

    #[test]
    fn readiness_and_fixed_frame_parsers_reject_truncated_or_malformed_records() {
        assert!(read_ready(&mut Cursor::new(b"short".to_vec())).is_err());

        let expected_payload_sha256 =
            frame_payload_sha256(test_config().validate().unwrap()).unwrap();
        let ready = ReadyRecord {
            width_px: 4,
            height_px: 3,
            payload_bytes: 36,
            warmup_frames: 2,
            measured_frames: 5,
            seed: 19,
            payload_sha256: expected_payload_sha256.clone(),
        };
        let mut bytes = Vec::new();
        write_ready(&mut bytes, ready).unwrap();
        let decoded = read_ready(&mut Cursor::new(bytes)).unwrap();
        validate_ready(
            decoded,
            test_config().validate().unwrap(),
            &expected_payload_sha256,
        )
        .unwrap();

        let mut malformed = Vec::new();
        malformed.extend_from_slice(FRAME_MAGIC);
        malformed.extend_from_slice(&[PHASE_MEASUREMENT, 0, 0, 1]);
        malformed.extend_from_slice(&0_u64.to_be_bytes());
        malformed.extend_from_slice(&36_u32.to_be_bytes());
        malformed.extend_from_slice(&[0; 36]);
        let mut payload = [0_u8; 36];
        assert!(read_frame(
            &mut Cursor::new(malformed),
            &mut payload,
            PHASE_MEASUREMENT,
            0
        )
        .is_err());
    }

    #[test]
    fn malformed_readiness_kills_and_reaps_the_helper() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let temp = tempfile::tempdir().unwrap();
            let script_path = temp.path().join("bad-ready.sh");
            let pid_path = temp.path().join("pid");
            std::fs::write(
                &script_path,
                "#!/bin/sh\nprintf '%s\\n' \"$$\" > \"$PID_FILE\"\nprintf '%116s' x\nexec sleep 30\n",
            )
            .unwrap();
            let mut permissions = std::fs::metadata(&script_path).unwrap().permissions();
            permissions.set_mode(0o700);
            std::fs::set_permissions(&script_path, permissions).unwrap();
            let mut command = Command::new(&script_path);
            command.env("PID_FILE", &pid_path);
            let error = run_stdout_pipe(test_config().validate().unwrap(), &mut command)
                .unwrap_err()
                .to_string();
            assert!(error.contains("invalid helper readiness magic"));

            let child_pid: i32 = std::fs::read_to_string(pid_path)
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            let alive = Command::new("kill")
                .args(["-0", &child_pid.to_string()])
                .stderr(Stdio::null())
                .status()
                .unwrap()
                .success();
            assert!(!alive, "failed helper process {child_pid} should be reaped");
        }
    }

    #[test]
    fn malformed_frame_kills_and_reaps_the_helper() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let temp = tempfile::tempdir().unwrap();
            let script_path = temp.path().join("bad-frame.sh");
            let pid_path = temp.path().join("pid");
            let expected_payload_sha256 =
                frame_payload_sha256(test_config().validate().unwrap()).unwrap();
            let mut ready = Vec::new();
            write_ready(
                &mut ready,
                ReadyRecord {
                    width_px: 4,
                    height_px: 3,
                    payload_bytes: 36,
                    warmup_frames: 2,
                    measured_frames: 5,
                    seed: 19,
                    payload_sha256: expected_payload_sha256,
                },
            )
            .unwrap();
            let ready_printf = ready
                .iter()
                .map(|byte| format!("\\{byte:03o}"))
                .collect::<String>();
            let script = format!(
                "#!/bin/sh\nprintf '%s\\n' \"$$\" > \"$PID_FILE\"\nprintf '{ready_printf}'\nprintf 'BAD!%16s' x\nexec sleep 30\n"
            );
            std::fs::write(&script_path, script).unwrap();
            let mut permissions = std::fs::metadata(&script_path).unwrap().permissions();
            permissions.set_mode(0o700);
            std::fs::set_permissions(&script_path, permissions).unwrap();
            let mut command = Command::new(&script_path);
            command.env("PID_FILE", &pid_path);
            let error = run_stdout_pipe(test_config().validate().unwrap(), &mut command)
                .unwrap_err()
                .to_string();
            assert!(error.contains("invalid frame magic"));

            let child_pid: i32 = std::fs::read_to_string(pid_path)
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            let alive = Command::new("kill")
                .args(["-0", &child_pid.to_string()])
                .stderr(Stdio::null())
                .status()
                .unwrap()
                .success();
            assert!(!alive, "failed helper process {child_pid} should be reaped");
        }
    }
}
