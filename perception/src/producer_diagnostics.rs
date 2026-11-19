//! Independent, bounded producer metadata transport. It never participates in
//! scene readiness or refreshes localization evidence. Serialization and socket
//! operations belong to the reporter thread, outside capture/inference work.

use std::net::{SocketAddr, TcpStream};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;
use tokio::sync::{watch, Semaphore};

use crate::framing::write_framed_message;
use crate::producer_diagnostics_contract::{
    ActiveModel, BuildInfo, DiagnosticsConfig, ModelState, ProducerDiagnosticsReport,
    ProducerExitReason, ProducerMetadata, ProducerStageName, ScheduledJobKind, ScheduledWorkState,
    StageObservation, StageStatus,
};

pub const REPORTER_ENV: &str = "BILLIARDS_PRODUCER_DIAGNOSTICS";

/// Private launch settings contain routing and policy, never model/config paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReporterLaunch {
    pub address: SocketAddr,
    pub generation_id: String,
    pub policy: DiagnosticsConfig,
    pub connect_timeout_ms: u64,
    pub write_timeout_ms: u64,
    pub reconnect_initial_delay_ms: u64,
    pub reconnect_maximum_delay_ms: u64,
}

impl ReporterLaunch {
    pub fn validate(&self) -> Result<(), String> {
        self.policy.validate()?;
        if !self.address.ip().is_loopback() || self.address.port() == 0 {
            return Err("diagnostics reporter requires an allocated loopback endpoint".into());
        }
        if self.generation_id.is_empty()
            || self.generation_id.len() > 128
            || self.generation_id.chars().any(char::is_control)
        {
            return Err("invalid diagnostics generation".into());
        }
        for value in [
            self.connect_timeout_ms,
            self.write_timeout_ms,
            self.reconnect_initial_delay_ms,
            self.reconnect_maximum_delay_ms,
        ] {
            if !(1..=300_000).contains(&value) {
                return Err("diagnostics transport deadlines must be 1..300000 ms".into());
            }
        }
        if self.reconnect_initial_delay_ms > self.reconnect_maximum_delay_ms {
            return Err("diagnostics reconnect bounds are inconsistent".into());
        }
        Ok(())
    }
}

/// Gateway-local timestamps are deliberately never serialized as producer time.
#[derive(Debug, Clone, Default)]
pub struct ReceivedDiagnostics {
    pub generation_id: Option<String>,
    pub report: Option<ProducerDiagnosticsReport>,
    pub received_at: Option<Instant>,
    pub exited: bool,
    pub exit_reason: Option<ProducerExitReason>,
    pub last_known: Option<LastKnownReport>,
}

/// One bounded report retained for history after expiry, retirement, or a
/// supervised generation replacement. Its timestamp is gateway-local.
#[derive(Debug, Clone)]
pub struct LastKnownReport {
    pub report: ProducerDiagnosticsReport,
    pub received_at: Instant,
}

#[derive(Debug)]
struct RegistryState {
    snapshot: ReceivedDiagnostics,
    process_id: Option<String>,
    sequence: u64,
}

/// Registered supervision generations are the only accepted producers. A new
/// generation atomically invalidates all active model identity from its predecessor.
#[derive(Debug, Clone)]
pub struct DiagnosticsRegistry {
    state: Arc<Mutex<RegistryState>>,
    updates: watch::Sender<ReceivedDiagnostics>,
}

impl Default for DiagnosticsRegistry {
    fn default() -> Self {
        let (updates, _) = watch::channel(ReceivedDiagnostics::default());
        Self {
            state: Arc::new(Mutex::new(RegistryState {
                snapshot: ReceivedDiagnostics::default(),
                process_id: None,
                sequence: 0,
            })),
            updates,
        }
    }
}

impl DiagnosticsRegistry {
    pub fn begin_generation(&self, generation: String) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let mut last_known = state.snapshot.last_known.clone();
        if let (Some(report), Some(received_at)) =
            (state.snapshot.report.clone(), state.snapshot.received_at)
        {
            last_known = Some(LastKnownReport {
                report,
                received_at,
            });
        }
        state.snapshot = ReceivedDiagnostics {
            generation_id: Some(generation),
            last_known,
            ..Default::default()
        };
        state.process_id = None;
        state.sequence = 0;
        self.updates.send_replace(state.snapshot.clone());
    }

    pub fn retire_generation(&self, generation: &str, reason: ProducerExitReason) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.snapshot.generation_id.as_deref() == Some(generation) {
            if let (Some(report), Some(received_at)) =
                (state.snapshot.report.clone(), state.snapshot.received_at)
            {
                state.snapshot.last_known = Some(LastKnownReport {
                    report,
                    received_at,
                });
            }
            state.snapshot.report = None;
            state.snapshot.received_at = None;
            state.snapshot.exited = true;
            state.snapshot.exit_reason = Some(reason);
            self.updates.send_replace(state.snapshot.clone());
        }
    }

    pub fn accept(
        &self,
        report: ProducerDiagnosticsReport,
        received_at: Instant,
    ) -> Result<(), String> {
        report.validate()?;
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.snapshot.exited
            || state.snapshot.generation_id.as_deref() != Some(&report.generation_id)
        {
            return Err("unregistered or retired diagnostics generation".into());
        }
        if report.sequence <= state.sequence
            || state
                .process_id
                .as_ref()
                .is_some_and(|id| id != &report.process_id)
        {
            return Err("diagnostics sequence or process identity changed".into());
        }
        state.process_id = Some(report.process_id.clone());
        state.sequence = report.sequence;
        state.snapshot.last_known = Some(LastKnownReport {
            report: report.clone(),
            received_at,
        });
        state.snapshot.report = Some(report);
        state.snapshot.received_at = Some(received_at);
        self.updates.send_replace(state.snapshot.clone());
        Ok(())
    }

    pub fn snapshot(&self) -> ReceivedDiagnostics {
        self.updates.borrow().clone()
    }

    /// Metadata consumers must opt into freshness instead of reusing the last
    /// active model after its reporter stops making progress.
    pub fn active_report_at(
        &self,
        now: Instant,
        expiry: Duration,
    ) -> Option<ProducerDiagnosticsReport> {
        let snapshot = self.snapshot();
        let age = now.checked_duration_since(snapshot.received_at?)?;
        if snapshot.exited || age >= expiry {
            return None;
        }
        snapshot.report
    }

    pub fn subscribe(&self) -> watch::Receiver<ReceivedDiagnostics> {
        self.updates.subscribe()
    }
}

/// The listener uses the existing bounded client policy. Every frame has a
/// deadline, including idle or partial clients, and length is checked before allocation.
pub async fn serve(
    listener: TcpListener,
    registry: DiagnosticsRegistry,
    policy: DiagnosticsConfig,
    maximum_clients: usize,
    mut shutdown: watch::Receiver<bool>,
) -> std::io::Result<()> {
    policy.validate().map_err(std::io::Error::other)?;
    if maximum_clients == 0 || !listener.local_addr()?.ip().is_loopback() {
        return Err(std::io::Error::other(
            "diagnostics requires loopback and bounded clients",
        ));
    }
    let slots = Arc::new(Semaphore::new(maximum_clients));
    let mut clients = tokio::task::JoinSet::new();
    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() { break; }
            }
            accepted = listener.accept() => {
                let (mut socket, peer) = accepted?;
                if !peer.ip().is_loopback() { continue; }
                let Ok(permit) = slots.clone().try_acquire_owned() else { continue; };
                let registry = registry.clone();
                let policy = policy.clone();
                clients.spawn(async move {
                    let _permit = permit;
                    loop {
                        let frame = async {
                            let length = socket.read_u32().await? as usize;
                            if length == 0 || length > policy.maximum_message_bytes {
                                return Err(std::io::Error::other("invalid diagnostics frame length"));
                            }
                            let mut bytes = vec![0; length];
                            socket.read_exact(&mut bytes).await?;
                            Ok(bytes)
                        };
                        let Ok(Ok(bytes)) = tokio::time::timeout(Duration::from_millis(policy.report_expiry_ms), frame).await else { break; };
                        let Ok(report) = ProducerDiagnosticsReport::decode(&bytes, policy.maximum_message_bytes) else { break; };
                        if registry.accept(report, Instant::now()).is_err() { break; }
                    }
                });
            }
            Some(_) = clients.join_next(), if !clients.is_empty() => {}
        }
    }
    clients.abort_all();
    while clients.join_next().await.is_some() {}
    Ok(())
}

struct Pending {
    metadata: ProducerMetadata,
    stages: Vec<StageTrack>,
    processing_stall_timeout_ms: u64,
    revision: u64,
    stopping: bool,
}

struct StageTrack {
    observation: StageObservation,
    progress_at: Option<Instant>,
    work_started_at: Option<Instant>,
    last_completion_at: Option<Instant>,
    last_failure_at: Option<Instant>,
}

/// Actual scheduler and worker state captured together at a report boundary.
#[derive(Debug, Clone, Copy)]
pub struct ScheduledWorkEvidence<'a> {
    pub scheduler_state: ScheduledWorkState,
    pub in_flight: bool,
    pub active_job_kind: Option<ScheduledJobKind>,
    pub work_started_at: Option<Instant>,
    pub expected_jobs: u64,
    pub completed_jobs: u64,
    pub failed_jobs: u64,
    pub last_completion_at: Option<Instant>,
    pub last_failure_code: Option<&'a str>,
    pub last_failure_at: Option<Instant>,
}

/// Cloneable mutation handle; no serialization/network calls occur in setters.
#[derive(Clone)]
pub struct ReporterHandle(Arc<(Mutex<Pending>, Condvar)>);

impl ReporterHandle {
    pub fn set_engine_identity(&self, engine: String) {
        self.update(|metadata| metadata.engine_instance_id = Some(engine));
    }

    /// Capture identity is published only after compatible source readiness.
    pub fn set_capture_session(&self, session: String) {
        self.update(|metadata| metadata.capture_session_id = Some(session));
    }

    pub fn activate_model(&self, model: ActiveModel) {
        self.update(|metadata| {
            metadata.model = Some(model);
            metadata.model_state = ModelState::Active;
            metadata.failure_code = None;
        });
        self.set_stage(
            ProducerStageName::ModelInitialization,
            StageStatus::Healthy,
            "model_loaded_and_warmed",
            None,
        );
    }

    /// Record a producer-level failure without claiming that model loading
    /// failed. Capture/readiness failures have their own stage observations.
    pub fn fail(&self) {
        self.update(|metadata| metadata.failure_code = Some("producer_failed".into()));
    }

    /// Model failure is asserted only by the explicit load/warm-up boundary.
    pub fn model_failed(&self) {
        self.update(|metadata| {
            metadata.model = None;
            metadata.model_state = ModelState::Failed;
            metadata.failure_code = Some("model_initialization_failed".into());
        });
        self.set_stage(
            ProducerStageName::ModelInitialization,
            StageStatus::Failed,
            "model_initialization_failed",
            None,
        );
    }

    pub fn stopped(&self) {
        self.update(|metadata| {
            metadata.model = None;
            metadata.model_state = ModelState::Stopped;
            metadata.failure_code = None;
        });
        self.set_stage(
            ProducerStageName::SourceLifecycle,
            StageStatus::Stopped,
            "producer_stopped",
            None,
        );
    }

    /// Record a lifecycle transition using stable, sanitized identifiers only.
    pub fn set_stage(
        &self,
        stage: ProducerStageName,
        status: StageStatus,
        reason_code: &str,
        detail: Option<&str>,
    ) {
        let (lock, wake) = &*self.0;
        let mut pending = lock.lock().unwrap_or_else(|error| error.into_inner());
        let now = Instant::now();
        let track = find_or_create_stage(&mut pending.stages, stage);
        let changed = track.observation.status != status
            || track.observation.reason_code != reason_code
            || track.observation.detail.as_deref() != detail;
        track.observation.status = status;
        track.observation.reason_code = reason_code.to_owned();
        track.observation.detail = detail.map(str::to_owned);
        if matches!(status, StageStatus::Starting | StageStatus::Healthy) {
            track.progress_at = Some(now);
        }
        if !matches!(status, StageStatus::Healthy) {
            track.observation.in_flight = false;
            track.observation.work_started_age_ms = None;
            track.work_started_at = None;
            track.observation.deadline_ms = None;
            if stage == ProducerStageName::ScheduledPerceptionWork {
                track.observation.active_job_kind = None;
                track.observation.scheduler_state = Some(match status {
                    StageStatus::Failed => ScheduledWorkState::Failed,
                    StageStatus::Starting => ScheduledWorkState::Due,
                    StageStatus::Stopped => ScheduledWorkState::Unknown,
                    StageStatus::Idle | StageStatus::Degraded => ScheduledWorkState::Idle,
                    StageStatus::Stalled => ScheduledWorkState::Failed,
                    StageStatus::Healthy | StageStatus::Unknown => ScheduledWorkState::Unknown,
                });
                if matches!(status, StageStatus::Failed | StageStatus::Stalled) {
                    track.observation.last_failure_code = Some(reason_code.to_owned());
                    track.observation.last_failure_age_ms = Some(0);
                    track.last_failure_at = Some(now);
                }
            }
        }
        if changed {
            pending.revision = pending.revision.saturating_add(1);
            wake.notify_one();
        }
    }

    /// Refresh actual source progress without waking the reporter for each frame;
    /// the next heartbeat serializes the current monotonic age off the hot path.
    pub fn stage_progress(&self, stage: ProducerStageName) {
        let (lock, _) = &*self.0;
        let mut pending = lock.lock().unwrap_or_else(|error| error.into_inner());
        let track = find_or_create_stage(&mut pending.stages, stage);
        track.progress_at = Some(Instant::now());
    }

    /// A closed receive stream ends both source lifecycle and capture progress.
    /// Update them under one lock so the reporter cannot publish a mixed state.
    pub fn capture_source_closed(&self) {
        let (lock, wake) = &*self.0;
        let mut pending = lock.lock().unwrap_or_else(|error| error.into_inner());
        let mut changed = false;
        for stage in [
            ProducerStageName::SourceLifecycle,
            ProducerStageName::CaptureProgress,
        ] {
            let track = find_or_create_stage(&mut pending.stages, stage);
            changed |= track.observation.status != StageStatus::Stopped
                || track.observation.reason_code != "capture_source_closed";
            track.observation.status = StageStatus::Stopped;
            track.observation.reason_code = "capture_source_closed".to_owned();
            track.observation.detail = None;
            track.observation.in_flight = false;
            track.observation.work_started_age_ms = None;
            track.observation.deadline_ms = None;
            track.work_started_at = None;
        }
        if changed {
            pending.revision = pending.revision.saturating_add(1);
            wake.notify_one();
        }
    }

    /// Publish current scheduler evidence. `work_started_at` is populated only
    /// for a successfully dispatched job which the worker still reports active.
    pub fn set_scheduled_work(&self, evidence: ScheduledWorkEvidence<'_>) {
        let ScheduledWorkEvidence {
            scheduler_state,
            in_flight,
            active_job_kind,
            work_started_at,
            expected_jobs,
            completed_jobs,
            failed_jobs,
            last_completion_at,
            last_failure_code,
            last_failure_at,
        } = evidence;
        let (lock, wake) = &*self.0;
        let mut pending = lock.lock().unwrap_or_else(|error| error.into_inner());
        let timeout = pending.processing_stall_timeout_ms;
        let track = find_or_create_stage(
            &mut pending.stages,
            ProducerStageName::ScheduledPerceptionWork,
        );
        let status = match scheduler_state {
            ScheduledWorkState::Unknown => StageStatus::Unknown,
            ScheduledWorkState::Idle
            | ScheduledWorkState::PhaseGated
            | ScheduledWorkState::ResultReady => StageStatus::Idle,
            ScheduledWorkState::Due => StageStatus::Starting,
            ScheduledWorkState::InFlight => StageStatus::Healthy,
            ScheduledWorkState::Failed => StageStatus::Failed,
        };
        let reason_code = match scheduler_state {
            ScheduledWorkState::Unknown => "scheduler_state_unknown",
            ScheduledWorkState::Idle => "no_job_due",
            ScheduledWorkState::PhaseGated => "work_phase_gated",
            ScheduledWorkState::Due => "scheduled_job_due",
            ScheduledWorkState::InFlight => "scheduled_job_in_flight",
            ScheduledWorkState::ResultReady => "scheduled_result_ready",
            ScheduledWorkState::Failed => "scheduled_job_failed",
        };
        let changed = track.observation.status != status
            || track.observation.in_flight != in_flight
            || track.observation.scheduler_state != Some(scheduler_state)
            || track.observation.active_job_kind != active_job_kind;
        track.observation.status = status;
        track.observation.reason_code = reason_code.to_owned();
        track.observation.detail = None;
        track.observation.in_flight = in_flight;
        track.observation.scheduler_state = Some(scheduler_state);
        track.observation.active_job_kind = active_job_kind;
        track.work_started_at = work_started_at;
        track.last_completion_at = last_completion_at;
        track.observation.last_failure_code = last_failure_code.map(str::to_owned);
        track.last_failure_at = last_failure_at;
        track.observation.last_failure_age_ms = last_failure_at
            .map(|failed_at| duration_millis(Instant::now().saturating_duration_since(failed_at)));
        track.observation.deadline_ms = (in_flight && work_started_at.is_some()).then_some(timeout);
        track.observation.expected_jobs = Some(expected_jobs);
        track.observation.completed_jobs = Some(completed_jobs);
        track.observation.failed_jobs = Some(failed_jobs);
        track.progress_at = [last_completion_at, work_started_at, track.progress_at]
            .into_iter()
            .flatten()
            .max();
        if changed {
            pending.revision = pending.revision.saturating_add(1);
            wake.notify_one();
        }
    }

    fn update(&self, update: impl FnOnce(&mut ProducerMetadata)) {
        let (lock, wake) = &*self.0;
        let mut pending = lock.lock().unwrap_or_else(|error| error.into_inner());
        update(&mut pending.metadata);
        pending.revision = pending.revision.saturating_add(1);
        wake.notify_one();
    }
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn find_or_create_stage(stages: &mut Vec<StageTrack>, stage: ProducerStageName) -> &mut StageTrack {
    if let Some(index) = stages
        .iter()
        .position(|track| track.observation.stage == stage)
    {
        return &mut stages[index];
    }
    let status = if stage == ProducerStageName::ScheduledPerceptionWork {
        StageStatus::Unknown
    } else {
        StageStatus::Starting
    };
    stages.push(StageTrack {
        observation: StageObservation {
            stage,
            status,
            reason_code: "stage_not_observed".into(),
            detail: None,
            last_progress_age_ms: None,
            in_flight: false,
            work_started_age_ms: None,
            expected_jobs: None,
            completed_jobs: None,
            failed_jobs: None,
            scheduler_state: (stage == ProducerStageName::ScheduledPerceptionWork)
                .then_some(ScheduledWorkState::Unknown),
            active_job_kind: None,
            last_completion_age_ms: None,
            last_failure_code: None,
            last_failure_age_ms: None,
            deadline_ms: None,
        },
        progress_at: None,
        work_started_at: None,
        last_completion_at: None,
        last_failure_at: None,
    });
    stages.last_mut().expect("new stage was pushed")
}

fn initial_stages() -> Vec<StageTrack> {
    [
        (
            ProducerStageName::SourceLifecycle,
            StageStatus::Starting,
            "capture_open_pending",
        ),
        (
            ProducerStageName::CaptureProgress,
            StageStatus::Starting,
            "awaiting_first_frame",
        ),
        (
            ProducerStageName::ModelInitialization,
            StageStatus::Starting,
            "model_initialization_pending",
        ),
        (
            ProducerStageName::ScheduledPerceptionWork,
            StageStatus::Unknown,
            "scheduler_not_initialized",
        ),
    ]
    .into_iter()
    .map(|(stage, status, reason_code)| StageTrack {
        observation: StageObservation {
            stage,
            status,
            reason_code: reason_code.into(),
            detail: None,
            last_progress_age_ms: None,
            in_flight: false,
            work_started_age_ms: None,
            expected_jobs: None,
            completed_jobs: None,
            failed_jobs: None,
            scheduler_state: (stage == ProducerStageName::ScheduledPerceptionWork)
                .then_some(ScheduledWorkState::Unknown),
            active_job_kind: None,
            last_completion_age_ms: None,
            last_failure_code: None,
            last_failure_age_ms: None,
            deadline_ms: None,
        },
        progress_at: None,
        work_started_at: None,
        last_completion_at: None,
        last_failure_at: None,
    })
    .collect()
}

pub struct ProducerReporter {
    handle: ReporterHandle,
    worker: Option<JoinHandle<()>>,
}

impl ProducerReporter {
    pub fn from_environment() -> Result<Option<Self>, String> {
        let Some(value) = std::env::var_os(REPORTER_ENV) else {
            return Ok(None);
        };
        let launch: ReporterLaunch = serde_json::from_str(&value.to_string_lossy())
            .map_err(|_| "invalid diagnostics launch settings".to_owned())?;
        Self::start(launch).map(Some)
    }

    pub fn start(launch: ReporterLaunch) -> Result<Self, String> {
        launch.validate()?;
        let pending = Pending {
            metadata: ProducerMetadata {
                build: BuildInfo::current(),
                model_state: ModelState::Loading,
                model: None,
                engine_instance_id: None,
                capture_session_id: None,
                failure_code: None,
            },
            stages: initial_stages(),
            processing_stall_timeout_ms: launch.policy.processing_stall_timeout_ms,
            revision: 1,
            stopping: false,
        };
        let handle = ReporterHandle(Arc::new((Mutex::new(pending), Condvar::new())));
        let shared = handle.clone();
        let worker = std::thread::Builder::new()
            .name("producer-diagnostics".into())
            .spawn(move || report_loop(launch, shared))
            .map_err(|_| "failed to start diagnostics worker".to_owned())?;
        Ok(Self {
            handle,
            worker: Some(worker),
        })
    }

    pub fn handle(&self) -> ReporterHandle {
        self.handle.clone()
    }

    pub fn finish(mut self) {
        self.stop();
    }

    fn stop(&mut self) {
        let (lock, wake) = &*self.handle.0;
        lock.lock()
            .unwrap_or_else(|error| error.into_inner())
            .stopping = true;
        wake.notify_one();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for ProducerReporter {
    fn drop(&mut self) {
        self.stop();
    }
}

fn report_loop(launch: ReporterLaunch, shared: ReporterHandle) {
    let process_id = uuid::Uuid::new_v4().to_string();
    let mut sequence = 0u64;
    let mut sent_revision = 0u64;
    let mut socket: Option<TcpStream> = None;
    let mut backoff = launch.reconnect_initial_delay_ms;
    let (lock, wake) = &*shared.0;
    loop {
        let pending = lock.lock().unwrap_or_else(|error| error.into_inner());
        let pending = if pending.revision == sent_revision && !pending.stopping {
            wake.wait_timeout(
                pending,
                Duration::from_millis(launch.policy.heartbeat_interval_ms),
            )
            .unwrap_or_else(|error| error.into_inner())
            .0
        } else {
            pending
        };
        let metadata = pending.metadata.clone();
        let now = Instant::now();
        let stages = pending
            .stages
            .iter()
            .map(|track| {
                let mut observation = track.observation.clone();
                observation.last_progress_age_ms = track
                    .progress_at
                    .map(|progress_at| duration_millis(now.saturating_duration_since(progress_at)));
                observation.work_started_age_ms = track
                    .work_started_at
                    .map(|started_at| duration_millis(now.saturating_duration_since(started_at)));
                observation.last_completion_age_ms = track.last_completion_at.map(|completed_at| {
                    duration_millis(now.saturating_duration_since(completed_at))
                });
                observation.last_failure_age_ms = track
                    .last_failure_at
                    .map(|failed_at| duration_millis(now.saturating_duration_since(failed_at)));
                observation
            })
            .collect();
        let revision = pending.revision;
        let stopping = pending.stopping;
        drop(pending);
        sequence = sequence.saturating_add(1);
        let report = ProducerDiagnosticsReport {
            schema_version: 1,
            generation_id: launch.generation_id.clone(),
            process_id: process_id.clone(),
            sequence,
            metadata,
            stages,
        };
        let bytes = serde_json::to_vec(&report)
            .ok()
            .filter(|bytes| bytes.len() <= launch.policy.maximum_message_bytes);
        if socket.is_none() {
            socket = TcpStream::connect_timeout(
                &launch.address,
                Duration::from_millis(launch.connect_timeout_ms),
            )
            .ok();
            if let Some(stream) = &socket {
                let _ =
                    stream.set_write_timeout(Some(Duration::from_millis(launch.write_timeout_ms)));
                let _ = stream.set_nodelay(true);
            }
        }
        let sent = match (&mut socket, bytes) {
            (Some(stream), Some(bytes)) => {
                write_framed_message(stream, &bytes, launch.policy.maximum_message_bytes).is_ok()
            }
            _ => false,
        };
        if stopping {
            return;
        }
        if sent {
            sent_revision = revision;
            backoff = launch.reconnect_initial_delay_ms;
        } else {
            socket = None;
            let pending = lock.lock().unwrap_or_else(|error| error.into_inner());
            if !pending.stopping && pending.revision == revision {
                drop(
                    wake.wait_timeout(pending, Duration::from_millis(backoff))
                        .unwrap_or_else(|error| error.into_inner()),
                );
            }
            backoff = backoff
                .saturating_mul(2)
                .min(launch.reconnect_maximum_delay_ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle() -> ReporterHandle {
        ReporterHandle(Arc::new((
            Mutex::new(Pending {
                metadata: ProducerMetadata {
                    build: BuildInfo::current(),
                    model_state: ModelState::Loading,
                    model: None,
                    engine_instance_id: None,
                    capture_session_id: None,
                    failure_code: None,
                },
                stages: initial_stages(),
                processing_stall_timeout_ms: 30_000,
                revision: 1,
                stopping: false,
            }),
            Condvar::new(),
        )))
    }

    #[test]
    fn phase_gated_and_result_ready_work_are_idle_without_deadlines() {
        let handle = handle();
        handle.set_scheduled_work(ScheduledWorkEvidence {
            scheduler_state: ScheduledWorkState::PhaseGated,
            in_flight: false,
            active_job_kind: None,
            work_started_at: None,
            expected_jobs: 4,
            completed_jobs: 4,
            failed_jobs: 0,
            last_completion_at: Some(Instant::now()),
            last_failure_code: None,
            last_failure_at: None,
        });
        {
            let pending = handle.0 .0.lock().unwrap();
            let stage = pending
                .stages
                .iter()
                .find(|stage| stage.observation.stage == ProducerStageName::ScheduledPerceptionWork)
                .unwrap();
            assert_eq!(stage.observation.status, StageStatus::Idle);
            assert!(!stage.observation.in_flight);
            assert_eq!(stage.observation.deadline_ms, None);
        }

        handle.set_scheduled_work(ScheduledWorkEvidence {
            scheduler_state: ScheduledWorkState::InFlight,
            in_flight: true,
            active_job_kind: Some(ScheduledJobKind::BallDetection),
            work_started_at: Some(Instant::now()),
            expected_jobs: 5,
            completed_jobs: 4,
            failed_jobs: 0,
            last_completion_at: Some(Instant::now()),
            last_failure_code: None,
            last_failure_at: None,
        });
        {
            let pending = handle.0 .0.lock().unwrap();
            let stage = pending
                .stages
                .iter()
                .find(|stage| stage.observation.stage == ProducerStageName::ScheduledPerceptionWork)
                .unwrap();
            assert_eq!(stage.observation.status, StageStatus::Healthy);
            assert!(stage.observation.in_flight);
            assert_eq!(stage.observation.deadline_ms, Some(30_000));
            assert_eq!(
                stage.observation.active_job_kind,
                Some(ScheduledJobKind::BallDetection)
            );
        }

        handle.set_scheduled_work(ScheduledWorkEvidence {
            scheduler_state: ScheduledWorkState::ResultReady,
            in_flight: false,
            active_job_kind: None,
            work_started_at: None,
            expected_jobs: 5,
            completed_jobs: 5,
            failed_jobs: 0,
            last_completion_at: Some(Instant::now()),
            last_failure_code: None,
            last_failure_at: None,
        });
        let pending = handle.0 .0.lock().unwrap();
        let stage = pending
            .stages
            .iter()
            .find(|stage| stage.observation.stage == ProducerStageName::ScheduledPerceptionWork)
            .unwrap();
        assert_eq!(stage.observation.status, StageStatus::Idle);
        assert!(!stage.observation.in_flight);
        assert_eq!(stage.observation.deadline_ms, None);
    }

    #[test]
    fn producer_failure_does_not_claim_model_initialization_failed() {
        let handle = handle();
        handle.fail();
        assert_eq!(
            handle.0 .0.lock().unwrap().metadata.model_state,
            ModelState::Loading
        );
        handle.model_failed();
        assert_eq!(
            handle.0 .0.lock().unwrap().metadata.model_state,
            ModelState::Failed
        );
    }

    #[test]
    fn closed_capture_stops_source_and_capture_stages_together() {
        let handle = handle();
        handle.capture_source_closed();
        let pending = handle.0 .0.lock().unwrap();
        for stage in [
            ProducerStageName::SourceLifecycle,
            ProducerStageName::CaptureProgress,
        ] {
            let observation = pending
                .stages
                .iter()
                .find(|track| track.observation.stage == stage)
                .unwrap();
            assert_eq!(observation.observation.status, StageStatus::Stopped);
            assert_eq!(observation.observation.reason_code, "capture_source_closed");
        }
    }
}
