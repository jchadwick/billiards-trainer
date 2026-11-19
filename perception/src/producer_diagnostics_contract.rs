//! Strict metadata contract for the independent producer diagnostics channel.
//! Generation and sequence ownership is receiver policy; reports carry relative
//! evidence ages, not producer monotonic instants or configuration/model paths.

use crate::detection_batch::ModelIdentity;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr};

pub const DIAGNOSTICS_SCHEMA_VERSION: u8 = 1;
pub const MAXIMUM_DIAGNOSTICS_MESSAGE_BYTES: usize = 65_536;
pub const MAXIMUM_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiagnosticsConfig {
    pub enabled: bool,
    pub bind_address: IpAddr,
    pub port: u16,
    pub maximum_message_bytes: usize,
    pub heartbeat_interval_ms: u64,
    pub report_expiry_ms: u64,
    /// Complete browser snapshots are pushed on change and at this heartbeat.
    pub stream_heartbeat_interval_ms: u64,
    /// Browser clients mark the stream stale after this much silence.
    pub stream_stale_after_ms: u64,
    /// Applied only while a named producer job is explicitly in flight.
    pub processing_stall_timeout_ms: u64,
}
impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bind_address: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 0,
            maximum_message_bytes: MAXIMUM_DIAGNOSTICS_MESSAGE_BYTES,
            heartbeat_interval_ms: 1_000,
            report_expiry_ms: 3_000,
            stream_heartbeat_interval_ms: 1_000,
            stream_stale_after_ms: 3_000,
            processing_stall_timeout_ms: 30_000,
        }
    }
}
impl DiagnosticsConfig {
    /// Missing sections retain backwards compatibility with deployed config.
    pub fn from_configuration(document: &Value) -> Result<Self, String> {
        let config: Self = match document.pointer("/perception/rust_live/diagnostics") {
            Some(value) => serde_json::from_value(value.clone())
                .map_err(|error| format!("invalid diagnostics configuration: {error}"))?,
            None => Self::default(),
        };
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !self.bind_address.is_loopback() {
            return Err("diagnostics bind address must be loopback".into());
        }
        if !(1..=MAXIMUM_DIAGNOSTICS_MESSAGE_BYTES).contains(&self.maximum_message_bytes) {
            return Err("diagnostics maximum bytes must be 1..=65536".into());
        }
        if !(1..=300_000).contains(&self.heartbeat_interval_ms)
            || !(1..=300_000).contains(&self.report_expiry_ms)
            || self.report_expiry_ms <= self.heartbeat_interval_ms
        {
            return Err(
                "diagnostics expiry must exceed heartbeat; both must be 1..=300000 ms".into(),
            );
        }
        if !(1..=300_000).contains(&self.stream_heartbeat_interval_ms)
            || !(1..=300_000).contains(&self.stream_stale_after_ms)
            || self.stream_stale_after_ms <= self.stream_heartbeat_interval_ms
            || !(1..=300_000).contains(&self.processing_stall_timeout_ms)
        {
            return Err("diagnostics stream stale interval must exceed its heartbeat; all stream and work deadlines must be 1..=300000 ms".into());
        }
        Ok(())
    }
}

/// Per-executable compile-time provenance. No runtime Git invocation or guessed revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildInfo {
    pub package_name: String,
    pub package_version: String,
    pub git_revision: Option<String>,
    pub dirty: Option<bool>,
}
impl BuildInfo {
    pub fn current() -> Self {
        Self {
            package_name: env!("CARGO_PKG_NAME").into(),
            package_version: env!("CARGO_PKG_VERSION").into(),
            git_revision: option_env!("BILLIARDS_BUILD_GIT_REVISION")
                .filter(|value| valid_git_revision(value))
                .map(str::to_owned),
            dirty: option_env!("BILLIARDS_BUILD_DIRTY").and_then(|value| match value {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            }),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        text(&self.package_name, "package name", 128)?;
        text(&self.package_version, "package version", 128)?;
        if self
            .git_revision
            .as_deref()
            .is_some_and(|value| !valid_git_revision(value))
        {
            return Err("Git revision must be 7..=64 hexadecimal characters".into());
        }
        Ok(())
    }
}
fn valid_git_revision(value: &str) -> bool {
    (7..=64).contains(&value.len()) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Identity of the verified, active inference session. A provider request alone
/// is not evidence that its device executed the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActiveModel {
    pub identity: ModelIdentity,
    pub input_width_px: u32,
    pub input_height_px: u32,
    pub requested_provider: String,
    pub execution_provider: Option<String>,
    pub execution_device: Option<String>,
    pub execution_proof: Option<String>,
}
impl ActiveModel {
    pub fn validate(&self) -> Result<(), String> {
        sha256(&self.identity.artifact_sha256)?;
        if let Some(hash) = &self.identity.source_checkpoint_sha256 {
            sha256(hash)?;
        }
        if self.identity.classes.is_empty() || self.identity.classes.len() > 256 {
            return Err("class map must contain 1..=256 classes".into());
        }
        let mut ids = HashSet::new();
        let mut names = HashSet::new();
        for class in &self.identity.classes {
            text(&class.class_name, "class name", 128)?;
            if !ids.insert(class.class_id) || !names.insert(&class.class_name) {
                return Err("model class IDs and names must be unique".into());
            }
        }
        if self.input_width_px == 0 || self.input_height_px == 0 {
            return Err("model input dimensions must be positive".into());
        }
        text(&self.requested_provider, "requested provider", 128)?;
        for (label, value) in [
            ("execution provider", &self.execution_provider),
            ("execution device", &self.execution_device),
            ("execution proof", &self.execution_proof),
        ] {
            if let Some(value) = value {
                text(value, label, 256)?;
            }
        }
        if self.execution_device.is_some() && self.execution_provider.is_none() {
            return Err("execution device requires execution provider evidence".into());
        }
        Ok(())
    }
}
fn sha256(value: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("model identity requires valid SHA-256 digests".into());
    }
    Ok(())
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    Loading,
    Active,
    Failed,
    Stopped,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerMetadata {
    pub build: BuildInfo,
    pub model_state: ModelState,
    pub model: Option<ActiveModel>,
    pub engine_instance_id: Option<String>,
    pub capture_session_id: Option<String>,
    pub failure_code: Option<String>,
}
impl ProducerMetadata {
    pub fn validate(&self) -> Result<(), String> {
        self.build.validate()?;
        if (self.model_state == ModelState::Active) != self.model.is_some() {
            return Err("only active model state must contain loaded model metadata".into());
        }
        if let Some(model) = &self.model {
            model.validate()?;
        }
        for (label, value) in [
            ("engine instance", &self.engine_instance_id),
            ("capture session", &self.capture_session_id),
            ("failure code", &self.failure_code),
        ] {
            if let Some(value) = value {
                text(value, label, 128)?;
            }
        }
        Ok(())
    }
}
/// Producer observations, not independent gateway health assessments. Ages are
/// relative to producer report time and are re-based at gateway receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProducerStageName {
    SourceLifecycle,
    CaptureProgress,
    ModelInitialization,
    ScheduledPerceptionWork,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Starting,
    Healthy,
    Idle,
    Degraded,
    Stalled,
    Failed,
    Stopped,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduledJobKind {
    MotionAssessment,
    BallDetection,
    CueGeometry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduledWorkState {
    Unknown,
    Idle,
    PhaseGated,
    Due,
    InFlight,
    ResultReady,
    Failed,
}

/// Gateway-observed outcome for the most recently supervised producer process.
/// These stable values avoid exposing operating-system errors or command paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProducerExitReason {
    GracefulStop,
    ProcessExitedSuccessfully,
    ProcessExitedNonzero,
    ProcessWaitFailed,
    SpawnFailed,
    LaunchRequestFailed,
    ForcedTermination,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageObservation {
    pub stage: ProducerStageName,
    pub status: StageStatus,
    pub reason_code: String,
    pub detail: Option<String>,
    pub last_progress_age_ms: Option<u64>,
    /// True only while a scheduler job is actually running, not merely due or
    /// waiting for a completed result to be consumed.
    pub in_flight: bool,
    /// Age of the current job's dispatch time, independent of prior progress.
    pub work_started_age_ms: Option<u64>,
    /// Only populated for scheduled work, from observed scheduler counters.
    pub expected_jobs: Option<u64>,
    pub completed_jobs: Option<u64>,
    pub failed_jobs: Option<u64>,
    pub scheduler_state: Option<ScheduledWorkState>,
    pub active_job_kind: Option<ScheduledJobKind>,
    pub last_completion_age_ms: Option<u64>,
    pub last_failure_code: Option<String>,
    pub last_failure_age_ms: Option<u64>,
    /// Maximum duration for the explicitly in-flight job.
    pub deadline_ms: Option<u64>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerDiagnosticsReport {
    pub schema_version: u8,
    pub generation_id: String,
    pub process_id: String,
    pub sequence: u64,
    pub metadata: ProducerMetadata,
    pub stages: Vec<StageObservation>,
}
impl ProducerDiagnosticsReport {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != DIAGNOSTICS_SCHEMA_VERSION {
            return Err("unsupported producer diagnostics schema version".into());
        }
        text(&self.generation_id, "generation ID", 128)?;
        text(&self.process_id, "process ID", 128)?;
        if self.sequence == 0 || self.sequence > MAXIMUM_SAFE_JSON_INTEGER {
            return Err("report sequence must be a positive safe JSON integer".into());
        }
        self.metadata.validate()?;
        if self.stages.len() > 16 {
            return Err("report exceeds 16 stage observations".into());
        }
        let mut stages = HashSet::new();
        for observation in &self.stages {
            text(&observation.reason_code, "stage reason", 128)?;
            if observation
                .detail
                .as_ref()
                .is_some_and(|detail| detail.len() > 256 || detail.chars().any(char::is_control))
            {
                return Err("stage detail must be at most 256 non-control bytes".into());
            }
            if !stages.insert(&observation.stage) {
                return Err("stage observations must be unique".into());
            }
            if observation
                .last_progress_age_ms
                .is_some_and(|age| age > MAXIMUM_SAFE_JSON_INTEGER)
                || observation
                    .work_started_age_ms
                    .is_some_and(|age| age > MAXIMUM_SAFE_JSON_INTEGER)
                || observation
                    .expected_jobs
                    .is_some_and(|count| count > MAXIMUM_SAFE_JSON_INTEGER)
                || observation
                    .completed_jobs
                    .is_some_and(|count| count > MAXIMUM_SAFE_JSON_INTEGER)
                || observation
                    .failed_jobs
                    .is_some_and(|count| count > MAXIMUM_SAFE_JSON_INTEGER)
                || observation
                    .last_completion_age_ms
                    .is_some_and(|age| age > MAXIMUM_SAFE_JSON_INTEGER)
                || observation
                    .last_failure_age_ms
                    .is_some_and(|age| age > MAXIMUM_SAFE_JSON_INTEGER)
                || observation
                    .deadline_ms
                    .is_some_and(|age| age > MAXIMUM_SAFE_JSON_INTEGER)
            {
                return Err("evidence ages must be safe JSON integers".into());
            }
            if let Some(code) = &observation.last_failure_code {
                text(code, "last scheduled-work failure", 128)?;
            }
            let scheduled_work = observation.stage == ProducerStageName::ScheduledPerceptionWork;
            let has_work_timer =
                observation.work_started_age_ms.is_some() && observation.deadline_ms.is_some();
            if (observation.in_flight
                && (!scheduled_work
                    || observation.scheduler_state != Some(ScheduledWorkState::InFlight)
                    || observation.active_job_kind.is_none()
                    || observation.work_started_age_ms.is_some()
                        != observation.deadline_ms.is_some()))
                || (!observation.in_flight
                    && (observation.active_job_kind.is_some()
                        || has_work_timer
                        || observation.scheduler_state == Some(ScheduledWorkState::InFlight)))
            {
                return Err(
                    "in-flight work requires an active job kind and consistent optional timer data"
                        .into(),
                );
            }
            if !scheduled_work
                && (observation.expected_jobs.is_some()
                    || observation.completed_jobs.is_some()
                    || observation.failed_jobs.is_some()
                    || observation.scheduler_state.is_some()
                    || observation.active_job_kind.is_some()
                    || observation.last_completion_age_ms.is_some()
                    || observation.last_failure_code.is_some()
                    || observation.last_failure_age_ms.is_some())
            {
                return Err("scheduler counters belong only to scheduled work".into());
            }
        }
        Ok(())
    }
    pub fn decode(bytes: &[u8], maximum_bytes: usize) -> Result<Self, String> {
        if maximum_bytes == 0
            || maximum_bytes > MAXIMUM_DIAGNOSTICS_MESSAGE_BYTES
            || bytes.is_empty()
            || bytes.len() > maximum_bytes
        {
            return Err("invalid diagnostics payload length or configured limit".into());
        }
        let report: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid diagnostics JSON: {error}"))?;
        report.validate()?;
        Ok(report)
    }
}
fn text(value: &str, label: &str, maximum: usize) -> Result<(), String> {
    if value.trim().is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(format!(
            "{label} must contain 1..={maximum} non-control bytes"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(name: &str) -> ProducerDiagnosticsReport {
        let bytes = match name {
            "active" => {
                include_bytes!("../contracts/producer-diagnostics/v1/active.json").as_slice()
            }
            "loading" => {
                include_bytes!("../contracts/producer-diagnostics/v1/loading.json").as_slice()
            }
            "failed" => {
                include_bytes!("../contracts/producer-diagnostics/v1/failed.json").as_slice()
            }
            _ => unreachable!(),
        };
        ProducerDiagnosticsReport::decode(bytes, MAXIMUM_DIAGNOSTICS_MESSAGE_BYTES).unwrap()
    }
    #[test]
    fn fixtures_cover_actual_active_model_and_pre_scene_loading_failure() {
        let active = fixture("active");
        assert!(active.metadata.model.is_some());
        assert_eq!(active.metadata.model_state, ModelState::Active);
        for name in ["loading", "failed"] {
            let report = fixture(name);
            assert!(report.metadata.model.is_none());
            assert!(report.metadata.capture_session_id.is_none());
        }
    }
    #[test]
    fn build_provenance_is_optional_and_independent_per_binary() {
        let build: BuildInfo =
            serde_json::from_str(r#"{"package_name":"producer","package_version":"1.2.3"}"#)
                .unwrap();
        assert_eq!(build.git_revision, None);
        assert_eq!(build.dirty, None);
        build.validate().unwrap();
        BuildInfo::current().validate().unwrap();
        let mut active = fixture("active");
        active.metadata.build.git_revision = Some("abcdef1234567".into());
        active.validate().unwrap();
        assert_ne!(
            active.metadata.build.git_revision,
            fixture("loading").metadata.build.git_revision
        );
    }
    #[test]
    fn rejects_invalid_versions_shapes_models_ages_and_payload_limits() {
        let report = fixture("active");
        let bytes = serde_json::to_vec(&report).unwrap();
        assert!(ProducerDiagnosticsReport::decode(&bytes, bytes.len() - 1).is_err());
        assert!(ProducerDiagnosticsReport::decode(&bytes, 65_537).is_err());
        assert!(ProducerDiagnosticsReport::decode(&[], 65_536).is_err());
        let mut wrong = report.clone();
        wrong.schema_version = 2;
        assert!(wrong.validate().is_err());
        wrong = report.clone();
        wrong.sequence = 0;
        assert!(wrong.validate().is_err());
        wrong = report.clone();
        wrong.metadata.model_state = ModelState::Loading;
        assert!(wrong.validate().is_err());
        wrong = report.clone();
        wrong
            .metadata
            .model
            .as_mut()
            .unwrap()
            .identity
            .artifact_sha256 = "filename.onnx".into();
        assert!(wrong.validate().is_err());
        wrong = report;
        wrong.stages = vec![StageObservation {
            stage: ProducerStageName::CaptureProgress,
            status: StageStatus::Starting,
            reason_code: "initializing".into(),
            detail: None,
            last_progress_age_ms: Some(MAXIMUM_SAFE_JSON_INTEGER + 1),
            in_flight: false,
            work_started_age_ms: None,
            expected_jobs: None,
            completed_jobs: None,
            failed_jobs: None,
            scheduler_state: None,
            active_job_kind: None,
            last_completion_age_ms: None,
            last_failure_code: None,
            last_failure_age_ms: None,
            deadline_ms: None,
        }];
        assert!(wrong.validate().is_err());
        let malformed = String::from_utf8(bytes).unwrap().replacen(
            "\"schema_version\":1",
            "\"schema_version\":1,\"schema_version\":1",
            1,
        );
        assert!(ProducerDiagnosticsReport::decode(malformed.as_bytes(), 65_536).is_err());
        assert!(ProducerDiagnosticsReport::decode(br#"{"schema_version":NaN}"#, 65_536).is_err());
    }
    #[test]
    fn config_defaults_are_backward_compatible_and_validate_cross_field_policy() {
        assert_eq!(
            DiagnosticsConfig::from_configuration(&serde_json::json!({})).unwrap(),
            DiagnosticsConfig::default()
        );
        let defaults: Value =
            serde_json::from_str(include_str!("../contracts/configuration/v1/defaults.json"))
                .unwrap();
        assert_eq!(
            DiagnosticsConfig::from_configuration(&defaults).unwrap(),
            DiagnosticsConfig::default()
        );
        for patch in [
            serde_json::json!({"bind_address":"0.0.0.0"}),
            serde_json::json!({"report_expiry_ms":1000}),
            serde_json::json!({"maximum_message_bytes":65537}),
            serde_json::json!({"heartbeat_interval_ms":0}),
        ] {
            assert!(DiagnosticsConfig::from_configuration(
                &serde_json::json!({"perception":{"rust_live":{"diagnostics":patch}}})
            )
            .is_err());
        }
    }
}
