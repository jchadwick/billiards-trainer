//! Versioned, full-snapshot contract for the browser diagnostics stream.
//!
//! This contract intentionally separates gateway assessment from producer
//! observations. Producer evidence is only exposed while its gateway-received
//! report is fresh; localization remains based on the scene lifecycle's own
//! evidence clocks.

use serde::{Deserialize, Serialize};

use crate::producer_diagnostics_contract::{
    BuildInfo, ProducerDiagnosticsReport, ProducerExitReason, ProducerMetadata, ScheduledJobKind,
    ScheduledWorkState, StageStatus,
};

pub const BROWSER_DIAGNOSTICS_SCHEMA_VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticsStageName {
    GatewayService,
    SourceLifecycle,
    CaptureProgress,
    ModelInitialization,
    ScheduledPerceptionWork,
    SceneDelivery,
    Localization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageEvidenceSource {
    GatewayRuntime,
    SupervisorRegistry,
    ProducerReport,
    SceneLifecycle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProducerReportState {
    Unknown,
    Fresh,
    Expired,
    Stopped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayRuntimeSnapshot {
    pub instance_id: String,
    pub revision: u64,
    pub assessed_at_utc: String,
    pub uptime_ms: u64,
    pub build: BuildInfo,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerReportSnapshot {
    pub state: ProducerReportState,
    pub generation_id: Option<String>,
    pub report_sequence: Option<u64>,
    pub report_age_ms: Option<u64>,
    pub expires_in_ms: Option<u64>,
    /// Present only when the active supervised report is still fresh.
    pub metadata: Option<ProducerMetadata>,
    /// Stable supervisor outcome; populated only after the active process retires.
    pub exit_reason: Option<ProducerExitReason>,
    /// Most recent accepted report, visible only when it is not active.
    pub last_known: Option<ProducerLastKnownSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerLastKnownSnapshot {
    pub report: ProducerDiagnosticsReport,
    pub report_age_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrowserDiagnosticsStage {
    pub stage: DiagnosticsStageName,
    pub status: StageStatus,
    pub reason_code: String,
    pub source: StageEvidenceSource,
    pub detail: Option<String>,
    pub last_progress_age_ms: Option<u64>,
    pub in_flight: bool,
    pub work_started_age_ms: Option<u64>,
    pub expected_jobs: Option<u64>,
    pub completed_jobs: Option<u64>,
    pub failed_jobs: Option<u64>,
    pub scheduler_state: Option<ScheduledWorkState>,
    pub active_job_kind: Option<ScheduledJobKind>,
    pub last_completion_age_ms: Option<u64>,
    pub last_failure_code: Option<String>,
    pub last_failure_age_ms: Option<u64>,
    pub deadline_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrowserDiagnosticsStreamPolicy {
    pub heartbeat_interval_ms: u64,
    pub stale_after_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrowserDiagnosticsSnapshot {
    pub schema_version: u8,
    pub gateway: GatewayRuntimeSnapshot,
    pub producer: ProducerReportSnapshot,
    pub stream: BrowserDiagnosticsStreamPolicy,
    pub stages: Vec<BrowserDiagnosticsStage>,
}

impl BrowserDiagnosticsSnapshot {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != BROWSER_DIAGNOSTICS_SCHEMA_VERSION {
            return Err("unsupported browser diagnostics schema version".into());
        }
        safe_text(&self.gateway.instance_id, "gateway instance ID", 128)?;
        safe_text(&self.gateway.assessed_at_utc, "assessment timestamp", 64)?;
        if self.gateway.revision == 0 {
            return Err("diagnostics revision must be positive".into());
        }
        self.gateway.build.validate()?;
        if self.stages.len() != 7 {
            return Err("browser diagnostics must contain all seven stages".into());
        }
        let mut stages = std::collections::HashSet::new();
        for stage in &self.stages {
            if !stages.insert(stage.stage) {
                return Err("browser diagnostic stages must be unique".into());
            }
            safe_text(&stage.reason_code, "stage reason", 128)?;
            if stage
                .detail
                .as_ref()
                .is_some_and(|detail| detail.len() > 256 || detail.chars().any(char::is_control))
            {
                return Err("stage detail must be at most 256 non-control bytes".into());
            }
            let has_work_timer = stage.work_started_age_ms.is_some() && stage.deadline_ms.is_some();
            if (stage.in_flight
                && (stage.stage != DiagnosticsStageName::ScheduledPerceptionWork
                    || stage.scheduler_state != Some(ScheduledWorkState::InFlight)
                    || stage.active_job_kind.is_none()
                    || stage.work_started_age_ms.is_some() != stage.deadline_ms.is_some()))
                || (!stage.in_flight
                    && (stage.active_job_kind.is_some()
                        || has_work_timer
                        || stage.scheduler_state == Some(ScheduledWorkState::InFlight)))
            {
                return Err(
                    "in-flight work requires a job kind and consistent optional timer data".into(),
                );
            }
            if stage.stage != DiagnosticsStageName::ScheduledPerceptionWork
                && (stage.expected_jobs.is_some()
                    || stage.completed_jobs.is_some()
                    || stage.failed_jobs.is_some()
                    || stage.scheduler_state.is_some()
                    || stage.active_job_kind.is_some()
                    || stage.last_completion_age_ms.is_some()
                    || stage.last_failure_code.is_some()
                    || stage.last_failure_age_ms.is_some())
            {
                return Err("scheduler counters belong only to scheduled work".into());
            }
            for age in [
                stage.last_progress_age_ms,
                stage.work_started_age_ms,
                stage.last_completion_age_ms,
                stage.last_failure_age_ms,
                stage.deadline_ms,
                stage.expected_jobs,
                stage.completed_jobs,
                stage.failed_jobs,
            ]
            .into_iter()
            .flatten()
            {
                if age > 9_007_199_254_740_991 {
                    return Err("stage counters and ages must be safe JSON integers".into());
                }
            }
            if let Some(code) = &stage.last_failure_code {
                safe_text(code, "last failure code", 128)?;
            }
        }
        if let Some(generation) = &self.producer.generation_id {
            safe_text(generation, "producer generation ID", 128)?;
        }
        if self.producer.state == ProducerReportState::Fresh
            && (self.producer.report_sequence.is_none()
                || self.producer.report_age_ms.is_none()
                || self.producer.metadata.is_none())
        {
            return Err("fresh producer report requires sequence, age, and metadata".into());
        }
        if self.producer.state != ProducerReportState::Fresh && self.producer.metadata.is_some() {
            return Err("stale producer metadata must not be exposed as active".into());
        }
        if let Some(last_known) = &self.producer.last_known {
            last_known.report.validate()?;
            if last_known.report_age_ms > 9_007_199_254_740_991 {
                return Err("last-known report age must be a safe JSON integer".into());
            }
            if self.producer.state == ProducerReportState::Fresh {
                return Err("fresh active reports must not be duplicated as last-known".into());
            }
        }
        if (self.producer.state == ProducerReportState::Stopped)
            != self.producer.exit_reason.is_some()
        {
            return Err("stopped producer requires a stable exit reason".into());
        }
        if !(1..=300_000).contains(&self.stream.heartbeat_interval_ms)
            || !(1..=300_000).contains(&self.stream.stale_after_ms)
            || self.stream.stale_after_ms <= self.stream.heartbeat_interval_ms
        {
            return Err("invalid browser diagnostics stream policy".into());
        }
        Ok(())
    }
}

fn safe_text(value: &str, label: &str, maximum: usize) -> Result<(), String> {
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

    #[test]
    fn no_scene_startup_fixture_is_a_valid_complete_snapshot() {
        let fixture: BrowserDiagnosticsSnapshot = serde_json::from_str(include_str!(
            "../contracts/browser-diagnostics/v1/no-scene-startup.json"
        ))
        .unwrap();
        fixture.validate().unwrap();
        assert_eq!(fixture.gateway.revision, 1);
        assert_eq!(fixture.producer.state, ProducerReportState::Unknown);
        assert_eq!(fixture.producer.metadata, None);
        assert_eq!(fixture.producer.last_known, None);
        assert_eq!(fixture.stages.len(), 7);
    }

    #[test]
    fn expired_fixture_keeps_history_separate_from_active_status() {
        let fixture: BrowserDiagnosticsSnapshot = serde_json::from_str(include_str!(
            "../contracts/browser-diagnostics/v1/last-known-expired.json"
        ))
        .unwrap();
        fixture.validate().unwrap();
        assert_eq!(fixture.producer.state, ProducerReportState::Expired);
        assert!(fixture.producer.metadata.is_none());
        let last_known = fixture.producer.last_known.unwrap();
        assert!(last_known.report.metadata.model.is_some());
        assert_eq!(last_known.report_age_ms, 3_500);
        assert_eq!(
            fixture.stages[1].status,
            StageStatus::Unknown,
            "historical positive evidence does not make the active stage healthy"
        );
    }

    #[test]
    fn rejects_stale_active_metadata_and_incomplete_stage_sets() {
        let mut fixture: BrowserDiagnosticsSnapshot = serde_json::from_str(include_str!(
            "../contracts/browser-diagnostics/v1/no-scene-startup.json"
        ))
        .unwrap();
        fixture.producer.state = ProducerReportState::Fresh;
        assert!(fixture.validate().is_err());
        fixture.producer.state = ProducerReportState::Unknown;
        fixture.stages.pop();
        assert!(fixture.validate().is_err());
    }

    #[test]
    fn browser_schema_keeps_embedded_producer_metadata_in_sync() {
        let browser: serde_json::Value = serde_json::from_str(include_str!(
            "../contracts/browser-diagnostics/v1/schema.json"
        ))
        .unwrap();
        let producer: serde_json::Value = serde_json::from_str(include_str!(
            "../contracts/producer-diagnostics/v1/schema.json"
        ))
        .unwrap();
        assert_eq!(
            browser["$defs"]["producerMetadata"],
            producer["properties"]["metadata"]
        );
        assert_eq!(
            browser["$defs"]["build"],
            producer["properties"]["metadata"]["properties"]["build"]
        );
        assert_eq!(browser["$defs"]["producerReport"], producer);
    }
}
