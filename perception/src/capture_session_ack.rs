//! Request-bound evidence that one opened capture session passed Rust
//! preflight. This is provenance and configuration correlation, not
//! authentication or proof of physical camera behavior.

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;

use crate::live_config::CaptureSessionCompatibility;

pub const CAPTURE_SESSION_ACK_SCHEMA_VERSION: u8 = 1;
pub const CAPTURE_SESSION_ACK_SCHEMA_VERSION_V2: u8 = 2;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum CaptureSessionOutcome {
    Compatible,
}

/// A compatible source report bound to the exact capture request.
///
/// Construction requires a `CaptureSessionCompatibility`, which can only be
/// produced after Rust compared the opened source report with requested
/// settings. The report is still source-provided metadata, not hardware proof.
#[derive(Debug, Clone, Serialize)]
pub struct CaptureSessionAcknowledgementV1 {
    schema_version: u8,
    configuration_revision: u64,
    request_sha256: String,
    outcome: CaptureSessionOutcome,
    source_report: Value,
    failure: Option<Value>,
}

impl CaptureSessionAcknowledgementV1 {
    pub fn from_compatible_session(compatibility: &CaptureSessionCompatibility) -> Result<Self> {
        ensure!(
            compatibility
                .expected_settings()
                .output_frame_geometry()
                .is_none(),
            "v2 capture settings require a v2 session acknowledgement"
        );
        let request_sha256 = compatibility
            .request_sha256()
            .context("live acknowledgement requires capture request bytes")?;
        ensure!(
            compatibility.source_report().session_id() == compatibility.session_id(),
            "capture compatibility report session identity is inconsistent"
        );
        let source_report = serde_json::to_value(compatibility.source_report())
            .context("failed to serialize compatible capture source report")?;
        Ok(Self {
            schema_version: CAPTURE_SESSION_ACK_SCHEMA_VERSION,
            configuration_revision: compatibility.configuration_revision(),
            request_sha256: request_sha256.to_owned(),
            outcome: CaptureSessionOutcome::Compatible,
            source_report,
            failure: None,
        })
    }

    pub fn configuration_revision(&self) -> u64 {
        self.configuration_revision
    }

    pub fn request_sha256(&self) -> &str {
        &self.request_sha256
    }

    pub fn session_id(&self) -> Option<&str> {
        self.source_report.get("session_id").and_then(Value::as_str)
    }
}

/// Request-bound v2 acknowledgement. Its source report must contain the
/// explicit emitted RGB8 geometry validated during preflight.
#[derive(Debug, Clone, Serialize)]
pub struct CaptureSessionAcknowledgementV2 {
    schema_version: u8,
    configuration_revision: u64,
    request_sha256: String,
    outcome: CaptureSessionOutcome,
    source_report: Value,
    failure: Option<Value>,
}

impl CaptureSessionAcknowledgementV2 {
    pub fn from_compatible_session(compatibility: &CaptureSessionCompatibility) -> Result<Self> {
        ensure!(
            compatibility
                .expected_settings()
                .output_frame_geometry()
                .is_some(),
            "v2 acknowledgement requires v2 capture settings"
        );
        let request_sha256 = compatibility
            .request_sha256()
            .context("live acknowledgement requires capture request bytes")?;
        ensure!(
            compatibility.source_report().session_id() == compatibility.session_id(),
            "capture compatibility report session identity is inconsistent"
        );
        let output_geometry = compatibility
            .source_report()
            .output_frame_geometry()
            .context("v2 source report omitted emitted RGB8 geometry")?;
        ensure!(
            output_geometry == compatibility.output_frame_geometry(),
            "v2 source report geometry differs from preflight compatibility"
        );
        let source_report = serde_json::to_value(compatibility.source_report())
            .context("failed to serialize compatible capture source report")?;
        Ok(Self {
            schema_version: CAPTURE_SESSION_ACK_SCHEMA_VERSION_V2,
            configuration_revision: compatibility.configuration_revision(),
            request_sha256: request_sha256.to_owned(),
            outcome: CaptureSessionOutcome::Compatible,
            source_report,
            failure: None,
        })
    }

    pub fn configuration_revision(&self) -> u64 {
        self.configuration_revision
    }

    pub fn request_sha256(&self) -> &str {
        &self.request_sha256
    }

    pub fn session_id(&self) -> Option<&str> {
        self.source_report.get("session_id").and_then(Value::as_str)
    }
}

/// Untagged sum type that preserves each acknowledgement's exact JSON shape.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum CaptureSessionAcknowledgement {
    V1(CaptureSessionAcknowledgementV1),
    V2(CaptureSessionAcknowledgementV2),
}

impl CaptureSessionAcknowledgement {
    pub fn from_compatible_session(compatibility: &CaptureSessionCompatibility) -> Result<Self> {
        if compatibility
            .expected_settings()
            .output_frame_geometry()
            .is_some()
        {
            Ok(Self::V2(
                CaptureSessionAcknowledgementV2::from_compatible_session(compatibility)?,
            ))
        } else {
            Ok(Self::V1(
                CaptureSessionAcknowledgementV1::from_compatible_session(compatibility)?,
            ))
        }
    }

    pub fn configuration_revision(&self) -> u64 {
        match self {
            Self::V1(acknowledgement) => acknowledgement.configuration_revision(),
            Self::V2(acknowledgement) => acknowledgement.configuration_revision(),
        }
    }

    pub fn request_sha256(&self) -> &str {
        match self {
            Self::V1(acknowledgement) => acknowledgement.request_sha256(),
            Self::V2(acknowledgement) => acknowledgement.request_sha256(),
        }
    }

    pub fn session_id(&self) -> Option<&str> {
        match self {
            Self::V1(acknowledgement) => acknowledgement.session_id(),
            Self::V2(acknowledgement) => acknowledgement.session_id(),
        }
    }
}

impl From<CaptureSessionAcknowledgementV1> for CaptureSessionAcknowledgement {
    fn from(value: CaptureSessionAcknowledgementV1) -> Self {
        Self::V1(value)
    }
}

impl From<CaptureSessionAcknowledgementV2> for CaptureSessionAcknowledgement {
    fn from(value: CaptureSessionAcknowledgementV2) -> Self {
        Self::V2(value)
    }
}

#[cfg(test)]
pub(crate) fn default_test_acknowledgement() -> CaptureSessionAcknowledgementV1 {
    let fixture: Value = serde_json::from_str(include_str!(
        "../contracts/capture-session-ack/v1/capture-session-ack-default.json"
    ))
    .expect("checked-in acknowledgement fixture must be valid JSON");
    CaptureSessionAcknowledgementV1 {
        schema_version: CAPTURE_SESSION_ACK_SCHEMA_VERSION,
        configuration_revision: fixture["configuration_revision"]
            .as_u64()
            .expect("fixture configuration revision must be an integer"),
        request_sha256: fixture["request_sha256"]
            .as_str()
            .expect("fixture request digest must be text")
            .to_owned(),
        outcome: CaptureSessionOutcome::Compatible,
        source_report: fixture["source_report"].clone(),
        failure: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{CaptureSessionAcknowledgementV1, CAPTURE_SESSION_ACK_SCHEMA_VERSION};

    #[test]
    fn acknowledgement_serializes_the_schema_fields_without_inventing_readbacks() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../contracts/capture-session-ack/v1/capture-session-ack-default.json"
        ))
        .unwrap();
        let acknowledgement = CaptureSessionAcknowledgementV1 {
            schema_version: CAPTURE_SESSION_ACK_SCHEMA_VERSION,
            configuration_revision: fixture["configuration_revision"].as_u64().unwrap(),
            request_sha256: fixture["request_sha256"].as_str().unwrap().to_owned(),
            outcome: super::CaptureSessionOutcome::Compatible,
            source_report: fixture["source_report"].clone(),
            failure: None,
        };

        let serialized = serde_json::to_value(acknowledgement).unwrap();
        assert_eq!(serialized, fixture);
    }
}
