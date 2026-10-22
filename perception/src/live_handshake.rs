//! Strict parser and validator for the live-v3 producer handshake.
//!
//! The acknowledgement is provenance and request correlation, not
//! authentication and not proof that a physical camera applied a setting.
//! Every value below is source-reported metadata. This module validates that
//! metadata against the exact Rust-resolved capture request before a caller
//! may mark the producer ready.

use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::de::Deserializer;
use serde::Deserialize;
use serde_json::Value;

use crate::capture::{
    CaptureAppliedCameraControls, CaptureAppliedFrameGeometry, CaptureCapability,
    CaptureControlReadback, CaptureExposureMode, CaptureFrameRate, CaptureOutputFrameGeometry,
    CapturePixelFormat, CaptureSessionReport, CaptureSourceCapabilities, CaptureSourceMetadata,
    CaptureTimestampSemantics, CaptureVideoProfile,
};
use crate::capture_settings_wire::ResolvedCaptureSettings;
use crate::detection_batch::{ClockKind, TimeUnit, TimestampStage};
use crate::live_config::{CapturePreflightError, CaptureSessionCompatibility};

const LIVE_PROTOCOL_VERSION: u8 = 3;
const ACK_SCHEMA_VERSION_V2: u8 = 2;
const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_IDENTIFIER_CHARACTERS: usize = 128;

/// Producer identity announced by a live-v3 scene-stream connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveProducerIdentity {
    pub engine_instance_id: String,
    pub source_session_id: String,
}

/// A handshake accepted against the exact resolved live capture request.
///
/// `last_transport_sequence` is intentionally absent: the producer hello does
/// not carry it. The receiving server owns that watermark and includes it in
/// its `ready` response after applying its epoch/restart policy.
#[derive(Debug, Clone)]
pub struct ValidatedLiveHandshake {
    pub identity: LiveProducerIdentity,
    pub capture_settings: ResolvedCaptureSettings,
    pub acknowledgement: ValidatedCaptureSessionAcknowledgement,
    pub compatibility: CaptureSessionCompatibility,
}

/// Validated request-bound evidence from the capture producer.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedCaptureSessionAcknowledgement {
    pub configuration_revision: u64,
    pub request_sha256: String,
    pub session_id: String,
    /// The original source report, retained for diagnostics and repeated-hello
    /// consistency checks. It remains source-reported evidence.
    pub source_report: Value,
}

/// Rejected live producer hello or request-bound capture acknowledgement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveHandshakeError(String);

impl Display for LiveHandshakeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for LiveHandshakeError {}

/// Parse and validate a complete live-v3 hello against the current resolved
/// capture settings. The caller must only send `ready` after this succeeds.
/// The request hash is compared with the digest computed from the exact
/// settings bytes and retained in `ResolvedCaptureSettings`; no request
/// normalization occurs here.
pub fn validate_live_v3_hello(
    hello_json: &[u8],
    capture_settings: &ResolvedCaptureSettings,
) -> Result<ValidatedLiveHandshake, LiveHandshakeError> {
    if capture_settings.schema_version != 3 {
        return Err(LiveHandshakeError(
            "live-v3 handshake requires capture settings schema v3".to_owned(),
        ));
    }

    let hello_value: Value = serde_json::from_slice(hello_json)
        .map_err(|error| LiveHandshakeError(format!("live-v3 hello is malformed: {error}")))?;
    // Decode from the original byte slice as well as Value. This catches
    // duplicate struct keys, which a generic JSON Value would otherwise
    // overwrite silently.
    let hello: LiveV3Hello = serde_json::from_slice(hello_json)
        .map_err(|error| LiveHandshakeError(format!("live-v3 hello is malformed: {error}")))?;
    if hello.message_type != "hello" || hello.protocol_version != LIVE_PROTOCOL_VERSION {
        return Err(LiveHandshakeError(
            "live-v3 hello type or protocol version is invalid".to_owned(),
        ));
    }
    validate_identifier(&hello.engine_instance_id, "engine_instance_id")?;
    validate_identifier(&hello.source_session_id, "source_session_id")?;

    let acknowledgement = hello.capture_session_ack;
    if acknowledgement.schema_version != ACK_SCHEMA_VERSION_V2 {
        return Err(LiveHandshakeError(
            "live-v3 capture acknowledgement must use schema v2".to_owned(),
        ));
    }
    if acknowledgement.configuration_revision > MAX_SAFE_JSON_INTEGER {
        return Err(LiveHandshakeError(
            "capture acknowledgement revision exceeds the safe JSON integer range".to_owned(),
        ));
    }
    if acknowledgement.configuration_revision != capture_settings.configuration_revision {
        return Err(LiveHandshakeError(
            "capture acknowledgement revision is stale".to_owned(),
        ));
    }
    if acknowledgement.request_sha256 != capture_settings.request_sha256 {
        return Err(LiveHandshakeError(
            "capture acknowledgement request digest differs".to_owned(),
        ));
    }
    validate_acknowledgement_shape(&acknowledgement)?;
    if acknowledgement.outcome != AcknowledgementOutcome::Compatible {
        return Err(LiveHandshakeError(
            "capture session outcome is not compatible".to_owned(),
        ));
    }
    let source_report = acknowledgement.source_report.ok_or_else(|| {
        LiveHandshakeError("compatible capture acknowledgement omitted source_report".to_owned())
    })?;
    if source_report.session_id != hello.source_session_id {
        return Err(LiveHandshakeError(
            "capture acknowledgement session ID differs from the producer hello".to_owned(),
        ));
    }

    let report_value = hello_value
        .get("capture_session_ack")
        .and_then(|acknowledgement| acknowledgement.get("source_report"))
        .cloned()
        .ok_or_else(|| {
            LiveHandshakeError(
                "compatible capture acknowledgement omitted source_report".to_owned(),
            )
        })?;
    let report = source_report.into_domain().map_err(|error| {
        LiveHandshakeError(format!("capture source report is invalid: {error}"))
    })?;
    let compatibility = capture_settings
        .expected
        .validate_session(&report)
        .map_err(|error| LiveHandshakeError(format_preflight_error(&error)))?;

    Ok(ValidatedLiveHandshake {
        identity: LiveProducerIdentity {
            engine_instance_id: hello.engine_instance_id,
            source_session_id: hello.source_session_id.clone(),
        },
        capture_settings: capture_settings.clone(),
        acknowledgement: ValidatedCaptureSessionAcknowledgement {
            configuration_revision: acknowledgement.configuration_revision,
            request_sha256: acknowledgement.request_sha256,
            session_id: hello.source_session_id,
            source_report: report_value,
        },
        compatibility,
    })
}

fn format_preflight_error(error: &CapturePreflightError) -> String {
    format!("capture source report is incompatible with the request: {error}")
}

fn validate_acknowledgement_shape(
    acknowledgement: &CaptureSessionAcknowledgementV2,
) -> Result<(), LiveHandshakeError> {
    let valid = match acknowledgement.outcome {
        AcknowledgementOutcome::Compatible => {
            acknowledgement.source_report.is_some() && acknowledgement.failure.is_none()
        }
        AcknowledgementOutcome::Rejected => {
            acknowledgement.source_report.is_some()
                && acknowledgement.failure.as_ref().is_some_and(|failure| {
                    failure.category == AcknowledgementFailureCategory::PreflightRejected
                })
        }
        AcknowledgementOutcome::Failed => acknowledgement.failure.as_ref().is_some_and(|failure| {
            matches!(
                failure.category,
                AcknowledgementFailureCategory::SourceOpenFailed
                    | AcknowledgementFailureCategory::RuntimeFailed
            )
        }),
        AcknowledgementOutcome::Stopped => acknowledgement
            .failure
            .as_ref()
            .is_some_and(|failure| failure.category == AcknowledgementFailureCategory::Stopped),
    };
    if valid {
        Ok(())
    } else {
        Err(LiveHandshakeError(
            "capture acknowledgement fields do not match its outcome".to_owned(),
        ))
    }
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), LiveHandshakeError> {
    if value.is_empty()
        || value.chars().count() > MAX_IDENTIFIER_CHARACTERS
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b':' | b'-'))
    {
        return Err(LiveHandshakeError(format!(
            "{field} must be a bounded ASCII identifier"
        )));
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LiveV3Hello {
    #[serde(rename = "type")]
    message_type: String,
    protocol_version: u8,
    engine_instance_id: String,
    source_session_id: String,
    capture_session_ack: CaptureSessionAcknowledgementV2,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureSessionAcknowledgementV2 {
    schema_version: u8,
    configuration_revision: u64,
    request_sha256: String,
    outcome: AcknowledgementOutcome,
    #[serde(deserialize_with = "required_option")]
    source_report: Option<CaptureSourceReportV2>,
    #[serde(deserialize_with = "required_option")]
    failure: Option<AcknowledgementFailure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AcknowledgementOutcome {
    Compatible,
    Rejected,
    Failed,
    Stopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AcknowledgementFailureCategory {
    PreflightRejected,
    SourceOpenFailed,
    RuntimeFailed,
    Stopped,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AcknowledgementFailure {
    category: AcknowledgementFailureCategory,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureSourceReportV2 {
    session_id: String,
    device_id: String,
    #[serde(deserialize_with = "required_option")]
    device_name: Option<String>,
    backend_id: String,
    requested_profile: CaptureProfileV2,
    negotiated_profile: CaptureProfileV2,
    capabilities: CaptureCapabilitiesV2,
    applied_frame_geometry: AppliedFrameGeometryV2,
    applied_controls: AppliedControlsV2,
    timestamp_semantics: TimestampSemanticsV2,
    output_frame_geometry: OutputFrameGeometryV2,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureProfileV2 {
    width_px: u32,
    height_px: u32,
    frame_rate: FrameRateV2,
    pixel_format: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrameRateV2 {
    frames_numerator: u32,
    seconds_denominator: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureCapabilitiesV2 {
    #[serde(deserialize_with = "required_option")]
    supported_pixel_formats: Option<Vec<String>>,
    manual_exposure: CaptureCapabilityV2,
    automatic_exposure: CaptureCapabilityV2,
    gain_control: CaptureCapabilityV2,
    source_timestamps: CaptureCapabilityV2,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CaptureCapabilityV2 {
    Supported,
    Unsupported,
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum TimestampSemanticsV2 {
    ReadCompletionOnly,
    SourceTimestamp {
        unit: TimeUnit,
        clock: ClockKind,
        domain_id: String,
        stage: TimestampStage,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AppliedFrameGeometryV2 {
    configuration_revision: u64,
    camera_profile_id: String,
    image_space: crate::ImageSpace,
    rectification_revision: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AppliedControlsV2 {
    exposure_mode: ControlReadbackV2<CaptureExposureModeV2>,
    exposure_value: ControlReadbackV2<f64>,
    gain: ControlReadbackV2<f64>,
    brightness: ControlReadbackV2<f64>,
    contrast: ControlReadbackV2<f64>,
    saturation: ControlReadbackV2<f64>,
    auto_focus: ControlReadbackV2<bool>,
    buffer_size: ControlReadbackV2<u32>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CaptureExposureModeV2 {
    Automatic,
    Manual,
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "status",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum ControlReadbackV2<T> {
    Applied(T),
    Unsupported,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputFrameGeometryV2 {
    width_px: u32,
    height_px: u32,
}

impl CaptureSourceReportV2 {
    fn into_domain(self) -> Result<CaptureSessionReport, String> {
        validate_identifier(&self.session_id, "source_report.session_id")
            .map_err(|error| error.to_string())?;
        validate_identifier(&self.device_id, "source_report.device_id")
            .map_err(|error| error.to_string())?;
        if let Some(device_name) = self.device_name.as_deref() {
            if device_name.chars().count() > MAX_IDENTIFIER_CHARACTERS
                || device_name.chars().any(char::is_control)
            {
                return Err("source_report.device_name is invalid".to_owned());
            }
        }
        if !matches!(
            self.backend_id.as_str(),
            "v4l2" | "dshow" | "gstreamer" | "opencv" | "kinect2"
        ) {
            return Err("source_report.backend_id is unsupported".to_owned());
        }
        if let TimestampSemanticsV2::SourceTimestamp { domain_id, .. } = &self.timestamp_semantics {
            validate_identifier(domain_id, "source_report.timestamp_semantics.domain_id")
                .map_err(|error| error.to_string())?;
        }

        let requested_profile = self.requested_profile.into_domain()?;
        let negotiated_profile = self.negotiated_profile.into_domain()?;
        let supported_pixel_formats = self
            .capabilities
            .supported_pixel_formats
            .map(|formats| {
                if formats.len() > 64 {
                    return Err(
                        "source_report.capabilities.supported_pixel_formats exceeds 64 entries"
                            .to_owned(),
                    );
                }
                formats
                    .into_iter()
                    .map(CapturePixelFormat::new)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| error.to_string())
            })
            .transpose()?;
        let capabilities = CaptureSourceCapabilities::new(
            supported_pixel_formats,
            self.capabilities.manual_exposure.into_domain(),
            self.capabilities.automatic_exposure.into_domain(),
            self.capabilities.gain_control.into_domain(),
            self.capabilities.source_timestamps.into_domain(),
        )
        .map_err(|error| error.to_string())?;
        let applied_frame_geometry = CaptureAppliedFrameGeometry::new(
            self.applied_frame_geometry.configuration_revision,
            self.applied_frame_geometry.camera_profile_id,
            self.applied_frame_geometry.image_space,
            self.applied_frame_geometry.rectification_revision,
        )
        .map_err(|error| error.to_string())?;
        let applied_controls = self.applied_controls.into_domain()?;
        let output_frame_geometry = CaptureOutputFrameGeometry::new(
            self.output_frame_geometry.width_px,
            self.output_frame_geometry.height_px,
        )
        .map_err(|error| error.to_string())?;
        let metadata = CaptureSourceMetadata::new(
            self.device_id,
            self.device_name,
            self.backend_id,
            requested_profile,
            negotiated_profile,
            capabilities,
            applied_frame_geometry,
            applied_controls,
            self.timestamp_semantics.into_domain(),
        )
        .map_err(|error| error.to_string())?
        .with_output_frame_geometry(output_frame_geometry);
        Ok(CaptureSessionReport::new(self.session_id, metadata))
    }
}

impl CaptureProfileV2 {
    fn into_domain(self) -> Result<CaptureVideoProfile, String> {
        let frame_rate = CaptureFrameRate::new(
            self.frame_rate.frames_numerator,
            self.frame_rate.seconds_denominator,
        )
        .map_err(|error| error.to_string())?;
        let pixel_format =
            CapturePixelFormat::new(self.pixel_format).map_err(|error| error.to_string())?;
        CaptureVideoProfile::new(self.width_px, self.height_px, frame_rate, pixel_format)
            .map_err(|error| error.to_string())
    }
}

impl CaptureCapabilityV2 {
    fn into_domain(self) -> CaptureCapability {
        match self {
            Self::Supported => CaptureCapability::Supported,
            Self::Unsupported => CaptureCapability::Unsupported,
            Self::Unknown => CaptureCapability::Unknown,
        }
    }
}

impl TimestampSemanticsV2 {
    fn into_domain(self) -> CaptureTimestampSemantics {
        match self {
            Self::ReadCompletionOnly => CaptureTimestampSemantics::ReadCompletionOnly,
            Self::SourceTimestamp {
                unit,
                clock,
                domain_id,
                stage,
            } => CaptureTimestampSemantics::SourceTimestamp {
                unit,
                clock,
                domain_id,
                stage,
            },
        }
    }
}

impl AppliedControlsV2 {
    fn into_domain(self) -> Result<CaptureAppliedCameraControls, String> {
        CaptureAppliedCameraControls::new(
            map_readback(self.exposure_mode, |mode| match mode {
                CaptureExposureModeV2::Automatic => CaptureExposureMode::Automatic,
                CaptureExposureModeV2::Manual => CaptureExposureMode::Manual,
            }),
            map_readback(self.exposure_value, |value| value),
            map_readback(self.gain, |value| value),
            map_readback(self.brightness, |value| value),
            map_readback(self.contrast, |value| value),
            map_readback(self.saturation, |value| value),
            map_readback(self.auto_focus, |value| value),
            map_readback(self.buffer_size, |value| value),
        )
        .map_err(|error| error.to_string())
    }
}

fn map_readback<T, U>(
    value: ControlReadbackV2<T>,
    map: impl FnOnce(T) -> U,
) -> CaptureControlReadback<U> {
    match value {
        ControlReadbackV2::Applied(value) => CaptureControlReadback::Applied(map(value)),
        ControlReadbackV2::Unsupported => CaptureControlReadback::Unsupported,
        ControlReadbackV2::Unknown => CaptureControlReadback::Unknown,
    }
}

fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(deserializer)
}
