//! Strict v1/v2/v3 capture request contracts and Rust-side request generation.
//!
//! The snapshot is a request. Decoding it does not open hardware or assert that
//! a camera applied any value. The capture coordinator still compares the
//! request with the actual source report before its first read, and that report
//! only verifies fields it explicitly contains.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::time::Duration;

use serde::de::Deserializer;
use serde::Deserialize;
use serde_json::{json, Number, Value};
use sha2::{Digest, Sha256};

use crate::calibration::{CalibrationError, TableCalibrationV1};
use crate::capture::{
    CaptureAggregateFrameLimit, CaptureExposureMode, CaptureFrameLimit, CaptureFrameLimitError,
    CaptureFrameRate, CaptureMetadataError, CaptureOutputFrameGeometry, CapturePixelFormat,
    CaptureTimestampSemantics, CaptureVideoProfile,
};
use crate::detection_batch::{ClockKind, TimeUnit, TimestampStage};
use crate::live_config::{
    CaptureControlRequirements, CapturePreflightError, ExpectedCaptureSettings,
    RequestedCaptureControlValues, SUPPORTED_CAPTURE_BACKENDS,
};
use crate::ImageSpace;

const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_CAPTURE_FRAME_BYTES: u64 = 1_073_741_824;
const HTTP_MJPEG_STREAM_DEVICE_ID: &str = "remote-http-mjpeg-stream";

/// Versioned configuration request consumed by the Rust capture runtime.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSettingsSnapshotV1 {
    pub schema_version: u8,
    pub configuration_revision: u64,
    pub requested: CaptureSettingsRequestV1,
}

/// Requested capture properties. Source-applied values arrive separately in a
/// `CaptureSessionReport`; they are never inferred from this request.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSettingsRequestV1 {
    pub device_id: String,
    pub backend: String,
    pub requested_profile: CaptureVideoProfileV1,
    pub expected_negotiated_profile: CaptureVideoProfileV1,
    pub camera_profile_id: String,
    pub image_space: ImageSpace,
    pub rectification_revision: u64,
    pub calibration_revision: u64,
    #[serde(deserialize_with = "required_option")]
    pub calibration: Option<TableCalibrationV1>,
    pub timestamp_semantics: CaptureTimestampSemanticsV1,
    pub control_requirements: CaptureControlRequirementsV1,
    pub requested_controls: RequestedCaptureControlsV1,
    pub reconnect_policy: CaptureReconnectPolicyV1,
    pub maximum_frame_age_ms: u64,
    pub maximum_rgb_frame_bytes: u64,
    pub maximum_aggregate_rgb_frame_bytes: u64,
}

/// Versioned v2 request. Its additional geometry is mandatory and describes
/// the RGB8 image emitted to inference, not the negotiated camera input mode.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSettingsSnapshotV2 {
    pub schema_version: u8,
    pub configuration_revision: u64,
    pub requested: CaptureSettingsRequestV2,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSettingsRequestV2 {
    pub device_id: String,
    pub backend: String,
    pub requested_profile: CaptureVideoProfileV1,
    pub expected_negotiated_profile: CaptureVideoProfileV1,
    pub output_frame_geometry: CaptureOutputFrameGeometryV2,
    pub camera_profile_id: String,
    pub image_space: ImageSpace,
    pub rectification_revision: u64,
    pub calibration_revision: u64,
    #[serde(deserialize_with = "required_option")]
    pub calibration: Option<TableCalibrationV1>,
    pub timestamp_semantics: CaptureTimestampSemanticsV1,
    pub control_requirements: CaptureControlRequirementsV1,
    pub requested_controls: RequestedCaptureControlsV1,
    pub reconnect_policy: CaptureReconnectPolicyV1,
    pub maximum_frame_age_ms: u64,
    pub maximum_rgb_frame_bytes: u64,
    pub maximum_aggregate_rgb_frame_bytes: u64,
}

/// Versioned v3 request. V3 retains v2's explicit emitted RGB8 geometry and
/// replaces the unconditional control request with an explicit policy.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSettingsSnapshotV3 {
    pub schema_version: u8,
    pub configuration_revision: u64,
    pub requested: CaptureSettingsRequestV3,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSettingsRequestV3 {
    pub device_id: String,
    pub backend: String,
    pub requested_profile: CaptureVideoProfileV1,
    pub expected_negotiated_profile: CaptureVideoProfileV1,
    pub output_frame_geometry: CaptureOutputFrameGeometryV2,
    pub camera_profile_id: String,
    pub image_space: ImageSpace,
    pub rectification_revision: u64,
    pub calibration_revision: u64,
    #[serde(deserialize_with = "required_option")]
    pub calibration: Option<TableCalibrationV1>,
    pub timestamp_semantics: CaptureTimestampSemanticsV1,
    pub camera_controls: CameraControlsPolicyV3,
    pub reconnect_policy: CaptureReconnectPolicyV1,
    pub maximum_frame_age_ms: u64,
    pub maximum_rgb_frame_bytes: u64,
    pub maximum_aggregate_rgb_frame_bytes: u64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureOutputFrameGeometryV2 {
    pub width_px: u32,
    pub height_px: u32,
}

/// Requested or expected mode in the source-neutral profile representation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureVideoProfileV1 {
    pub width_px: u32,
    pub height_px: u32,
    pub frame_rate: CaptureFrameRateV1,
    pub pixel_format: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureFrameRateV1 {
    pub frames_numerator: u32,
    pub seconds_denominator: u32,
}

/// Requested source timestamp semantics. Read-completion timestamps remain
/// distinct from camera exposure timestamps throughout the pipeline.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CaptureTimestampSemanticsV1 {
    ReadCompletionOnly,
    SourceTimestamp {
        unit: TimeUnit,
        clock: ClockKind,
        domain_id: String,
        stage: TimestampStage,
    },
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureControlRequirementsV1 {
    pub manual_exposure: bool,
    pub automatic_exposure: bool,
    pub gain_control: bool,
}

/// V3 explicitly distinguishes source-owned defaults from strict requested
/// camera control values. The tag is closed: unknown modes and extra fields
/// fail deserialization.
#[derive(Debug, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum CameraControlsPolicyV3 {
    SourceDefaults {},
    Require {
        requirements: CaptureControlRequirementsV1,
        requested: RequestedCaptureControlsV1,
    },
}

/// Concrete values the source is asked to use. They remain requested until
/// the adapter adds and reports actual applied controls.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestedCaptureControlsV1 {
    pub exposure_mode: ExposureModeV1,
    #[serde(deserialize_with = "required_option")]
    pub exposure_value: Option<f64>,
    pub gain: f64,
    pub brightness: f64,
    pub contrast: f64,
    pub saturation: f64,
    pub auto_focus: bool,
    pub buffer_size: u32,
}

/// Requested lifecycle behavior retained from Python's effective camera
/// settings. It is not enabled until a Rust provider implements it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureReconnectPolicyV1 {
    pub enabled: bool,
    pub delay_seconds: f64,
    pub maximum_attempts: u32,
    pub stop_timeout_seconds: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExposureModeV1 {
    Auto,
    Manual,
}

/// Validated request ready to pass into capture preflight and a future adapter.
#[derive(Debug, Clone)]
pub struct ResolvedCaptureSettings {
    /// The exact supported request contract used to create this value.
    pub schema_version: u8,
    pub configuration_revision: u64,
    /// SHA-256 of the exact request bytes provided to the Rust capture runtime.
    /// Supervisors must pass those bytes through unchanged after generation.
    pub request_sha256: String,
    pub expected: ExpectedCaptureSettings,
    pub requested_backend: String,
    pub camera_controls: ResolvedCameraControlsPolicy,
    pub reconnect_policy: CaptureReconnectPolicyV1,
}

/// Validated policy retained by the runtime-facing settings value. Source
/// defaults carry no invented requested values; strict requests retain the
/// exact requested settings and capability requirements.
#[derive(Debug, Clone)]
pub enum ResolvedCameraControlsPolicy {
    SourceDefaults,
    Require {
        requirements: CaptureControlRequirementsV1,
        requested: RequestedCaptureControlsV1,
    },
}

/// Backward-compatible name retained for callers that parse v1 directly.
pub type ResolvedCaptureSettingsV1 = ResolvedCaptureSettings;

/// Parse a supported strict capture-settings contract by its explicit
/// version. V1 callers that need to reject any version other than 1 should use
/// [`CaptureSettingsSnapshotV1::parse_json`] directly.
pub fn parse_capture_settings_json(
    json: &str,
) -> Result<ResolvedCaptureSettingsV1, CaptureSettingsWireError> {
    #[derive(Deserialize)]
    struct Version {
        schema_version: u8,
    }

    let version: Version = serde_json::from_str(json).map_err(CaptureSettingsWireError::Json)?;
    match version.schema_version {
        1 => CaptureSettingsSnapshotV1::parse_json(json),
        2 => CaptureSettingsSnapshotV2::parse_json(json),
        3 => CaptureSettingsSnapshotV3::parse_json(json),
        unsupported => Err(CaptureSettingsWireError::UnsupportedSchemaVersion(
            unsupported,
        )),
    }
}

#[derive(Debug)]
pub enum CaptureSettingsWireError {
    Json(serde_json::Error),
    UnsupportedSchemaVersion(u8),
    InvalidConfiguration(String),
    UnsafeRevision(&'static str),
    InvalidRequest(&'static str),
    InvalidFrameLimit(CaptureFrameLimitError),
    InvalidMetadata(CaptureMetadataError),
    InvalidCalibration(CalibrationError),
    InvalidPreflight(CapturePreflightError),
}

impl Display for CaptureSettingsWireError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "capture settings JSON: {error}"),
            Self::UnsupportedSchemaVersion(version) => {
                write!(
                    formatter,
                    "unsupported capture settings schema version {version}"
                )
            }
            Self::InvalidConfiguration(reason) => {
                write!(
                    formatter,
                    "invalid effective capture configuration: {reason}"
                )
            }
            Self::UnsafeRevision(field) => write!(formatter, "unsafe capture revision: {field}"),
            Self::InvalidRequest(reason) => write!(formatter, "invalid capture request: {reason}"),
            Self::InvalidFrameLimit(error) => {
                write!(formatter, "invalid capture frame limit: {error}")
            }
            Self::InvalidMetadata(error) => write!(formatter, "invalid capture metadata: {error}"),
            Self::InvalidCalibration(error) => {
                write!(formatter, "invalid calibration bundle: {error}")
            }
            Self::InvalidPreflight(error) => {
                write!(formatter, "invalid capture preflight: {error}")
            }
        }
    }
}

/// Build the production v3 request from an effective unified configuration.
///
/// The caller supplies the complete configuration with the versioned Rust
/// defaults merged into it. Keeping this builder in the Rust domain crate
/// makes configuration resolution, profile identity and the strict wire
/// parser one implementation. Calibration is deliberately emitted as null:
/// the production supervisor currently supplies no provenance-bound bundle.
///
/// The returned `Value` is serialized with `serde_json::to_string` by the
/// authority command. The JSON map implementation sorts object keys, matching
/// Python's compact `sort_keys=True` encoding for the ASCII-only request fields
/// used by v3. The strict parser is run over those exact bytes before return,
/// so the request digest in the resolved value is the same digest used by the
/// capture helper and acknowledgement contract.
pub fn build_capture_settings_v3(configuration: &Value) -> Result<Value, CaptureSettingsWireError> {
    let configuration_revision = config_integer(
        configuration,
        "metadata.configuration_revision",
        1,
        MAX_SAFE_JSON_INTEGER,
    )?;
    let video_file_path = config_value(configuration, "vision.camera.video_file_path")?;
    match video_file_path {
        Value::Null => {}
        Value::String(path) if path.is_empty() => {}
        Value::String(_) => {
            return Err(invalid_configuration(
                "vision.camera.video_file_path must be cleared for live Rust capture",
            ));
        }
        _ => {
            return Err(invalid_configuration(
                "vision.camera.video_file_path must be a string or null",
            ));
        }
    }

    let raw_device = config_value(configuration, "vision.camera.device_id")?;
    let raw_selector = config_value(configuration, "perception.capture.device_selector")?;
    let backend = config_string(configuration, "vision.camera.backend")?;
    let selector = match raw_selector {
        Value::Null => None,
        Value::String(value) if valid_gstreamer_selector(value) => Some(value.clone()),
        _ => {
            return Err(invalid_configuration(
                "perception.capture.device_selector must be null or a stable GStreamer selector",
            ));
        }
    };
    let mut source_uri_sha256 = None;
    let device_id = if let Some(selector) = selector.as_ref() {
        if raw_device
            .as_str()
            .is_some_and(crate::configuration_authority::valid_http_uri)
        {
            return Err(invalid_configuration(
                "HTTP-MJPEG URI cannot be combined with a physical camera selector",
            ));
        }
        selector.clone()
    } else {
        if let Some(index) = raw_device.as_u64() {
            format!("device-index:{index}")
        } else if let Some(uri) = raw_device
            .as_str()
            .filter(|value| crate::configuration_authority::valid_http_uri(value))
        {
            if backend != "auto" && backend != "gstreamer" {
                return Err(invalid_configuration(
                    "HTTP-MJPEG input requires the auto or gstreamer backend",
                ));
            }
            source_uri_sha256 = Some(format!("{:x}", Sha256::digest(uri.as_bytes())));
            HTTP_MJPEG_STREAM_DEVICE_ID.to_owned()
        } else {
            return Err(invalid_configuration(
                "vision.camera.device_id must be a nonnegative index, HTTP(S) MJPEG URI, or use a stable selector",
            ));
        }
    };

    if !SUPPORTED_CAPTURE_BACKENDS.contains(&backend.as_str()) {
        return Err(invalid_configuration(
            "vision.camera.backend is unsupported",
        ));
    }
    if selector.is_some() && backend != "auto" && backend != "gstreamer" {
        return Err(invalid_configuration(
            "perception.capture.device_selector requires the auto or gstreamer backend",
        ));
    }

    let (width_px, height_px) = config_resolution(configuration, "vision.camera.resolution")?;
    let (output_width_px, output_height_px) = match config_value(
        configuration,
        "perception.capture.output_resolution",
    )? {
        Value::Null => (width_px, height_px),
        Value::Array(_) => {
            config_resolution(configuration, "perception.capture.output_resolution")?
        }
        _ => {
            return Err(invalid_configuration(
                "perception.capture.output_resolution must be null or contain two positive integers",
            ));
        }
    };
    let fps = config_integer(configuration, "vision.camera.fps", 1, 120)?;
    let pixel_format = config_string(configuration, "perception.capture.pixel_format")?;
    if !valid_pixel_format(&pixel_format) {
        return Err(invalid_configuration(
            "perception.capture.pixel_format must be a bounded adapter-neutral identifier",
        ));
    }
    let image_space = config_string(configuration, "perception.capture.image_space")?;
    if image_space != "raw" && image_space != "rectified" {
        return Err(invalid_configuration(
            "perception.capture.image_space must be 'raw' or 'rectified'",
        ));
    }
    let rectification_revision = config_integer(
        configuration,
        "perception.capture.rectification_revision",
        0,
        MAX_SAFE_JSON_INTEGER,
    )?;
    let calibration_revision = config_integer(
        configuration,
        "perception.capture.calibration_revision",
        0,
        MAX_SAFE_JSON_INTEGER,
    )?;
    if config_string(configuration, "perception.capture.timestamp_semantics")?
        != "read_completion_only"
    {
        return Err(invalid_configuration(
            "perception.capture.timestamp_semantics must be read_completion_only until a source clock is qualified",
        ));
    }

    let control_policy = config_string(configuration, "vision.camera.control_policy")?;
    if control_policy != "source_defaults" && control_policy != "require" {
        return Err(invalid_configuration(
            "vision.camera.control_policy must be 'source_defaults' or 'require'",
        ));
    }
    if source_uri_sha256.is_some() && control_policy != "source_defaults" {
        return Err(invalid_configuration(
            "HTTP-MJPEG streams do not support requested camera controls; use source_defaults",
        ));
    }
    let camera_controls = if control_policy == "source_defaults" {
        json!({"mode": "source_defaults"})
    } else {
        let exposure_mode = config_string(configuration, "vision.camera.exposure_mode")?;
        if exposure_mode != "auto" && exposure_mode != "manual" {
            return Err(invalid_configuration(
                "vision.camera.exposure_mode must be 'auto' or 'manual'",
            ));
        }
        let exposure_value =
            config_nullable_finite_number(configuration, "vision.camera.exposure_value")?;
        if exposure_mode == "manual" && exposure_value.is_none() {
            return Err(invalid_configuration(
                "manual exposure requires vision.camera.exposure_value",
            ));
        }
        let gain = config_finite_number(configuration, "vision.camera.gain")?;
        if gain < 0.0 {
            return Err(invalid_configuration(
                "vision.camera.gain must be nonnegative",
            ));
        }
        let brightness = config_control(configuration, "vision.camera.brightness")?;
        let contrast = config_control(configuration, "vision.camera.contrast")?;
        let saturation = config_control(configuration, "vision.camera.saturation")?;
        let auto_focus = config_bool(configuration, "vision.camera.auto_focus")?;
        let buffer_size = config_integer(
            configuration,
            "vision.camera.buffer_size",
            1,
            u64::from(u32::MAX),
        )?;
        json!({
            "mode": "require",
            "requirements": {
                "manual_exposure": exposure_mode == "manual",
                "automatic_exposure": exposure_mode == "auto",
                "gain_control": true
            },
            "requested": {
                "exposure_mode": exposure_mode,
                "exposure_value": exposure_value.map(number_from_f64).transpose()?,
                "gain": number_from_f64(gain)?,
                "brightness": number_from_f64(brightness)?,
                "contrast": number_from_f64(contrast)?,
                "saturation": number_from_f64(saturation)?,
                "auto_focus": auto_focus,
                "buffer_size": buffer_size
            }
        })
    };

    let reconnect_enabled = config_bool(configuration, "vision.camera.auto_reconnect")?;
    let reconnect_delay = config_finite_number(configuration, "vision.camera.reconnect_delay")?;
    if reconnect_delay <= 0.0 {
        return Err(invalid_configuration(
            "vision.camera.reconnect_delay must be positive",
        ));
    }
    let reconnect_attempts = config_integer(
        configuration,
        "vision.camera.max_reconnect_attempts",
        0,
        u64::from(u32::MAX),
    )?;
    let stop_timeout = config_finite_number(configuration, "vision.camera.stop_timeout")?;
    if stop_timeout <= 0.0 {
        return Err(invalid_configuration(
            "vision.camera.stop_timeout must be positive",
        ));
    }

    let maximum_frame_age_ms = config_integer(
        configuration,
        "perception.capture.maximum_frame_age_ms",
        1,
        MAX_SAFE_JSON_INTEGER,
    )?;
    let maximum_rgb_frame_bytes = config_integer(
        configuration,
        "perception.capture.maximum_rgb_frame_bytes",
        1,
        MAX_CAPTURE_FRAME_BYTES,
    )?;
    let maximum_aggregate_rgb_frame_bytes = config_integer(
        configuration,
        "perception.capture.maximum_aggregate_rgb_frame_bytes",
        1,
        MAX_CAPTURE_FRAME_BYTES,
    )?;
    if maximum_aggregate_rgb_frame_bytes < maximum_rgb_frame_bytes {
        return Err(invalid_configuration(
            "aggregate RGB frame limit must be at least the per-frame limit",
        ));
    }
    let minimum_frame_bytes = u64::from(output_width_px)
        .checked_mul(u64::from(output_height_px))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| invalid_configuration("emitted RGB8 frame size overflows u64"))?;
    if minimum_frame_bytes > maximum_rgb_frame_bytes {
        return Err(invalid_configuration(
            "emitted RGB8 frame exceeds maximum_rgb_frame_bytes",
        ));
    }
    if minimum_frame_bytes > maximum_aggregate_rgb_frame_bytes {
        return Err(invalid_configuration(
            "emitted RGB8 frame exceeds maximum_aggregate_rgb_frame_bytes",
        ));
    }

    let profile = json!({
        "width_px": width_px,
        "height_px": height_px,
        "frame_rate": {"frames_numerator": fps, "seconds_denominator": 1},
        "pixel_format": pixel_format
    });
    let mut profile_identity_source = json!({
        "device_id": device_id,
        "backend": backend,
        "profile": profile,
        "image_space": image_space,
        "rectification_revision": rectification_revision,
        "output_frame_geometry": {"width_px": output_width_px, "height_px": output_height_px},
        "output_pixel_format": "RGB8"
    });
    if let Some(uri_sha256) = source_uri_sha256 {
        // Bind profile identity to this stream without copying its URL into
        // the capture-settings payload or pretending it is camera hardware.
        profile_identity_source["source_uri_sha256"] = json!(uri_sha256);
    }
    let profile_bytes = serde_json::to_vec(&profile_identity_source).map_err(|error| {
        invalid_configuration(format!("cannot encode camera profile identity: {error}"))
    })?;
    let camera_profile_id = format!("camera-profile-v3:{:x}", Sha256::digest(profile_bytes));

    let requested = json!({
        "device_id": device_id,
        "backend": backend,
        "requested_profile": profile,
        "expected_negotiated_profile": profile,
        "output_frame_geometry": {"width_px": output_width_px, "height_px": output_height_px},
        "camera_profile_id": camera_profile_id,
        "image_space": image_space,
        "rectification_revision": rectification_revision,
        "calibration_revision": calibration_revision,
        "calibration": Value::Null,
        "timestamp_semantics": {"kind": "read_completion_only"},
        "camera_controls": camera_controls,
        "reconnect_policy": {
            "enabled": reconnect_enabled,
            "delay_seconds": number_from_f64(reconnect_delay)?,
            "maximum_attempts": reconnect_attempts,
            "stop_timeout_seconds": number_from_f64(stop_timeout)?
        },
        "maximum_frame_age_ms": maximum_frame_age_ms,
        "maximum_rgb_frame_bytes": maximum_rgb_frame_bytes,
        "maximum_aggregate_rgb_frame_bytes": maximum_aggregate_rgb_frame_bytes
    });
    let snapshot = json!({
        "schema_version": 3,
        "configuration_revision": configuration_revision,
        "requested": requested
    });
    let encoded = serde_json::to_string(&snapshot).map_err(|error| {
        invalid_configuration(format!("cannot encode capture settings request: {error}"))
    })?;
    parse_capture_settings_json(&encoded).map_err(|error| {
        invalid_configuration(format!(
            "generated v3 request failed strict parsing: {error}"
        ))
    })?;
    Ok(snapshot)
}

fn invalid_configuration(reason: impl Into<String>) -> CaptureSettingsWireError {
    CaptureSettingsWireError::InvalidConfiguration(reason.into())
}

fn config_value<'a>(
    configuration: &'a Value,
    path: &str,
) -> Result<&'a Value, CaptureSettingsWireError> {
    let mut current = configuration;
    for component in path.split('.') {
        let Some(next) = current.get(component) else {
            return Err(invalid_configuration(format!("{path} is missing")));
        };
        current = next;
    }
    Ok(current)
}

fn config_string(configuration: &Value, path: &str) -> Result<String, CaptureSettingsWireError> {
    config_value(configuration, path)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid_configuration(format!("{path} must be a string")))
}

fn config_bool(configuration: &Value, path: &str) -> Result<bool, CaptureSettingsWireError> {
    config_value(configuration, path)?
        .as_bool()
        .ok_or_else(|| invalid_configuration(format!("{path} must be a boolean")))
}

fn config_integer(
    configuration: &Value,
    path: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, CaptureSettingsWireError> {
    let value = config_value(configuration, path)?
        .as_u64()
        .filter(|value| (minimum..=maximum).contains(value))
        .ok_or_else(|| {
            invalid_configuration(format!(
                "{path} must be an integer in [{minimum}, {maximum}]"
            ))
        })?;
    Ok(value)
}

fn config_resolution(
    configuration: &Value,
    path: &str,
) -> Result<(u32, u32), CaptureSettingsWireError> {
    let value = config_value(configuration, path)?;
    let Some(parts) = value.as_array() else {
        return Err(invalid_configuration(format!(
            "{path} must contain two positive integers"
        )));
    };
    if parts.len() != 2 {
        return Err(invalid_configuration(format!(
            "{path} must contain two positive integers"
        )));
    }
    let width = parts[0]
        .as_u64()
        .filter(|value| (1..=u64::from(u32::MAX)).contains(value))
        .ok_or_else(|| invalid_configuration(format!("{path}[0] must be a positive u32")))?;
    let height = parts[1]
        .as_u64()
        .filter(|value| (1..=u64::from(u32::MAX)).contains(value))
        .ok_or_else(|| invalid_configuration(format!("{path}[1] must be a positive u32")))?;
    Ok((width as u32, height as u32))
}

fn config_finite_number(
    configuration: &Value,
    path: &str,
) -> Result<f64, CaptureSettingsWireError> {
    let value = config_value(configuration, path)?
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| invalid_configuration(format!("{path} must be a finite number")))?;
    Ok(value)
}

fn config_nullable_finite_number(
    configuration: &Value,
    path: &str,
) -> Result<Option<f64>, CaptureSettingsWireError> {
    match config_value(configuration, path)? {
        Value::Null => Ok(None),
        value => value
            .as_f64()
            .filter(|value| value.is_finite())
            .map(Some)
            .ok_or_else(|| invalid_configuration(format!("{path} must be null or finite"))),
    }
}

fn config_control(configuration: &Value, path: &str) -> Result<f64, CaptureSettingsWireError> {
    let value = config_finite_number(configuration, path)?;
    if !(0.0..=1.0).contains(&value) {
        return Err(invalid_configuration(format!("{path} must be in [0, 1]")));
    }
    Ok(value)
}

fn number_from_f64(value: f64) -> Result<Number, CaptureSettingsWireError> {
    Number::from_f64(value).ok_or_else(|| invalid_configuration("number must be finite"))
}

fn valid_gstreamer_selector(value: &str) -> bool {
    value
        .strip_prefix("gstreamer-device-v1:")
        .is_some_and(|fingerprint| {
            fingerprint.len() == 64
                && fingerprint
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
}

fn valid_pixel_format(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
}

impl Error for CaptureSettingsWireError {}

impl CaptureSettingsSnapshotV1 {
    /// Parse the strict Python v1 request and validate it through Rust domain
    /// constructors rather than deserializing directly into runtime types.
    /// The digest covers these exact input bytes; live callers must pass
    /// Python's deterministic `to_json()` output unchanged.
    pub fn parse_json(json: &str) -> Result<ResolvedCaptureSettingsV1, CaptureSettingsWireError> {
        let request_sha256 = format!("{:x}", Sha256::digest(json.as_bytes()));
        let snapshot: Self = serde_json::from_str(json).map_err(CaptureSettingsWireError::Json)?;
        if snapshot.schema_version != 1 {
            return Err(CaptureSettingsWireError::UnsupportedSchemaVersion(
                snapshot.schema_version,
            ));
        }
        let requested = snapshot.requested;
        resolve_capture_settings(
            CaptureSettingsParts {
                schema_version: 1,
                configuration_revision: snapshot.configuration_revision,
                device_id: requested.device_id,
                backend: requested.backend,
                requested_profile: requested.requested_profile,
                expected_negotiated_profile: requested.expected_negotiated_profile,
                output_frame_geometry: None,
                camera_profile_id: requested.camera_profile_id,
                image_space: requested.image_space,
                rectification_revision: requested.rectification_revision,
                calibration_revision: requested.calibration_revision,
                calibration: requested.calibration,
                timestamp_semantics: requested.timestamp_semantics,
                camera_controls: CameraControlsInput::Require {
                    requirements: requested.control_requirements,
                    requested: requested.requested_controls,
                },
                reconnect_policy: requested.reconnect_policy,
                maximum_frame_age_ms: requested.maximum_frame_age_ms,
                maximum_rgb_frame_bytes: requested.maximum_rgb_frame_bytes,
                maximum_aggregate_rgb_frame_bytes: requested.maximum_aggregate_rgb_frame_bytes,
            },
            request_sha256,
        )
    }
}

impl CaptureSettingsSnapshotV2 {
    /// Parse the strict v2 request while hashing its exact input bytes.
    pub fn parse_json(json: &str) -> Result<ResolvedCaptureSettingsV1, CaptureSettingsWireError> {
        let request_sha256 = format!("{:x}", Sha256::digest(json.as_bytes()));
        let snapshot: Self = serde_json::from_str(json).map_err(CaptureSettingsWireError::Json)?;
        if snapshot.schema_version != 2 {
            return Err(CaptureSettingsWireError::UnsupportedSchemaVersion(
                snapshot.schema_version,
            ));
        }
        let CaptureSettingsRequestV2 {
            device_id,
            backend,
            requested_profile,
            expected_negotiated_profile,
            output_frame_geometry,
            camera_profile_id,
            image_space,
            rectification_revision,
            calibration_revision,
            calibration,
            timestamp_semantics,
            control_requirements,
            requested_controls,
            reconnect_policy,
            maximum_frame_age_ms,
            maximum_rgb_frame_bytes,
            maximum_aggregate_rgb_frame_bytes,
        } = snapshot.requested;
        let output_frame_geometry = CaptureOutputFrameGeometry::new(
            output_frame_geometry.width_px,
            output_frame_geometry.height_px,
        )
        .map_err(CaptureSettingsWireError::InvalidMetadata)?;
        resolve_capture_settings(
            CaptureSettingsParts {
                schema_version: 2,
                configuration_revision: snapshot.configuration_revision,
                device_id,
                backend,
                requested_profile,
                expected_negotiated_profile,
                output_frame_geometry: Some(output_frame_geometry),
                camera_profile_id,
                image_space,
                rectification_revision,
                calibration_revision,
                calibration,
                timestamp_semantics,
                camera_controls: CameraControlsInput::Require {
                    requirements: control_requirements,
                    requested: requested_controls,
                },
                reconnect_policy,
                maximum_frame_age_ms,
                maximum_rgb_frame_bytes,
                maximum_aggregate_rgb_frame_bytes,
            },
            request_sha256,
        )
    }
}

impl CaptureSettingsSnapshotV3 {
    /// Parse the v3 request and hash the exact request bytes.
    pub fn parse_json(json: &str) -> Result<ResolvedCaptureSettingsV1, CaptureSettingsWireError> {
        let request_sha256 = format!("{:x}", Sha256::digest(json.as_bytes()));
        let snapshot: Self = serde_json::from_str(json).map_err(CaptureSettingsWireError::Json)?;
        if snapshot.schema_version != 3 {
            return Err(CaptureSettingsWireError::UnsupportedSchemaVersion(
                snapshot.schema_version,
            ));
        }
        let CaptureSettingsRequestV3 {
            device_id,
            backend,
            requested_profile,
            expected_negotiated_profile,
            output_frame_geometry,
            camera_profile_id,
            image_space,
            rectification_revision,
            calibration_revision,
            calibration,
            timestamp_semantics,
            camera_controls,
            reconnect_policy,
            maximum_frame_age_ms,
            maximum_rgb_frame_bytes,
            maximum_aggregate_rgb_frame_bytes,
        } = snapshot.requested;
        let output_frame_geometry = CaptureOutputFrameGeometry::new(
            output_frame_geometry.width_px,
            output_frame_geometry.height_px,
        )
        .map_err(CaptureSettingsWireError::InvalidMetadata)?;
        let camera_controls = match camera_controls {
            CameraControlsPolicyV3::SourceDefaults {} => CameraControlsInput::SourceDefaults,
            CameraControlsPolicyV3::Require {
                requirements,
                requested,
            } => CameraControlsInput::Require {
                requirements,
                requested,
            },
        };
        resolve_capture_settings(
            CaptureSettingsParts {
                schema_version: 3,
                configuration_revision: snapshot.configuration_revision,
                device_id,
                backend,
                requested_profile,
                expected_negotiated_profile,
                output_frame_geometry: Some(output_frame_geometry),
                camera_profile_id,
                image_space,
                rectification_revision,
                calibration_revision,
                calibration,
                timestamp_semantics,
                camera_controls,
                reconnect_policy,
                maximum_frame_age_ms,
                maximum_rgb_frame_bytes,
                maximum_aggregate_rgb_frame_bytes,
            },
            request_sha256,
        )
    }
}

struct CaptureSettingsParts {
    schema_version: u8,
    configuration_revision: u64,
    device_id: String,
    backend: String,
    requested_profile: CaptureVideoProfileV1,
    expected_negotiated_profile: CaptureVideoProfileV1,
    output_frame_geometry: Option<CaptureOutputFrameGeometry>,
    camera_profile_id: String,
    image_space: ImageSpace,
    rectification_revision: u64,
    calibration_revision: u64,
    calibration: Option<TableCalibrationV1>,
    timestamp_semantics: CaptureTimestampSemanticsV1,
    camera_controls: CameraControlsInput,
    reconnect_policy: CaptureReconnectPolicyV1,
    maximum_frame_age_ms: u64,
    maximum_rgb_frame_bytes: u64,
    maximum_aggregate_rgb_frame_bytes: u64,
}

enum CameraControlsInput {
    SourceDefaults,
    Require {
        requirements: CaptureControlRequirementsV1,
        requested: RequestedCaptureControlsV1,
    },
}

fn resolve_capture_settings(
    requested: CaptureSettingsParts,
    request_sha256: String,
) -> Result<ResolvedCaptureSettingsV1, CaptureSettingsWireError> {
    validate_revision(requested.configuration_revision, "configuration_revision")?;
    validate_identifier(&requested.device_id, "device_id")?;
    if !SUPPORTED_CAPTURE_BACKENDS.contains(&requested.backend.as_str()) {
        return Err(CaptureSettingsWireError::InvalidRequest(
            "unsupported capture backend",
        ));
    }
    validate_identifier(&requested.camera_profile_id, "camera_profile_id")?;
    validate_revision(requested.rectification_revision, "rectification_revision")?;
    validate_revision(requested.calibration_revision, "calibration_revision")?;
    validate_bounded_positive(
        requested.maximum_frame_age_ms,
        MAX_SAFE_JSON_INTEGER,
        "maximum_frame_age_ms",
    )?;
    validate_bounded_positive(
        requested.maximum_rgb_frame_bytes,
        MAX_CAPTURE_FRAME_BYTES,
        "maximum_rgb_frame_bytes",
    )?;
    validate_bounded_positive(
        requested.maximum_aggregate_rgb_frame_bytes,
        MAX_CAPTURE_FRAME_BYTES,
        "maximum_aggregate_rgb_frame_bytes",
    )?;
    validate_reconnect_policy(&requested.reconnect_policy)?;

    let (control_requirements, requested_controls, resolved_camera_controls) =
        match requested.camera_controls {
            CameraControlsInput::SourceDefaults => (
                CaptureControlRequirements::default(),
                None,
                ResolvedCameraControlsPolicy::SourceDefaults,
            ),
            CameraControlsInput::Require {
                requirements,
                requested: requested_controls,
            } => {
                validate_requested_controls(&requested_controls)?;
                if requirements.manual_exposure
                    != (requested_controls.exposure_mode == ExposureModeV1::Manual)
                    || requirements.automatic_exposure
                        != (requested_controls.exposure_mode == ExposureModeV1::Auto)
                    || !requirements.gain_control
                {
                    return Err(CaptureSettingsWireError::InvalidRequest(
                        "control requirements must cover the requested exposure mode and gain",
                    ));
                }
                let values = RequestedCaptureControlValues::new(
                    match requested_controls.exposure_mode {
                        ExposureModeV1::Auto => CaptureExposureMode::Automatic,
                        ExposureModeV1::Manual => CaptureExposureMode::Manual,
                    },
                    requested_controls.exposure_value,
                    requested_controls.gain,
                )
                .map_err(CaptureSettingsWireError::InvalidPreflight)?;
                let domain_requirements = CaptureControlRequirements {
                    manual_exposure: requirements.manual_exposure,
                    automatic_exposure: requirements.automatic_exposure,
                    gain_control: requirements.gain_control,
                };
                (
                    domain_requirements,
                    Some(values),
                    ResolvedCameraControlsPolicy::Require {
                        requirements,
                        requested: requested_controls,
                    },
                )
            }
        };

    let requested_profile = requested.requested_profile.into_profile()?;
    let expected_negotiated_profile = requested.expected_negotiated_profile.into_profile()?;
    let calibration = requested
        .calibration
        .map(TableCalibrationV1::into_validated)
        .transpose()
        .map_err(CaptureSettingsWireError::InvalidCalibration)?;
    let timestamp_semantics = requested.timestamp_semantics.into_domain()?;
    let frame_limit = CaptureFrameLimit::new(as_usize(
        requested.maximum_rgb_frame_bytes,
        "maximum_rgb_frame_bytes",
    )?)
    .map_err(CaptureSettingsWireError::InvalidFrameLimit)?;
    if requested.maximum_aggregate_rgb_frame_bytes < requested.maximum_rgb_frame_bytes {
        return Err(CaptureSettingsWireError::InvalidRequest(
            "aggregate frame limit must be at least the per-frame limit",
        ));
    }
    let aggregate_frame_limit = CaptureAggregateFrameLimit::new(as_usize(
        requested.maximum_aggregate_rgb_frame_bytes,
        "maximum_aggregate_rgb_frame_bytes",
    )?)
    .map_err(CaptureSettingsWireError::InvalidFrameLimit)?;
    let requested_backend = requested.backend;
    let expected_settings = match requested.output_frame_geometry {
        Some(output_geometry) => ExpectedCaptureSettings::new_with_output_frame_geometry(
            requested.configuration_revision,
            requested.device_id,
            requested_profile,
            expected_negotiated_profile,
            output_geometry,
            requested.camera_profile_id,
            requested.image_space,
            requested.rectification_revision,
            requested.calibration_revision,
            calibration,
            timestamp_semantics,
            control_requirements,
            Duration::from_millis(requested.maximum_frame_age_ms),
            frame_limit,
            aggregate_frame_limit,
        ),
        None => ExpectedCaptureSettings::new(
            requested.configuration_revision,
            requested.device_id,
            requested_profile,
            expected_negotiated_profile,
            requested.camera_profile_id,
            requested.image_space,
            requested.rectification_revision,
            requested.calibration_revision,
            calibration,
            timestamp_semantics,
            control_requirements,
            Duration::from_millis(requested.maximum_frame_age_ms),
            frame_limit,
            aggregate_frame_limit,
        ),
    };
    let mut expected = expected_settings
        .map_err(CaptureSettingsWireError::InvalidPreflight)?
        .with_requested_backend(requested_backend.clone())
        .map_err(CaptureSettingsWireError::InvalidPreflight)?
        .with_request_sha256(request_sha256.clone());
    if let Some(requested_control_values) = requested_controls {
        expected = expected
            .with_requested_control_values(requested_control_values)
            .map_err(CaptureSettingsWireError::InvalidPreflight)?;
    }

    Ok(ResolvedCaptureSettingsV1 {
        schema_version: requested.schema_version,
        configuration_revision: requested.configuration_revision,
        request_sha256,
        expected,
        requested_backend,
        camera_controls: resolved_camera_controls,
        reconnect_policy: requested.reconnect_policy,
    })
}

impl CaptureVideoProfileV1 {
    fn into_profile(self) -> Result<CaptureVideoProfile, CaptureSettingsWireError> {
        let frame_rate = CaptureFrameRate::new(
            self.frame_rate.frames_numerator,
            self.frame_rate.seconds_denominator,
        )
        .map_err(CaptureSettingsWireError::InvalidMetadata)?;
        let pixel_format = CapturePixelFormat::new(self.pixel_format)
            .map_err(CaptureSettingsWireError::InvalidMetadata)?;
        CaptureVideoProfile::new(self.width_px, self.height_px, frame_rate, pixel_format)
            .map_err(CaptureSettingsWireError::InvalidMetadata)
    }
}

impl CaptureTimestampSemanticsV1 {
    fn into_domain(self) -> Result<CaptureTimestampSemantics, CaptureSettingsWireError> {
        match self {
            Self::ReadCompletionOnly => Ok(CaptureTimestampSemantics::ReadCompletionOnly),
            Self::SourceTimestamp {
                unit,
                clock,
                domain_id,
                stage,
            } => {
                validate_identifier(&domain_id, "timestamp.domain_id")?;
                if matches!(
                    stage,
                    TimestampStage::InferenceStarted
                        | TimestampStage::InferenceComplete
                        | TimestampStage::BatchPublication
                ) {
                    return Err(CaptureSettingsWireError::InvalidRequest(
                        "source timestamp stage must precede inference",
                    ));
                }
                Ok(CaptureTimestampSemantics::SourceTimestamp {
                    unit,
                    clock,
                    domain_id,
                    stage,
                })
            }
        }
    }
}

impl From<serde_json::Error> for CaptureSettingsWireError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

fn validate_requested_controls(
    controls: &RequestedCaptureControlsV1,
) -> Result<(), CaptureSettingsWireError> {
    if !controls.gain.is_finite() || controls.gain < 0.0 {
        return Err(CaptureSettingsWireError::InvalidRequest(
            "requested gain must be finite and nonnegative",
        ));
    }
    if controls
        .exposure_value
        .is_some_and(|value| !value.is_finite())
    {
        return Err(CaptureSettingsWireError::InvalidRequest(
            "requested exposure must be finite",
        ));
    }
    if [controls.brightness, controls.contrast, controls.saturation]
        .into_iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(CaptureSettingsWireError::InvalidRequest(
            "requested brightness, contrast and saturation must be in [0, 1]",
        ));
    }
    if controls.exposure_mode == ExposureModeV1::Manual && controls.exposure_value.is_none() {
        return Err(CaptureSettingsWireError::InvalidRequest(
            "manual exposure requires a requested value",
        ));
    }
    if controls.buffer_size == 0 {
        return Err(CaptureSettingsWireError::InvalidRequest(
            "requested buffer size must be positive",
        ));
    }
    Ok(())
}

fn validate_reconnect_policy(
    policy: &CaptureReconnectPolicyV1,
) -> Result<(), CaptureSettingsWireError> {
    if !policy.delay_seconds.is_finite()
        || policy.delay_seconds <= 0.0
        || !policy.stop_timeout_seconds.is_finite()
        || policy.stop_timeout_seconds <= 0.0
    {
        return Err(CaptureSettingsWireError::InvalidRequest(
            "reconnect delay and stop timeout must be finite and positive",
        ));
    }
    Ok(())
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), CaptureSettingsWireError> {
    // Match Python's identifier pattern exactly: bounded ASCII alphanumeric
    // identifiers with only underscore, period, colon, and hyphen separators.
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b':' | b'-'))
    {
        return Err(CaptureSettingsWireError::InvalidRequest(field));
    }
    Ok(())
}

fn validate_bounded_positive(
    value: u64,
    maximum: u64,
    field: &'static str,
) -> Result<(), CaptureSettingsWireError> {
    if value == 0 || value > maximum {
        return Err(CaptureSettingsWireError::InvalidRequest(field));
    }
    Ok(())
}

fn validate_revision(value: u64, field: &'static str) -> Result<(), CaptureSettingsWireError> {
    if value > MAX_SAFE_JSON_INTEGER {
        return Err(CaptureSettingsWireError::UnsafeRevision(field));
    }
    Ok(())
}

fn as_usize(value: u64, field: &'static str) -> Result<usize, CaptureSettingsWireError> {
    usize::try_from(value).map_err(|_| CaptureSettingsWireError::InvalidRequest(field))
}

fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

#[cfg(test)]
mod tests {
    use super::{build_capture_settings_v3, HTTP_MJPEG_STREAM_DEVICE_ID};
    use serde_json::{json, Value};

    fn remote_stream_configuration(uri: &str) -> Value {
        let mut configuration: Value =
            serde_json::from_str(include_str!("../contracts/configuration/v1/defaults.json"))
                .expect("configuration defaults are valid JSON");
        configuration["vision"]["camera"]["device_id"] = json!(uri);
        configuration["vision"]["camera"]["video_file_path"] = Value::Null;
        configuration
    }

    #[test]
    fn remote_mjpeg_configuration_uses_distinct_identity_and_hashes_uri_into_profile() {
        let first_uri = "http://127.0.0.1:8001/stream.mjpg";
        let second_uri = "http://127.0.0.1:8002/stream.mjpg";
        let first = build_capture_settings_v3(&remote_stream_configuration(first_uri)).unwrap();
        let second = build_capture_settings_v3(&remote_stream_configuration(second_uri)).unwrap();

        assert_eq!(
            first["requested"]["device_id"],
            json!(HTTP_MJPEG_STREAM_DEVICE_ID)
        );
        assert_ne!(
            first["requested"]["camera_profile_id"],
            second["requested"]["camera_profile_id"]
        );
        assert!(!serde_json::to_string(&first).unwrap().contains(first_uri));
    }

    #[test]
    fn remote_mjpeg_configuration_rejects_physical_controls_and_camera_selector() {
        let uri = "http://127.0.0.1:8001/stream.mjpg";
        let mut controls = remote_stream_configuration(uri);
        controls["vision"]["camera"]["control_policy"] = json!("require");
        assert!(build_capture_settings_v3(&controls).is_err());

        let mut selector = remote_stream_configuration(uri);
        selector["perception"]["capture"]["device_selector"] =
            json!(format!("gstreamer-device-v1:{}", "a".repeat(64)));
        assert!(build_capture_settings_v3(&selector).is_err());
    }
}
