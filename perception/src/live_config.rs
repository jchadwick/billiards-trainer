//! Pure validation for revisioned expected live-capture settings.
//!
//! This module does not open a device, set controls, or select a provider. It
//! checks a caller-supplied expected configuration for
//! internal consistency, then compares the subset observable in an opened
//! [`CaptureSessionReport`]. Explicit backend requests must match the
//! canonical backend identifier in that report. The returned compatibility
//! value binds the configuration revision to that source session; it is not an
//! application acknowledgement.
//!
//! Camera control readbacks and emitted-frame geometry are carried separately
//! from source capabilities and compared with the strict request contract.
//! This verifies reported values and software ordering only; it does not prove
//! physical-device behavior. Table calibration remains a pipeline-owned fact.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::time::Duration;

use crate::calibration::ValidatedTableCalibrationV1;
use crate::capture::{
    CaptureAggregateFrameLimit, CaptureAppliedCameraControls, CaptureAppliedFrameGeometry,
    CaptureCapability, CaptureControlReadback, CaptureExposureMode, CaptureFrameLimit,
    CaptureOutputFrameGeometry, CaptureSessionReport, CaptureTimestampSemantics,
    CaptureVideoProfile,
};
use crate::detection_batch::MAX_IDENTIFIER_CHARACTERS;
use crate::ImageSpace;

const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
const RGB8_CHANNELS: u64 = 3;

/// Backend request identifiers shared by the capture-settings v1 and v2 contracts.
pub const SUPPORTED_CAPTURE_BACKENDS: &[&str] =
    &["auto", "v4l2", "dshow", "gstreamer", "opencv", "kinect2"];

/// Capability requirements for controls exposed by a capture adapter.
///
/// These flags require the source to declare a capability as supported. The
/// separate `CaptureAppliedCameraControls` readback verifies required values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CaptureControlRequirements {
    pub manual_exposure: bool,
    pub automatic_exposure: bool,
    pub gain_control: bool,
}

/// Requested camera controls needed for strict source readback comparison.
///
/// Optional controls remain available in the source report but do not block
/// startup. Required exposure mode/value and gain must be reported as applied
/// with exact values; any future device-specific normalization policy requires
/// its own measured contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RequestedCaptureControlValues {
    pub exposure_mode: CaptureExposureMode,
    pub exposure_value: Option<f64>,
    pub gain: f64,
}

impl RequestedCaptureControlValues {
    pub fn new(
        exposure_mode: CaptureExposureMode,
        exposure_value: Option<f64>,
        gain: f64,
    ) -> Result<Self, CapturePreflightError> {
        if exposure_value.is_some_and(|value| !value.is_finite()) {
            return Err(CapturePreflightError::InvalidRequestedControl(
                "exposure_value",
            ));
        }
        if !gain.is_finite() || gain < 0.0 {
            return Err(CapturePreflightError::InvalidRequestedControl("gain"));
        }
        if exposure_mode == CaptureExposureMode::Manual && exposure_value.is_none() {
            return Err(CapturePreflightError::InvalidRequestedControl(
                "manual exposure requires exposure_value",
            ));
        }
        Ok(Self {
            exposure_mode,
            exposure_value,
            gain,
        })
    }
}

/// Revisioned expected capture configuration resolved by the control owner.
///
/// Every limit and requested mode must be supplied by the caller. This type
/// validates those expectations and their consistency with an optional
/// calibration. It does not prove they have been applied to a camera.
#[derive(Debug, Clone)]
pub struct ExpectedCaptureSettings {
    configuration_revision: u64,
    device_id: String,
    requested_backend: Option<String>,
    requested_profile: CaptureVideoProfile,
    expected_negotiated_profile: CaptureVideoProfile,
    /// Explicit v2 output dimensions. `None` is the unchanged v1 contract in
    /// which emitted RGB8 dimensions equal the negotiated camera profile.
    output_frame_geometry: Option<CaptureOutputFrameGeometry>,
    camera_profile_id: String,
    image_space: ImageSpace,
    rectification_revision: u64,
    calibration_revision: u64,
    calibration: Option<ValidatedTableCalibrationV1>,
    /// Digest of the exact canonical request bytes, when created from
    /// the strict wire parser. Hand-built expectations remain replay-compatible.
    request_sha256: Option<String>,
    timestamp_semantics: CaptureTimestampSemantics,
    control_requirements: CaptureControlRequirements,
    requested_control_values: Option<RequestedCaptureControlValues>,
    maximum_frame_age: Duration,
    frame_limit: CaptureFrameLimit,
    aggregate_frame_limit: CaptureAggregateFrameLimit,
}

impl ExpectedCaptureSettings {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        configuration_revision: u64,
        device_id: impl Into<String>,
        requested_profile: CaptureVideoProfile,
        expected_negotiated_profile: CaptureVideoProfile,
        camera_profile_id: impl Into<String>,
        image_space: ImageSpace,
        rectification_revision: u64,
        calibration_revision: u64,
        calibration: Option<ValidatedTableCalibrationV1>,
        timestamp_semantics: CaptureTimestampSemantics,
        control_requirements: CaptureControlRequirements,
        maximum_frame_age: Duration,
        frame_limit: CaptureFrameLimit,
        aggregate_frame_limit: CaptureAggregateFrameLimit,
    ) -> Result<Self, CapturePreflightError> {
        Self::build(
            configuration_revision,
            device_id,
            requested_profile,
            expected_negotiated_profile,
            None,
            camera_profile_id,
            image_space,
            rectification_revision,
            calibration_revision,
            calibration,
            timestamp_semantics,
            control_requirements,
            maximum_frame_age,
            frame_limit,
            aggregate_frame_limit,
        )
    }

    /// Construct v2 settings with separate dimensions for emitted RGB8 frames.
    /// Calibration dimensions are expressed in that emitted image space.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_output_frame_geometry(
        configuration_revision: u64,
        device_id: impl Into<String>,
        requested_profile: CaptureVideoProfile,
        expected_negotiated_profile: CaptureVideoProfile,
        output_frame_geometry: CaptureOutputFrameGeometry,
        camera_profile_id: impl Into<String>,
        image_space: ImageSpace,
        rectification_revision: u64,
        calibration_revision: u64,
        calibration: Option<ValidatedTableCalibrationV1>,
        timestamp_semantics: CaptureTimestampSemantics,
        control_requirements: CaptureControlRequirements,
        maximum_frame_age: Duration,
        frame_limit: CaptureFrameLimit,
        aggregate_frame_limit: CaptureAggregateFrameLimit,
    ) -> Result<Self, CapturePreflightError> {
        Self::build(
            configuration_revision,
            device_id,
            requested_profile,
            expected_negotiated_profile,
            Some(output_frame_geometry),
            camera_profile_id,
            image_space,
            rectification_revision,
            calibration_revision,
            calibration,
            timestamp_semantics,
            control_requirements,
            maximum_frame_age,
            frame_limit,
            aggregate_frame_limit,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        configuration_revision: u64,
        device_id: impl Into<String>,
        requested_profile: CaptureVideoProfile,
        expected_negotiated_profile: CaptureVideoProfile,
        output_frame_geometry: Option<CaptureOutputFrameGeometry>,
        camera_profile_id: impl Into<String>,
        image_space: ImageSpace,
        rectification_revision: u64,
        calibration_revision: u64,
        calibration: Option<ValidatedTableCalibrationV1>,
        timestamp_semantics: CaptureTimestampSemantics,
        control_requirements: CaptureControlRequirements,
        maximum_frame_age: Duration,
        frame_limit: CaptureFrameLimit,
        aggregate_frame_limit: CaptureAggregateFrameLimit,
    ) -> Result<Self, CapturePreflightError> {
        let device_id = device_id.into();
        let camera_profile_id = camera_profile_id.into();
        validate_identifier(&device_id, "device_id")?;
        if !valid_profile_identifier(&camera_profile_id) {
            return Err(CapturePreflightError::InvalidIdentifier(
                "camera_profile_id",
            ));
        }
        if configuration_revision == 0 {
            return Err(CapturePreflightError::InvalidRevision(
                "configuration_revision",
            ));
        }
        validate_revision(configuration_revision, "configuration_revision")?;
        validate_revision(rectification_revision, "rectification_revision")?;
        validate_revision(calibration_revision, "calibration_revision")?;
        if maximum_frame_age.is_zero() {
            return Err(CapturePreflightError::ZeroMaximumFrameAge);
        }

        let (output_width_px, output_height_px) = output_frame_geometry
            .map(|geometry| (geometry.width_px(), geometry.height_px()))
            .unwrap_or_else(|| {
                (
                    expected_negotiated_profile.width_px(),
                    expected_negotiated_profile.height_px(),
                )
            });
        let minimum_rgb8_bytes = rgb8_size(output_width_px, output_height_px)?;
        let minimum_rgb8_bytes = usize::try_from(minimum_rgb8_bytes)
            .map_err(|_| CapturePreflightError::FrameSizeOverflow)?;
        if minimum_rgb8_bytes > frame_limit.maximum_rgb8_bytes() {
            return Err(CapturePreflightError::FrameExceedsConfiguredLimit {
                required_bytes: minimum_rgb8_bytes,
                frame_limit_bytes: frame_limit.maximum_rgb8_bytes(),
            });
        }
        if minimum_rgb8_bytes > aggregate_frame_limit.maximum_rgb8_bytes() {
            return Err(CapturePreflightError::FrameExceedsAggregateLimit {
                required_bytes: minimum_rgb8_bytes,
                aggregate_limit_bytes: aggregate_frame_limit.maximum_rgb8_bytes(),
            });
        }

        if let Some(bundle) = &calibration {
            if bundle.calibration_revision() != calibration_revision {
                return Err(CapturePreflightError::CalibrationMismatch(
                    "bundle calibration revision differs from the expected revision",
                ));
            }
            if bundle.camera_profile_id() != camera_profile_id {
                return Err(CapturePreflightError::CalibrationMismatch(
                    "bundle camera profile differs from the expected profile",
                ));
            }
            if bundle.rectification_revision() != rectification_revision {
                return Err(CapturePreflightError::CalibrationMismatch(
                    "bundle rectification revision differs from the expected revision",
                ));
            }
            if bundle.image_space() != image_space {
                return Err(CapturePreflightError::CalibrationMismatch(
                    "bundle image space differs from the expected image space",
                ));
            }
            // V2 calibration dimensions are defined in emitted image space,
            // whether raw or rectified. Keep the legacy v1 raw-space rule
            // unchanged while clients migrate to the versioned contract.
            if output_frame_geometry.is_some()
                && bundle.source_dimensions_px() != (output_width_px, output_height_px)
            {
                return Err(CapturePreflightError::CalibrationMismatch(
                    "calibration dimensions differ from emitted RGB8 frame geometry",
                ));
            }
            if output_frame_geometry.is_none()
                && image_space == ImageSpace::Raw
                && bundle.source_dimensions_px()
                    != (
                        expected_negotiated_profile.width_px(),
                        expected_negotiated_profile.height_px(),
                    )
            {
                return Err(CapturePreflightError::CalibrationMismatch(
                    "raw calibration dimensions differ from the expected negotiated mode",
                ));
            }
        }

        Ok(Self {
            configuration_revision,
            device_id,
            requested_backend: None,
            requested_profile,
            expected_negotiated_profile,
            output_frame_geometry,
            camera_profile_id,
            image_space,
            rectification_revision,
            calibration_revision,
            calibration,
            request_sha256: None,
            timestamp_semantics,
            control_requirements,
            requested_control_values: None,
            maximum_frame_age,
            frame_limit,
            aggregate_frame_limit,
        })
    }

    /// Add the requested backend contract after constructing validated modes.
    /// `auto` accepts any reported backend; explicit values must match the
    /// backend identifier reported by the opened source.
    pub fn with_requested_backend(
        mut self,
        requested_backend: impl Into<String>,
    ) -> Result<Self, CapturePreflightError> {
        let requested_backend = requested_backend.into();
        if !SUPPORTED_CAPTURE_BACKENDS.contains(&requested_backend.as_str()) {
            return Err(CapturePreflightError::UnsupportedCaptureBackend);
        }
        self.requested_backend = Some(requested_backend);
        Ok(self)
    }

    /// Bind the exact serialized request to the settings validated by
    /// this expectation. This is crate-only because the wire parser computes
    /// the digest from the original bytes before parsing can normalize them.
    pub(crate) fn with_request_sha256(mut self, request_sha256: String) -> Self {
        self.request_sha256 = Some(request_sha256);
        self
    }

    /// Attach concrete requested values from the strict settings wire contract.
    pub fn with_requested_control_values(
        mut self,
        values: RequestedCaptureControlValues,
    ) -> Result<Self, CapturePreflightError> {
        if self.control_requirements.manual_exposure
            && values.exposure_mode != CaptureExposureMode::Manual
        {
            return Err(CapturePreflightError::InvalidRequestedControl(
                "manual exposure requirement differs from requested mode",
            ));
        }
        if self.control_requirements.automatic_exposure
            && values.exposure_mode != CaptureExposureMode::Automatic
        {
            return Err(CapturePreflightError::InvalidRequestedControl(
                "automatic exposure requirement differs from requested mode",
            ));
        }
        self.requested_control_values = Some(values);
        Ok(self)
    }

    /// Compare expected settings with the source metadata from an opened
    /// session. Success confirms only the metadata that the report contains.
    pub fn validate_session(
        &self,
        report: &CaptureSessionReport,
    ) -> Result<CaptureSessionCompatibility, CapturePreflightError> {
        if report.device_id() != self.device_id {
            return Err(CapturePreflightError::DeviceMismatch);
        }
        if self
            .requested_backend
            .as_deref()
            .is_some_and(|backend| backend != "auto" && backend != report.backend_id())
        {
            return Err(CapturePreflightError::RequestedBackendMismatch);
        }
        if report.requested_profile() != &self.requested_profile {
            return Err(CapturePreflightError::RequestedProfileMismatch);
        }
        if report.negotiated_profile() != &self.expected_negotiated_profile {
            return Err(CapturePreflightError::NegotiatedProfileMismatch);
        }
        let effective_output_geometry = match self.output_frame_geometry {
            Some(expected) => {
                let Some(reported) = report.output_frame_geometry() else {
                    return Err(CapturePreflightError::OutputFrameGeometryMissing);
                };
                if reported != expected {
                    return Err(CapturePreflightError::OutputFrameGeometryMismatch);
                }
                expected
            }
            None => {
                // V1 remains valid only when emitted frame dimensions equal
                // the negotiated camera mode. An explicit transformed output
                // must use v2, even when the source can report it.
                let legacy = CaptureOutputFrameGeometry::new(
                    self.expected_negotiated_profile.width_px(),
                    self.expected_negotiated_profile.height_px(),
                )
                .expect("validated profile dimensions are non-zero");
                if report.output_frame_geometry().is_some() {
                    // Keep v1's closed source-report shape byte/schema stable;
                    // v2-only metadata must be negotiated as v2 even when
                    // dimensions happen to match the legacy profile size.
                    return Err(CapturePreflightError::OutputFrameGeometryMismatch);
                }
                legacy
            }
        };
        let applied_geometry = report.applied_frame_geometry();
        if applied_geometry.configuration_revision() != self.configuration_revision {
            return Err(CapturePreflightError::ConfigurationRevisionMismatch);
        }
        if applied_geometry.camera_profile_id() != self.camera_profile_id {
            return Err(CapturePreflightError::CameraProfileMismatch);
        }
        if applied_geometry.image_space() != self.image_space {
            return Err(CapturePreflightError::ImageSpaceMismatch);
        }
        if applied_geometry.rectification_revision() != self.rectification_revision {
            return Err(CapturePreflightError::RectificationRevisionMismatch);
        }
        let capabilities = report.capabilities();
        require_capability(
            "manual_exposure",
            self.control_requirements.manual_exposure,
            capabilities.manual_exposure(),
        )?;
        require_capability(
            "automatic_exposure",
            self.control_requirements.automatic_exposure,
            capabilities.automatic_exposure(),
        )?;
        require_capability(
            "gain_control",
            self.control_requirements.gain_control,
            capabilities.gain_control(),
        )?;
        if report.timestamp_semantics() != &self.timestamp_semantics {
            return Err(CapturePreflightError::TimestampSemanticsMismatch);
        }
        if matches!(
            self.timestamp_semantics,
            CaptureTimestampSemantics::SourceTimestamp { .. }
        ) && capabilities.source_timestamps() != CaptureCapability::Supported
        {
            return Err(CapturePreflightError::SourceTimestampsNotSupported(
                capabilities.source_timestamps(),
            ));
        }
        let controls_required = self.control_requirements.manual_exposure
            || self.control_requirements.automatic_exposure
            || self.control_requirements.gain_control;
        if controls_required && self.requested_control_values.is_none() {
            return Err(CapturePreflightError::RequestedControlValuesMissing);
        }
        if let Some(requested) = self.requested_control_values {
            validate_required_readbacks(
                &self.control_requirements,
                requested,
                report.applied_controls(),
            )?;
        }

        Ok(CaptureSessionCompatibility {
            expected_settings: self.clone(),
            session_id: report.session_id().to_owned(),
            device_id: report.device_id().to_owned(),
            backend_id: report.backend_id().to_owned(),
            requested_profile: report.requested_profile().clone(),
            negotiated_profile: report.negotiated_profile().clone(),
            output_frame_geometry: effective_output_geometry,
            timestamp_semantics: report.timestamp_semantics().clone(),
            applied_frame_geometry: applied_geometry.clone(),
            applied_controls: report.applied_controls().clone(),
            source_report: report.clone(),
        })
    }

    pub fn configuration_revision(&self) -> u64 {
        self.configuration_revision
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    pub fn requested_backend(&self) -> Option<&str> {
        self.requested_backend.as_deref()
    }

    pub fn requested_profile(&self) -> &CaptureVideoProfile {
        &self.requested_profile
    }

    pub fn expected_negotiated_profile(&self) -> &CaptureVideoProfile {
        &self.expected_negotiated_profile
    }

    /// Requested emitted RGB8 dimensions, present only for the v2 contract.
    pub fn output_frame_geometry(&self) -> Option<CaptureOutputFrameGeometry> {
        self.output_frame_geometry
    }

    pub fn camera_profile_id(&self) -> &str {
        &self.camera_profile_id
    }

    pub fn image_space(&self) -> ImageSpace {
        self.image_space
    }

    pub fn rectification_revision(&self) -> u64 {
        self.rectification_revision
    }

    pub fn calibration_revision(&self) -> u64 {
        self.calibration_revision
    }

    pub fn calibration(&self) -> Option<&ValidatedTableCalibrationV1> {
        self.calibration.as_ref()
    }

    /// Replace request-carried calibration with the Rust-owned durable store
    /// state while leaving the exact request digest unchanged. A never-written
    /// store is represented by revision zero; a cleared store retains its
    /// tombstone revision with no active bundle.
    /// This is used only when the live runtime has explicitly enabled the
    /// calibration store.
    pub fn apply_calibration_store_state(
        &mut self,
        calibration_revision: u64,
        calibration: Option<ValidatedTableCalibrationV1>,
    ) -> Result<(), CapturePreflightError> {
        validate_revision(calibration_revision, "calibration_revision")?;
        if let Some(bundle) = &calibration {
            if bundle.calibration_revision() != calibration_revision {
                return Err(CapturePreflightError::CalibrationMismatch(
                    "stored calibration revision differs from its bundle",
                ));
            }
            if bundle.camera_profile_id() != self.camera_profile_id {
                return Err(CapturePreflightError::CalibrationMismatch(
                    "stored calibration profile differs from the capture request",
                ));
            }
            if bundle.rectification_revision() != self.rectification_revision {
                return Err(CapturePreflightError::CalibrationMismatch(
                    "stored calibration rectification differs from the capture request",
                ));
            }
            if bundle.image_space() != self.image_space {
                return Err(CapturePreflightError::CalibrationMismatch(
                    "stored calibration image space differs from the capture request",
                ));
            }
            let expected_dimensions = self
                .output_frame_geometry
                .map(|geometry| (geometry.width_px(), geometry.height_px()))
                .unwrap_or_else(|| {
                    (
                        self.expected_negotiated_profile.width_px(),
                        self.expected_negotiated_profile.height_px(),
                    )
                });
            if bundle.source_dimensions_px() != expected_dimensions {
                return Err(CapturePreflightError::CalibrationMismatch(
                    "stored calibration dimensions differ from emitted RGB8 geometry",
                ));
            }
        }
        self.calibration_revision = calibration_revision;
        self.calibration = calibration;
        Ok(())
    }

    pub fn timestamp_semantics(&self) -> &CaptureTimestampSemantics {
        &self.timestamp_semantics
    }

    pub fn maximum_frame_age(&self) -> Duration {
        self.maximum_frame_age
    }

    pub fn frame_limit(&self) -> CaptureFrameLimit {
        self.frame_limit
    }

    pub fn aggregate_frame_limit(&self) -> CaptureAggregateFrameLimit {
        self.aggregate_frame_limit
    }

    pub fn requested_control_values(&self) -> Option<RequestedCaptureControlValues> {
        self.requested_control_values
    }

    pub(crate) fn request_sha256(&self) -> Option<&str> {
        self.request_sha256.as_deref()
    }
}

/// Metadata proven compatible with one opened source session.
///
/// Required controls and emitted-frame geometry matched the effective request;
/// source values remain separate from requested settings. This software
/// comparison does not prove physical-device behavior, and it does not claim
/// that the source applied the pipeline-owned table calibration.
#[derive(Debug, Clone)]
pub struct CaptureSessionCompatibility {
    expected_settings: ExpectedCaptureSettings,
    session_id: String,
    device_id: String,
    backend_id: String,
    requested_profile: CaptureVideoProfile,
    negotiated_profile: CaptureVideoProfile,
    output_frame_geometry: CaptureOutputFrameGeometry,
    timestamp_semantics: CaptureTimestampSemantics,
    applied_frame_geometry: CaptureAppliedFrameGeometry,
    applied_controls: CaptureAppliedCameraControls,
    source_report: CaptureSessionReport,
}

impl CaptureSessionCompatibility {
    pub fn configuration_revision(&self) -> u64 {
        self.expected_settings.configuration_revision()
    }

    /// The exact revisioned expectations checked against this session report.
    /// Source-reported geometry and required controls are copied separately
    /// below; values absent from the report remain expectations only.
    pub fn expected_settings(&self) -> &ExpectedCaptureSettings {
        &self.expected_settings
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// Backend identifier reported by the opened source.
    pub fn backend_id(&self) -> &str {
        &self.backend_id
    }

    pub fn requested_profile(&self) -> &CaptureVideoProfile {
        &self.requested_profile
    }

    pub fn negotiated_profile(&self) -> &CaptureVideoProfile {
        &self.negotiated_profile
    }

    /// Emitted RGB8 dimensions validated during preflight. For v1, this is
    /// the negotiated camera-profile size; for v2, it is explicitly reported.
    pub fn output_frame_geometry(&self) -> CaptureOutputFrameGeometry {
        self.output_frame_geometry
    }

    pub fn timestamp_semantics(&self) -> &CaptureTimestampSemantics {
        &self.timestamp_semantics
    }

    /// Actual source readbacks, preserved independently from requested values.
    pub fn applied_controls(&self) -> &CaptureAppliedCameraControls {
        &self.applied_controls
    }

    /// Output geometry applied by this source before it emitted RGB8 frames.
    pub fn applied_frame_geometry(&self) -> &CaptureAppliedFrameGeometry {
        &self.applied_frame_geometry
    }

    /// Full source report checked during preflight, preserved for the
    /// request-bound live startup acknowledgement.
    pub fn source_report(&self) -> &CaptureSessionReport {
        &self.source_report
    }

    /// SHA-256 of the exact bytes from the deterministic request encoder.
    /// `None` means this compatibility was created from hand-built settings
    /// and is suitable for the existing replay protocol only.
    pub fn request_sha256(&self) -> Option<&str> {
        self.expected_settings.request_sha256()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapturePreflightError {
    InvalidIdentifier(&'static str),
    InvalidRevision(&'static str),
    UnsafeRevision(&'static str),
    ZeroMaximumFrameAge,
    FrameSizeOverflow,
    FrameExceedsConfiguredLimit {
        required_bytes: usize,
        frame_limit_bytes: usize,
    },
    FrameExceedsAggregateLimit {
        required_bytes: usize,
        aggregate_limit_bytes: usize,
    },
    CalibrationMismatch(&'static str),
    DeviceMismatch,
    UnsupportedCaptureBackend,
    RequestedBackendMismatch,
    RequestedProfileMismatch,
    NegotiatedProfileMismatch,
    OutputFrameGeometryMissing,
    OutputFrameGeometryMismatch,
    ConfigurationRevisionMismatch,
    CameraProfileMismatch,
    ImageSpaceMismatch,
    RectificationRevisionMismatch,
    RequiredCapabilityUnavailable {
        capability: &'static str,
        reported: CaptureCapability,
    },
    TimestampSemanticsMismatch,
    SourceTimestampsNotSupported(CaptureCapability),
    InvalidRequestedControl(&'static str),
    RequestedControlValuesMissing,
    RequiredControlReadbackUnavailable(&'static str),
    RequiredControlReadbackMismatch(&'static str),
}

impl Display for CapturePreflightError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "capture preflight failed: {self:?}")
    }
}

impl Error for CapturePreflightError {}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), CapturePreflightError> {
    if value.is_empty()
        || value.chars().count() > MAX_IDENTIFIER_CHARACTERS
        || value.chars().any(char::is_control)
    {
        return Err(CapturePreflightError::InvalidIdentifier(field));
    }
    Ok(())
}

fn valid_profile_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_IDENTIFIER_CHARACTERS
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b':' | b'-'))
}

fn validate_revision(value: u64, field: &'static str) -> Result<(), CapturePreflightError> {
    if value > MAX_SAFE_JSON_INTEGER {
        return Err(CapturePreflightError::UnsafeRevision(field));
    }
    Ok(())
}

fn rgb8_size(width_px: u32, height_px: u32) -> Result<u64, CapturePreflightError> {
    u64::from(width_px)
        .checked_mul(u64::from(height_px))
        .and_then(|pixels| pixels.checked_mul(RGB8_CHANNELS))
        .ok_or(CapturePreflightError::FrameSizeOverflow)
}

fn require_capability(
    capability_name: &'static str,
    required: bool,
    reported: CaptureCapability,
) -> Result<(), CapturePreflightError> {
    if required && reported != CaptureCapability::Supported {
        return Err(CapturePreflightError::RequiredCapabilityUnavailable {
            capability: capability_name,
            reported,
        });
    }
    Ok(())
}

fn validate_required_readbacks(
    requirements: &CaptureControlRequirements,
    requested: RequestedCaptureControlValues,
    applied: &CaptureAppliedCameraControls,
) -> Result<(), CapturePreflightError> {
    if requirements.manual_exposure || requirements.automatic_exposure {
        match applied.exposure_mode() {
            CaptureControlReadback::Applied(actual) if *actual == requested.exposure_mode => {}
            CaptureControlReadback::Applied(_) => {
                return Err(CapturePreflightError::RequiredControlReadbackMismatch(
                    "exposure_mode",
                ));
            }
            CaptureControlReadback::Unsupported | CaptureControlReadback::Unknown => {
                return Err(CapturePreflightError::RequiredControlReadbackUnavailable(
                    "exposure_mode",
                ));
            }
        }
    }
    if requirements.manual_exposure {
        if let Some(expected) = requested.exposure_value {
            validate_required_numeric_readback(
                "exposure_value",
                applied.exposure_value(),
                expected,
            )?;
        }
    }
    if requirements.gain_control {
        validate_required_numeric_readback("gain", applied.gain(), requested.gain)?;
    }
    Ok(())
}

fn validate_required_numeric_readback(
    field: &'static str,
    actual: &CaptureControlReadback<f64>,
    expected: f64,
) -> Result<(), CapturePreflightError> {
    match actual {
        CaptureControlReadback::Applied(value) if *value == expected => Ok(()),
        CaptureControlReadback::Applied(_) => Err(
            CapturePreflightError::RequiredControlReadbackMismatch(field),
        ),
        CaptureControlReadback::Unsupported | CaptureControlReadback::Unknown => Err(
            CapturePreflightError::RequiredControlReadbackUnavailable(field),
        ),
    }
}
