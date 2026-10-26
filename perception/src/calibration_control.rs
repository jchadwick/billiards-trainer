//! Versioned, bounded loopback protocol for the Rust-owned calibration store.
//!
//! The protocol is multiplexed on the frame-preview listener so the gateway
//! has one local endpoint and one readiness record. `BTCR0001` reads the
//! durable snapshot. `BTCA0001` is followed by a four-byte big-endian JSON
//! request length and the exact JSON body. Responses are a four-byte
//! big-endian length and one bounded JSON object. The preview listener limits
//! concurrent handlers and applies read/write deadlines before calling here.

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::Duration;

use serde::de::Deserializer;
use serde::{Deserialize, Serialize};

use crate::calibration::{ImagePointPx, TableCalibrationSolverInput};
use crate::calibration_store::{CalibrationStore, CalibrationStoreError, CalibrationStoreSnapshot};
use crate::frame_preview::{FramePreviewDescriptorV1, LatestCapturedFrameSlot};
use crate::table_visibility::{
    derive_reference_from_confirmed_frame, NormalizedPoint, OperatorConfirmedFrame, Rgb8Frame,
    TableRegion, TableVisibilityConfig,
};

pub(crate) const CALIBRATION_READ_REQUEST_V1: &[u8; 8] = b"BTCR0001";
pub(crate) const CALIBRATION_APPLY_REQUEST_V1: &[u8; 8] = b"BTCA0001";
pub(crate) const CALIBRATION_CONTROL_SCHEMA_VERSION: u8 = 1;
const MINIMUM_CALIBRATION_REQUEST_BYTES: usize = 1_024;
const MAXIMUM_CALIBRATION_REQUEST_BYTES: usize = 1_048_576;

/// Rust-owned durable store and request-body bound for the control commands.
#[derive(Debug, Clone)]
pub struct CalibrationControlConfig {
    pub store: CalibrationStore,
    pub maximum_request_bytes: usize,
    pub table_visibility: TableVisibilityConfig,
}

impl CalibrationControlConfig {
    pub fn new(
        store_path: impl Into<PathBuf>,
        lock_timeout: Duration,
        maximum_store_file_bytes: usize,
        maximum_request_bytes: usize,
        table_visibility: TableVisibilityConfig,
    ) -> Result<Self, CalibrationControlConfigError> {
        if lock_timeout.is_zero() {
            return Err(CalibrationControlConfigError::ZeroLockTimeout);
        }
        if !(MINIMUM_CALIBRATION_REQUEST_BYTES..=MAXIMUM_CALIBRATION_REQUEST_BYTES)
            .contains(&maximum_request_bytes)
        {
            return Err(CalibrationControlConfigError::InvalidMaximumRequestBytes);
        }
        let store = CalibrationStore::new(store_path, lock_timeout, maximum_store_file_bytes)
            .map_err(CalibrationControlConfigError::InvalidStore)?;
        Ok(Self {
            store,
            maximum_request_bytes,
            table_visibility,
        })
    }

    pub fn validate(&self) -> Result<(), CalibrationControlConfigError> {
        if !(MINIMUM_CALIBRATION_REQUEST_BYTES..=MAXIMUM_CALIBRATION_REQUEST_BYTES)
            .contains(&self.maximum_request_bytes)
        {
            return Err(CalibrationControlConfigError::InvalidMaximumRequestBytes);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum CalibrationControlConfigError {
    InvalidMaximumRequestBytes,
    ZeroLockTimeout,
    InvalidStore(CalibrationStoreError),
}

impl std::fmt::Display for CalibrationControlConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMaximumRequestBytes => formatter
                .write_str("calibration request maximum must be between 1024 and 1048576 bytes"),
            Self::ZeroLockTimeout => {
                formatter.write_str("calibration store lock timeout must be positive")
            }
            Self::InvalidStore(error) => write!(formatter, "invalid calibration store: {error}"),
        }
    }
}

impl std::error::Error for CalibrationControlConfigError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CalibrationControlStatusV1 {
    Ok,
    InvalidRequest,
    StaleContext,
    StaleRevision,
    Busy,
    StoreError,
    Unsupported,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct CalibrationControlResponseV1 {
    schema_version: u8,
    status: CalibrationControlStatusV1,
    snapshot: Option<CalibrationStoreSnapshot>,
    error: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    table_visibility_reference_issues: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationApplyRequestV1 {
    schema_version: u8,
    descriptor: FramePreviewDescriptorV1,
    table_width_mm: f64,
    table_height_mm: f64,
    corners_px: [ImagePointPx; 4],
    #[serde(deserialize_with = "required_option")]
    expected_store_revision: Option<u64>,
}

/// Handle a command after its fixed eight-byte magic has been read.
pub(crate) fn handle_request(
    mut stream: TcpStream,
    magic: [u8; 8],
    slot: &LatestCapturedFrameSlot,
    config: Option<&CalibrationControlConfig>,
    maximum_response_bytes: usize,
) -> io::Result<()> {
    let Some(config) = config else {
        return write_response(
            &mut stream,
            response(
                CalibrationControlStatusV1::Unsupported,
                None,
                Some("disabled"),
            ),
            maximum_response_bytes,
        );
    };
    if magic == *CALIBRATION_READ_REQUEST_V1 {
        return read_snapshot(&mut stream, config, maximum_response_bytes);
    }
    if magic == *CALIBRATION_APPLY_REQUEST_V1 {
        return apply_calibration(&mut stream, slot, config, maximum_response_bytes);
    }
    write_response(
        &mut stream,
        response(
            CalibrationControlStatusV1::Unsupported,
            None,
            Some("unknown_command"),
        ),
        maximum_response_bytes,
    )
}

pub(crate) fn write_busy_response(
    stream: &mut TcpStream,
    maximum_response_bytes: usize,
) -> io::Result<()> {
    write_response(
        stream,
        response(
            CalibrationControlStatusV1::Busy,
            None,
            Some("request_capacity_full"),
        ),
        maximum_response_bytes,
    )
}

fn read_snapshot(
    stream: &mut TcpStream,
    config: &CalibrationControlConfig,
    maximum_response_bytes: usize,
) -> io::Result<()> {
    match config.store.load() {
        Ok(snapshot) => write_response(
            stream,
            response(CalibrationControlStatusV1::Ok, Some(snapshot), None),
            maximum_response_bytes,
        ),
        Err(_) => write_response(
            stream,
            response(
                CalibrationControlStatusV1::StoreError,
                None,
                Some("store_unavailable"),
            ),
            maximum_response_bytes,
        ),
    }
}

fn apply_calibration(
    stream: &mut TcpStream,
    slot: &LatestCapturedFrameSlot,
    config: &CalibrationControlConfig,
    maximum_response_bytes: usize,
) -> io::Result<()> {
    let mut length_bytes = [0_u8; 4];
    stream.read_exact(&mut length_bytes)?;
    let request_length = u32::from_be_bytes(length_bytes) as usize;
    if request_length == 0 || request_length > config.maximum_request_bytes {
        return write_response(
            stream,
            response(
                CalibrationControlStatusV1::InvalidRequest,
                None,
                Some("request_size_limit"),
            ),
            maximum_response_bytes,
        );
    }
    let mut body = vec![0_u8; request_length];
    stream.read_exact(&mut body)?;
    let request: CalibrationApplyRequestV1 =
        match serde_json::from_slice::<CalibrationApplyRequestV1>(&body) {
            Ok(request) if request.schema_version == CALIBRATION_CONTROL_SCHEMA_VERSION => request,
            Ok(_) => {
                return write_response(
                    stream,
                    response(
                        CalibrationControlStatusV1::InvalidRequest,
                        None,
                        Some("unsupported_schema"),
                    ),
                    maximum_response_bytes,
                )
            }
            Err(_) => {
                return write_response(
                    stream,
                    response(
                        CalibrationControlStatusV1::InvalidRequest,
                        None,
                        Some("invalid_json"),
                    ),
                    maximum_response_bytes,
                )
            }
        };

    // Session transitions are serialized against validation and the short
    // durable commit, while ordinary frame publication never takes this gate.
    let _context_guard = slot.lock_calibration_context();
    let Some(preview) = slot.confirmed_calibration_preview(&request.descriptor) else {
        return write_response(
            stream,
            response(
                CalibrationControlStatusV1::StaleContext,
                None,
                Some("confirmed_preview_unavailable"),
            ),
            maximum_response_bytes,
        );
    };
    let binding_context = match slot.calibration_binding_context(&request.descriptor) {
        Ok(context) => context,
        Err(_) => {
            return write_response(
                stream,
                response(
                    CalibrationControlStatusV1::StaleContext,
                    None,
                    Some("capture_context_mismatch"),
                ),
                maximum_response_bytes,
            )
        }
    };
    let input = TableCalibrationSolverInput {
        calibration_revision: 0,
        camera_profile_id: binding_context.camera_profile_id.clone(),
        rectification_revision: binding_context.rectification_revision,
        image_space: binding_context.image_space,
        source_width_px: binding_context.source_width_px,
        source_height_px: binding_context.source_height_px,
        table_width_mm: request.table_width_mm,
        table_height_mm: request.table_height_mm,
        corners_px: request.corners_px,
    };
    let table_region = TableRegion::new(request.corners_px.map(|point| NormalizedPoint {
        x: (point.x_px / f64::from(preview.descriptor().width_px.saturating_sub(1))) as f32,
        y: (point.y_px / f64::from(preview.descriptor().height_px.saturating_sub(1))) as f32,
    }));
    let reference = match derive_reference_from_confirmed_frame(
        OperatorConfirmedFrame::new(
            Rgb8Frame {
                width: preview.descriptor().width_px,
                height: preview.descriptor().height_px,
                pixels: preview.rgb8(),
            },
            table_region,
        ),
        &config.table_visibility,
    ) {
        Ok(result) if result.validity.valid => result.reference,
        Ok(result) => {
            let mut response = response(
                CalibrationControlStatusV1::InvalidRequest,
                None,
                Some("table_visibility_reference_quality"),
            );
            response.table_visibility_reference_issues = Some(
                result
                    .validity
                    .reasons
                    .iter()
                    .map(|issue| format!("{issue:?}"))
                    .collect(),
            );
            return write_response(stream, response, maximum_response_bytes);
        }
        Err(_) => {
            return write_response(
                stream,
                response(
                    CalibrationControlStatusV1::InvalidRequest,
                    None,
                    Some("invalid_visibility_reference_input"),
                ),
                maximum_response_bytes,
            )
        }
    }
    .expect("valid reference decision must contain its feature bundle");
    match config.store.save_with_context_and_visibility(
        request.expected_store_revision,
        input,
        Some(binding_context),
        Some(reference),
    ) {
        Ok(snapshot) => write_response(
            stream,
            response(CalibrationControlStatusV1::Ok, Some(snapshot), None),
            maximum_response_bytes,
        ),
        Err(CalibrationStoreError::StaleExpectedRevision { .. }) => write_response(
            stream,
            response(
                CalibrationControlStatusV1::StaleRevision,
                None,
                Some("stale_store_revision"),
            ),
            maximum_response_bytes,
        ),
        Err(
            CalibrationStoreError::InvalidCalibration(_)
            | CalibrationStoreError::InvalidCaptureContext,
        ) => write_response(
            stream,
            response(
                CalibrationControlStatusV1::InvalidRequest,
                None,
                Some("invalid_calibration"),
            ),
            maximum_response_bytes,
        ),
        Err(_) => write_response(
            stream,
            response(
                CalibrationControlStatusV1::StoreError,
                None,
                Some("store_unavailable"),
            ),
            maximum_response_bytes,
        ),
    }
}

fn response(
    status: CalibrationControlStatusV1,
    snapshot: Option<CalibrationStoreSnapshot>,
    error: Option<&'static str>,
) -> CalibrationControlResponseV1 {
    CalibrationControlResponseV1 {
        schema_version: CALIBRATION_CONTROL_SCHEMA_VERSION,
        status,
        snapshot,
        error,
        table_visibility_reference_issues: None,
    }
}

fn write_response(
    stream: &mut TcpStream,
    response: CalibrationControlResponseV1,
    maximum_response_bytes: usize,
) -> io::Result<()> {
    let body = serde_json::to_vec(&response)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if body.len() > maximum_response_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "calibration control response exceeds configured byte limit",
        ));
    }
    let length = u32::try_from(body.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "calibration control response exceeds protocol length",
        )
    })?;
    stream.write_all(&length.to_be_bytes())?;
    stream.write_all(&body)
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
    use super::*;

    #[test]
    fn request_config_rejects_unbounded_body_size() {
        let directory = tempfile::tempdir().unwrap();
        let result = CalibrationControlConfig::new(
            directory.path().join("table.json"),
            Duration::from_millis(10),
            64 * 1024,
            MAXIMUM_CALIBRATION_REQUEST_BYTES + 1,
            TableVisibilityConfig::default(),
        );
        assert!(matches!(
            result,
            Err(CalibrationControlConfigError::InvalidMaximumRequestBytes)
        ));
    }
}
