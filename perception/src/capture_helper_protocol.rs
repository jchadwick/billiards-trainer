//! Bounded framing and identity validation for supervised capture helpers.
//!
//! The stream begins with one bounded schema-v1 synthetic or schema-v2
//! physical helper-ready record containing a big-endian `u32` JSON length and
//! JSON metadata. It is followed by frame
//! records containing a big-endian `u32` versioned JSON-header length, the
//! header, a big-endian `u32` RGB payload length, and packed RGB8 bytes. Limits
//! are supplied by the caller so this module does not impose production-specific
//! image budgets.

use std::fmt;
use std::io::{self, Read, Write};
use std::ops::DerefMut;

use serde::{Deserialize, Serialize};

use crate::capture::CaptureHelperTimingNs;

const MAX_DIAGNOSTIC_CHARS: usize = 192;
const CAPTURE_HELPER_READY_SCHEMA_VERSION_V1: u8 = 1;
const CAPTURE_HELPER_READY_SCHEMA_VERSION_V2: u8 = 2;

/// Strict, versioned acknowledgement emitted after a helper has negotiated an
/// output RGB sample and before it emits frame records. Schema v2 additionally
/// binds physical helpers to a stable source ID and the actual source input
/// mode, independently of the emitted RGB geometry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureHelperReady {
    pub record_schema_version: u8,
    pub session_id: String,
    pub configuration_revision: u64,
    pub camera_profile_id: String,
    pub capture_settings_sha256: String,
    pub width_px: u32,
    pub height_px: u32,
    pub frame_rate_numerator: u32,
    pub frame_rate_seconds_denominator: u32,
    pub pixel_format: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_device_id: Option<String>,
    /// Raw persistent provider ID and key are diagnostics that let the parent
    /// recompute the selector; they are never interpreted as camera exposure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_identifier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_identifier_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_profile: Option<CaptureHelperInputProfile>,
}

/// Actual source-side capture mode before decoding/scaling to RGB8.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureHelperInputProfile {
    pub width_px: u32,
    pub height_px: u32,
    pub frame_rate_numerator: u32,
    pub frame_rate_seconds_denominator: u32,
    pub pixel_format: String,
}

/// Write one length-bounded helper-ready record as `[u32 BE JSON length][JSON]`.
///
/// Serialization is capped before bytes are written to the destination, so a
/// caller-selected limit is respected even when metadata strings are invalidly
/// large. `max_json_bytes` must be positive.
pub fn write_ready_record<W: Write>(
    writer: &mut W,
    ready: &CaptureHelperReady,
    max_json_bytes: usize,
) -> Result<(), FrameReaderError> {
    validate_ready_limit(max_json_bytes)?;
    validate_ready(ready)?;

    let mut json = BoundedJsonBuffer::new(max_json_bytes);
    serde_json::to_writer(&mut json, ready)
        .map_err(|error| invalid(format!("could not serialize helper-ready JSON: {error}")))?;
    let length = u32::try_from(json.bytes.len())
        .map_err(|_| invalid("helper-ready JSON exceeds framing limit"))?;
    writer
        .write_all(&length.to_be_bytes())
        .and_then(|()| writer.write_all(&json.bytes))
        .map_err(|error| io_error("helper-ready record write", error))
}

/// Read one required, length-bounded helper-ready record.
///
/// Clean EOF is an error because a helper that exits before acknowledging its
/// negotiated settings is not ready. The untrusted length is checked before
/// allocating the JSON buffer.
pub fn read_ready_record<R: Read>(
    reader: &mut R,
    max_json_bytes: usize,
) -> Result<CaptureHelperReady, FrameReaderError> {
    validate_ready_limit(max_json_bytes)?;
    let length_bytes = read_prefix(reader, "ready length prefix", "ready length prefix")?
        .ok_or_else(|| invalid("unexpected EOF before helper-ready record"))?;
    let json_len = usize::try_from(u32::from_be_bytes(length_bytes))
        .map_err(|_| invalid("helper-ready length does not fit address space"))?;
    if json_len == 0 {
        return Err(invalid("helper-ready JSON length must be positive"));
    }
    if json_len > max_json_bytes {
        return Err(invalid("helper-ready JSON length exceeds configured limit"));
    }

    let mut json = Vec::new();
    json.try_reserve_exact(json_len)
        .map_err(|error| allocation_error("helper-ready JSON", error))?;
    json.resize(json_len, 0);
    read_exact_context(reader, &mut json, "helper-ready JSON")?;
    let ready: CaptureHelperReady = serde_json::from_slice(&json).map_err(|error| {
        invalid(format!(
            "invalid helper-ready JSON: {}",
            bounded_diagnostic(error.to_string())
        ))
    })?;
    validate_ready(&ready)?;
    Ok(ready)
}

fn validate_ready_limit(max_json_bytes: usize) -> Result<(), FrameReaderError> {
    if max_json_bytes == 0 {
        return Err(FrameReaderError::InvalidLimit("ready JSON byte"));
    }
    Ok(())
}

fn validate_ready(ready: &CaptureHelperReady) -> Result<(), FrameReaderError> {
    if !matches!(
        ready.record_schema_version,
        CAPTURE_HELPER_READY_SCHEMA_VERSION_V1 | CAPTURE_HELPER_READY_SCHEMA_VERSION_V2
    ) {
        return Err(invalid(format!(
            "unsupported helper-ready schema version {}; expected {} or {}",
            ready.record_schema_version,
            CAPTURE_HELPER_READY_SCHEMA_VERSION_V1,
            CAPTURE_HELPER_READY_SCHEMA_VERSION_V2
        )));
    }
    if ready.session_id.is_empty()
        || ready.camera_profile_id.is_empty()
        || ready.capture_settings_sha256.is_empty()
        || ready.pixel_format.is_empty()
    {
        return Err(invalid(
            "helper-ready identity and format fields must be non-empty",
        ));
    }
    if ready.width_px == 0
        || ready.height_px == 0
        || ready.frame_rate_numerator == 0
        || ready.frame_rate_seconds_denominator == 0
    {
        return Err(invalid(
            "helper-ready geometry and frame rate must be positive",
        ));
    }
    match ready.record_schema_version {
        CAPTURE_HELPER_READY_SCHEMA_VERSION_V1 => {
            if ready.source_device_id.is_some()
                || ready.source_identifier.is_some()
                || ready.source_identifier_source.is_some()
                || ready.input_profile.is_some()
            {
                return Err(invalid(
                    "schema-v1 helper-ready records cannot include physical source fields",
                ));
            }
        }
        CAPTURE_HELPER_READY_SCHEMA_VERSION_V2 => {
            let Some(device_id) = ready.source_device_id.as_deref() else {
                return Err(invalid(
                    "schema-v2 helper-ready record must include source_device_id",
                ));
            };
            let Some(input_profile) = ready.input_profile.as_ref() else {
                return Err(invalid(
                    "schema-v2 helper-ready record must include input_profile",
                ));
            };
            let Some(source_identifier) = ready.source_identifier.as_deref() else {
                return Err(invalid(
                    "schema-v2 helper-ready record must include source_identifier",
                ));
            };
            let Some(source_identifier_source) = ready.source_identifier_source.as_deref() else {
                return Err(invalid(
                    "schema-v2 helper-ready record must include source_identifier_source",
                ));
            };
            if device_id.is_empty()
                || source_identifier.is_empty()
                || source_identifier_source.is_empty()
                || input_profile.pixel_format.is_empty()
            {
                return Err(invalid(
                    "schema-v2 helper-ready device, provider, and format identities must be non-empty",
                ));
            }
            if input_profile.width_px == 0
                || input_profile.height_px == 0
                || input_profile.frame_rate_numerator == 0
                || input_profile.frame_rate_seconds_denominator == 0
            {
                return Err(invalid(
                    "schema-v2 helper-ready input profile geometry and frame rate must be positive",
                ));
            }
        }
        _ => unreachable!("version checked above"),
    }
    Ok(())
}

struct BoundedJsonBuffer {
    bytes: Vec<u8>,
    maximum: usize,
}

impl BoundedJsonBuffer {
    fn new(maximum: usize) -> Self {
        Self {
            bytes: Vec::new(),
            maximum,
        }
    }
}

impl Write for BoundedJsonBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next_len = self.bytes.len().checked_add(bytes.len()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "helper-ready JSON length overflow",
            )
        })?;
        if next_len > self.maximum || next_len > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "helper-ready JSON exceeds configured framing limit",
            ));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(|error| io::Error::other(error.to_string()))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Strict metadata emitted for each helper frame. Version 3 is used only when
/// the live audit requests helper-side timing segments; version 2 remains the
/// normal untimed form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameHeader {
    pub record_schema_version: u8,
    pub session_id: String,
    pub frame_id: String,
    pub configuration_revision: u64,
    pub camera_profile_id: String,
    pub capture_settings_sha256: String,
    pub width_px: u32,
    pub height_px: u32,
    pub pixel_format: String,
    pub row_stride_bytes: usize,
    /// Pipeline PTS is deliberately not represented as a verified source time.
    #[serde(deserialize_with = "required_option")]
    pub unverified_pipeline_pts_ticks: Option<String>,
    /// `None` says that source/exposure time was not provided or verified.
    #[serde(deserialize_with = "required_option")]
    pub source_timestamp_ticks: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_read_completed_elapsed_ns: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_timings_ns: Option<CaptureHelperTimingNs>,
}

/// A decoded record with caller-selected payload storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameRecord<P> {
    pub header: FrameHeader,
    pub payload: P,
}

/// Failure while reading one bounded helper record.
#[derive(Debug)]
pub enum FrameReaderError {
    InvalidLimit(&'static str),
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
        detail: String,
    },
    InvalidRecord(String),
    BufferAllocation(String),
    BufferSizeMismatch {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for FrameReaderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimit(name) => write!(formatter, "{name} limit must be positive"),
            Self::Io {
                operation,
                kind,
                detail,
            } => write!(formatter, "{operation} failed ({kind:?}): {detail}"),
            Self::InvalidRecord(message) => formatter.write_str(message),
            Self::BufferAllocation(message) => {
                write!(formatter, "could not allocate payload buffer: {message}")
            }
            Self::BufferSizeMismatch { expected, actual } => write!(
                formatter,
                "payload allocator returned {actual} bytes; record requires {expected}"
            ),
        }
    }
}

impl std::error::Error for FrameReaderError {}

/// Reads helper records without allocating from untrusted lengths before
/// checking caller-supplied limits.
pub struct FrameRecordReader<R> {
    reader: R,
    max_header_bytes: usize,
    max_payload_bytes: usize,
}

impl<R: Read> FrameRecordReader<R> {
    /// Create a reader with explicit positive limits for both framed sections.
    pub fn new(
        reader: R,
        max_header_bytes: usize,
        max_payload_bytes: usize,
    ) -> Result<Self, FrameReaderError> {
        if max_header_bytes == 0 {
            return Err(FrameReaderError::InvalidLimit("header byte"));
        }
        if max_payload_bytes == 0 {
            return Err(FrameReaderError::InvalidLimit("payload byte"));
        }
        Ok(Self {
            reader,
            max_header_bytes,
            max_payload_bytes,
        })
    }

    /// Read one record into a buffer allocated by the caller.
    ///
    /// The allocator runs only after the header has been parsed and the
    /// payload length has passed the configured limit. The returned buffer's
    /// length is checked before any payload bytes are read. `Ok(None)` means
    /// clean EOF before the first byte of a new record's header-length prefix.
    pub fn read_next_with_buffer<B, F, E>(
        &mut self,
        allocate: F,
    ) -> Result<Option<FrameRecord<B>>, FrameReaderError>
    where
        B: DerefMut<Target = [u8]>,
        F: FnOnce(&FrameHeader, usize) -> Result<B, E>,
        E: fmt::Display,
    {
        let Some(header_length_bytes) = read_prefix(
            &mut self.reader,
            "header length prefix",
            "header length prefix",
        )?
        else {
            return Ok(None);
        };
        let header_len = usize::try_from(u32::from_be_bytes(header_length_bytes))
            .map_err(|_| invalid("header length does not fit address space"))?;
        if header_len == 0 {
            return Err(invalid("header length must be positive"));
        }
        if header_len > self.max_header_bytes {
            return Err(invalid("header length exceeds configured limit"));
        }

        let mut header_json = Vec::new();
        header_json
            .try_reserve_exact(header_len)
            .map_err(|error| allocation_error("header", error))?;
        header_json.resize(header_len, 0);
        read_exact_context(&mut self.reader, &mut header_json, "JSON header")?;
        let header: FrameHeader = serde_json::from_slice(&header_json).map_err(|error| {
            invalid(format!(
                "invalid frame JSON header: {}",
                bounded_diagnostic(error.to_string())
            ))
        })?;

        let payload_length_bytes = read_prefix(
            &mut self.reader,
            "payload length prefix",
            "payload length prefix",
        )?
        .ok_or_else(|| invalid("truncated record: missing payload length prefix"))?;
        let payload_len = usize::try_from(u32::from_be_bytes(payload_length_bytes))
            .map_err(|_| invalid("payload length does not fit address space"))?;
        if payload_len == 0 {
            return Err(invalid("payload length must be positive"));
        }
        if payload_len > self.max_payload_bytes {
            return Err(invalid("payload length exceeds configured limit"));
        }

        let mut payload = allocate(&header, payload_len).map_err(|error| {
            FrameReaderError::BufferAllocation(bounded_diagnostic(error.to_string()))
        })?;
        if payload.deref_mut().len() != payload_len {
            return Err(FrameReaderError::BufferSizeMismatch {
                expected: payload_len,
                actual: payload.deref_mut().len(),
            });
        }
        read_exact_context(&mut self.reader, payload.deref_mut(), "RGB payload")?;

        Ok(Some(FrameRecord { header, payload }))
    }

    /// Convenience method that allocates a bounded `Vec<u8>` payload.
    pub fn read_next(&mut self) -> Result<Option<FrameRecord<Vec<u8>>>, FrameReaderError> {
        self.read_next_with_buffer(|_, payload_len| {
            let mut payload = Vec::new();
            payload
                .try_reserve_exact(payload_len)
                .map_err(|error| error.to_string())?;
            payload.resize(payload_len, 0);
            Ok::<_, String>(payload)
        })
    }

    pub fn into_inner(self) -> R {
        self.reader
    }
}

/// Expected identity and geometry for all records in one helper session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameBinding {
    pub session_id: String,
    pub configuration_revision: u64,
    pub camera_profile_id: String,
    pub capture_settings_sha256: String,
    pub width_px: u32,
    pub height_px: u32,
    pub max_payload_bytes: usize,
}

/// Stateful validation for the identity and packed geometry of a helper run.
///
/// Sequence state advances only after the entire record has passed validation,
/// so a malformed record cannot silently consume an expected frame ID.
pub struct FrameBindingValidator {
    binding: FrameBinding,
    next_sequence: u64,
    exhausted: bool,
}

impl FrameBindingValidator {
    pub fn new(binding: FrameBinding) -> Result<Self, FrameBindingError> {
        if binding.width_px == 0 || binding.height_px == 0 {
            return Err(FrameBindingError::InvalidExpectedGeometry);
        }
        if binding.max_payload_bytes == 0 {
            return Err(FrameBindingError::InvalidPayloadLimit);
        }
        Ok(Self {
            binding,
            next_sequence: 0,
            exhausted: false,
        })
    }

    /// Validate one header and its packed payload length against the session.
    pub fn validate(
        &mut self,
        header: &FrameHeader,
        payload_len: usize,
    ) -> Result<(), FrameBindingError> {
        if !matches!(header.record_schema_version, 2 | 3) {
            return Err(FrameBindingError::SchemaVersion {
                actual: header.record_schema_version,
            });
        }
        if (header.record_schema_version == 3) != header.capture_timings_ns.is_some() {
            return Err(FrameBindingError::CaptureTimingSchemaMismatch);
        }
        if header.session_id != self.binding.session_id {
            return Err(FrameBindingError::SessionIdentity);
        }
        if self.exhausted {
            return Err(FrameBindingError::SequenceExhausted);
        }
        let expected_frame_id = self.next_sequence.to_string();
        if header.frame_id != expected_frame_id {
            return Err(FrameBindingError::FrameSequence {
                expected: self.next_sequence,
            });
        }
        if header.configuration_revision != self.binding.configuration_revision {
            return Err(FrameBindingError::ConfigurationRevision {
                expected: self.binding.configuration_revision,
            });
        }
        if header.camera_profile_id != self.binding.camera_profile_id {
            return Err(FrameBindingError::CameraProfileIdentity);
        }
        if header.capture_settings_sha256 != self.binding.capture_settings_sha256 {
            return Err(FrameBindingError::CaptureSettingsDigest);
        }
        if header.pixel_format != "rgb8" {
            return Err(FrameBindingError::PixelFormat);
        }
        if header.width_px != self.binding.width_px || header.height_px != self.binding.height_px {
            return Err(FrameBindingError::Geometry {
                expected_width: self.binding.width_px,
                expected_height: self.binding.height_px,
            });
        }

        let width = usize::try_from(self.binding.width_px)
            .map_err(|_| FrameBindingError::GeometryOverflow)?;
        let height = usize::try_from(self.binding.height_px)
            .map_err(|_| FrameBindingError::GeometryOverflow)?;
        let row_bytes = width
            .checked_mul(3)
            .ok_or(FrameBindingError::GeometryOverflow)?;
        if header.row_stride_bytes < row_bytes {
            return Err(FrameBindingError::RowStride { minimum: row_bytes });
        }
        let expected_payload_len = row_bytes
            .checked_mul(height)
            .ok_or(FrameBindingError::GeometryOverflow)?;
        if expected_payload_len > self.binding.max_payload_bytes {
            return Err(FrameBindingError::ExpectedPayloadExceedsLimit {
                expected: expected_payload_len,
                limit: self.binding.max_payload_bytes,
            });
        }
        if payload_len > self.binding.max_payload_bytes {
            return Err(FrameBindingError::PayloadExceedsLimit {
                actual: payload_len,
                limit: self.binding.max_payload_bytes,
            });
        }
        if payload_len != expected_payload_len {
            return Err(FrameBindingError::PayloadLength {
                expected: expected_payload_len,
                actual: payload_len,
            });
        }

        if self.next_sequence == u64::MAX {
            self.exhausted = true;
        } else {
            self.next_sequence += 1;
        }
        Ok(())
    }
}

/// A failed identity, sequence, or payload-contract check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameBindingError {
    InvalidExpectedGeometry,
    InvalidPayloadLimit,
    SchemaVersion {
        actual: u8,
    },
    CaptureTimingSchemaMismatch,
    SessionIdentity,
    SequenceExhausted,
    FrameSequence {
        expected: u64,
    },
    ConfigurationRevision {
        expected: u64,
    },
    CameraProfileIdentity,
    CaptureSettingsDigest,
    PixelFormat,
    Geometry {
        expected_width: u32,
        expected_height: u32,
    },
    GeometryOverflow,
    RowStride {
        minimum: usize,
    },
    ExpectedPayloadExceedsLimit {
        expected: usize,
        limit: usize,
    },
    PayloadExceedsLimit {
        actual: usize,
        limit: usize,
    },
    PayloadLength {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for FrameBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExpectedGeometry => {
                formatter.write_str("expected frame geometry must be positive")
            }
            Self::InvalidPayloadLimit => {
                formatter.write_str("maximum payload byte limit must be positive")
            }
            Self::SchemaVersion { actual } => {
                write!(
                    formatter,
                    "unsupported frame schema version {actual}; expected 2 or 3"
                )
            }
            Self::CaptureTimingSchemaMismatch => formatter.write_str(
                "frame capture timing fields do not match the declared frame schema version",
            ),
            Self::SessionIdentity => formatter.write_str("frame session identity does not match"),
            Self::SequenceExhausted => formatter.write_str("frame sequence exhausted"),
            Self::FrameSequence { expected } => {
                write!(
                    formatter,
                    "frame sequence is not contiguous; expected {expected}"
                )
            }
            Self::ConfigurationRevision { expected } => write!(
                formatter,
                "configuration revision does not match expected {expected}"
            ),
            Self::CameraProfileIdentity => {
                formatter.write_str("camera profile identity does not match")
            }
            Self::CaptureSettingsDigest => {
                formatter.write_str("capture settings digest does not match")
            }
            Self::PixelFormat => formatter.write_str("pixel format must be rgb8"),
            Self::Geometry {
                expected_width,
                expected_height,
            } => write!(
                formatter,
                "frame geometry does not match expected {expected_width}x{expected_height}"
            ),
            Self::GeometryOverflow => formatter.write_str("frame geometry arithmetic overflow"),
            Self::RowStride { minimum } => write!(
                formatter,
                "row stride is shorter than packed RGB row of {minimum} bytes"
            ),
            Self::ExpectedPayloadExceedsLimit { expected, limit } => write!(
                formatter,
                "expected packed payload of {expected} bytes exceeds limit {limit}"
            ),
            Self::PayloadExceedsLimit { actual, limit } => {
                write!(formatter, "payload of {actual} bytes exceeds limit {limit}")
            }
            Self::PayloadLength { expected, actual } => write!(
                formatter,
                "packed payload length is {actual} bytes; expected {expected}"
            ),
        }
    }
}

impl std::error::Error for FrameBindingError {}

fn read_prefix<R: Read>(
    reader: &mut R,
    operation: &'static str,
    truncated_label: &'static str,
) -> Result<Option<[u8; 4]>, FrameReaderError> {
    let mut prefix = [0; 4];
    let mut filled = 0;
    while filled < prefix.len() {
        match reader.read(&mut prefix[filled..]) {
            Ok(0) if filled == 0 => return Ok(None),
            Ok(0) => {
                return Err(invalid(format!(
                    "truncated {truncated_label}: received {filled} of 4 bytes"
                )))
            }
            Ok(count) => filled += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(io_error(operation, error)),
        }
    }
    Ok(Some(prefix))
}

fn read_exact_context<R: Read>(
    reader: &mut R,
    target: &mut [u8],
    operation: &'static str,
) -> Result<(), FrameReaderError> {
    let mut filled = 0;
    while filled < target.len() {
        match reader.read(&mut target[filled..]) {
            Ok(0) => {
                return Err(invalid(format!(
                    "truncated {operation}: received {filled} of {} bytes",
                    target.len()
                )))
            }
            Ok(count) => filled += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(io_error(operation, error)),
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> FrameReaderError {
    FrameReaderError::InvalidRecord(bounded_diagnostic(message.into()))
}

fn io_error(operation: &'static str, error: io::Error) -> FrameReaderError {
    FrameReaderError::Io {
        operation,
        kind: error.kind(),
        detail: bounded_diagnostic(error.to_string()),
    }
}

fn allocation_error(operation: &str, error: impl fmt::Display) -> FrameReaderError {
    FrameReaderError::BufferAllocation(format!(
        "{operation}: {}",
        bounded_diagnostic(error.to_string())
    ))
}

fn bounded_diagnostic(message: String) -> String {
    let mut chars = message.chars();
    let bounded: String = chars.by_ref().take(MAX_DIAGNOSTIC_CHARS).collect();
    if chars.next().is_some() {
        format!("{bounded}…")
    } else {
        bounded
    }
}

fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(deserializer)
}

#[cfg(test)]
mod ready_record_tests {
    use std::io::Cursor;

    use super::{
        read_ready_record, write_ready_record, CaptureHelperInputProfile, CaptureHelperReady,
        FrameReaderError,
    };

    const MAX_READY_JSON_BYTES: usize = 1_024;

    fn ready() -> CaptureHelperReady {
        CaptureHelperReady {
            record_schema_version: 1,
            session_id: "helper-session".to_owned(),
            configuration_revision: 19,
            camera_profile_id: "profile-a".to_owned(),
            capture_settings_sha256: "a".repeat(64),
            width_px: 640,
            height_px: 480,
            frame_rate_numerator: 30000,
            frame_rate_seconds_denominator: 1001,
            pixel_format: "RGB8".to_owned(),
            source_device_id: None,
            source_identifier: None,
            source_identifier_source: None,
            input_profile: None,
        }
    }

    fn encoded_ready(value: &CaptureHelperReady) -> Vec<u8> {
        let json = serde_json::to_vec(value).unwrap();
        let mut bytes = Vec::with_capacity(4 + json.len());
        bytes.extend_from_slice(&(json.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&json);
        bytes
    }

    #[test]
    fn ready_record_round_trips_and_requires_a_record_before_eof() {
        let expected = ready();
        let mut bytes = Vec::new();
        write_ready_record(&mut bytes, &expected, MAX_READY_JSON_BYTES).unwrap();
        assert_eq!(
            u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize,
            bytes.len() - 4
        );
        assert_eq!(
            read_ready_record(&mut Cursor::new(bytes), MAX_READY_JSON_BYTES).unwrap(),
            expected
        );
        assert!(
            read_ready_record(&mut Cursor::new([]), MAX_READY_JSON_BYTES)
                .unwrap_err()
                .to_string()
                .contains("unexpected EOF before helper-ready record")
        );
    }

    #[test]
    fn ready_record_requires_positive_limits_and_caps_serialized_output() {
        let mut output = Vec::new();
        assert!(matches!(
            write_ready_record(&mut output, &ready(), 0),
            Err(FrameReaderError::InvalidLimit("ready JSON byte"))
        ));
        assert!(matches!(
            read_ready_record(&mut Cursor::new([]), 0),
            Err(FrameReaderError::InvalidLimit("ready JSON byte"))
        ));

        let error = write_ready_record(&mut output, &ready(), 1).unwrap_err();
        assert!(error.to_string().contains("helper-ready JSON exceeds"));
        assert!(
            output.is_empty(),
            "failed serialization must write no prefix"
        );
    }

    #[test]
    fn ready_reader_rejects_zero_oversized_and_truncated_lengths_before_json() {
        assert!(
            read_ready_record(&mut Cursor::new([0, 0, 0, 0]), MAX_READY_JSON_BYTES)
                .unwrap_err()
                .to_string()
                .contains("length must be positive")
        );

        let oversized_length = (MAX_READY_JSON_BYTES as u32 + 1).to_be_bytes();
        assert!(
            read_ready_record(&mut Cursor::new(oversized_length), MAX_READY_JSON_BYTES)
                .unwrap_err()
                .to_string()
                .contains("length exceeds configured limit")
        );

        for prefix_length in 1..4 {
            assert!(read_ready_record(
                &mut Cursor::new(vec![0; prefix_length]),
                MAX_READY_JSON_BYTES
            )
            .unwrap_err()
            .to_string()
            .contains("truncated ready length prefix"));
        }

        let full = encoded_ready(&ready());
        assert!(read_ready_record(
            &mut Cursor::new(full[..full.len() - 1].to_vec()),
            MAX_READY_JSON_BYTES
        )
        .unwrap_err()
        .to_string()
        .contains("truncated helper-ready JSON"));
    }

    #[test]
    fn ready_reader_rejects_malformed_unknown_and_unsupported_records() {
        assert!(read_ready_record(
            &mut Cursor::new({
                let mut bytes = Vec::from(4_u32.to_be_bytes());
                bytes.extend_from_slice(b"nope");
                bytes
            }),
            MAX_READY_JSON_BYTES
        )
        .unwrap_err()
        .to_string()
        .contains("invalid helper-ready JSON"));

        let mut unknown = serde_json::to_value(ready()).unwrap();
        unknown["unexpected"] = serde_json::json!(true);
        assert!(read_ready_record(
            &mut Cursor::new(encoded_ready_json(unknown)),
            MAX_READY_JSON_BYTES
        )
        .unwrap_err()
        .to_string()
        .contains("invalid helper-ready JSON"));

        let mut unsupported = ready();
        unsupported.record_schema_version = 3;
        assert!(read_ready_record(
            &mut Cursor::new(encoded_ready(&unsupported)),
            MAX_READY_JSON_BYTES
        )
        .unwrap_err()
        .to_string()
        .contains("unsupported helper-ready schema version"));

        let mut invalid_geometry = ready();
        invalid_geometry.width_px = 0;
        assert!(read_ready_record(
            &mut Cursor::new(encoded_ready(&invalid_geometry)),
            MAX_READY_JSON_BYTES
        )
        .unwrap_err()
        .to_string()
        .contains("geometry and frame rate must be positive"));
    }

    #[test]
    fn ready_v2_binds_physical_identity_and_distinct_input_mode() {
        let mut value = ready();
        value.record_schema_version = 2;
        value.source_device_id = Some("camera-serial-42".to_owned());
        value.source_identifier = Some("provider-camera-42".to_owned());
        value.source_identifier_source = Some("avf.unique_id".to_owned());
        value.source_identifier = Some("provider-camera-42".to_owned());
        value.source_identifier_source = Some("avf.unique_id".to_owned());
        value.input_profile = Some(CaptureHelperInputProfile {
            width_px: 1280,
            height_px: 720,
            frame_rate_numerator: 30,
            frame_rate_seconds_denominator: 1,
            pixel_format: "MJPG".to_owned(),
        });
        let encoded = encoded_ready(&value);
        assert_eq!(
            read_ready_record(&mut Cursor::new(encoded), MAX_READY_JSON_BYTES).unwrap(),
            value
        );

        value.source_device_id = None;
        assert!(read_ready_record(
            &mut Cursor::new(encoded_ready(&value)),
            MAX_READY_JSON_BYTES
        )
        .is_err());
    }

    fn encoded_ready_json(value: serde_json::Value) -> Vec<u8> {
        let json = serde_json::to_vec(&value).unwrap();
        let mut bytes = Vec::with_capacity(4 + json.len());
        bytes.extend_from_slice(&(json.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&json);
        bytes
    }
}
