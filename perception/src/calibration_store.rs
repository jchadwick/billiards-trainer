//! Durable, revisioned ownership for the active table calibration.
//!
//! The store is deliberately a single local record. All reads and mutations
//! take a stable sibling advisory lock so compare-and-swap checks remain valid
//! across both threads and processes. Mutations write and synchronize a sibling
//! temporary file before atomically replacing the record.

use std::error::Error;
use std::ffi::OsString;
use std::fmt::{Display, Formatter};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use fs2::FileExt;
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};

use crate::calibration::{
    solve_table_calibration, CalibrationError, TableCalibrationSolverInput, TableCalibrationV1,
};
use crate::table_visibility::{validate_reference_structure, TableVisibilityReference};

const STORE_SCHEMA_VERSION: u8 = 3;
const MINIMUM_SUPPORTED_STORE_SCHEMA_VERSION: u8 = 1;
const MAX_SAFE_STORE_REVISION: u64 = 9_007_199_254_740_991;
const MAXIMUM_VISIBILITY_FRAME_DIMENSION: u32 = 16_384;
const MAXIMUM_VISIBILITY_FEATURES: usize = 16_384;
// Calibration records are small fixed-size matrices and metadata. Bound reads
// so a damaged or externally replaced file cannot cause unbounded allocation.
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(5);

/// The observable state of the calibration store.
///
/// `(None, None)` means no store has ever been written. A tombstone has a
/// revision and no calibration, preserving the compare-and-swap history after
/// a clear operation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CalibrationStoreSnapshot {
    pub revision: Option<u64>,
    pub calibration: Option<TableCalibrationV1>,
    /// Capture context that produced this calibration. Legacy schema-v1
    /// records have no binding context and are deliberately not activated by
    /// the live application, though they remain readable for migration.
    pub binding_context: Option<CalibrationCaptureContextV1>,
    /// Immutable appearance evidence captured with operator-confirmed
    /// calibration. Legacy records have no reference and cannot enable aim.
    pub visibility_reference: Option<TableVisibilityReference>,
}

impl CalibrationStoreSnapshot {
    fn empty() -> Self {
        Self {
            revision: None,
            calibration: None,
            binding_context: None,
            visibility_reference: None,
        }
    }
}

/// Stable capture identity recorded with a calibration. The process, session,
/// and source-frame identifiers are provenance; activation on a later process
/// matches the configuration, image space, geometry, profile and rectification
/// fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationCaptureContextV1 {
    pub engine_instance_id: String,
    pub capture_session_id: String,
    pub source_frame_id: String,
    pub configuration_revision: u64,
    pub camera_profile_id: String,
    pub image_space: crate::ImageSpace,
    pub source_width_px: u32,
    pub source_height_px: u32,
    pub calibration_revision: u64,
    pub rectification_revision: u64,
}

/// Errors returned while loading or updating durable calibration state.
#[derive(Debug)]
pub enum CalibrationStoreError {
    InvalidPath,
    InvalidMaximumFileBytes,
    Io(io::Error),
    DurabilityUnconfirmed {
        revision: u64,
        source: io::Error,
    },
    MalformedStore(String),
    StoreFileLimitExceeded {
        maximum: usize,
        actual: usize,
    },
    UnsupportedStoreSchemaVersion(u8),
    InvalidStoreRevision,
    InvalidCaptureContext,
    InvalidVisibilityReference,
    CalibrationRevisionMismatch {
        store_revision: u64,
        calibration_revision: u64,
    },
    InvalidCalibration(CalibrationError),
    StaleExpectedRevision {
        expected: Option<u64>,
        actual: Option<u64>,
    },
    RevisionExhausted,
    LockTimeout,
}

impl Display for CalibrationStoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPath => write!(formatter, "calibration store path must name a file"),
            Self::InvalidMaximumFileBytes => {
                write!(formatter, "calibration store maximum file size must be positive")
            }
            Self::Io(error) => write!(formatter, "calibration store I/O error: {error}"),
            Self::DurabilityUnconfirmed { revision, source } => write!(
                formatter,
                "calibration revision {revision} was replaced but its directory entry could not be synchronized; reread the store before retrying: {source}"
            ),
            Self::MalformedStore(reason) => {
                write!(formatter, "malformed calibration store: {reason}")
            }
            Self::StoreFileLimitExceeded { maximum, actual } => write!(
                formatter,
                "serialized calibration store is {actual} bytes, exceeding the configured {maximum}-byte limit"
            ),
            Self::UnsupportedStoreSchemaVersion(version) => write!(
                formatter,
                "unsupported calibration store schema version: {version}"
            ),
            Self::InvalidStoreRevision => write!(formatter, "invalid calibration store revision"),
            Self::InvalidCaptureContext => {
                write!(formatter, "calibration capture context does not match its bundle")
            }
            Self::InvalidVisibilityReference => {
                write!(formatter, "calibration table-visibility reference is invalid")
            }
            Self::CalibrationRevisionMismatch {
                store_revision,
                calibration_revision,
            } => write!(
                formatter,
                "store revision {store_revision} does not match calibration revision {calibration_revision}"
            ),
            Self::InvalidCalibration(error) => {
                write!(formatter, "invalid stored calibration: {error}")
            }
            Self::StaleExpectedRevision { expected, actual } => write!(
                formatter,
                "stale calibration revision: expected {expected:?}, current {actual:?}"
            ),
            Self::RevisionExhausted => write!(formatter, "calibration store revision exhausted"),
            Self::LockTimeout => write!(formatter, "timed out waiting for calibration store lock"),
        }
    }
}

impl Error for CalibrationStoreError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::DurabilityUnconfirmed { source, .. } => Some(source),
            Self::InvalidCalibration(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for CalibrationStoreError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Rust-owned durable storage for one provenance-bound table calibration.
///
/// The caller supplies both a lock timeout and read-size limit because waiting
/// and resource policies belong to the gateway/runtime, not this domain store.
/// The path's parent directory must already exist; the store never silently
/// changes deployment layout.
#[derive(Debug, Clone)]
pub struct CalibrationStore {
    path: PathBuf,
    lock_timeout: Duration,
    maximum_file_bytes: usize,
}

impl CalibrationStore {
    pub fn new(
        path: impl Into<PathBuf>,
        lock_timeout: Duration,
        maximum_file_bytes: usize,
    ) -> Result<Self, CalibrationStoreError> {
        let path = path.into();
        if path.file_name().is_none() {
            return Err(CalibrationStoreError::InvalidPath);
        }
        if maximum_file_bytes == 0 {
            return Err(CalibrationStoreError::InvalidMaximumFileBytes);
        }
        Ok(Self {
            path,
            lock_timeout,
            maximum_file_bytes,
        })
    }

    /// Load and validate the current record. Missing storage is an initial
    /// state, while malformed or unsupported files are returned as errors.
    pub fn load(&self) -> Result<CalibrationStoreSnapshot, CalibrationStoreError> {
        let _lock = self.acquire_lock()?;
        self.load_unlocked()
    }

    /// Solve, validate, and persist a calibration at the next store revision.
    /// The caller must present the revision it observed, including `None` for
    /// a never-created store. The solver input's revision is intentionally
    /// overwritten so callers cannot choose or reuse canonical revisions.
    pub fn save(
        &self,
        expected_current_revision: Option<u64>,
        input: TableCalibrationSolverInput,
    ) -> Result<CalibrationStoreSnapshot, CalibrationStoreError> {
        self.save_with_context(expected_current_revision, input, None)
    }

    /// Solve and persist a calibration while retaining its source capture
    /// provenance. The context is validated against the solved calibration
    /// before any bytes are written.
    pub fn save_with_context(
        &self,
        expected_current_revision: Option<u64>,
        input: TableCalibrationSolverInput,
        binding_context: Option<CalibrationCaptureContextV1>,
    ) -> Result<CalibrationStoreSnapshot, CalibrationStoreError> {
        self.save_with_context_and_visibility(
            expected_current_revision,
            input,
            binding_context,
            None,
        )
    }

    /// Solve and persist calibration provenance and an optional immutable
    /// operator-confirmed visibility reference in one atomic store update.
    pub fn save_with_context_and_visibility(
        &self,
        expected_current_revision: Option<u64>,
        mut input: TableCalibrationSolverInput,
        binding_context: Option<CalibrationCaptureContextV1>,
        visibility_reference: Option<TableVisibilityReference>,
    ) -> Result<CalibrationStoreSnapshot, CalibrationStoreError> {
        let _lock = self.acquire_lock()?;
        let current = self.load_unlocked()?;
        self.ensure_expected_revision(expected_current_revision, current.revision)?;
        let next_revision = next_revision(current.revision)?;

        input.calibration_revision = next_revision;
        let calibration =
            solve_table_calibration(input).map_err(CalibrationStoreError::InvalidCalibration)?;
        calibration
            .validate()
            .map_err(CalibrationStoreError::InvalidCalibration)?;
        if binding_context
            .as_ref()
            .is_some_and(|context| !context.matches_calibration(&calibration))
        {
            return Err(CalibrationStoreError::InvalidCaptureContext);
        }
        if visibility_reference.as_ref().is_some_and(|reference| {
            !validate_reference_structure(
                reference,
                MAXIMUM_VISIBILITY_FRAME_DIMENSION,
                MAXIMUM_VISIBILITY_FEATURES,
            )
        }) {
            return Err(CalibrationStoreError::InvalidVisibilityReference);
        }
        if visibility_reference.is_some() && binding_context.is_none() {
            return Err(CalibrationStoreError::InvalidCaptureContext);
        }

        let snapshot = CalibrationStoreSnapshot {
            revision: Some(next_revision),
            calibration: Some(calibration),
            binding_context,
            visibility_reference,
        };
        self.write_snapshot(&snapshot)?;
        Ok(snapshot)
    }

    /// Clear the calibration by writing a tombstone at the next revision.
    /// Repeated clears also advance the revision, so history never resets.
    pub fn clear(
        &self,
        expected_current_revision: Option<u64>,
    ) -> Result<CalibrationStoreSnapshot, CalibrationStoreError> {
        let _lock = self.acquire_lock()?;
        let current = self.load_unlocked()?;
        self.ensure_expected_revision(expected_current_revision, current.revision)?;
        let snapshot = CalibrationStoreSnapshot {
            revision: Some(next_revision(current.revision)?),
            calibration: None,
            binding_context: None,
            visibility_reference: None,
        };
        self.write_snapshot(&snapshot)?;
        Ok(snapshot)
    }

    fn ensure_expected_revision(
        &self,
        expected: Option<u64>,
        actual: Option<u64>,
    ) -> Result<(), CalibrationStoreError> {
        if expected == actual {
            Ok(())
        } else {
            Err(CalibrationStoreError::StaleExpectedRevision { expected, actual })
        }
    }

    fn acquire_lock(&self) -> Result<File, CalibrationStoreError> {
        let lock_path = self.lock_path();
        let lock_file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path)?;
        let started = Instant::now();
        loop {
            match lock_file.try_lock_exclusive() {
                Ok(()) => return Ok(lock_file),
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    let elapsed = started.elapsed();
                    if elapsed >= self.lock_timeout {
                        return Err(CalibrationStoreError::LockTimeout);
                    }
                    thread::sleep(LOCK_RETRY_INTERVAL.min(self.lock_timeout - elapsed));
                }
                Err(error) => return Err(CalibrationStoreError::Io(error)),
            }
        }
    }

    fn lock_path(&self) -> PathBuf {
        let mut lock_name: OsString = self
            .path
            .file_name()
            .expect("constructor validates calibration store file name")
            .to_os_string();
        lock_name.push(".lock");
        self.parent_directory().join(lock_name)
    }

    fn parent_directory(&self) -> &Path {
        self.path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
    }

    fn load_unlocked(&self) -> Result<CalibrationStoreSnapshot, CalibrationStoreError> {
        let file = match File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(CalibrationStoreSnapshot::empty());
            }
            Err(error) => return Err(CalibrationStoreError::Io(error)),
        };
        let mut bytes = Vec::new();
        file.take((self.maximum_file_bytes as u64).saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(CalibrationStoreError::Io)?;
        if bytes.len() > self.maximum_file_bytes {
            return Err(CalibrationStoreError::StoreFileLimitExceeded {
                maximum: self.maximum_file_bytes,
                actual: bytes.len(),
            });
        }

        let envelope: StoreEnvelope = serde_json::from_slice(&bytes)
            .map_err(|error| CalibrationStoreError::MalformedStore(error.to_string()))?;
        if !(MINIMUM_SUPPORTED_STORE_SCHEMA_VERSION..=STORE_SCHEMA_VERSION)
            .contains(&envelope.schema_version)
        {
            return Err(CalibrationStoreError::UnsupportedStoreSchemaVersion(
                envelope.schema_version,
            ));
        }
        if envelope.store_revision == 0 || envelope.store_revision > MAX_SAFE_STORE_REVISION {
            return Err(CalibrationStoreError::InvalidStoreRevision);
        }
        if let Some(calibration) = &envelope.calibration {
            calibration
                .validate()
                .map_err(CalibrationStoreError::InvalidCalibration)?;
            if calibration.calibration_revision != envelope.store_revision {
                return Err(CalibrationStoreError::CalibrationRevisionMismatch {
                    store_revision: envelope.store_revision,
                    calibration_revision: calibration.calibration_revision,
                });
            }
            if envelope
                .binding_context
                .as_ref()
                .is_some_and(|context| !context.matches_calibration(calibration))
            {
                return Err(CalibrationStoreError::InvalidCaptureContext);
            }
        }
        if envelope
            .visibility_reference
            .as_ref()
            .is_some_and(|reference| {
                let Some(context) = envelope.binding_context.as_ref() else {
                    return true;
                };
                envelope.calibration.is_none()
                    || !validate_reference_structure(
                        reference,
                        MAXIMUM_VISIBILITY_FRAME_DIMENSION,
                        MAXIMUM_VISIBILITY_FEATURES,
                    )
                    || reference.source_width != context.source_width_px
                    || reference.source_height != context.source_height_px
            })
        {
            return Err(CalibrationStoreError::InvalidVisibilityReference);
        }

        Ok(CalibrationStoreSnapshot {
            revision: Some(envelope.store_revision),
            calibration: envelope.calibration,
            binding_context: envelope.binding_context,
            visibility_reference: envelope.visibility_reference,
        })
    }

    fn write_snapshot(
        &self,
        snapshot: &CalibrationStoreSnapshot,
    ) -> Result<(), CalibrationStoreError> {
        let revision = snapshot
            .revision
            .ok_or(CalibrationStoreError::InvalidStoreRevision)?;
        if revision == 0 || revision > MAX_SAFE_STORE_REVISION {
            return Err(CalibrationStoreError::InvalidStoreRevision);
        }
        if let Some(calibration) = &snapshot.calibration {
            calibration
                .validate()
                .map_err(CalibrationStoreError::InvalidCalibration)?;
            if calibration.calibration_revision != revision {
                return Err(CalibrationStoreError::CalibrationRevisionMismatch {
                    store_revision: revision,
                    calibration_revision: calibration.calibration_revision,
                });
            }
        }
        if snapshot
            .visibility_reference
            .as_ref()
            .is_some_and(|reference| {
                let Some(context) = snapshot.binding_context.as_ref() else {
                    return true;
                };
                snapshot.calibration.is_none()
                    || !validate_reference_structure(
                        reference,
                        MAXIMUM_VISIBILITY_FRAME_DIMENSION,
                        MAXIMUM_VISIBILITY_FEATURES,
                    )
                    || reference.source_width != context.source_width_px
                    || reference.source_height != context.source_height_px
            })
        {
            return Err(CalibrationStoreError::InvalidVisibilityReference);
        }
        let envelope = StoreEnvelope {
            schema_version: STORE_SCHEMA_VERSION,
            store_revision: revision,
            calibration: snapshot.calibration.clone(),
            binding_context: snapshot.binding_context.clone(),
            visibility_reference: snapshot.visibility_reference.clone(),
        };
        let bytes = serde_json::to_vec(&envelope)
            .map_err(|error| CalibrationStoreError::MalformedStore(error.to_string()))?;
        if bytes.len() > self.maximum_file_bytes {
            return Err(CalibrationStoreError::StoreFileLimitExceeded {
                maximum: self.maximum_file_bytes,
                actual: bytes.len(),
            });
        }
        match write_atomically(&self.path, &bytes) {
            Ok(()) => Ok(()),
            Err(AtomicWriteError::BeforeReplacement(error)) => {
                Err(CalibrationStoreError::Io(error))
            }
            Err(AtomicWriteError::DurabilityUnconfirmed(error)) => {
                Err(CalibrationStoreError::DurabilityUnconfirmed {
                    revision,
                    source: error,
                })
            }
        }
    }
}

fn next_revision(current: Option<u64>) -> Result<u64, CalibrationStoreError> {
    let next = match current {
        None => 1,
        Some(revision) => revision
            .checked_add(1)
            .ok_or(CalibrationStoreError::RevisionExhausted)?,
    };
    if next > MAX_SAFE_STORE_REVISION {
        return Err(CalibrationStoreError::RevisionExhausted);
    }
    Ok(next)
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoreEnvelope {
    schema_version: u8,
    store_revision: u64,
    #[serde(deserialize_with = "required_option")]
    calibration: Option<TableCalibrationV1>,
    #[serde(default)]
    binding_context: Option<CalibrationCaptureContextV1>,
    #[serde(default)]
    visibility_reference: Option<TableVisibilityReference>,
}

impl CalibrationCaptureContextV1 {
    fn matches_calibration(&self, calibration: &TableCalibrationV1) -> bool {
        self.configuration_revision > 0
            && !self.engine_instance_id.is_empty()
            && !self.capture_session_id.is_empty()
            && !self.source_frame_id.is_empty()
            && self.source_width_px > 0
            && self.source_height_px > 0
            && self.calibration_revision <= MAX_SAFE_STORE_REVISION
            && self.rectification_revision == calibration.rectification_revision
            && self.camera_profile_id == calibration.camera_profile_id
            && self.image_space == calibration.image_space
            && self.source_width_px == calibration.source_width_px
            && self.source_height_px == calibration.source_height_px
    }
}

fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

#[derive(Debug)]
enum AtomicWriteError {
    BeforeReplacement(io::Error),
    DurabilityUnconfirmed(io::Error),
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), AtomicWriteError> {
    write_atomically_with(
        path,
        bytes,
        |temporary, destination| {
            temporary
                .persist(destination)
                .map(|_| ())
                .map_err(|error| AtomicWriteError::BeforeReplacement(error.error))
        },
        sync_parent_directory,
    )
}

fn write_atomically_with(
    path: &Path,
    bytes: &[u8],
    replace: impl FnOnce(tempfile::NamedTempFile, &Path) -> Result<(), AtomicWriteError>,
    sync_directory: impl FnOnce(&Path) -> io::Result<()>,
) -> Result<(), AtomicWriteError> {
    let mut temporary = tempfile::NamedTempFile::new_in(parent_directory(path))
        .map_err(AtomicWriteError::BeforeReplacement)?;
    temporary
        .write_all(bytes)
        .map_err(AtomicWriteError::BeforeReplacement)?;
    temporary
        .flush()
        .map_err(AtomicWriteError::BeforeReplacement)?;
    temporary
        .as_file()
        .sync_all()
        .map_err(AtomicWriteError::BeforeReplacement)?;
    replace(temporary, path)?;
    sync_directory(parent_directory(path)).map_err(AtomicWriteError::DurabilityUnconfirmed)
}

fn parent_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> io::Result<()> {
    let directory = File::open(parent)?;
    match directory.sync_all() {
        Ok(()) => Ok(()),
        Err(error) if directory_sync_unsupported(&error) => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path) -> io::Result<()> {
    // Directory handles cannot be synchronized portably through std on every
    // supported non-Unix target. File contents are still flushed before rename.
    Ok(())
}

#[cfg(unix)]
fn directory_sync_unsupported(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::Unsupported {
        return true;
    }
    match error.raw_os_error() {
        // EINVAL is returned by filesystems that do not implement directory
        // synchronization. ENOTSUP differs between Linux and macOS.
        Some(22) => true,
        #[cfg(target_os = "linux")]
        Some(95) => true,
        #[cfg(target_os = "macos")]
        Some(45) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{write_atomically_with, AtomicWriteError};

    #[test]
    fn replacement_failure_keeps_the_previous_file_intact() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("calibration.json");
        fs::write(&path, b"previous committed value").unwrap();

        let error = write_atomically_with(
            &path,
            b"new value",
            |temporary, _| {
                // Simulate a failed atomic replace after the new content is fully
                // written and synchronized, but before the commit point.
                drop(temporary);
                Err(AtomicWriteError::BeforeReplacement(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "injected replacement failure",
                )))
            },
            |_| Ok(()),
        )
        .unwrap_err();

        assert!(matches!(error, AtomicWriteError::BeforeReplacement(_)));
        assert_eq!(fs::read(path).unwrap(), b"previous committed value");
    }

    #[test]
    fn directory_sync_failure_reports_committed_value_with_uncertain_durability() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("calibration.json");
        fs::write(&path, b"previous committed value").unwrap();

        let error = write_atomically_with(
            &path,
            b"new value",
            |temporary, destination| {
                temporary
                    .persist(destination)
                    .map(|_| ())
                    .map_err(|error| AtomicWriteError::BeforeReplacement(error.error))
            },
            |_| {
                Err(std::io::Error::other(
                    "injected directory synchronization failure",
                ))
            },
        )
        .unwrap_err();

        assert!(matches!(error, AtomicWriteError::DurabilityUnconfirmed(_)));
        assert_eq!(fs::read(path).unwrap(), b"new value");
    }
}
