//! Rust-owned game-session metadata, kept separate from measured scene state.
//!
//! The session store owns only game mode and lightweight session bookkeeping.
//! It never stores ball positions, cue geometry, calibration, tracking state, or
//! predictions. Session reset therefore cannot erase or synthesize a location.
//! A small locked JSON record lets the Rust gateway share session metadata
//! across its request handlers and runtime components.

use std::error::Error;
use std::ffi::OsString;
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use fs2::FileExt;
#[cfg(unix)]
use rustix::fs::{mkdirat, openat, renameat, unlinkat, AtFlags, Mode, OFlags};
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::os::fd::OwnedFd;

const STORE_SCHEMA_VERSION: u8 = 1;
const AUTHORITY_PROTOCOL_VERSION: u8 = 1;
const MAX_SAFE_REVISION: u64 = 9_007_199_254_740_991;
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(5);
pub const DEFAULT_MAX_INPUT_BYTES: usize = 4_096;
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 8_192;
pub const DEFAULT_MAX_STORE_BYTES: usize = 4_096;
pub const DEFAULT_LOCK_TIMEOUT_MS: u64 = 2_000;
pub const MINIMUM_AUTHORITY_RESPONSE_BYTES: usize = 256;

/// Game modes supported by the existing public API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameType {
    #[serde(rename = "practice")]
    Practice,
    #[serde(rename = "8ball")]
    EightBall,
    #[serde(rename = "9ball")]
    NineBall,
    #[serde(rename = "straight")]
    Straight,
}

impl GameType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Practice => "practice",
            Self::EightBall => "8ball",
            Self::NineBall => "9ball",
            Self::Straight => "straight",
        }
    }
}

/// Current game/session metadata. `revision` advances only when a session
/// command commits; capture frames do not touch it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameSessionSnapshotV1 {
    pub schema_version: u8,
    pub revision: u64,
    pub game_type: GameType,
    pub active_player: u8,
    /// Scores are indexed by player number minus one: `[player 1, player 2]`.
    pub scores: [u32; 2],
    pub is_break: bool,
}

impl GameSessionSnapshotV1 {
    pub fn initial() -> Self {
        Self {
            schema_version: STORE_SCHEMA_VERSION,
            revision: 0,
            game_type: GameType::Practice,
            active_player: 1,
            scores: [0, 0],
            is_break: true,
        }
    }

    fn validate(&self) -> Result<(), GameSessionStoreError> {
        if self.schema_version != STORE_SCHEMA_VERSION {
            return Err(GameSessionStoreError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if self.revision > MAX_SAFE_REVISION || !(1..=2).contains(&self.active_player) {
            return Err(GameSessionStoreError::InvalidSnapshot);
        }
        Ok(())
    }
}

/// Errors for the Rust-owned game-session store.
#[derive(Debug)]
pub enum GameSessionStoreError {
    InvalidPath,
    InvalidMaximumFileBytes,
    InvalidLockTimeout,
    Io(io::Error),
    DurabilityUnconfirmed { revision: u64, source: io::Error },
    MalformedStore(String),
    StoreFileLimitExceeded { maximum: usize, actual: usize },
    UnsupportedSchemaVersion(u8),
    InvalidSnapshot,
    StaleExpectedRevision { expected: u64, actual: u64 },
    RevisionExhausted,
    LockTimeout,
}

impl From<io::Error> for GameSessionStoreError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl Display for GameSessionStoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPath => formatter.write_str("game-session store path must be an absolute file path"),
            Self::InvalidMaximumFileBytes => formatter.write_str("game-session store file limit must be positive"),
            Self::InvalidLockTimeout => formatter.write_str("game-session lock timeout must be between 1 and 60000 ms"),
            Self::Io(error) => write!(formatter, "game-session store I/O error: {error}"),
            Self::DurabilityUnconfirmed { revision, source } => write!(
                formatter,
                "game-session revision {revision} was replaced but directory durability is uncertain; reread before retrying: {source}"
            ),
            Self::MalformedStore(reason) => write!(formatter, "malformed game-session store: {reason}"),
            Self::StoreFileLimitExceeded { maximum, actual } => write!(
                formatter,
                "game-session store is at least {actual} bytes, exceeding the configured {maximum}-byte limit"
            ),
            Self::UnsupportedSchemaVersion(version) => write!(formatter, "unsupported game-session schema version: {version}"),
            Self::InvalidSnapshot => formatter.write_str("game-session snapshot is invalid"),
            Self::StaleExpectedRevision { expected, actual } => write!(
                formatter,
                "stale game-session revision: expected {expected}, current {actual}"
            ),
            Self::RevisionExhausted => formatter.write_str("game-session revision exhausted"),
            Self::LockTimeout => formatter.write_str("timed out waiting for game-session store lock"),
        }
    }
}

impl Error for GameSessionStoreError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::DurabilityUnconfirmed { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Cross-process-safe, revisioned persistence for session metadata.
#[derive(Debug, Clone)]
pub struct GameSessionStore {
    path: PathBuf,
    lock_timeout: Duration,
    maximum_file_bytes: usize,
    confined_path: Option<Arc<ConfinedPath>>,
}

struct StoreLock {
    _file: File,
    #[cfg(unix)]
    secure_parent: Option<OwnedFd>,
}

#[derive(Debug)]
struct ConfinedPath {
    /// Anchors each operation to the already-open deployment root directory.
    root: File,
    parent_components: Vec<OsString>,
    file_name: OsString,
}

impl GameSessionStore {
    pub fn new(
        path: impl Into<PathBuf>,
        lock_timeout: Duration,
        maximum_file_bytes: usize,
    ) -> Result<Self, GameSessionStoreError> {
        let path = path.into();
        if !path.is_absolute() || path.file_name().is_none() {
            return Err(GameSessionStoreError::InvalidPath);
        }
        if lock_timeout.is_zero() || lock_timeout > Duration::from_secs(60) {
            return Err(GameSessionStoreError::InvalidLockTimeout);
        }
        if !(1..=65_536).contains(&maximum_file_bytes) {
            return Err(GameSessionStoreError::InvalidMaximumFileBytes);
        }
        Ok(Self {
            path,
            lock_timeout,
            maximum_file_bytes,
            confined_path: None,
        })
    }

    /// Construct a store whose file operations remain beneath `root` even if
    /// a checked directory is later replaced by a symlink. On Unix, every path
    /// component is opened relative to a held directory descriptor with
    /// `O_NOFOLLOW`; the root descriptor also prevents path replacement above
    /// the deployment root from redirecting store access.
    pub fn new_confined(
        path: impl Into<PathBuf>,
        root: impl AsRef<Path>,
        lock_timeout: Duration,
        maximum_file_bytes: usize,
    ) -> Result<Self, GameSessionStoreError> {
        let root = fs::canonicalize(root)?;
        if !root.is_dir() {
            return Err(GameSessionStoreError::InvalidPath);
        }
        let mut store = Self::new(path, lock_timeout, maximum_file_bytes)?;
        let relative = store
            .path
            .strip_prefix(&root)
            .map_err(|_| GameSessionStoreError::InvalidPath)?;
        let components = relative.components().collect::<Vec<_>>();
        if components.is_empty()
            || components
                .iter()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            return Err(GameSessionStoreError::InvalidPath);
        }
        let file_name = match components.last() {
            Some(std::path::Component::Normal(name)) => name.to_os_string(),
            _ => return Err(GameSessionStoreError::InvalidPath),
        };
        let parent_components = components[..components.len() - 1]
            .iter()
            .filter_map(|component| match component {
                std::path::Component::Normal(name) => Some(name.to_os_string()),
                _ => None,
            })
            .collect();
        #[cfg(unix)]
        let root = open_absolute_directory_no_follow(&root)?;
        #[cfg(not(unix))]
        let root = File::open(root)?;
        store.confined_path = Some(Arc::new(ConfinedPath {
            root,
            parent_components,
            file_name,
        }));
        Ok(store)
    }

    /// Read the current session. A missing file is the documented initial
    /// practice session at revision zero; the session record is not created.
    /// A sibling lock file is maintained to serialize reads with mutations.
    pub fn load(&self) -> Result<GameSessionSnapshotV1, GameSessionStoreError> {
        let lock = self.acquire_lock()?;
        self.load_unlocked(&lock)
    }

    /// Start a fresh game while preserving no historical or location data.
    pub fn reset(
        &self,
        expected_revision: u64,
        game_type: GameType,
    ) -> Result<GameSessionSnapshotV1, GameSessionStoreError> {
        let lock = self.acquire_lock()?;
        let current = self.load_unlocked(&lock)?;
        if expected_revision != current.revision {
            return Err(GameSessionStoreError::StaleExpectedRevision {
                expected: expected_revision,
                actual: current.revision,
            });
        }
        let revision = current
            .revision
            .checked_add(1)
            .filter(|value| *value <= MAX_SAFE_REVISION)
            .ok_or(GameSessionStoreError::RevisionExhausted)?;
        let next = GameSessionSnapshotV1 {
            schema_version: STORE_SCHEMA_VERSION,
            revision,
            game_type,
            active_player: 1,
            scores: [0, 0],
            is_break: true,
        };
        self.write_unlocked(&next, &lock)?;
        Ok(next)
    }

    fn acquire_lock(&self) -> Result<StoreLock, GameSessionStoreError> {
        #[cfg(unix)]
        if let Some(confined) = &self.confined_path {
            return self.acquire_confined_lock(confined);
        }
        fs::create_dir_all(self.parent_directory())?;
        let mut lock_name: OsString = self
            .path
            .file_name()
            .expect("constructor validates the game-session file name")
            .to_os_string();
        lock_name.push(".lock");
        let lock_path = self.parent_directory().join(lock_name);
        let lock_file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path)?;
        let started = Instant::now();
        loop {
            match lock_file.try_lock_exclusive() {
                Ok(()) => {
                    return Ok(StoreLock {
                        _file: lock_file,
                        #[cfg(unix)]
                        secure_parent: None,
                    })
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    let elapsed = started.elapsed();
                    if elapsed >= self.lock_timeout {
                        return Err(GameSessionStoreError::LockTimeout);
                    }
                    thread::sleep(LOCK_RETRY_INTERVAL.min(self.lock_timeout - elapsed));
                }
                Err(error) => return Err(GameSessionStoreError::Io(error)),
            }
        }
    }

    #[cfg(unix)]
    fn acquire_confined_lock(
        &self,
        confined: &ConfinedPath,
    ) -> Result<StoreLock, GameSessionStoreError> {
        let parent = open_confined_parent(confined)?;
        let mut lock_name = confined.file_name.clone();
        lock_name.push(".lock");
        let lock_file = openat(
            &parent,
            &lock_name,
            OFlags::RDWR | OFlags::CREATE | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::from_raw_mode(0o600),
        )
        .map(File::from)
        .map_err(std::io::Error::from)?;
        if !lock_file.metadata()?.is_file() {
            return Err(GameSessionStoreError::InvalidPath);
        }
        let started = Instant::now();
        loop {
            match lock_file.try_lock_exclusive() {
                Ok(()) => {
                    return Ok(StoreLock {
                        _file: lock_file,
                        secure_parent: Some(parent),
                    })
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    let elapsed = started.elapsed();
                    if elapsed >= self.lock_timeout {
                        return Err(GameSessionStoreError::LockTimeout);
                    }
                    thread::sleep(LOCK_RETRY_INTERVAL.min(self.lock_timeout - elapsed));
                }
                Err(error) => return Err(GameSessionStoreError::Io(error)),
            }
        }
    }

    fn load_unlocked(
        &self,
        lock: &StoreLock,
    ) -> Result<GameSessionSnapshotV1, GameSessionStoreError> {
        #[cfg(unix)]
        if let Some(confined) = &self.confined_path {
            let parent = lock
                .secure_parent
                .as_ref()
                .expect("confined stores retain their locked parent descriptor");
            let file = match openat(
                parent,
                &confined.file_name,
                OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
                Mode::empty(),
            ) {
                Ok(file) => {
                    let file = File::from(file);
                    if !file.metadata()?.is_file() {
                        return Err(GameSessionStoreError::InvalidPath);
                    }
                    file
                }
                Err(error) if error == rustix::io::Errno::NOENT => {
                    return Ok(GameSessionSnapshotV1::initial())
                }
                Err(error) => return Err(GameSessionStoreError::Io(error.into())),
            };
            return self.read_snapshot_file(file);
        }
        let file = match File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(GameSessionSnapshotV1::initial());
            }
            Err(error) => return Err(GameSessionStoreError::Io(error)),
        };
        self.read_snapshot_file(file)
    }

    fn read_snapshot_file(
        &self,
        file: File,
    ) -> Result<GameSessionSnapshotV1, GameSessionStoreError> {
        let mut bytes = Vec::new();
        file.take((self.maximum_file_bytes as u64).saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() > self.maximum_file_bytes {
            return Err(GameSessionStoreError::StoreFileLimitExceeded {
                maximum: self.maximum_file_bytes,
                actual: bytes.len(),
            });
        }
        let snapshot: GameSessionSnapshotV1 = serde_json::from_slice(&bytes)
            .map_err(|error| GameSessionStoreError::MalformedStore(error.to_string()))?;
        snapshot.validate()?;
        Ok(snapshot)
    }

    fn write_unlocked(
        &self,
        snapshot: &GameSessionSnapshotV1,
        lock: &StoreLock,
    ) -> Result<(), GameSessionStoreError> {
        snapshot.validate()?;
        let bytes = serde_json::to_vec(snapshot)
            .map_err(|error| GameSessionStoreError::MalformedStore(error.to_string()))?;
        if bytes.len() > self.maximum_file_bytes {
            return Err(GameSessionStoreError::StoreFileLimitExceeded {
                maximum: self.maximum_file_bytes,
                actual: bytes.len(),
            });
        }
        #[cfg(unix)]
        if let Some(confined) = &self.confined_path {
            let parent = lock
                .secure_parent
                .as_ref()
                .expect("confined stores retain their locked parent descriptor");
            return self.write_confined(confined, parent, snapshot, &bytes);
        }
        let parent = self.parent_directory();
        fs::create_dir_all(parent)?;
        let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
        temporary.write_all(&bytes)?;
        temporary.flush()?;
        temporary.as_file().sync_all()?;
        temporary
            .persist(&self.path)
            .map_err(|error| GameSessionStoreError::Io(error.error))?;
        sync_parent_directory(parent).map_err(|source| {
            GameSessionStoreError::DurabilityUnconfirmed {
                revision: snapshot.revision,
                source,
            }
        })
    }

    #[cfg(unix)]
    fn write_confined(
        &self,
        confined: &ConfinedPath,
        parent: &OwnedFd,
        snapshot: &GameSessionSnapshotV1,
        bytes: &[u8],
    ) -> Result<(), GameSessionStoreError> {
        let temporary_name = format!(
            ".{}.{}.tmp",
            confined.file_name.to_string_lossy(),
            uuid::Uuid::new_v4()
        );
        let mut temporary = openat(
            parent,
            temporary_name.as_str(),
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::from_raw_mode(0o600),
        )
        .map(File::from)
        .map_err(std::io::Error::from)?;
        let commit = (|| {
            temporary.write_all(bytes)?;
            temporary.flush()?;
            temporary.sync_all()?;
            renameat(parent, temporary_name.as_str(), parent, &confined.file_name)
                .map_err(|error| GameSessionStoreError::Io(error.into()))?;
            Ok::<_, GameSessionStoreError>(())
        })();
        if commit.is_err() {
            let _ = unlinkat(parent, temporary_name.as_str(), AtFlags::empty());
            return commit;
        }
        match rustix::fs::fsync(parent) {
            Ok(()) => Ok(()),
            Err(error) => {
                let source = std::io::Error::from(error);
                if directory_sync_is_unsupported(&source) {
                    Ok(())
                } else {
                    Err(GameSessionStoreError::DurabilityUnconfirmed {
                        revision: snapshot.revision,
                        source,
                    })
                }
            }
        }
    }

    fn parent_directory(&self) -> &Path {
        self.path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("/"))
    }
}

#[cfg(unix)]
fn open_confined_parent(confined: &ConfinedPath) -> Result<OwnedFd, GameSessionStoreError> {
    let mut parent = openat(
        &confined.root,
        ".",
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty(),
    )
    .map_err(|error| GameSessionStoreError::Io(error.into()))?;
    for component in &confined.parent_components {
        let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW;
        match openat(&parent, component, flags, Mode::empty()) {
            Ok(next) => parent = next,
            Err(error) if error == rustix::io::Errno::NOENT => {
                match mkdirat(&parent, component, Mode::from_raw_mode(0o700)) {
                    Ok(()) => {}
                    Err(create_error) if create_error == rustix::io::Errno::EXIST => {}
                    Err(create_error) => {
                        return Err(GameSessionStoreError::Io(create_error.into()))
                    }
                }
                parent = openat(&parent, component, flags, Mode::empty())
                    .map_err(|open_error| GameSessionStoreError::Io(open_error.into()))?;
            }
            Err(error) => return Err(GameSessionStoreError::Io(error.into())),
        }
    }
    Ok(parent)
}

#[cfg(unix)]
fn open_absolute_directory_no_follow(path: &Path) -> io::Result<File> {
    if !path.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "confined store root must be absolute",
        ));
    }
    let mut current = File::open("/")?;
    let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW;
    for component in path.components() {
        match component {
            std::path::Component::RootDir | std::path::Component::CurDir => {}
            std::path::Component::Normal(name) => {
                current = File::from(
                    openat(&current, name, flags, Mode::empty()).map_err(std::io::Error::from)?,
                );
            }
            std::path::Component::ParentDir | std::path::Component::Prefix(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "confined store root must be normalized",
                ));
            }
        }
    }
    Ok(current)
}

#[cfg(unix)]
fn directory_sync_is_unsupported(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::Unsupported
        || matches!(error.raw_os_error(), Some(22))
        || cfg!(target_os = "linux") && matches!(error.raw_os_error(), Some(95))
        || cfg!(target_os = "macos") && matches!(error.raw_os_error(), Some(45))
}

/// Bounded JSON request handled by the one-shot Rust authority process.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorityRequest {
    protocol_version: u8,
    operation: String,
    #[serde(default)]
    expected_revision: Option<u64>,
    #[serde(default)]
    game_type: Option<GameType>,
}

/// Handle one strict JSON request and return its protocol response.
pub fn handle_authority_request(
    store: &GameSessionStore,
    input: &[u8],
    maximum_input_bytes: usize,
    maximum_output_bytes: usize,
) -> Result<Vec<u8>, String> {
    if !(1..=65_536).contains(&maximum_input_bytes)
        || !(MINIMUM_AUTHORITY_RESPONSE_BYTES..=65_536).contains(&maximum_output_bytes)
    {
        return Err("game-session authority I/O limits are outside their safe bounds".to_owned());
    }
    let request = if input.len() > maximum_input_bytes {
        Err((
            "request_too_large",
            "Request exceeds the configured input limit".to_owned(),
        ))
    } else {
        serde_json::from_slice::<AuthorityRequest>(input)
            .map_err(|error| ("invalid_request", error.to_string()))
    };
    let response = match request {
        Err((code, message)) => error_response("unknown", code, message, None),
        Ok(request) if request.protocol_version != AUTHORITY_PROTOCOL_VERSION => error_response(
            &request.operation,
            "unsupported_protocol_version",
            "protocol_version must be 1".to_owned(),
            None,
        ),
        Ok(request) => match request.operation.as_str() {
            "snapshot" if request.expected_revision.is_none() && request.game_type.is_none() => {
                match store.load() {
                    Ok(session) => serde_json::json!({
                        "protocol_version": AUTHORITY_PROTOCOL_VERSION,
                        "ok": true,
                        "operation": "snapshot",
                        "session": session,
                    }),
                    Err(error) => store_error_response("snapshot", error),
                }
            }
            "reset" => match (request.expected_revision, request.game_type) {
                (Some(expected_revision), Some(game_type)) => {
                    match store.reset(expected_revision, game_type) {
                        Ok(session) => serde_json::json!({
                            "protocol_version": AUTHORITY_PROTOCOL_VERSION,
                            "ok": true,
                            "operation": "reset",
                            "session": session,
                        }),
                        Err(error) => store_error_response("reset", error),
                    }
                }
                _ => error_response(
                    "reset",
                    "invalid_request",
                    "reset requires expected_revision and game_type".to_owned(),
                    None,
                ),
            },
            _ => error_response(
                &request.operation,
                "unsupported_operation",
                "operation must be snapshot or reset".to_owned(),
                None,
            ),
        },
    };
    let mut bytes = serde_json::to_vec(&response).unwrap_or_else(|_| {
        br#"{"protocol_version":1,"ok":false,"operation":"unknown","error":{"code":"internal_error","message":"Response encoding failed"}}"#.to_vec()
    });
    if bytes.len() > maximum_output_bytes {
        bytes = br#"{"protocol_version":1,"ok":false,"operation":"unknown","error":{"code":"response_too_large","message":"Response exceeds the configured output limit"}}"#.to_vec();
    }
    if bytes.len() > maximum_output_bytes {
        return Err("bounded game-session response did not fit output limit".to_owned());
    }
    Ok(bytes)
}

fn store_error_response(operation: &str, error: GameSessionStoreError) -> serde_json::Value {
    match error {
        GameSessionStoreError::StaleExpectedRevision { expected, actual } => error_response(
            operation,
            "stale_revision",
            format!("expected session revision {expected}, current revision is {actual}"),
            Some(actual),
        ),
        GameSessionStoreError::RevisionExhausted => {
            error_response(operation, "revision_exhausted", error.to_string(), None)
        }
        other => error_response(operation, "store_error", other.to_string(), None),
    }
}

fn error_response(
    operation: &str,
    code: &str,
    message: String,
    current_revision: Option<u64>,
) -> serde_json::Value {
    let mut error = serde_json::json!({"code": code, "message": message});
    if let Some(revision) = current_revision {
        error["current_revision"] = serde_json::json!(revision);
    }
    serde_json::json!({
        "protocol_version": AUTHORITY_PROTOCOL_VERSION,
        "ok": false,
        "operation": operation,
        "error": error,
    })
}

fn sync_parent_directory(parent: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        let directory = File::open(parent)?;
        match directory.sync_all() {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::Unsupported => Ok(()),
            Err(error) if matches!(error.raw_os_error(), Some(22)) => Ok(()),
            #[cfg(target_os = "linux")]
            Err(error) if matches!(error.raw_os_error(), Some(95)) => Ok(()),
            #[cfg(target_os = "macos")]
            Err(error) if matches!(error.raw_os_error(), Some(45)) => Ok(()),
            Err(error) => Err(error),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = parent;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{
        handle_authority_request, GameSessionSnapshotV1, GameSessionStore, GameSessionStoreError,
        GameType,
    };
    use std::time::Duration;

    fn store(path: &std::path::Path) -> GameSessionStore {
        GameSessionStore::new(path, Duration::from_millis(100), 4096).unwrap()
    }

    #[cfg(unix)]
    #[test]
    fn confined_store_rejects_parent_symlink_swapped_after_construction() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let data_directory = root.path().join("data");
        fs::create_dir(&data_directory).unwrap();
        let canonical_root = fs::canonicalize(root.path()).unwrap();
        let store = GameSessionStore::new_confined(
            canonical_root.join("data/session.json"),
            &canonical_root,
            Duration::from_millis(100),
            4096,
        )
        .unwrap();

        fs::remove_dir(&data_directory).unwrap();
        symlink(outside.path(), &data_directory).unwrap();

        assert!(store.load().is_err());
        assert!(store.reset(0, GameType::EightBall).is_err());
        assert!(!outside.path().join("session.json").exists());
        assert!(!outside.path().join("session.json.lock").exists());
    }

    #[cfg(unix)]
    #[test]
    fn confined_root_open_rejects_symlink_components() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let canonical_root = fs::canonicalize(root.path()).unwrap();
        let root_link = canonical_root.join("project-link");
        symlink(outside.path(), &root_link).unwrap();

        assert!(super::open_absolute_directory_no_follow(&root_link).is_err());
    }

    #[test]
    fn missing_store_is_a_read_only_practice_session() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.json");
        let current = store(&path).load().unwrap();
        assert_eq!(current, GameSessionSnapshotV1::initial());
        assert!(!path.exists());
    }

    #[test]
    fn reset_changes_only_session_metadata_and_advances_revision() {
        let directory = tempfile::tempdir().unwrap();
        let store = store(&directory.path().join("session.json"));
        let next = store.reset(0, GameType::NineBall).unwrap();
        assert_eq!(next.revision, 1);
        assert_eq!(next.game_type, GameType::NineBall);
        assert_eq!(next.active_player, 1);
        assert_eq!(next.scores, [0, 0]);
        assert!(next.is_break);
        assert_eq!(store.load().unwrap(), next);
    }

    #[test]
    fn reset_rejects_stale_revision_without_overwriting_current_session() {
        let directory = tempfile::tempdir().unwrap();
        let store = store(&directory.path().join("session.json"));
        store.reset(0, GameType::EightBall).unwrap();
        let error = store.reset(0, GameType::Practice).unwrap_err();
        assert!(matches!(
            error,
            GameSessionStoreError::StaleExpectedRevision {
                expected: 0,
                actual: 1
            }
        ));
        assert_eq!(store.load().unwrap().game_type, GameType::EightBall);
    }

    #[test]
    fn simultaneous_resets_from_one_revision_allow_only_one_commit() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(store(&directory.path().join("session.json")));
        let barrier = Arc::new(Barrier::new(3));
        let workers: Vec<_> = [GameType::EightBall, GameType::NineBall]
            .into_iter()
            .map(|game_type| {
                let store = Arc::clone(&store);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    store.reset(0, game_type)
                })
            })
            .collect();
        barrier.wait();
        let results: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(
                    result,
                    Err(GameSessionStoreError::StaleExpectedRevision { .. })
                ))
                .count(),
            1
        );
        assert_eq!(store.load().unwrap().revision, 1);
    }

    #[test]
    fn store_rejects_malformed_and_oversized_records() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.json");
        fs::write(&path, b"not json").unwrap();
        assert!(matches!(
            store(&path).load(),
            Err(GameSessionStoreError::MalformedStore(_))
        ));
        fs::write(&path, vec![b' '; 4097]).unwrap();
        assert!(matches!(
            store(&path).load(),
            Err(GameSessionStoreError::StoreFileLimitExceeded { .. })
        ));
    }

    #[test]
    fn authority_requires_revision_and_rejects_unknown_fields() {
        let directory = tempfile::tempdir().unwrap();
        let store = store(&directory.path().join("session.json"));
        let missing_revision = handle_authority_request(
            &store,
            br#"{"protocol_version":1,"operation":"reset","game_type":"8ball"}"#,
            1024,
            4096,
        )
        .unwrap();
        let response: serde_json::Value = serde_json::from_slice(&missing_revision).unwrap();
        assert_eq!(response["ok"], false);
        assert_eq!(response["error"]["code"], "invalid_request");

        let extra_field = handle_authority_request(
            &store,
            br#"{"protocol_version":1,"operation":"snapshot","surprise":true}"#,
            1024,
            4096,
        )
        .unwrap();
        let response: serde_json::Value = serde_json::from_slice(&extra_field).unwrap();
        assert_eq!(response["ok"], false);
    }

    #[test]
    fn authority_snapshot_and_reset_emit_versioned_session_contract() {
        let directory = tempfile::tempdir().unwrap();
        let store = store(&directory.path().join("session.json"));
        let snapshot = handle_authority_request(
            &store,
            br#"{"protocol_version":1,"operation":"snapshot"}"#,
            1024,
            4096,
        )
        .unwrap();
        let snapshot: serde_json::Value = serde_json::from_slice(&snapshot).unwrap();
        assert_eq!(snapshot["session"]["game_type"], "practice");
        let reset = handle_authority_request(
            &store,
            br#"{"protocol_version":1,"operation":"reset","expected_revision":0,"game_type":"8ball"}"#,
            1024,
            4096,
        )
        .unwrap();
        let reset: serde_json::Value = serde_json::from_slice(&reset).unwrap();
        assert_eq!(reset["session"]["revision"], 1);
        assert_eq!(reset["session"]["game_type"], "8ball");
    }

    #[test]
    fn authority_rejects_an_output_limit_that_cannot_hold_a_complete_envelope() {
        let directory = tempfile::tempdir().unwrap();
        let store = store(&directory.path().join("session.json"));
        assert!(handle_authority_request(
            &store,
            br#"{"protocol_version":1,"operation":"snapshot"}"#,
            1024,
            128,
        )
        .is_err());
    }

    #[test]
    fn authority_rejects_limits_above_the_protocol_cap() {
        let directory = tempfile::tempdir().unwrap();
        let store = store(&directory.path().join("session.json"));
        assert!(handle_authority_request(
            &store,
            br#"{"protocol_version":1,"operation":"snapshot"}"#,
            65_537,
            4096,
        )
        .is_err());
        assert!(handle_authority_request(
            &store,
            br#"{"protocol_version":1,"operation":"snapshot"}"#,
            1024,
            65_537,
        )
        .is_err());
    }
}
