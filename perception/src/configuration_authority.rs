//! Rust-owned authority for the application's unified configuration document.
//!
//! The versioned defaults and JSON Schema in `contracts/configuration/v1` are
//! embedded so the one-shot command and the future HTTP adapter share one
//! source of truth. Mutations re-read the file under a cross-process lock,
//! compare the caller's revision, validate the complete merged document, and
//! atomically persist before returning the committed snapshot.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::IpAddr;
use std::path::{Component, Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::capture_settings_wire::build_capture_settings_v3;

pub const CONFIGURATION_CONTRACT_VERSION: u8 = 1;
pub const DEFAULT_MAX_INPUT_BYTES: usize = 1_048_576;
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 2_097_152;
pub const DEFAULT_MAX_CONFIG_BYTES: usize = 1_048_576;
pub const DEFAULT_LOCK_TIMEOUT_MS: u64 = 2_000;
const MAX_INPUT_BYTES: usize = 16_777_216;
const MAX_OUTPUT_BYTES: usize = 33_554_432;
const MAX_CONFIG_BYTES: usize = 16_777_216;
const MAX_LOCK_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
const DEFAULTS_JSON: &str = include_str!("../contracts/configuration/v1/defaults.json");
const SCHEMA_JSON: &str = include_str!("../contracts/configuration/v1/schema.json");

#[derive(Debug, Clone, Copy)]
pub struct AuthorityLimits {
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_config_bytes: usize,
    pub lock_timeout: Duration,
}

impl Default for AuthorityLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            max_config_bytes: DEFAULT_MAX_CONFIG_BYTES,
            lock_timeout: Duration::from_millis(DEFAULT_LOCK_TIMEOUT_MS),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConfigurationAuthority {
    config_path: PathBuf,
    limits: AuthorityLimits,
    defaults: Value,
    schema: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    protocol_version: u8,
    operation: String,
    #[serde(default)]
    expected_revision: Option<u64>,
    #[serde(default)]
    data: Option<Value>,
    #[serde(default)]
    sections: Option<Vec<String>>,
    #[serde(default)]
    section: Option<String>,
    #[serde(default)]
    merge: Option<bool>,
    #[serde(default)]
    force_update: bool,
    #[serde(default)]
    include_defaults: bool,
    #[serde(default = "default_true")]
    include_metadata: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize)]
struct ValidationError {
    field: String,
    message: String,
    current_value: Value,
    expected_type: String,
}

#[derive(Debug)]
struct AuthorityError {
    code: &'static str,
    message: String,
    current_revision: Option<u64>,
    validation_errors: Vec<ValidationError>,
    warnings: Vec<String>,
}

impl AuthorityError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            current_revision: None,
            validation_errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn validation(errors: Vec<ValidationError>, warnings: Vec<String>) -> Self {
        Self {
            code: "validation_failed",
            message: "Configuration validation failed".to_owned(),
            current_revision: None,
            validation_errors: errors,
            warnings,
        }
    }
}

impl ConfigurationAuthority {
    pub fn new(config_path: impl Into<PathBuf>, limits: AuthorityLimits) -> Result<Self, String> {
        if limits.max_input_bytes < 1
            || limits.max_input_bytes > MAX_INPUT_BYTES
            || limits.max_output_bytes < 256
            || limits.max_output_bytes > MAX_OUTPUT_BYTES
            || limits.max_config_bytes < 1
            || limits.max_config_bytes > MAX_CONFIG_BYTES
            || limits.lock_timeout > MAX_LOCK_TIMEOUT
        {
            return Err(
                "I/O, configuration, or lock limits are outside their safe bounds".to_owned(),
            );
        }
        let defaults: Value = serde_json::from_str(DEFAULTS_JSON)
            .map_err(|error| format!("embedded configuration defaults are invalid: {error}"))?;
        let schema: Value = serde_json::from_str(SCHEMA_JSON)
            .map_err(|error| format!("embedded configuration schema is invalid: {error}"))?;
        if !defaults.is_object() || !schema.is_object() {
            return Err("embedded configuration contract roots must be objects".to_owned());
        }
        let initial_revision = revision_from_document(&defaults)?;
        if initial_revision == 0 {
            return Err("embedded configuration revision must be positive".to_owned());
        }
        let config_path = normalize_config_path(config_path.into())?;
        Ok(Self {
            config_path,
            limits,
            defaults,
            schema,
        })
    }

    /// Return the configured maximum size of one authority protocol request.
    /// HTTP adapters use this to bound request-body buffering before adding
    /// their protocol envelope.
    pub fn max_input_bytes(&self) -> usize {
        self.limits.max_input_bytes
    }

    /// Return the configured maximum size of one authority protocol response.
    /// HTTP adapters also use this bound for serialized exports and downloads.
    pub fn max_output_bytes(&self) -> usize {
        self.limits.max_output_bytes
    }

    /// Return the current effective configuration for another in-process Rust
    /// server domain that consumes the same revisioned document. This read is
    /// bounded by the authority's file and lock limits and never persists a
    /// legacy-revision migration as a side effect.
    pub fn effective_configuration(&self) -> Result<Value, String> {
        self.with_store_lock(|| {
            let loaded = self.load_current()?;
            Ok(deep_merge(&self.defaults, &loaded.document))
        })
        .map_err(|error| format!("{}: {}", error.code, error.message))
    }

    /// Process exactly one bounded JSON request and return exactly one JSON response.
    /// Errors are serialized to stdout as protocol responses; the boolean is the
    /// process success status used by the command wrapper.
    pub fn handle_json(&self, input: &[u8]) -> (Vec<u8>, bool) {
        if input.len() > self.limits.max_input_bytes {
            return self.encode_response(error_response(AuthorityError::new(
                "request_too_large",
                format!(
                    "Request exceeds the {} byte input limit",
                    self.limits.max_input_bytes
                ),
            )));
        }
        let request = match serde_json::from_slice::<Request>(input) {
            Ok(request) => request,
            Err(error) => {
                return self.encode_response(error_response(AuthorityError::new(
                    "invalid_request",
                    format!("Request JSON is malformed or has invalid fields: {error}"),
                )))
            }
        };
        if request.protocol_version != CONFIGURATION_CONTRACT_VERSION {
            return self.encode_response(error_response(AuthorityError::new(
                "unsupported_protocol_version",
                format!("protocol_version must be {CONFIGURATION_CONTRACT_VERSION}"),
            )));
        }
        let operation = request.operation.clone();
        let response = match self.handle_request(request) {
            Ok(response) => response,
            Err(error) => error_response(error),
        };
        let (bytes, ok) = self.encode_response(response);
        let mut value: Value = serde_json::from_slice(&bytes).unwrap_or_else(|_| {
            json!({
                "protocol_version": CONFIGURATION_CONTRACT_VERSION,
                "ok": false,
                "operation": operation,
                "error": {"code": "internal_error", "message": "Response encoding failed"}
            })
        });
        if let Some(object) = value.as_object_mut() {
            object.insert("operation".to_owned(), Value::String(operation));
        }
        self.serialize_bounded(value, ok)
    }

    fn handle_request(&self, request: Request) -> Result<Value, AuthorityError> {
        match request.operation.as_str() {
            "schema" => self.schema_response(request.section.as_deref()),
            "defaults" => Ok(json!({
                "protocol_version": CONFIGURATION_CONTRACT_VERSION,
                "ok": true,
                "configuration": self.defaults.clone(),
            })),
            "snapshot" => self.with_store_lock(|| {
                let loaded = self.load_current()?;
                if loaded.needs_revision_migration {
                    // A legacy revision migration changes disk state. Verify that
                    // the complete success response can be delivered before the
                    // atomic replace, otherwise the caller would observe failure
                    // after an unacknowledged mutation.
                    let mut response = snapshot_response(loaded.document.clone(), loaded.revision);
                    response
                        .as_object_mut()
                        .expect("snapshot response object")
                        .insert(
                            "migration".to_owned(),
                            Value::String("legacy_revision_added".to_owned()),
                        );
                    response
                        .as_object_mut()
                        .expect("snapshot response object")
                        .insert(
                            "durability".to_owned(),
                            Value::String("uncertain".to_owned()),
                        );
                    self.ensure_response_fits(&response, "snapshot")?;
                    let durability = self.persist(&loaded.document)?;
                    if durability == PersistenceStatus::Durable {
                        response
                            .as_object_mut()
                            .expect("snapshot response object")
                            .remove("durability");
                    }
                    return Ok(response);
                }
                let mut response = snapshot_response(loaded.document, loaded.revision);
                if loaded.needs_revision_migration {
                    response
                        .as_object_mut()
                        .expect("snapshot response object")
                        .insert(
                            "migration".to_owned(),
                            Value::String("legacy_revision_added".to_owned()),
                        );
                }
                Ok(response)
            }),
            "capture_settings" => self.with_store_lock(|| {
                let data = required_object_data(&request)?;
                if data.len() != 1 || data.get("schema_version").and_then(Value::as_u64) != Some(3)
                {
                    return Err(AuthorityError::new(
                        "unsupported_capture_settings_version",
                        "capture_settings data must be exactly {schema_version: 3}",
                    ));
                }
                let loaded = self.load_current()?;
                let configuration = normalize_document_revision(
                    deep_merge(&self.defaults, &loaded.document),
                    loaded.revision,
                )?;
                let capture_settings =
                    build_capture_settings_v3(&configuration).map_err(|error| {
                        AuthorityError::new("invalid_capture_settings", error.to_string())
                    })?;
                Ok(json!({
                    "protocol_version": CONFIGURATION_CONTRACT_VERSION,
                    "ok": true,
                    "revision": loaded.revision,
                    "capture_settings": capture_settings,
                }))
            }),
            "export" => self.with_store_lock(|| {
                let loaded = self.load_current()?;
                let exported = self.export_document(
                    &loaded.document,
                    request.sections.as_deref(),
                    request.include_defaults,
                    request.include_metadata,
                )?;
                let mut response = json!({
                    "protocol_version": CONFIGURATION_CONTRACT_VERSION,
                    "ok": true,
                    "revision": loaded.revision,
                    "configuration": exported,
                });
                if loaded.needs_revision_migration {
                    response
                        .as_object_mut()
                        .expect("export response object")
                        .insert(
                            "migration".to_owned(),
                            Value::String("legacy_revision_added".to_owned()),
                        );
                }
                Ok(response)
            }),
            "validate" => self.with_store_lock(|| self.validate_request(&request)),
            "update" => self.mutate(&request, Mutation::Update),
            "replace" => self.mutate(&request, Mutation::Replace),
            "reset_section" => self.mutate(&request, Mutation::ResetSections),
            operation => Err(AuthorityError::new(
                "unknown_operation",
                format!("Unsupported configuration operation '{operation}'"),
            )),
        }
    }

    fn schema_response(&self, section: Option<&str>) -> Result<Value, AuthorityError> {
        let schema = if let Some(section) = section {
            let sections = self.section_names();
            if !sections.iter().any(|item| item == section) {
                return Err(AuthorityError::new(
                    "invalid_section",
                    format!("Unknown configuration section '{section}'"),
                ));
            }
            self.schema
                .get("properties")
                .and_then(Value::as_object)
                .and_then(|properties| properties.get(section))
                .cloned()
                .ok_or_else(|| AuthorityError::new("contract_error", "Section schema is missing"))?
        } else {
            self.schema.clone()
        };
        Ok(json!({
            "protocol_version": CONFIGURATION_CONTRACT_VERSION,
            "ok": true,
            "schema": schema,
        }))
    }

    fn validate_request(&self, request: &Request) -> Result<Value, AuthorityError> {
        let loaded = self.load_current()?;
        let data = required_object_data(request)?;
        let merge = request.merge.unwrap_or(true);
        let candidate = if merge {
            reject_store_revision_in_update(data)?;
            deep_merge(&loaded.document, &Value::Object(data.clone()))
        } else {
            normalize_document_revision(Value::Object(data.clone()), loaded.revision)?
        };
        let validation = self.validate_document(&candidate);
        Ok(json!({
            "protocol_version": CONFIGURATION_CONTRACT_VERSION,
            "ok": true,
            "revision": loaded.revision,
            "valid": validation.errors.is_empty(),
            "validation_errors": validation.errors,
            "warnings": validation.warnings,
        }))
    }

    fn mutate(&self, request: &Request, mutation: Mutation) -> Result<Value, AuthorityError> {
        self.with_store_lock(|| {
            let loaded = self.load_current()?;
            let expected = request.expected_revision.ok_or_else(|| {
                AuthorityError::new(
                    "expected_revision_required",
                    "Mutation requires expected_revision",
                )
            })?;
            if expected != loaded.revision {
                let mut error = AuthorityError::new(
                    "revision_conflict",
                    format!(
                        "Expected revision {expected}, current revision is {}",
                        loaded.revision
                    ),
                );
                error.current_revision = Some(loaded.revision);
                return Err(error);
            }

            let (mut candidate, force_update) = match mutation {
                Mutation::Update => {
                    let data = required_object_data(request)?;
                    reject_store_revision_in_update(data)?;
                    (
                        deep_merge(&loaded.document, &Value::Object(data.clone())),
                        request.force_update,
                    )
                }
                Mutation::Replace => {
                    let data = required_object_data(request)?;
                    (
                        normalize_document_revision(Value::Object(data.clone()), loaded.revision)?,
                        request.force_update,
                    )
                }
                Mutation::ResetSections => {
                    let sections = request.sections.as_deref().ok_or_else(|| {
                        AuthorityError::new(
                            "sections_required",
                            "Section reset requires a non-empty sections list",
                        )
                    })?;
                    if sections.is_empty() {
                        return Err(AuthorityError::new(
                            "sections_required",
                            "Section reset requires a non-empty sections list",
                        ));
                    }
                    let mut reset = loaded.document.clone();
                    let reset_object = reset.as_object_mut().ok_or_else(|| {
                        AuthorityError::new(
                            "store_error",
                            "Stored configuration root is not an object",
                        )
                    })?;
                    let default_object = self
                        .defaults
                        .as_object()
                        .expect("embedded defaults validated");
                    let mut seen = std::collections::BTreeSet::new();
                    for section in sections {
                        if !seen.insert(section) {
                            continue;
                        }
                        let default_value = default_object.get(section).ok_or_else(|| {
                            AuthorityError::new(
                                "invalid_section",
                                format!("Unknown configuration section '{section}'"),
                            )
                        })?;
                        reset_object.insert(section.clone(), default_value.clone());
                    }
                    (reset, request.force_update)
                }
            };

            // Section reset may include `metadata`; normalize its imported or
            // default revision before validation and change reporting.
            candidate = normalize_document_revision(candidate, loaded.revision)?;

            let validation = self.validate_document(&candidate);
            if !validation.errors.is_empty() {
                return Err(AuthorityError::validation(
                    validation.errors,
                    validation.warnings,
                ));
            }
            if !validation.warnings.is_empty() && !force_update {
                let mut error = AuthorityError::new(
                    "warnings_require_force",
                    "Configuration has warnings; set force_update=true to proceed",
                );
                error.current_revision = Some(loaded.revision);
                error.warnings = validation.warnings;
                return Err(error);
            }

            let mut changed_fields = changed_leaf_paths(&loaded.document, &candidate, "");
            if changed_fields.is_empty() {
                return Ok(update_response(
                    loaded.document,
                    loaded.revision,
                    changed_fields,
                    validation.warnings,
                ));
            }
            if loaded.revision >= MAX_SAFE_JSON_INTEGER {
                return Err(AuthorityError::new(
                    "revision_exhausted",
                    "Configuration revision is exhausted",
                ));
            }
            let next_revision = loaded.revision + 1;
            candidate = normalize_document_revision(candidate, next_revision)?;
            // `metadata.configuration_revision` is store-owned and does not
            // appear as a changed field because it was normalized before diff.
            changed_fields.sort();
            let mut response = update_response(
                candidate.clone(),
                next_revision,
                changed_fields,
                validation.warnings,
            );
            // Reserve the larger of the two possible durability encodings. The
            // actual commit status is only known after rename and directory sync.
            response
                .as_object_mut()
                .expect("update response object")
                .insert(
                    "durability".to_owned(),
                    Value::String("uncertain".to_owned()),
                );
            self.ensure_response_fits(&response, &request.operation)?;
            let durability = self.persist(&candidate)?;
            if durability == PersistenceStatus::Durable {
                response
                    .as_object_mut()
                    .expect("update response object")
                    .remove("durability");
            }
            Ok(response)
        })
    }

    fn validate_document(&self, candidate: &Value) -> ValidationResult {
        let complete = deep_merge(&self.defaults, candidate);
        let mut errors = Vec::new();
        validate_against_schema(&complete, &self.schema, "", &mut errors);
        validate_camera_source(&complete, "vision.camera.device_id", &mut errors);
        validate_viewer_source(&complete, "api.stream.viewer_url", &mut errors);
        validate_selector(&complete, "perception.capture.device_selector", &mut errors);
        validate_cross_field_rules(&complete, &self.schema, &mut errors);
        validate_runtime_rules(&complete, &mut errors);
        let warnings = configuration_warnings(&complete);
        ValidationResult { errors, warnings }
    }

    fn export_document(
        &self,
        document: &Value,
        sections: Option<&[String]>,
        include_defaults: bool,
        include_metadata: bool,
    ) -> Result<Value, AuthorityError> {
        let mut result = if include_defaults {
            deep_merge(&self.defaults, document)
        } else {
            document.clone()
        };
        if let Some(sections) = sections {
            let names = self.section_names();
            let mut selected = Map::new();
            for section in sections {
                if !names.iter().any(|name| name == section) {
                    return Err(AuthorityError::new(
                        "invalid_section",
                        format!("Unknown configuration section '{section}'"),
                    ));
                }
                if let Some(value) = result.get(section) {
                    selected.insert(section.clone(), value.clone());
                }
            }
            result = Value::Object(selected);
        }
        if !include_metadata {
            if let Some(object) = result.as_object_mut() {
                object.remove("metadata");
            }
        }
        Ok(result)
    }

    fn section_names(&self) -> Vec<String> {
        self.defaults
            .as_object()
            .map(|object| object.keys().cloned().collect())
            .unwrap_or_default()
    }

    fn with_store_lock<T>(
        &self,
        operation: impl FnOnce() -> Result<T, AuthorityError>,
    ) -> Result<T, AuthorityError> {
        let parent = self
            .config_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent).map_err(|error| {
            AuthorityError::new(
                "store_error",
                format!("Unable to create configuration directory: {error}"),
            )
        })?;
        let lock_path = lock_path_for(&self.config_path)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|error| {
                AuthorityError::new(
                    "store_error",
                    format!("Unable to open configuration lock: {error}"),
                )
            })?;
        acquire_lock(&lock, self.limits.lock_timeout)?;
        let result = operation();
        let _ = FileExt::unlock(&lock);
        result
    }

    fn load_current(&self) -> Result<LoadedDocument, AuthorityError> {
        let mut file = match File::open(&self.config_path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let document = self.defaults.clone();
                let revision = revision_from_document(&document)
                    .map_err(|message| AuthorityError::new("contract_error", message))?;
                return Ok(LoadedDocument {
                    document,
                    revision,
                    needs_revision_migration: false,
                });
            }
            Err(error) => {
                return Err(AuthorityError::new(
                    "store_error",
                    format!("Unable to read configuration: {error}"),
                ))
            }
        };
        let file_size = file
            .metadata()
            .map_err(|error| {
                AuthorityError::new(
                    "store_error",
                    format!("Unable to inspect configuration file: {error}"),
                )
            })?
            .len();
        if file_size > self.limits.max_config_bytes as u64 {
            return Err(AuthorityError::new(
                "store_too_large",
                format!(
                    "Stored configuration exceeds the {} byte limit",
                    self.limits.max_config_bytes
                ),
            ));
        }
        let bytes = read_bounded(&mut file, self.limits.max_config_bytes).map_err(|error| {
            AuthorityError::new(
                "store_error",
                format!("Unable to read configuration: {error}"),
            )
        })?;
        if bytes.len() > self.limits.max_config_bytes {
            return Err(AuthorityError::new(
                "store_too_large",
                format!(
                    "Stored configuration exceeds the {} byte limit",
                    self.limits.max_config_bytes
                ),
            ));
        }
        let mut document: Value = serde_json::from_slice(&bytes).map_err(|error| {
            AuthorityError::new(
                "store_error",
                format!("Stored configuration is malformed JSON: {error}"),
            )
        })?;
        if !document.is_object() {
            return Err(AuthorityError::new(
                "store_error",
                "Stored configuration root must be an object",
            ));
        }
        let (revision, needs_revision_migration) = match revision_from_document(&document) {
            Ok(revision) => (revision, false),
            Err(message) if message == "legacy_revision_missing" => {
                let initial_revision =
                    revision_from_document(&self.defaults).map_err(|contract_error| {
                        AuthorityError::new("contract_error", contract_error)
                    })?;
                document = normalize_document_revision(document, initial_revision)?;
                self.validate_store_document(&document)?;
                (initial_revision, true)
            }
            Err(message) => return Err(AuthorityError::new("store_error", message)),
        };
        self.validate_store_document(&document)?;
        Ok(LoadedDocument {
            document,
            revision,
            needs_revision_migration,
        })
    }

    fn validate_store_document(&self, document: &Value) -> Result<(), AuthorityError> {
        let validation = self.validate_document(document);
        if validation.errors.is_empty() {
            Ok(())
        } else {
            Err(AuthorityError::validation(
                validation.errors,
                validation.warnings,
            ))
        }
    }

    fn persist(&self, document: &Value) -> Result<PersistenceStatus, AuthorityError> {
        let mut bytes = serde_json::to_vec_pretty(document).map_err(|error| {
            AuthorityError::new(
                "store_error",
                format!("Unable to serialize configuration: {error}"),
            )
        })?;
        bytes.push(b'\n');
        if bytes.len() > self.limits.max_config_bytes {
            return Err(AuthorityError::new(
                "configuration_too_large",
                format!(
                    "Configuration exceeds the {} byte storage limit",
                    self.limits.max_config_bytes
                ),
            ));
        }
        atomic_write(&self.config_path, &bytes).map_err(|error| {
            AuthorityError::new(
                "persistence_failed",
                format!("Configuration was not persisted: {error}"),
            )
        })
    }

    fn encode_response(&self, mut response: Value) -> (Vec<u8>, bool) {
        if let Some(object) = response.as_object_mut() {
            object
                .entry("protocol_version".to_owned())
                .or_insert(Value::from(CONFIGURATION_CONTRACT_VERSION));
        }
        let ok = response.get("ok").and_then(Value::as_bool).unwrap_or(false);
        self.serialize_bounded(response, ok)
    }

    /// Ensure a mutating operation's success payload fits before changing disk.
    /// `serialize_bounded` remains the final guard for read-only and error output.
    fn ensure_response_fits(
        &self,
        response: &Value,
        operation: &str,
    ) -> Result<(), AuthorityError> {
        let mut response = response.clone();
        let Some(object) = response.as_object_mut() else {
            return Err(AuthorityError::new(
                "internal_error",
                "Configuration response must be a JSON object",
            ));
        };
        object
            .entry("protocol_version".to_owned())
            .or_insert(Value::from(CONFIGURATION_CONTRACT_VERSION));
        object.insert("operation".to_owned(), Value::String(operation.to_owned()));
        let serialized_length = serde_json::to_vec(&response)
            .map_err(|error| {
                AuthorityError::new(
                    "internal_error",
                    format!("Unable to size configuration response: {error}"),
                )
            })?
            .len()
            .saturating_add(1); // newline written by the one-shot protocol
        if serialized_length > self.limits.max_output_bytes {
            return Err(AuthorityError::new(
                "response_too_large",
                format!(
                    "Response exceeds the {} byte output limit",
                    self.limits.max_output_bytes
                ),
            ));
        }
        Ok(())
    }

    fn serialize_bounded(&self, response: Value, ok: bool) -> (Vec<u8>, bool) {
        let mut bytes = serde_json::to_vec(&response).unwrap_or_else(|_| {
            br#"{"protocol_version":1,"ok":false,"error":{"code":"internal_error","message":"Response serialization failed"}}"#.to_vec()
        });
        bytes.push(b'\n');
        if bytes.len() > self.limits.max_output_bytes {
            let mut limited = serde_json::to_vec(&json!({
                "protocol_version": CONFIGURATION_CONTRACT_VERSION,
                "ok": false,
                "error": {
                    "code": "response_too_large",
                    "message": format!("Response exceeds the {} byte output limit", self.limits.max_output_bytes)
                }
            })).unwrap_or_else(|_| b"{\"protocol_version\":1,\"ok\":false}\n".to_vec());
            limited.push(b'\n');
            return (limited, false);
        }
        (bytes, ok)
    }
}

#[derive(Debug, Clone, Copy)]
enum Mutation {
    Update,
    Replace,
    ResetSections,
}

#[derive(Debug)]
struct LoadedDocument {
    document: Value,
    revision: u64,
    needs_revision_migration: bool,
}

#[derive(Debug)]
struct ValidationResult {
    errors: Vec<ValidationError>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PersistenceStatus {
    Durable,
    Uncertain,
}

fn snapshot_response(document: Value, revision: u64) -> Value {
    json!({
        "protocol_version": CONFIGURATION_CONTRACT_VERSION,
        "ok": true,
        "revision": revision,
        "configuration": document,
    })
}

fn update_response(
    document: Value,
    revision: u64,
    updated_fields: Vec<String>,
    warnings: Vec<String>,
) -> Value {
    json!({
        "protocol_version": CONFIGURATION_CONTRACT_VERSION,
        "ok": true,
        "revision": revision,
        "configuration": document,
        "updated_fields": updated_fields,
        "warnings": warnings,
    })
}

fn error_response(error: AuthorityError) -> Value {
    let mut body = json!({
        "protocol_version": CONFIGURATION_CONTRACT_VERSION,
        "ok": false,
        "error": {"code": error.code, "message": error.message},
    });
    let object = body.as_object_mut().expect("error response is an object");
    if let Some(revision) = error.current_revision {
        object.insert("current_revision".to_owned(), Value::from(revision));
    }
    if !error.validation_errors.is_empty() {
        object.insert(
            "validation_errors".to_owned(),
            json!(error.validation_errors),
        );
    }
    if !error.warnings.is_empty() {
        object.insert("warnings".to_owned(), json!(error.warnings));
    }
    body
}

fn required_object_data(request: &Request) -> Result<&Map<String, Value>, AuthorityError> {
    let data = request.data.as_ref().ok_or_else(|| {
        AuthorityError::new("data_required", "Operation requires an object in data")
    })?;
    let object = data.as_object().ok_or_else(|| {
        AuthorityError::new("invalid_data", "Configuration data must be an object")
    })?;
    if object.is_empty() {
        return Err(AuthorityError::new(
            "invalid_data",
            "Configuration data cannot be empty",
        ));
    }
    Ok(object)
}

fn reject_store_revision_in_update(data: &Map<String, Value>) -> Result<(), AuthorityError> {
    if data
        .get("metadata")
        .and_then(Value::as_object)
        .is_some_and(|metadata| metadata.contains_key("configuration_revision"))
    {
        return Err(AuthorityError::new(
            "store_managed_revision",
            "metadata.configuration_revision is managed by the configuration authority",
        ));
    }
    Ok(())
}

fn normalize_document_revision(
    mut document: Value,
    revision: u64,
) -> Result<Value, AuthorityError> {
    if revision == 0 || revision > MAX_SAFE_JSON_INTEGER {
        return Err(AuthorityError::new(
            "invalid_revision",
            "Configuration revision must be a positive safe JSON integer",
        ));
    }
    let object = document.as_object_mut().ok_or_else(|| {
        AuthorityError::new("invalid_data", "Configuration root must be an object")
    })?;
    let metadata = object
        .entry("metadata")
        .or_insert_with(|| Value::Object(Map::new()));
    let metadata = metadata
        .as_object_mut()
        .ok_or_else(|| AuthorityError::new("invalid_data", "metadata must be an object"))?;
    metadata.insert("configuration_revision".to_owned(), Value::from(revision));
    Ok(document)
}

fn revision_from_document(document: &Value) -> Result<u64, String> {
    let metadata = match document.get("metadata") {
        None => return Err("legacy_revision_missing".to_owned()),
        Some(Value::Object(metadata)) => metadata,
        Some(_) => return Err("metadata must be an object".to_owned()),
    };
    let Some(revision_value) = metadata.get("configuration_revision") else {
        return Err("legacy_revision_missing".to_owned());
    };
    let Some(revision) = revision_value.as_u64() else {
        return Err("metadata.configuration_revision must be a positive safe integer".to_owned());
    };
    if revision == 0 || revision > MAX_SAFE_JSON_INTEGER {
        return Err("metadata.configuration_revision must be a positive safe integer".to_owned());
    }
    Ok(revision)
}

fn deep_merge(base: &Value, overlay: &Value) -> Value {
    match (base, overlay) {
        (Value::Object(base), Value::Object(overlay)) => {
            let mut result = base.clone();
            for (key, value) in overlay {
                let merged = match result.get(key) {
                    Some(existing) if existing.is_object() && value.is_object() => {
                        deep_merge(existing, value)
                    }
                    _ => value.clone(),
                };
                result.insert(key.clone(), merged);
            }
            Value::Object(result)
        }
        (_, overlay) => overlay.clone(),
    }
}

fn changed_leaf_paths(before: &Value, after: &Value, prefix: &str) -> Vec<String> {
    match (before.as_object(), after.as_object()) {
        (Some(before), Some(after)) => {
            let keys: std::collections::BTreeSet<&String> =
                before.keys().chain(after.keys()).collect();
            let mut paths = Vec::new();
            for key in keys {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                match (before.get(key), after.get(key)) {
                    (Some(before), Some(after)) => {
                        paths.extend(changed_leaf_paths(before, after, &path))
                    }
                    (Some(_), None) | (None, Some(_)) => paths.push(path),
                    (None, None) => {}
                }
            }
            paths
        }
        _ if before == after => Vec::new(),
        _ => vec![prefix.to_owned()],
    }
}

fn validate_against_schema(
    value: &Value,
    schema: &Value,
    path: &str,
    errors: &mut Vec<ValidationError>,
) {
    let types: Vec<&str> = match schema.get("type") {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(items)) => items.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    if !types.is_empty() && !types.iter().any(|kind| value_matches_type(value, kind)) {
        errors.push(validation_error(
            path,
            format!("Expected {}", types.join(" or ")),
            value,
            &types.join("|"),
        ));
        return;
    }

    if let Some(alternatives) = schema.get("anyOf").and_then(Value::as_array) {
        if alternatives
            .iter()
            .any(|alternative| schema_matches(value, alternative, path))
        {
            // Combinators add constraints; keep validating sibling keywords.
        } else {
            errors.push(validation_error(
                path,
                "Value does not match any allowed schema",
                value,
                "anyOf",
            ));
            return;
        }
    }
    if let Some(alternatives) = schema.get("oneOf").and_then(Value::as_array) {
        if alternatives
            .iter()
            .filter(|alternative| schema_matches(value, alternative, path))
            .count()
            == 1
        {
            // Continue with any constraints alongside oneOf.
        } else {
            errors.push(validation_error(
                path,
                "Value must match exactly one allowed schema",
                value,
                "oneOf",
            ));
            return;
        }
    }

    if let Some(enum_values) = schema.get("enum").and_then(Value::as_array) {
        if !enum_values.contains(value) {
            errors.push(validation_error(
                path,
                "Value is not one of the allowed values",
                value,
                "enumerated value",
            ));
            return;
        }
    }
    validate_numeric_schema(value, schema, path, errors);
    validate_string_schema(value, schema, path, errors);

    if let Some(array) = value.as_array() {
        if schema
            .get("minItems")
            .and_then(Value::as_u64)
            .is_some_and(|minimum| array.len() < minimum as usize)
        {
            errors.push(validation_error(
                path,
                "Array has fewer items than allowed",
                value,
                "array with minimum length",
            ));
        }
        if schema
            .get("maxItems")
            .and_then(Value::as_u64)
            .is_some_and(|maximum| array.len() > maximum as usize)
        {
            errors.push(validation_error(
                path,
                "Array has more items than allowed",
                value,
                "array with maximum length",
            ));
        }
        if let Some(item_schema) = schema.get("items") {
            for (index, item) in array.iter().enumerate() {
                validate_against_schema(item, item_schema, &format!("{path}[{index}]"), errors);
            }
        }
    }

    if let Some(object) = value.as_object() {
        let properties = schema.get("properties").and_then(Value::as_object);
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for key in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(key) {
                    errors.push(validation_error(
                        &join_path(path, key),
                        "Missing required property",
                        &Value::Null,
                        "required property",
                    ));
                }
            }
        }
        if let Some(properties) = properties {
            for (key, child) in object {
                if let Some(child_schema) = properties.get(key) {
                    validate_against_schema(child, child_schema, &join_path(path, key), errors);
                    continue;
                }
                match schema.get("additionalProperties") {
                    Some(Value::Bool(false)) => errors.push(validation_error(
                        &join_path(path, key),
                        "Unknown configuration key",
                        child,
                        "known key",
                    )),
                    Some(Value::Object(additional_schema)) => validate_against_schema(
                        child,
                        &Value::Object(additional_schema.clone()),
                        &join_path(path, key),
                        errors,
                    ),
                    _ => {}
                }
            }
        }
    }
}

fn schema_matches(value: &Value, schema: &Value, path: &str) -> bool {
    let mut errors = Vec::new();
    validate_against_schema(value, schema, path, &mut errors);
    errors.is_empty()
}

fn value_matches_type(value: &Value, expected: &str) -> bool {
    match expected {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.as_f64().is_some(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn validate_numeric_schema(
    value: &Value,
    schema: &Value,
    path: &str,
    errors: &mut Vec<ValidationError>,
) {
    let Some(number) = value.as_f64() else { return };
    for (key, comparison, message) in [
        ("minimum", 0_u8, "Value is below the minimum"),
        ("maximum", 1_u8, "Value exceeds the maximum"),
        (
            "exclusiveMinimum",
            2_u8,
            "Value must exceed the exclusive minimum",
        ),
        (
            "exclusiveMaximum",
            3_u8,
            "Value must be below the exclusive maximum",
        ),
    ] {
        if let Some(bound) = schema.get(key).and_then(Value::as_f64) {
            let failed = match comparison {
                0 => number < bound,
                1 => number > bound,
                2 => number <= bound,
                _ => number >= bound,
            };
            if failed {
                errors.push(validation_error(
                    path,
                    message,
                    value,
                    "number in configured range",
                ));
            }
        }
    }
}

fn validate_string_schema(
    value: &Value,
    schema: &Value,
    path: &str,
    errors: &mut Vec<ValidationError>,
) {
    let Some(string) = value.as_str() else { return };
    if schema
        .get("maxLength")
        .and_then(Value::as_u64)
        .is_some_and(|limit| string.chars().count() > limit as usize)
    {
        errors.push(validation_error(
            path,
            "String exceeds maximum length",
            value,
            "string within configured length",
        ));
    }
    if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
        if pattern == "^gstreamer-device-v1:[0-9a-f]{64}$" && !is_gstreamer_selector(string) {
            errors.push(validation_error(
                path,
                "String does not match the required pattern",
                value,
                "versioned GStreamer selector",
            ));
        }
    }
    if schema.get("format").and_then(Value::as_str) == Some("uri") && !valid_http_uri(string) {
        errors.push(validation_error(
            path,
            "Value must be a valid HTTP(S) URI without credentials or fragments",
            value,
            "HTTP(S) URI",
        ));
    }
    if schema.get("format").and_then(Value::as_str) == Some("ipv4-or-ipv6-cidr")
        && !valid_ip_network(string)
    {
        errors.push(validation_error(
            path,
            "Value must be a valid IPv4 or IPv6 address or CIDR",
            value,
            "IPv4 or IPv6 CIDR",
        ));
    }
}

/// Match `ipaddress.ip_network(value, strict=False)` for trusted proxy entries.
/// Host bits are permitted because the consumer canonicalizes to a network, but
/// address families, prefix syntax, and prefix widths remain strictly checked.
fn valid_ip_network(value: &str) -> bool {
    let mut parts = value.split('/');
    let Some(address_text) = parts.next() else {
        return false;
    };
    let Ok(address) = address_text.parse::<IpAddr>() else {
        return false;
    };
    let Some(prefix_text) = parts.next() else {
        return true;
    };
    if parts.next().is_some()
        || prefix_text.is_empty()
        || !prefix_text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return false;
    }
    let Ok(prefix) = prefix_text.parse::<u8>() else {
        return false;
    };
    let maximum = match address {
        IpAddr::V4(_) => 32,
        IpAddr::V6(_) => 128,
    };
    prefix <= maximum
}

fn validate_cross_field_rules(document: &Value, schema: &Value, errors: &mut Vec<ValidationError>) {
    let Some(rules) = schema.get("x-cross-field-rules").and_then(Value::as_array) else {
        return;
    };
    for rule in rules {
        let Some(fields) = rule.get("fields").and_then(Value::as_array) else {
            continue;
        };
        let fields: Vec<&str> = fields.iter().filter_map(Value::as_str).collect();
        let values: Vec<&Value> = fields
            .iter()
            .filter_map(|path| value_at_path(document, path))
            .collect();
        if fields.len() != values.len() || values.iter().any(|value| !value.as_f64().is_some()) {
            continue;
        }
        let kind = rule.get("kind").and_then(Value::as_str).unwrap_or_default();
        let numbers: Vec<f64> = values.iter().filter_map(|value| value.as_f64()).collect();
        let valid = numbers.windows(2).all(|pair| match kind {
            "strictly_increasing" => pair[0] < pair[1],
            _ => pair[0] <= pair[1],
        });
        if !valid {
            errors.push(ValidationError {
                field: fields.last().copied().unwrap_or("config_data").to_owned(),
                message: rule
                    .get("rule")
                    .and_then(Value::as_str)
                    .unwrap_or("Cross-field constraint failed")
                    .to_owned(),
                current_value: values.last().cloned().cloned().unwrap_or(Value::Null),
                expected_type: "values satisfying cross-field rule".to_owned(),
            });
        }
    }
}

fn validate_runtime_rules(document: &Value, errors: &mut Vec<ValidationError>) {
    let path = "perception.rust_live.image_size";
    if let Some(size) = value_at_path(document, path).and_then(Value::as_u64) {
        if size % 32 != 0 {
            errors.push(validation_error(
                path,
                "Image size must be divisible by 32",
                &Value::from(size),
                "integer multiple of 32",
            ));
        }
    }
    let control_enabled = value_at_path(
        document,
        "perception.rust_live.frame_preview_calibration_control_enabled",
    )
    .and_then(Value::as_bool)
        == Some(true);
    if control_enabled {
        let preview_enabled = value_at_path(document, "perception.rust_live.frame_preview_enabled");
        if preview_enabled.and_then(Value::as_bool) != Some(true) {
            errors.push(validation_error(
                "perception.rust_live.frame_preview_enabled",
                "Calibration control requires the Rust frame-preview listener",
                preview_enabled.unwrap_or(&Value::Null),
                "true when calibration control is enabled",
            ));
        }
        let header = value_at_path(
            document,
            "perception.rust_live.frame_preview_max_header_bytes",
        );
        if header
            .and_then(Value::as_u64)
            .is_some_and(|size| size < 4096)
        {
            errors.push(validation_error(
                "perception.rust_live.frame_preview_max_header_bytes",
                "Calibration control requires a response limit of at least 4096 bytes",
                header.unwrap_or(&Value::Null),
                "integer >= 4096",
            ));
        }
        let store_path = value_at_path(
            document,
            "perception.rust_live.frame_preview_calibration_store_path",
        )
        .and_then(Value::as_str);
        if store_path.is_some_and(|path| path.trim().is_empty() || path.chars().count() > 1024) {
            errors.push(validation_error(
                "perception.rust_live.frame_preview_calibration_store_path",
                "Calibration store path must be a non-empty path of at most 1024 characters",
                value_at_path(
                    document,
                    "perception.rust_live.frame_preview_calibration_store_path",
                )
                .unwrap_or(&Value::Null),
                "non-empty path <= 1024 characters",
            ));
        }
    }
}

fn configuration_warnings(document: &Value) -> Vec<String> {
    value_at_path(document, "vision.camera.fps")
        .and_then(Value::as_f64)
        .filter(|fps| *fps > 60.0)
        .map(|fps| vec![format!("vision.camera.fps={fps} may increase CPU usage")])
        .unwrap_or_default()
}

fn validate_camera_source(document: &Value, path: &str, errors: &mut Vec<ValidationError>) {
    if let Some(value) = value_at_path(document, path) {
        let valid = if let Some(index) = value.as_i64() {
            index >= 0
        } else if let Some(index) = value.as_u64() {
            index <= MAX_SAFE_JSON_INTEGER
        } else if let Some(source) = value.as_str() {
            is_gstreamer_selector(source) || valid_http_uri(source)
        } else {
            false
        };
        if !valid {
            errors.push(validation_error(path, "Camera source must be a nonnegative device index, HTTP(S) URI, or stable GStreamer selector", value, "camera source"));
        }
    }
}

fn validate_viewer_source(document: &Value, path: &str, errors: &mut Vec<ValidationError>) {
    if let Some(value) = value_at_path(document, path) {
        if !value.is_null() && !value.as_str().is_some_and(valid_http_uri) {
            errors.push(validation_error(
                path,
                "Viewer URL must be an HTTP(S) URI without credentials or fragments",
                value,
                "HTTP(S) URI or null",
            ));
        }
    }
}

fn validate_selector(document: &Value, path: &str, errors: &mut Vec<ValidationError>) {
    if let Some(value) = value_at_path(document, path) {
        if !value.is_null() && !value.as_str().is_some_and(is_gstreamer_selector) {
            errors.push(validation_error(
                path,
                "Capture device selector must be null or a versioned GStreamer selector",
                value,
                "null or GStreamer selector",
            ));
        }
    }
}

fn is_gstreamer_selector(value: &str) -> bool {
    value
        .strip_prefix("gstreamer-device-v1:")
        .is_some_and(|hash| {
            hash.len() == 64
                && hash
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        })
}

/// Validate a bounded credential-free HTTP(S) URI accepted by runtime
/// configuration and explicit local capture tools.
pub fn valid_http_uri(value: &str) -> bool {
    if value.len() > 2048
        || value.trim() != value
        || value.contains('#')
        || value
            .chars()
            .any(|character| character.is_whitespace() || character.is_control())
    {
        return false;
    }
    let Some((scheme, rest)) = value.split_once("://") else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return false;
    }
    let authority = rest.split(['/', '?']).next().unwrap_or_default();
    if authority.is_empty() || authority.contains('@') || authority.chars().any(char::is_whitespace)
    {
        return false;
    }
    let port_part = if authority.starts_with('[') {
        let Some(close) = authority.find(']') else {
            return false;
        };
        let host = &authority[1..close];
        if host.parse::<std::net::Ipv6Addr>().is_err() {
            return false;
        }
        let tail = &authority[close + 1..];
        if tail.is_empty() {
            None
        } else if let Some(port) = tail.strip_prefix(':') {
            Some(port)
        } else {
            return false;
        }
    } else if let Some((host, port)) = authority.rsplit_once(':') {
        if host.is_empty() || host.contains(':') || !valid_hostname(host) {
            return false;
        }
        Some(port)
    } else {
        if !valid_hostname(authority) {
            return false;
        }
        None
    };
    if let Some(port) = port_part {
        if !port.is_empty() && port.parse::<u16>().is_err() {
            return false;
        }
    }
    true
}

fn valid_hostname(host: &str) -> bool {
    !host.is_empty()
        && host.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_')
        })
}

fn value_at_path<'a>(document: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.')
        .try_fold(document, |value, key| value.get(key))
}

fn join_path(parent: &str, child: &str) -> String {
    if parent.is_empty() {
        child.to_owned()
    } else {
        format!("{parent}.{child}")
    }
}

fn validation_error(
    path: &str,
    message: impl Into<String>,
    value: &Value,
    expected: &str,
) -> ValidationError {
    ValidationError {
        field: if path.is_empty() {
            "config_data".to_owned()
        } else {
            path.to_owned()
        },
        message: message.into(),
        current_value: value.clone(),
        expected_type: expected.to_owned(),
    }
}

fn lock_path_for(config_path: &Path) -> Result<PathBuf, AuthorityError> {
    let file_name = config_path.file_name().ok_or_else(|| {
        AuthorityError::new("invalid_config_path", "Configuration path must name a file")
    })?;
    let parent = config_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok(parent.join(format!(".{}.lock", file_name.to_string_lossy())))
}

fn normalize_config_path(path: PathBuf) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .map_err(|error| format!("Unable to resolve configuration path: {error}"))?
            .join(path)
    };
    let mut lexical = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                lexical.pop();
            }
            other => lexical.push(other.as_os_str()),
        }
    }
    if lexical.file_name().is_none() {
        return Err("Configuration path must name a file".to_owned());
    }
    if lexical.exists() {
        return fs::canonicalize(&lexical)
            .map_err(|error| format!("Unable to resolve configuration path: {error}"));
    }

    let file_name = lexical
        .file_name()
        .ok_or_else(|| "Configuration path must name a file".to_owned())?
        .to_owned();
    let mut ancestor = lexical
        .parent()
        .ok_or_else(|| "Configuration path must have a parent directory".to_owned())?
        .to_path_buf();
    let mut missing = Vec::new();
    while !ancestor.exists() {
        let component = ancestor
            .file_name()
            .ok_or_else(|| "No existing parent directory for configuration path".to_owned())?
            .to_owned();
        missing.push(component);
        ancestor = ancestor
            .parent()
            .ok_or_else(|| "No existing parent directory for configuration path".to_owned())?
            .to_path_buf();
    }
    let mut resolved = fs::canonicalize(&ancestor)
        .map_err(|error| format!("Unable to resolve configuration parent: {error}"))?;
    for component in missing.iter().rev() {
        resolved.push(component);
    }
    resolved.push(file_name);
    Ok(resolved)
}

fn acquire_lock(lock: &File, timeout: Duration) -> Result<(), AuthorityError> {
    let start = Instant::now();
    loop {
        match lock.try_lock_exclusive() {
            Ok(()) => return Ok(()),
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock && start.elapsed() < timeout =>
            {
                thread::sleep(
                    Duration::from_millis(5).min(timeout.saturating_sub(start.elapsed())),
                );
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                return Err(AuthorityError::new(
                    "lock_timeout",
                    format!(
                        "Timed out waiting for the configuration lock after {} ms",
                        timeout.as_millis()
                    ),
                ));
            }
            Err(error) => {
                return Err(AuthorityError::new(
                    "store_error",
                    format!("Unable to lock configuration: {error}"),
                ))
            }
        }
    }
}

fn atomic_write(target: &Path, bytes: &[u8]) -> io::Result<PersistenceStatus> {
    let parent = target
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".billiards-config-")
        .tempfile_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    temporary.persist(target).map_err(|error| error.error)?;
    Ok(
        if File::open(parent)
            .and_then(|directory| directory.sync_all())
            .is_ok()
        {
            PersistenceStatus::Durable
        } else {
            // The atomic rename already made the complete new file visible. Report
            // the commit as successful while exposing that crash durability could
            // not be confirmed; returning failure here could cause a retry against
            // a revision that has in fact advanced.
            PersistenceStatus::Uncertain
        },
    )
}

/// Read at most the configured limit plus one byte, returning an explicit error
/// before parsing an oversized payload or allocating an unbounded buffer.
pub fn read_bounded<R: Read>(mut reader: R, limit: usize) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(limit.min(64 * 1024));
    let mut limited = reader.by_ref().take(limit.saturating_add(1) as u64);
    limited.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// The one-shot command's shared implementation; used by the binary and tests.
pub fn run_one_shot<R: Read, W: Write>(
    authority: &ConfigurationAuthority,
    mut input: R,
    mut output: W,
) -> io::Result<bool> {
    let bytes = read_bounded(&mut input, authority.limits.max_input_bytes)?;
    let (mut response, ok) = authority.handle_json(&bytes);
    if bytes.len() > authority.limits.max_input_bytes {
        response = authority
            .encode_response(error_response(AuthorityError::new(
                "request_too_large",
                format!(
                    "Request exceeds the {} byte input limit",
                    authority.limits.max_input_bytes
                ),
            )))
            .0;
    }
    output.write_all(&response)?;
    output.flush()?;
    Ok(ok && bytes.len() <= authority.limits.max_input_bytes)
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn contract_files_parse_and_defaults_are_valid_under_embedded_schema() {
        let defaults: Value = serde_json::from_str(DEFAULTS_JSON).expect("defaults JSON");
        let schema: Value = serde_json::from_str(SCHEMA_JSON).expect("schema JSON");
        let mut errors = Vec::new();
        validate_against_schema(&defaults, &schema, "", &mut errors);
        assert!(
            errors.is_empty(),
            "contract defaults failed schema: {errors:?}"
        );
    }

    #[test]
    fn uri_validator_rejects_credentials_and_fragments() {
        assert!(valid_http_uri("http://127.0.0.1:8080/stream"));
        assert!(valid_http_uri("https://example.test/path?q=1"));
        assert!(!valid_http_uri("http://user:pass@example.test/stream"));
        assert!(!valid_http_uri("http://example.test/stream#fragment"));
        assert!(!valid_http_uri("https://example.test/path with space"));
        assert!(!valid_http_uri("https://example.test/path\nwith-control"));
        assert!(!valid_http_uri("https://example.test/search?q=two words"));
        assert!(!valid_http_uri("file:///tmp/video.mp4"));
    }

    #[test]
    fn trusted_proxy_network_validator_accepts_ipaddress_network_forms() {
        assert!(valid_ip_network("192.0.2.1"));
        assert!(valid_ip_network("192.0.2.129/24"));
        assert!(valid_ip_network("2001:db8::1/64"));
        assert!(valid_ip_network("2001:db8::1/128"));
        assert!(!valid_ip_network("nonsense"));
        assert!(!valid_ip_network("192.0.2.1/33"));
        assert!(!valid_ip_network("2001:db8::1/129"));
        assert!(!valid_ip_network("192.0.2.1/24/8"));
        assert!(!valid_ip_network("192.0.2.1/-1"));
    }

    #[test]
    fn atomic_replace_failure_leaves_existing_target_directory_intact() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let target = directory.path().join("config.json");
        fs::create_dir(&target).expect("target directory");
        let marker = target.join("previous-state");
        fs::write(&marker, b"unchanged").expect("marker file");

        let result = atomic_write(&target, b"new configuration");
        assert!(result.is_err());
        assert_eq!(
            fs::read(marker).expect("original marker remains"),
            b"unchanged"
        );
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }
}
