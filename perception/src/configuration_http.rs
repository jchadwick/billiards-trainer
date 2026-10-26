//! Browser-compatible configuration routes hosted directly by the Rust gateway.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Request, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use chrono::{SecondsFormat, Utc};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, OwnedMutexGuard, Semaphore};

use crate::configuration_authority::{ConfigurationAuthority, DEFAULT_MAX_INPUT_BYTES};

const CONFIGURATION_API_SCHEMA_VERSION: &str = "1.0.0";
const DEFAULT_EXPORT_FORMAT: &str = "json";
const DEFAULT_IMPORT_STRATEGY: &str = "replace";
const CONFIGURATION_IMPORT_MESSAGE: &str = "Configuration imported successfully";
const CONFIGURATION_VALIDATION_MESSAGE: &str = "Configuration file validation completed";
const ALLOWED_IMPORT_EXTENSIONS: [&str; 3] = ["json", "yaml", "yml"];
// Leave room for the authority protocol fields wrapped around an HTTP JSON body.
const CONFIGURATION_PROTOCOL_ENVELOPE_RESERVE_BYTES: usize = 256;
const DEPLOYMENT_EXECUTABLE_FIELDS: [&str; 2] = [
    "perception.rust_live.executable_path",
    "perception.rust_live.capture_helper_executable_path",
];
const RESTART_REQUIRED_PREFIXES: [&str; 14] = [
    "perception.",
    "vision.",
    "core.",
    "integration.",
    "camera.",
    "cue.",
    "projector.",
    "api.server.",
    "api.network.",
    "api.middleware.",
    "api.rate_limiting.",
    "api.websocket.",
    "streaming.",
    "game_session.",
];

#[derive(Clone)]
struct ConfigurationHttpState {
    authority: Option<ConfigurationAuthority>,
    slots: Arc<Semaphore>,
    mutation_lock: Arc<Mutex<()>>,
}

pub(crate) fn router(
    authority: Option<ConfigurationAuthority>,
    slots: Arc<Semaphore>,
    mutation_lock: Arc<Mutex<()>>,
) -> Router {
    let maximum_body_bytes = authority
        .as_ref()
        .map(ConfigurationAuthority::max_input_bytes)
        .unwrap_or(DEFAULT_MAX_INPUT_BYTES);
    Router::new()
        .route(
            "/api/v1/config",
            get(get_configuration).put(update_configuration),
        )
        // The legacy FastAPI route is declared with a trailing slash. Accept
        // both browser-used URLs directly so callers do not need a redirect.
        .route("/api/v1/config/", put(update_configuration))
        .route("/api/v1/config/schema", get(get_configuration_schema))
        .route("/api/v1/config/reset", post(reset_configuration))
        .route("/api/v1/config/import", post(import_configuration))
        .route("/api/v1/config/export", get(export_configuration))
        .route(
            "/api/v1/config/export/download",
            get(download_configuration),
        )
        .layer(DefaultBodyLimit::max(maximum_body_bytes))
        .with_state(ConfigurationHttpState {
            authority,
            slots,
            mutation_lock,
        })
}

#[derive(Debug)]
struct ConfigurationQuery {
    section: Option<String>,
    // The legacy endpoint accepts this flag but currently returns stored metadata either way.
    _include_metadata: bool,
}

#[derive(Debug)]
struct ConfigurationUpdateQuery {
    section: Option<String>,
    validate_only: bool,
    force_update: bool,
}

#[derive(Debug)]
struct QueryValidationError {
    field: &'static str,
    message: &'static str,
    error_type: &'static str,
    input: Value,
}

#[derive(Debug)]
struct HttpContext {
    path: String,
    method: String,
    request_id: String,
}

impl HttpContext {
    fn from_request(request: &Request) -> Self {
        let request_id = request
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        Self {
            path: request.uri().path().to_owned(),
            method: request.method().as_str().to_owned(),
            request_id,
        }
    }
}

fn parse_configuration_query(
    query: Option<&str>,
    parse_metadata_flag: bool,
) -> Result<ConfigurationQuery, QueryValidationError> {
    let mut section = None;
    let mut include_metadata_value = None;
    for (name, value) in form_urlencoded::parse(query.unwrap_or_default().as_bytes()) {
        match name.as_ref() {
            // Starlette's query parameter lookup returns the last occurrence
            // when a key appears more than once. Preserve that behavior here.
            "section" => section = Some(value.into_owned()),
            "include_metadata" if parse_metadata_flag => {
                include_metadata_value = Some(value.into_owned())
            }
            // FastAPI ignores undeclared query parameters on these read routes.
            _ => {}
        }
    }

    let include_metadata = match include_metadata_value.as_deref() {
        None => true,
        Some(value) => match value.to_ascii_lowercase().as_str() {
            "true" | "1" | "on" | "yes" | "y" | "t" => true,
            "false" | "0" | "off" | "no" | "n" | "f" => false,
            _ => {
                return Err(QueryValidationError {
                    field: "query -> include_metadata",
                    message: "Input should be a valid boolean, unable to interpret input",
                    error_type: "bool_parsing",
                    input: Value::String(value.to_owned()),
                });
            }
        },
    };
    Ok(ConfigurationQuery {
        section,
        _include_metadata: include_metadata,
    })
}

fn parse_boolean_query(
    field: &'static str,
    value: Option<String>,
    default: bool,
) -> Result<bool, QueryValidationError> {
    let Some(value) = value else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "on" | "yes" | "y" | "t" => Ok(true),
        "false" | "0" | "off" | "no" | "n" | "f" => Ok(false),
        _ => Err(QueryValidationError {
            field,
            message: "Input should be a valid boolean, unable to interpret input",
            error_type: "bool_parsing",
            input: Value::String(value),
        }),
    }
}

fn parse_configuration_update_query(
    query: Option<&str>,
) -> Result<ConfigurationUpdateQuery, QueryValidationError> {
    let mut section = None;
    let mut validate_only = None;
    let mut force_update = None;
    for (name, value) in form_urlencoded::parse(query.unwrap_or_default().as_bytes()) {
        match name.as_ref() {
            "section" => section = Some(value.into_owned()),
            "validate_only" => validate_only = Some(value.into_owned()),
            "force_update" => force_update = Some(value.into_owned()),
            // FastAPI ignores query parameters that are not declared by the route.
            _ => {}
        }
    }
    Ok(ConfigurationUpdateQuery {
        section: section.filter(|section| !section.is_empty()),
        validate_only: parse_boolean_query("query -> validate_only", validate_only, false)?,
        force_update: parse_boolean_query("query -> force_update", force_update, false)?,
    })
}

fn parse_configuration_import_query(
    query: Option<&str>,
) -> Result<ConfigurationImportQuery, QueryValidationError> {
    let mut merge_strategy = None;
    let mut validate_only = None;
    let mut force_update = None;
    for (name, value) in form_urlencoded::parse(query.unwrap_or_default().as_bytes()) {
        match name.as_ref() {
            "merge_strategy" => merge_strategy = Some(value.into_owned()),
            "validate_only" => validate_only = Some(value.into_owned()),
            "force_update" => force_update = Some(value.into_owned()),
            _ => {}
        }
    }
    let merge_strategy = merge_strategy.unwrap_or_else(|| DEFAULT_IMPORT_STRATEGY.to_owned());
    if !matches!(merge_strategy.as_str(), "replace" | "merge") {
        return Err(QueryValidationError {
            field: "query -> merge_strategy",
            message: "String should match pattern '^(replace|merge)$'",
            error_type: "string_pattern_mismatch",
            input: Value::String(merge_strategy),
        });
    }
    Ok(ConfigurationImportQuery {
        merge_strategy,
        validate_only: parse_boolean_query("query -> validate_only", validate_only, false)?,
        force_update: parse_boolean_query("query -> force_update", force_update, false)?,
    })
}

fn parse_configuration_export_query(
    query: Option<&str>,
    include_options: bool,
) -> Result<ConfigurationExportQuery, QueryValidationError> {
    let mut format = None;
    let mut sections = Vec::new();
    let mut has_sections = false;
    let mut include_defaults = None;
    let mut include_metadata = None;
    for (name, value) in form_urlencoded::parse(query.unwrap_or_default().as_bytes()) {
        match name.as_ref() {
            "format" => format = Some(value.into_owned()),
            "sections" => {
                has_sections = true;
                sections.push(value.into_owned());
            }
            "include_defaults" if include_options => include_defaults = Some(value.into_owned()),
            "include_metadata" if include_options => include_metadata = Some(value.into_owned()),
            _ => {}
        }
    }
    let format = format.unwrap_or_else(|| DEFAULT_EXPORT_FORMAT.to_owned());
    if !matches!(format.as_str(), "json" | "yaml") {
        return Err(QueryValidationError {
            field: "query -> format",
            message: "String should match pattern '^(json|yaml)$'",
            error_type: "string_pattern_mismatch",
            input: Value::String(format),
        });
    }
    Ok(ConfigurationExportQuery {
        format,
        sections: has_sections.then_some(sections),
        include_defaults: parse_boolean_query(
            "query -> include_defaults",
            include_defaults,
            false,
        )?,
        include_metadata: parse_boolean_query("query -> include_metadata", include_metadata, true)?,
    })
}

#[derive(Debug)]
struct ConfigurationResetQuery {
    confirm: bool,
    reset_type: String,
    sections: Option<Vec<String>>,
}

#[derive(Debug)]
struct ConfigurationImportQuery {
    merge_strategy: String,
    validate_only: bool,
    force_update: bool,
}

#[derive(Debug, Clone)]
struct ConfigurationExportQuery {
    format: String,
    sections: Option<Vec<String>>,
    include_defaults: bool,
    include_metadata: bool,
}

fn parse_configuration_reset_query(
    query: Option<&str>,
) -> Result<ConfigurationResetQuery, Vec<QueryValidationError>> {
    let mut confirm = None;
    let mut reset_type = None;
    let mut sections = Vec::new();
    let mut has_sections = false;
    for (name, value) in form_urlencoded::parse(query.unwrap_or_default().as_bytes()) {
        match name.as_ref() {
            "confirm" => confirm = Some(value.into_owned()),
            "reset_type" => reset_type = Some(value.into_owned()),
            "sections" => {
                has_sections = true;
                sections.push(value.into_owned());
            }
            // The legacy route ignores every undeclared parameter.
            _ => {}
        }
    }

    let mut errors = Vec::new();
    let parsed_confirm = match confirm {
        Some(value) => match parse_boolean_query("query -> confirm", Some(value), false) {
            Ok(value) => Some(value),
            Err(error) => {
                errors.push(error);
                None
            }
        },
        None => {
            errors.push(QueryValidationError {
                field: "query -> confirm",
                message: "Field required",
                error_type: "missing",
                input: Value::Null,
            });
            None
        }
    };
    let reset_type = reset_type.unwrap_or_else(|| "all".to_owned());
    if !matches!(reset_type.as_str(), "all" | "user" | "section") {
        errors.push(QueryValidationError {
            field: "query -> reset_type",
            message: "String should match pattern '^(all|user|section)$'",
            error_type: "string_pattern_mismatch",
            input: Value::String(reset_type.clone()),
        });
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(ConfigurationResetQuery {
        confirm: parsed_confirm.expect("missing or invalid confirm is reported above"),
        reset_type,
        sections: has_sections.then_some(sections),
    })
}

async fn configuration_operation(
    state: &ConfigurationHttpState,
    operation: &'static str,
    section: Option<String>,
) -> Result<Value, StatusCode> {
    let authority = state
        .authority
        .as_ref()
        .cloned()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let permit = Arc::clone(&state.slots)
        .try_acquire_owned()
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;
    let mut request = json!({
        "protocol_version": 1,
        "operation": operation,
    });
    if let Some(section) = section {
        request["section"] = Value::String(section);
    }
    let request = serde_json::to_vec(&request).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let (response_bytes, process_ok) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        authority.handle_json(&request)
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let response: Value =
        serde_json::from_slice(&response_bytes).map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;
    if !process_ok || response.get("ok").and_then(Value::as_bool) != Some(true) {
        let code = response
            .get("error")
            .and_then(|error| error.get("code"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        return Err(match code {
            "invalid_section" => StatusCode::BAD_REQUEST,
            "validation_failed" => StatusCode::UNPROCESSABLE_ENTITY,
            _ => StatusCode::SERVICE_UNAVAILABLE,
        });
    }
    Ok(response)
}

fn section_names(defaults_response: &Value) -> Option<Vec<String>> {
    defaults_response
        .get("configuration")?
        .as_object()
        .map(|configuration| configuration.keys().cloned().collect())
}

fn section_error_message(section: &str, sections: &[String]) -> String {
    let allowed = sections
        .iter()
        .map(|section| format!("'{}'", section.replace('\\', "\\\\").replace('\'', "\\'")))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Invalid configuration section. Section '{section}' is not valid. Allowed sections: [{allowed}]"
    )
}

fn attach_request_id(mut response: Response, request_id: &str) -> Response {
    if let Ok(value) = HeaderValue::from_str(request_id) {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}

fn api_error_response(
    context: &HttpContext,
    status: StatusCode,
    message: &str,
    details: Value,
) -> Response {
    let (error, message) = match status {
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => ("VAL_001", message),
        StatusCode::CONFLICT => ("GAME_002", message),
        StatusCode::SERVICE_UNAVAILABLE => ("SYS_002", "Service unavailable"),
        _ => ("SYS_001", "Internal server error"),
    };
    let body = json!({
        "error": error,
        "message": message,
        "details": details,
        "timestamp": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
        "request_id": context.request_id,
        "path": context.path,
        "method": context.method,
    });
    attach_request_id((status, Json(body)).into_response(), &context.request_id)
}

fn query_error_response(context: &HttpContext, error: QueryValidationError) -> Response {
    query_errors_response(context, vec![error])
}

fn query_errors_response(context: &HttpContext, errors: Vec<QueryValidationError>) -> Response {
    api_error_response(
        context,
        StatusCode::UNPROCESSABLE_ENTITY,
        "Request validation failed",
        json!({
            "validation_errors": errors.iter().map(|error| json!({
                "field": error.field,
                "message": error.message,
                "type": error.error_type,
                "input": error.input,
            })).collect::<Vec<_>>()
        }),
    )
}

#[derive(Debug)]
struct UpdateFailure {
    status: StatusCode,
    message: String,
    details: Value,
}

impl UpdateFailure {
    fn route_error(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            details: Value::Null,
        }
    }

    fn service_unavailable() -> Self {
        Self::route_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration update failed",
        )
    }
}

async fn update_configuration(
    State(state): State<ConfigurationHttpState>,
    request: Request,
) -> Response {
    let context = HttpContext::from_request(&request);
    let query = match parse_configuration_update_query(request.uri().query()) {
        Ok(query) => query,
        Err(error) => return query_error_response(&context, error),
    };
    let Some(authority) = state.authority.as_ref().cloned() else {
        return api_error_response(
            &context,
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration update failed",
            Value::Null,
        );
    };
    let body_limit = authority
        .max_input_bytes()
        .saturating_sub(CONFIGURATION_PROTOCOL_ENVELOPE_RESERVE_BYTES);
    let bytes = match to_bytes(request.into_body(), body_limit).await {
        Ok(bytes) if !bytes.is_empty() => bytes,
        Ok(_) => {
            return query_error_response(
                &context,
                QueryValidationError {
                    field: "body",
                    message: "Field required",
                    error_type: "missing",
                    input: Value::Null,
                },
            )
        }
        Err(_) => {
            return query_error_response(
                &context,
                QueryValidationError {
                    field: "body",
                    message: "Request body is invalid or exceeds the configured limit",
                    error_type: "json_invalid",
                    input: Value::Null,
                },
            )
        }
    };
    let body: Value = match serde_json::from_slice(&bytes) {
        Ok(body) => body,
        Err(_) => {
            return query_error_response(
                &context,
                QueryValidationError {
                    field: "body",
                    message: "JSON decode error",
                    error_type: "json_invalid",
                    input: Value::Null,
                },
            )
        }
    };
    if !body.is_object() {
        return query_error_response(
            &context,
            QueryValidationError {
                field: "body",
                message: "Input should be a valid dictionary",
                error_type: "dict_type",
                input: body,
            },
        );
    }
    let permit = match Arc::clone(&state.slots).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return api_error_response(
                &context,
                StatusCode::SERVICE_UNAVAILABLE,
                "Configuration update failed",
                Value::Null,
            )
        }
    };
    let mutation_guard = Arc::clone(&state.mutation_lock).lock_owned().await;
    let validate_only = query.validate_only;
    let force_update = query.force_update;
    let section = query.section;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _mutation_guard: OwnedMutexGuard<()> = mutation_guard;
        update_configuration_sync(
            &authority,
            &body,
            section.as_deref(),
            validate_only,
            force_update,
        )
    })
    .await;
    match result {
        Ok(Ok(value)) => attach_request_id(Json(value).into_response(), &context.request_id),
        Ok(Err(error)) => api_error_response(&context, error.status, &error.message, error.details),
        Err(_) => api_error_response(
            &context,
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration update failed",
            Value::Null,
        ),
    }
}

async fn reset_configuration(
    State(state): State<ConfigurationHttpState>,
    request: Request,
) -> Response {
    let context = HttpContext::from_request(&request);
    let Some(authority) = state.authority.as_ref().cloned() else {
        return api_error_response(
            &context,
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration reset failed",
            Value::Null,
        );
    };
    let query = match parse_configuration_reset_query(request.uri().query()) {
        Ok(query) => query,
        Err(errors) => return query_errors_response(&context, errors),
    };
    if !query.confirm {
        return api_error_response(
            &context,
            StatusCode::BAD_REQUEST,
            "Configuration reset must be explicitly confirmed",
            Value::Null,
        );
    }
    let permit = match Arc::clone(&state.slots).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return api_error_response(
                &context,
                StatusCode::SERVICE_UNAVAILABLE,
                "Configuration reset failed",
                Value::Null,
            )
        }
    };
    let mutation_guard = Arc::clone(&state.mutation_lock).lock_owned().await;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _mutation_guard: OwnedMutexGuard<()> = mutation_guard;
        reset_configuration_sync(&authority, &query)
    })
    .await;
    match result {
        Ok(Ok(value)) => attach_request_id(Json(value).into_response(), &context.request_id),
        Ok(Err(error)) => api_error_response(&context, error.status, &error.message, error.details),
        Err(_) => api_error_response(
            &context,
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration reset failed",
            Value::Null,
        ),
    }
}

async fn import_configuration(
    State(state): State<ConfigurationHttpState>,
    request: Request,
) -> Response {
    let context = HttpContext::from_request(&request);
    let query = match parse_configuration_import_query(request.uri().query()) {
        Ok(query) => query,
        Err(error) => return query_error_response(&context, error),
    };
    let Some(authority) = state.authority.as_ref().cloned() else {
        return api_error_response(
            &context,
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration import failed",
            Value::Null,
        );
    };
    // Reserve capacity before the multipart parser reads file bytes so
    // concurrent uploads cannot bypass the gateway's configured work bound.
    let permit = match Arc::clone(&state.slots).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return api_error_response(
                &context,
                StatusCode::SERVICE_UNAVAILABLE,
                "Configuration import failed",
                Value::Null,
            )
        }
    };
    let mut multipart = match Multipart::from_request(request, &state).await {
        Ok(multipart) => multipart,
        Err(_) => {
            return api_error_response(
                &context,
                StatusCode::BAD_REQUEST,
                "Multipart configuration upload is invalid or exceeds the configured limit",
                Value::Null,
            )
        }
    };
    let mut uploaded_file = None;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(_) => {
                return api_error_response(
                    &context,
                    StatusCode::BAD_REQUEST,
                    "Multipart configuration upload is invalid",
                    Value::Null,
                )
            }
        };
        if field.name() != Some("file") {
            if field.bytes().await.is_err() {
                return api_error_response(
                    &context,
                    StatusCode::BAD_REQUEST,
                    "Multipart configuration upload is invalid",
                    Value::Null,
                );
            }
            continue;
        }
        let filename = field.file_name().map(str::to_owned);
        let extension = filename
            .as_deref()
            .and_then(|filename| std::path::Path::new(filename).extension())
            .and_then(std::ffi::OsStr::to_str)
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        if !ALLOWED_IMPORT_EXTENSIONS.contains(&extension.as_str()) {
            let allowed = ALLOWED_IMPORT_EXTENSIONS
                .iter()
                .map(|extension| format!(".{extension}"))
                .collect::<Vec<_>>();
            return api_error_response(
                &context,
                StatusCode::BAD_REQUEST,
                &format!(
                    "Invalid file format. File format '.{extension}' not supported. Allowed: {allowed:?}"
                ),
                Value::Null,
            );
        }
        let bytes = match field.bytes().await {
            Ok(bytes) => bytes,
            Err(_) => {
                return api_error_response(
                    &context,
                    StatusCode::BAD_REQUEST,
                    "Configuration file could not be read",
                    Value::Null,
                )
            }
        };
        // Starlette's form multi-dict returns the last value for `.get("file")`.
        uploaded_file = Some((filename, extension, bytes.to_vec()));
    }
    let Some((filename, extension, bytes)) = uploaded_file else {
        return query_error_response(
            &context,
            QueryValidationError {
                field: "body -> file",
                message: "Field required",
                error_type: "missing",
                input: Value::Null,
            },
        );
    };

    // Keep baseline read, validation and commit serial with other local config
    // mutations; the authority revision still protects independent processes.
    let mutation_guard = Arc::clone(&state.mutation_lock).lock_owned().await;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _mutation_guard: OwnedMutexGuard<()> = mutation_guard;
        import_configuration_sync(&authority, filename, &extension, &bytes, &query)
    })
    .await;
    match result {
        Ok(Ok(value)) => attach_request_id(Json(value).into_response(), &context.request_id),
        Ok(Err(error)) => api_error_response(&context, error.status, &error.message, error.details),
        Err(_) => api_error_response(
            &context,
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration import failed",
            Value::Null,
        ),
    }
}

fn import_configuration_sync(
    authority: &ConfigurationAuthority,
    filename: Option<String>,
    extension: &str,
    bytes: &[u8],
    query: &ConfigurationImportQuery,
) -> Result<Value, UpdateFailure> {
    let mut data = match extension {
        "json" => serde_json::from_slice::<Value>(bytes)
            .map_err(|error| parse_import_failure(error.to_string()))?,
        "yaml" | "yml" => serde_yaml_ng::from_slice::<Value>(bytes)
            .map_err(|error| parse_import_failure(error.to_string()))?,
        _ => {
            return Err(UpdateFailure::route_error(
                StatusCode::BAD_REQUEST,
                "Configuration file format is not supported",
            ));
        }
    };
    let Some(object) = data.as_object_mut() else {
        return Err(UpdateFailure::route_error(
            StatusCode::BAD_REQUEST,
            "Invalid configuration structure. Configuration must be a JSON/YAML object",
        ));
    };
    object.remove("_metadata");

    let current = authority_operation(
        authority,
        json!({
            "operation": "export",
            "include_defaults": false,
            "include_metadata": true,
        }),
    )
    .map_err(|_| UpdateFailure::service_unavailable())?;
    let current_configuration = current
        .get("configuration")
        .filter(|configuration| configuration.is_object())
        .ok_or_else(UpdateFailure::service_unavailable)?;
    let revision = current
        .get("revision")
        .and_then(Value::as_u64)
        .ok_or_else(UpdateFailure::service_unavailable)?;
    let changed_executables = changed_deployment_executable_fields(&data, current_configuration);
    if !changed_executables.is_empty() {
        let detail = vec![
            (
                "message",
                Value::String(
                    "Server executable paths are deployment-managed and cannot be changed through the HTTP configuration API"
                        .to_owned(),
                ),
            ),
            ("fields", json!(changed_executables)),
        ];
        return Err(UpdateFailure::route_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            compatibility_dict_repr(&detail),
        ));
    }

    let merge = query.merge_strategy == "merge";
    let validation = authority_operation(
        authority,
        json!({"operation":"validate", "merge":merge, "data":data}),
    )
    .map_err(|_| UpdateFailure::service_unavailable())?;
    let validation_errors = validation
        .get("validation_errors")
        .cloned()
        .unwrap_or_else(|| json!([]));
    let warnings = validation
        .get("warnings")
        .cloned()
        .unwrap_or_else(|| json!([]));
    let sections = data
        .as_object()
        .map(|object| object.keys().cloned().collect::<Vec<_>>())
        .ok_or_else(UpdateFailure::service_unavailable)?;

    if query.validate_only {
        return Ok(configuration_success(
            CONFIGURATION_VALIDATION_MESSAGE,
            json!({
                "is_valid": validation.get("valid").and_then(Value::as_bool) == Some(true),
                "validation_errors": validation_errors,
                "warnings": warnings,
                "sections_found": sections,
            }),
        ));
    }
    if validation.get("valid").and_then(Value::as_bool) != Some(true) {
        let detail = vec![
            (
                "message",
                Value::String("Configuration validation failed".to_owned()),
            ),
            ("errors", validation_errors),
        ];
        return Err(UpdateFailure::route_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            compatibility_dict_repr(&detail),
        ));
    }
    if warnings.as_array().is_some_and(|items| !items.is_empty()) && !query.force_update {
        let detail = vec![
            (
                "message",
                Value::String(
                    "Configuration has warnings. Use force_update=true to proceed".to_owned(),
                ),
            ),
            ("warnings", warnings),
        ];
        return Err(UpdateFailure::route_error(
            StatusCode::BAD_REQUEST,
            compatibility_dict_repr(&detail),
        ));
    }

    let operation = if merge { "update" } else { "replace" };
    let updated = authority_operation(
        authority,
        json!({
            "operation": operation,
            "expected_revision": revision,
            "force_update": query.force_update,
            "data": data,
        }),
    )
    .map_err(|error| authority_update_failure(&error))?;
    let data = json!({
        "filename": filename,
        "format": format!(".{extension}"),
        "merge_strategy": query.merge_strategy,
        "sections_imported": sections,
        "updated_fields": updated.get("updated_fields").cloned().unwrap_or_else(|| json!([])),
        "warnings": updated.get("warnings").cloned().unwrap_or(warnings),
    });
    Ok(configuration_success(CONFIGURATION_IMPORT_MESSAGE, data))
}

fn parse_import_failure(detail: String) -> UpdateFailure {
    UpdateFailure::route_error(
        StatusCode::BAD_REQUEST,
        format!("File parse error. Unable to parse configuration file: {detail}"),
    )
}

fn configuration_success(message: &str, data: Value) -> Value {
    json!({
        "success": true,
        "message": message,
        "timestamp": Utc::now().naive_utc().format("%Y-%m-%dT%H:%M:%S%.6f").to_string(),
        "data": data,
    })
}

async fn export_configuration(
    State(state): State<ConfigurationHttpState>,
    request: Request,
) -> Response {
    export_configuration_response(&state, request, false).await
}

async fn download_configuration(
    State(state): State<ConfigurationHttpState>,
    request: Request,
) -> Response {
    export_configuration_response(&state, request, true).await
}

async fn export_configuration_response(
    state: &ConfigurationHttpState,
    request: Request,
    download: bool,
) -> Response {
    let context = HttpContext::from_request(&request);
    let query = match parse_configuration_export_query(request.uri().query(), !download) {
        Ok(query) => query,
        Err(error) => return query_error_response(&context, error),
    };
    let Some(authority) = state.authority.as_ref().cloned() else {
        return api_error_response(
            &context,
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration export failed",
            Value::Null,
        );
    };
    let permit = match Arc::clone(&state.slots).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return api_error_response(
                &context,
                StatusCode::SERVICE_UNAVAILABLE,
                "Configuration export failed",
                Value::Null,
            )
        }
    };
    let export_query = query.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        // Both file serialization and the JSON HTTP envelope stay off the
        // async executor, under the same bounded export permit.
        build_configuration_export(&authority, &export_query, !download)
    })
    .await;
    let export = match result {
        Ok(Ok(export)) => export,
        Ok(Err(error)) => {
            return api_error_response(&context, error.status, &error.message, error.details)
        }
        Err(_) => {
            return api_error_response(
                &context,
                StatusCode::SERVICE_UNAVAILABLE,
                "Configuration export failed",
                Value::Null,
            )
        }
    };
    if download {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("billiards_config_{timestamp}.{}", query.format);
        let content_length = export.serialized.len().to_string();
        let mut response = Response::new(Body::from(export.serialized));
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/octet-stream"),
        );
        if let Ok(value) = HeaderValue::from_str(&content_length) {
            response
                .headers_mut()
                .insert(axum::http::header::CONTENT_LENGTH, value);
        }
        if let Ok(value) = HeaderValue::from_str(&format!("attachment; filename=\"{filename}\"")) {
            response
                .headers_mut()
                .insert(axum::http::header::CONTENT_DISPOSITION, value);
        }
        return attach_request_id(response, &context.request_id);
    }
    let Some(response_body) = export.response_body else {
        return api_error_response(
            &context,
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration export failed",
            Value::Null,
        );
    };
    let content_length = response_body.len().to_string();
    let mut response = Response::new(Body::from(response_body));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    if let Ok(value) = HeaderValue::from_str(&content_length) {
        response
            .headers_mut()
            .insert(axum::http::header::CONTENT_LENGTH, value);
    }
    attach_request_id(response, &context.request_id)
}

struct ConfigurationExport {
    serialized: String,
    response_body: Option<Vec<u8>>,
}

fn build_configuration_export(
    authority: &ConfigurationAuthority,
    query: &ConfigurationExportQuery,
    include_http_envelope: bool,
) -> Result<ConfigurationExport, UpdateFailure> {
    let exported = authority_operation(
        authority,
        json!({
            "operation": "export",
            "sections": query.sections,
            "include_defaults": query.include_defaults,
            // The route flag controls only its `_metadata` envelope. Preserve
            // stored runtime metadata whether or not that envelope is wanted.
            "include_metadata": true,
        }),
    )
    .map_err(|error| authority_update_failure(&error))?;
    let mut data = exported
        .get("configuration")
        .filter(|configuration| configuration.is_object())
        .cloned()
        .ok_or_else(UpdateFailure::service_unavailable)?;
    if query.include_metadata {
        let sections = data
            .as_object()
            .map(|object| object.keys().cloned().collect::<Vec<_>>())
            .ok_or_else(UpdateFailure::service_unavailable)?;
        let export_timestamp = Utc::now().to_rfc3339_opts(SecondsFormat::Micros, false);
        data.as_object_mut()
            .ok_or_else(UpdateFailure::service_unavailable)?
            .insert(
                "_metadata".to_owned(),
                json!({
                    "exported_at": export_timestamp,
                    "exported_by": "api",
                    "version": CONFIGURATION_API_SCHEMA_VERSION,
                    "sections": sections,
                }),
            );
    }
    let serialized = match query.format.as_str() {
        "json" => {
            serde_json::to_string_pretty(&data).map_err(|_| UpdateFailure::service_unavailable())
        }
        "yaml" => serde_yaml_ng::to_string(&data).map_err(|_| UpdateFailure::service_unavailable()),
        _ => return Err(UpdateFailure::service_unavailable()),
    }?;
    if serialized.len() > authority.max_output_bytes() {
        return Err(UpdateFailure::service_unavailable());
    }
    let checksum = format!("{:x}", Sha256::digest(serialized.as_bytes()));
    let response_body = if include_http_envelope {
        let response = json!({
            "format": query.format,
            "size": serialized.len(),
            "checksum": checksum,
            "timestamp": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
            "data": data,
        });
        let bytes =
            serde_json::to_vec(&response).map_err(|_| UpdateFailure::service_unavailable())?;
        if bytes.len() > authority.max_output_bytes() {
            return Err(UpdateFailure::service_unavailable());
        }
        Some(bytes)
    } else {
        None
    };
    Ok(ConfigurationExport {
        serialized,
        response_body,
    })
}

fn reset_configuration_sync(
    authority: &ConfigurationAuthority,
    query: &ConfigurationResetQuery,
) -> Result<Value, UpdateFailure> {
    let defaults_response = authority_operation(authority, json!({"operation": "defaults"}))
        .map_err(|_| UpdateFailure::service_unavailable())?;
    let section_names =
        section_names(&defaults_response).ok_or_else(UpdateFailure::service_unavailable)?;
    if query.reset_type == "section" {
        if let Some(sections) = query.sections.as_ref() {
            for section in sections {
                if !section_names.iter().any(|known| known == section) {
                    return Err(UpdateFailure::route_error(
                        StatusCode::BAD_REQUEST,
                        section_error_message(section, &section_names),
                    ));
                }
            }
        }
        if query.sections.as_ref().is_none_or(Vec::is_empty) {
            return Err(UpdateFailure::route_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "At least one section is required for a section reset",
            ));
        }
    }
    let defaults = defaults_response
        .get("configuration")
        .filter(|value| value.is_object())
        .cloned()
        .ok_or_else(UpdateFailure::service_unavailable)?;
    let baseline = authority_operation(
        authority,
        json!({
            "operation": "export",
            "include_defaults": false,
            "include_metadata": true,
        }),
    )
    .map_err(|_| UpdateFailure::service_unavailable())?;
    let expected_revision = baseline
        .get("revision")
        .and_then(Value::as_u64)
        .ok_or_else(UpdateFailure::service_unavailable)?;
    if query.reset_type == "section" {
        authority_operation(
            authority,
            json!({
                "operation": "reset_section",
                "expected_revision": expected_revision,
                "force_update": false,
                "sections": query.sections,
            }),
        )
        .map_err(|error| authority_update_failure(&error))?;
    } else {
        authority_operation(
            authority,
            json!({
                "operation": "replace",
                "expected_revision": expected_revision,
                "force_update": false,
                "data": defaults,
            }),
        )
        .map_err(|error| authority_update_failure(&error))?;
    }
    let sections_reset = if query.reset_type == "section" {
        json!(query.sections)
    } else {
        json!(section_names)
    };
    let data = json!({
        "reset_type": query.reset_type,
        "sections_reset": sections_reset,
    });
    Ok(json!({
        "success": true,
        "message": "Configuration reset to defaults successfully",
        // FastAPI/Pydantic serializes datetime.utcnow() as ISO-8601 with a T
        // separator and no timezone suffix because the legacy value is naive.
        "timestamp": Utc::now().naive_utc().format("%Y-%m-%dT%H:%M:%S%.6f").to_string(),
        "data": data,
    }))
}

fn update_configuration_sync(
    authority: &ConfigurationAuthority,
    body: &Value,
    section: Option<&str>,
    validate_only: bool,
    force_update: bool,
) -> Result<Value, UpdateFailure> {
    let defaults = authority_operation(authority, json!({"operation": "defaults"}))
        .map_err(|_| UpdateFailure::service_unavailable())?;
    let sections = section_names(&defaults).ok_or_else(UpdateFailure::service_unavailable)?;
    if let Some(section) = section {
        if !sections.iter().any(|known| known == section) {
            return Err(UpdateFailure::route_error(
                StatusCode::BAD_REQUEST,
                section_error_message(section, &sections),
            ));
        }
    }
    let object = body
        .as_object()
        .ok_or_else(UpdateFailure::service_unavailable)?;
    if object.is_empty() {
        return Err(UpdateFailure::route_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Configuration data cannot be empty",
        ));
    }
    let updates = normalize_section_update(body, section)?;
    // `snapshot` may persist the legacy revision migration, which would make
    // validate-only observably mutate the configuration file. Export is read-only.
    let baseline = authority_operation(
        authority,
        json!({
            "operation": "export",
            "include_defaults": false,
            "include_metadata": true,
        }),
    )
    .map_err(|_| UpdateFailure::service_unavailable())?;
    let current_configuration = baseline
        .get("configuration")
        .filter(|value| value.is_object())
        .ok_or_else(UpdateFailure::service_unavailable)?;
    let baseline_revision = baseline
        .get("revision")
        .and_then(Value::as_u64)
        .ok_or_else(UpdateFailure::service_unavailable)?;

    let changed_executable_fields =
        changed_deployment_executable_fields(&updates, current_configuration);
    if !changed_executable_fields.is_empty() {
        let detail = vec![
            (
                "message",
                Value::String(
                    "Server executable paths are deployment-managed and cannot be changed through the HTTP configuration API"
                        .to_owned(),
                ),
            ),
            ("fields", json!(changed_executable_fields)),
        ];
        return Err(UpdateFailure::route_error(
            StatusCode::BAD_REQUEST,
            compatibility_dict_repr(&detail),
        ));
    }

    let validation = authority_operation(
        authority,
        json!({"operation":"validate", "merge":true, "data":updates}),
    )
    .map_err(|_| UpdateFailure::service_unavailable())?;
    let validation_errors = validation
        .get("validation_errors")
        .cloned()
        .unwrap_or_else(|| json!([]));
    let warnings = validation
        .get("warnings")
        .cloned()
        .unwrap_or_else(|| json!([]));
    if validate_only {
        return Ok(json!({
            "success": validation.get("valid").and_then(Value::as_bool) == Some(true),
            "updated_fields": [],
            "validation_errors": validation_errors,
            "warnings": warnings,
            "rollback_available": false,
            "restart_required": false,
        }));
    }
    if validation.get("valid").and_then(Value::as_bool) != Some(true) {
        let detail = vec![
            (
                "message",
                Value::String("Configuration validation failed".to_owned()),
            ),
            ("errors", validation_errors),
        ];
        return Err(UpdateFailure::route_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            compatibility_dict_repr(&detail),
        ));
    }
    if warnings
        .as_array()
        .is_some_and(|warnings| !warnings.is_empty())
        && !force_update
    {
        let detail = vec![
            (
                "message",
                Value::String(
                    "Configuration has warnings. Use force_update=true to proceed".to_owned(),
                ),
            ),
            ("warnings", warnings),
        ];
        return Err(UpdateFailure::route_error(
            StatusCode::BAD_REQUEST,
            compatibility_dict_repr(&detail),
        ));
    }
    let updated = match authority_operation(
        authority,
        json!({
            "operation": "update",
            "expected_revision": baseline_revision,
            "force_update": force_update,
            "data": updates,
        }),
    ) {
        Ok(updated) => updated,
        Err(error) => return Err(authority_update_failure(&error)),
    };
    let updated_fields = updated
        .get("updated_fields")
        .cloned()
        .filter(Value::is_array)
        .unwrap_or_else(|| json!([]));
    let restart_required = updated_fields.as_array().is_some_and(|fields| {
        fields.iter().any(|field| {
            field.as_str().is_some_and(|field| {
                RESTART_REQUIRED_PREFIXES
                    .iter()
                    .any(|prefix| field.starts_with(prefix))
            })
        })
    });
    Ok(json!({
        "success": true,
        "updated_fields": updated_fields,
        "validation_errors": [],
        "warnings": updated.get("warnings").cloned().unwrap_or(warnings),
        "rollback_available": true,
        "restart_required": restart_required,
    }))
}

fn normalize_section_update(body: &Value, section: Option<&str>) -> Result<Value, UpdateFailure> {
    let Some(section) = section else {
        return Ok(body.clone());
    };
    let Some(object) = body.as_object() else {
        return Err(UpdateFailure::service_unavailable());
    };
    if object.len() == 1 && object.contains_key(section) {
        if !object.get(section).is_some_and(Value::is_object) {
            return Err(UpdateFailure::route_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "Section update must be an object",
            ));
        }
        return Ok(body.clone());
    }
    Ok(json!({(section): body}))
}

fn authority_operation(
    authority: &ConfigurationAuthority,
    mut request: Value,
) -> Result<Value, Value> {
    request["protocol_version"] = Value::from(1);
    let bytes = serde_json::to_vec(&request).map_err(|_| Value::Null)?;
    let (response, process_ok) = authority.handle_json(&bytes);
    let response: Value = serde_json::from_slice(&response).map_err(|_| Value::Null)?;
    if process_ok && response.get("ok").and_then(Value::as_bool) == Some(true) {
        Ok(response)
    } else {
        Err(response)
    }
}

fn authority_update_failure(error: &Value) -> UpdateFailure {
    let error_body = error.get("error").unwrap_or(&Value::Null);
    let code = error_body.get("code").and_then(Value::as_str).unwrap_or("");
    let error_message = error_body
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("Configuration update failed");
    match code {
        "validation_failed" => {
            let detail = vec![
                (
                    "message",
                    Value::String("Configuration validation failed".to_owned()),
                ),
                (
                    "errors",
                    error
                        .get("validation_errors")
                        .cloned()
                        .unwrap_or_else(|| json!([])),
                ),
            ];
            UpdateFailure::route_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                compatibility_dict_repr(&detail),
            )
        }
        "warnings_require_force" => {
            let detail = vec![
                (
                    "message",
                    Value::String(
                        "Configuration has warnings. Use force_update=true to proceed".to_owned(),
                    ),
                ),
                (
                    "warnings",
                    error.get("warnings").cloned().unwrap_or_else(|| json!([])),
                ),
            ];
            UpdateFailure::route_error(StatusCode::BAD_REQUEST, compatibility_dict_repr(&detail))
        }
        "revision_conflict" => {
            let mut detail = vec![("message", Value::String(error_message.to_owned()))];
            if let Some(revision) = error.get("current_revision") {
                detail.push(("current_revision", revision.clone()));
            }
            UpdateFailure::route_error(StatusCode::CONFLICT, compatibility_dict_repr(&detail))
        }
        "invalid_section" => UpdateFailure::route_error(StatusCode::BAD_REQUEST, error_message),
        _ => UpdateFailure::service_unavailable(),
    }
}

fn changed_deployment_executable_fields(updates: &Value, current: &Value) -> Vec<String> {
    let mut changed = DEPLOYMENT_EXECUTABLE_FIELDS
        .iter()
        .filter_map(|path| {
            let updated_value = nested_value(updates, path)?;
            let current_value = nested_value(current, path).unwrap_or(&Value::Null);
            (updated_value != current_value).then(|| (*path).to_owned())
        })
        .collect::<Vec<_>>();
    changed.sort();
    changed
}

fn nested_value<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.')
        .try_fold(root, |value, component| value.get(component))
}

fn compatibility_dict_repr(entries: &[(&str, Value)]) -> String {
    // The established error response formats HTTP exception dictionaries with
    // a stable dictionary representation while leaving the structured `details` field null.
    let fields = entries
        .iter()
        .map(|(key, value)| format!("'{}': {}", key, compatibility_value_repr(value)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{fields}}}")
}

fn compatibility_value_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => compatibility_string_repr(value),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(compatibility_value_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}: {}",
                        compatibility_string_repr(key),
                        compatibility_value_repr(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn compatibility_string_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push(quote);
    for character in value.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character == quote => {
                escaped.push('\\');
                escaped.push(character);
            }
            character if character.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(escaped, "\\x{:02x}", character as u32);
            }
            character => escaped.push(character),
        }
    }
    escaped.push(quote);
    escaped
}

async fn get_configuration(
    State(state): State<ConfigurationHttpState>,
    request: Request,
) -> Response {
    let context = HttpContext::from_request(&request);
    let query = match parse_configuration_query(request.uri().query(), true) {
        Ok(query) => query,
        Err(error) => return query_error_response(&context, error),
    };
    let defaults = match configuration_operation(&state, "defaults", None).await {
        Ok(defaults) => defaults,
        Err(status) => {
            return api_error_response(
                &context,
                status,
                "Configuration retrieval failed",
                Value::Null,
            )
        }
    };
    let sections = match section_names(&defaults) {
        Some(sections) => sections,
        None => {
            return api_error_response(
                &context,
                StatusCode::SERVICE_UNAVAILABLE,
                "Configuration retrieval failed",
                Value::Null,
            )
        }
    };
    let section = query.section.filter(|section| !section.is_empty());
    if let Some(section) = section.as_deref() {
        if !sections.iter().any(|known| known == section) {
            return api_error_response(
                &context,
                StatusCode::BAD_REQUEST,
                &section_error_message(section, &sections),
                Value::Null,
            );
        }
    }
    let snapshot = match configuration_operation(&state, "snapshot", None).await {
        Ok(snapshot) => snapshot,
        Err(status) => {
            return api_error_response(
                &context,
                status,
                "Configuration retrieval failed",
                Value::Null,
            )
        }
    };
    let Some(configuration) = snapshot.get("configuration").and_then(Value::as_object) else {
        return api_error_response(
            &context,
            StatusCode::SERVICE_UNAVAILABLE,
            "Configuration retrieval failed",
            Value::Null,
        );
    };
    let values = match section {
        Some(section) => {
            json!({(section.clone()): configuration.get(&section).cloned().unwrap_or_else(|| json!({}))})
        }
        None => Value::Object(configuration.clone()),
    };
    let timestamp = Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true);
    let body = json!({
        "timestamp": timestamp.clone(),
        "values": values,
        "schema_version": CONFIGURATION_API_SCHEMA_VERSION,
        "last_modified": timestamp,
        "is_valid": true,
        "validation_errors": [],
    });
    attach_request_id(Json(body).into_response(), &context.request_id)
}

async fn get_configuration_schema(
    State(state): State<ConfigurationHttpState>,
    request: Request,
) -> Response {
    let context = HttpContext::from_request(&request);
    let query = match parse_configuration_query(request.uri().query(), false) {
        Ok(query) => query,
        Err(error) => return query_error_response(&context, error),
    };
    let defaults = match configuration_operation(&state, "defaults", None).await {
        Ok(defaults) => defaults,
        Err(status) => {
            return api_error_response(&context, status, "Schema retrieval failed", Value::Null)
        }
    };
    let sections = match section_names(&defaults) {
        Some(sections) => sections,
        None => {
            return api_error_response(
                &context,
                StatusCode::SERVICE_UNAVAILABLE,
                "Schema retrieval failed",
                Value::Null,
            )
        }
    };
    let section = query.section.filter(|section| !section.is_empty());
    if let Some(section) = section.as_deref() {
        if !sections.iter().any(|known| known == section) {
            return api_error_response(
                &context,
                StatusCode::BAD_REQUEST,
                &section_error_message(section, &sections),
                Value::Null,
            );
        }
    }
    let schema = match configuration_operation(&state, "schema", section).await {
        Ok(schema) => schema,
        Err(status) => {
            return api_error_response(&context, status, "Schema retrieval failed", Value::Null)
        }
    };
    match schema.get("schema") {
        Some(schema) => {
            attach_request_id(Json(schema.clone()).into_response(), &context.request_id)
        }
        None => api_error_response(
            &context,
            StatusCode::SERVICE_UNAVAILABLE,
            "Schema retrieval failed",
            Value::Null,
        ),
    }
}
