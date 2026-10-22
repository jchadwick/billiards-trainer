use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use billiards_perception::server::{Gateway, ServerConfig};
use fs2::FileExt;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tempfile::tempdir;
use tower::ServiceExt;

async fn get(gateway: Gateway, uri: &str) -> (StatusCode, axum::http::HeaderMap, Value) {
    let response = gateway
        .router()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("configuration response is bounded");
    let body = serde_json::from_slice(&bytes).expect("configuration response is JSON");
    (status, headers, body)
}

async fn put_raw(
    gateway: Gateway,
    uri: &str,
    body: &str,
) -> (StatusCode, axum::http::HeaderMap, Value) {
    let response = gateway
        .router()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_owned()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("configuration response is bounded");
    let body = serde_json::from_slice(&bytes).expect("configuration response is JSON");
    (status, headers, body)
}

async fn post_raw(
    gateway: Gateway,
    uri: &str,
    body: &str,
) -> (StatusCode, axum::http::HeaderMap, Value) {
    let response = gateway
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .header("x-request-id", "configuration-reset-test")
                .body(Body::from(body.to_owned()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("configuration response is bounded");
    let body = serde_json::from_slice(&bytes).expect("configuration response is JSON");
    (status, headers, body)
}

fn multipart_file_body(boundary: &str, filename: &str, content: &[u8]) -> Vec<u8> {
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(content);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

async fn post_multipart(
    gateway: Gateway,
    uri: &str,
    filename: &str,
    content: &[u8],
) -> (StatusCode, axum::http::HeaderMap, Value) {
    let boundary = "configuration-audit-boundary";
    let response = gateway
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(multipart_file_body(boundary, filename, content)))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("configuration response is bounded");
    let body = serde_json::from_slice(&bytes).expect("configuration response is JSON");
    (status, headers, body)
}

async fn get_raw(gateway: Gateway, uri: &str) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let response = gateway
        .router()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("configuration response is bounded");
    (status, headers, bytes.to_vec())
}

fn gateway_with_config_path(path: Option<std::path::PathBuf>) -> Gateway {
    let config = ServerConfig {
        configuration_path: path,
        ..ServerConfig::default()
    };
    Gateway::new(config).expect("valid gateway configuration")
}

#[tokio::test]
async fn configuration_reads_keep_the_api_shape_and_legacy_metadata_behavior() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    let (status, headers, whole) =
        get(gateway.clone(), "/api/v1/config?include_metadata=false").await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers.contains_key("x-request-id"));
    assert!(whole["timestamp"].is_string());
    assert_eq!(whole["timestamp"], whole["last_modified"]);
    assert_eq!(whole["schema_version"], "1.0.0");
    assert_eq!(whole["is_valid"], true);
    assert_eq!(whole["validation_errors"], serde_json::json!([]));
    assert!(whole["values"]["metadata"]["configuration_revision"].is_number());

    let (_, _, section) = get(gateway.clone(), "/api/v1/config?section=vision").await;
    assert!(section["values"].get("vision").is_some());
    assert_eq!(section["values"].as_object().unwrap().len(), 1);

    let (_, _, empty_section) = get(gateway, "/api/v1/config?section=").await;
    assert!(empty_section["values"].get("metadata").is_some());
}

#[tokio::test]
async fn schema_endpoint_returns_the_raw_full_or_section_schema() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    let (status, _, whole) = get(gateway.clone(), "/api/v1/config/schema").await;
    assert_eq!(status, StatusCode::OK);
    assert!(whole["properties"]["vision"].is_object());

    let (status, _, vision) = get(
        gateway,
        "/api/v1/config/schema?section=vision&include_metadata=not-a-boolean",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(vision["type"], "object");
    assert!(vision.get("properties").is_some());
}

#[tokio::test]
async fn query_validation_and_section_errors_use_the_public_error_envelope() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    let (status, headers, invalid_query) =
        get(gateway.clone(), "/api/v1/config?include_metadata=maybe").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(invalid_query["error"], "VAL_001");
    assert_eq!(invalid_query["message"], "Request validation failed");
    assert_eq!(
        invalid_query["details"]["validation_errors"][0]["field"],
        "query -> include_metadata"
    );
    assert_eq!(
        headers.get("x-request-id").unwrap(),
        invalid_query["request_id"].as_str().unwrap()
    );
    assert_eq!(invalid_query["path"], "/api/v1/config");
    assert_eq!(invalid_query["method"], "GET");

    let (status, _, invalid_section) = get(gateway, "/api/v1/config?section=not-a-section").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(invalid_section["error"], "VAL_001");
    assert!(invalid_section["message"]
        .as_str()
        .unwrap()
        .contains("Section 'not-a-section' is not valid"));
    assert!(invalid_section["details"].is_null());
}

#[tokio::test]
async fn repeated_query_parameters_use_the_last_value_like_the_legacy_service() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    let (status, _, response) = get(
        gateway,
        "/api/v1/config?section=not-a-section&section=vision&include_metadata=maybe&include_metadata=false",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(response["values"].get("vision").is_some());
    assert_eq!(response["values"].as_object().unwrap().len(), 1);
}

#[tokio::test]
async fn configuration_updates_keep_the_public_shape_and_accept_both_route_forms() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    let (status, headers, update) = put_raw(
        gateway.clone(),
        "/api/v1/config",
        r#"{"metadata":{"version":"1.0.1"}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{update}");
    assert!(headers.contains_key("x-request-id"));
    assert_eq!(update["success"], true);
    assert_eq!(
        update["updated_fields"],
        serde_json::json!(["metadata.version"])
    );
    assert_eq!(update["validation_errors"], serde_json::json!([]));
    assert_eq!(update["rollback_available"], true);
    assert_eq!(update["restart_required"], false);

    let (status, _, update) = put_raw(
        gateway,
        "/api/v1/config/?section=vision",
        r#"{"vision":{"camera":{"fps":45}}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        update["updated_fields"],
        serde_json::json!(["vision.camera.fps"])
    );
    assert_eq!(update["restart_required"], true);
}

#[tokio::test]
async fn validation_only_reports_errors_and_does_not_mutate_legacy_files() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("config.json");
    let original = br#"{"metadata":{"application":"billiards-trainer","environment":"production","version":"1.0.0"}}"#;
    std::fs::write(&path, original).unwrap();
    let gateway = gateway_with_config_path(Some(path.clone()));

    let (status, _, validation) = put_raw(
        gateway.clone(),
        "/api/v1/config?validate_only=true",
        r#"{"metadata":{"version":"1.0.1"}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(validation["success"], true, "{validation}");
    assert_eq!(validation["updated_fields"], serde_json::json!([]));
    assert_eq!(validation["rollback_available"], false);
    assert_eq!(validation["restart_required"], false);
    assert_eq!(std::fs::read(&path).unwrap(), original);

    let (status, _, invalid) = put_raw(
        gateway,
        "/api/v1/config?validate_only=true",
        r#"{"vision":{"camera":{"fps":0}}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(invalid["success"], false);
    assert_eq!(
        invalid["validation_errors"][0]["field"],
        "vision.camera.fps"
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[tokio::test]
async fn update_validation_warning_precedence_and_force_behavior_match_the_legacy_api() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    let (status, _, invalid) = put_raw(
        gateway.clone(),
        "/api/v1/config?force_update=true",
        r#"{"vision":{"camera":{"fps":0}}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(invalid["error"], "VAL_001");
    assert!(invalid["message"]
        .as_str()
        .unwrap()
        .contains("Configuration validation failed"));

    let (status, _, warning) = put_raw(
        gateway.clone(),
        "/api/v1/config",
        r#"{"vision":{"camera":{"fps":90}}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(warning["message"]
        .as_str()
        .unwrap()
        .contains("force_update=true"));
    assert!(warning["details"].is_null());

    let (status, _, forced) = put_raw(
        gateway,
        "/api/v1/config?force_update=on",
        r#"{"vision":{"camera":{"fps":90}}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(forced["success"], true);
    assert_eq!(forced["restart_required"], true);
    assert_eq!(forced["warnings"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn deployment_executable_changes_are_rejected_even_when_forced_or_validation_only() {
    for path in [
        "perception.rust_live.executable_path",
        "perception.rust_live.capture_helper_executable_path",
        "perception.rust_replay.executable_path",
    ] {
        let directory = tempdir().unwrap();
        let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));
        let mut body = serde_json::Map::new();
        let mut cursor = &mut body;
        let components = path.split('.').collect::<Vec<_>>();
        for component in &components[..components.len() - 1] {
            let value = cursor
                .entry((*component).to_owned())
                .or_insert_with(|| serde_json::json!({}));
            cursor = value.as_object_mut().unwrap();
        }
        cursor.insert(
            components[components.len() - 1].to_owned(),
            serde_json::json!("/unapproved/server-binary"),
        );
        let (status, _, rejected) = put_raw(
            gateway,
            "/api/v1/config?validate_only=true&force_update=true",
            &Value::Object(body).to_string(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}");
        assert_eq!(rejected["error"], "VAL_001");
        assert!(rejected["message"].as_str().unwrap().contains(path));
    }
}

#[tokio::test]
async fn malformed_missing_and_non_object_update_bodies_use_request_validation_envelopes() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    for (body, expected_type) in [("", "missing"), ("{", "json_invalid"), ("[]", "dict_type")] {
        let (status, _, response) = put_raw(gateway.clone(), "/api/v1/config", body).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(response["error"], "VAL_001");
        assert_eq!(
            response["details"]["validation_errors"][0]["type"],
            expected_type
        );
    }
}

#[tokio::test]
async fn update_query_parameters_use_the_last_value_like_the_legacy_service() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));
    let (status, _, response) = put_raw(
        gateway,
        "/api/v1/config?section=unknown&section=vision&validate_only=maybe&validate_only=true",
        r#"{"vision":{"camera":{"fps":0}}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["success"], false);
    assert_eq!(
        response["validation_errors"][0]["field"],
        "vision.camera.fps"
    );
}

#[tokio::test]
async fn update_rejects_invalid_sections_empty_data_and_section_scalars() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    let (status, _, unknown_section) = put_raw(
        gateway.clone(),
        "/api/v1/config?section=not-a-section",
        r#"{"vision":{"camera":{"fps":30}}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(unknown_section["message"]
        .as_str()
        .unwrap()
        .contains("Section 'not-a-section' is not valid"));

    let (status, _, empty) = put_raw(gateway.clone(), "/api/v1/config", "{}").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(empty["message"], "Configuration data cannot be empty");

    let (status, _, scalar) =
        put_raw(gateway, "/api/v1/config?section=vision", r#"{"vision":17}"#).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(scalar["message"], "Section update must be an object");
}

#[tokio::test]
async fn no_op_updates_report_no_changed_fields_but_keep_legacy_rollback_flag() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    let (status, _, response) = put_raw(
        gateway,
        "/api/v1/config",
        r#"{"metadata":{"version":"1.0.0"}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["updated_fields"], serde_json::json!([]));
    assert_eq!(response["rollback_available"], true);
    assert_eq!(response["restart_required"], false);
}

#[tokio::test]
async fn reset_query_validation_aggregates_required_and_pattern_errors() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    let (status, _, missing) =
        post_raw(gateway.clone(), "/api/v1/config/reset", "{malformed").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        missing["details"]["validation_errors"][0]["field"],
        "query -> confirm"
    );
    assert_eq!(
        missing["details"]["validation_errors"][0]["type"],
        "missing"
    );

    let (status, _, invalid) = post_raw(
        gateway,
        "/api/v1/config/reset?confirm=maybe&reset_type=unknown",
        "{malformed",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        invalid["details"]["validation_errors"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        invalid["details"]["validation_errors"][0]["field"],
        "query -> confirm"
    );
    assert_eq!(
        invalid["details"]["validation_errors"][1]["field"],
        "query -> reset_type"
    );
    assert_eq!(
        invalid["details"]["validation_errors"][1]["type"],
        "string_pattern_mismatch"
    );
}

#[tokio::test]
async fn reset_requires_true_confirmation_and_ignores_the_request_body() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    let (status, _, unconfirmed) = post_raw(
        gateway.clone(),
        "/api/v1/config/reset?confirm=false",
        "{malformed",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        unconfirmed["message"],
        "Configuration reset must be explicitly confirmed"
    );

    let (status, _, reset) = post_raw(
        gateway,
        "/api/v1/config/reset?confirm=false&confirm=true&reset_type=section&reset_type=all",
        "{malformed",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{reset}");
    assert_eq!(reset["success"], true);
    assert_eq!(
        reset["message"],
        "Configuration reset to defaults successfully"
    );
    assert_eq!(reset["data"]["reset_type"], "all");
    let timestamp = reset["timestamp"].as_str().unwrap();
    assert!(timestamp.contains('T'));
    assert!(!timestamp.ends_with('Z'));
}

#[tokio::test]
async fn section_reset_preserves_unselected_configuration_and_reports_repeated_sections() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));
    let (status, _, update) = put_raw(
        gateway.clone(),
        "/api/v1/config?force_update=true",
        r#"{"metadata":{"version":"1.0.1"},"vision":{"camera":{"fps":45}}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{update}");

    let (status, _, reset) = post_raw(
        gateway.clone(),
        "/api/v1/config/reset?confirm=true&reset_type=section&sections=vision&sections=vision",
        "",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{reset}");
    assert_eq!(
        reset["data"]["sections_reset"],
        serde_json::json!(["vision", "vision"])
    );

    let (status, _, current) = get(gateway, "/api/v1/config").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(current["values"]["metadata"]["version"], "1.0.1");
    assert_eq!(current["values"]["vision"]["camera"]["fps"], 30);
}

#[tokio::test]
async fn reset_modes_match_legacy_behavior_and_section_errors_keep_precedence() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));
    let (status, _, update) = put_raw(
        gateway.clone(),
        "/api/v1/config?force_update=true",
        r#"{"vision":{"camera":{"fps":45}}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{update}");

    let (status, _, missing_sections) = post_raw(
        gateway.clone(),
        "/api/v1/config/reset?confirm=true&reset_type=section",
        "",
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        missing_sections["message"],
        "At least one section is required for a section reset"
    );

    let (status, _, empty_section) = post_raw(
        gateway.clone(),
        "/api/v1/config/reset?confirm=true&reset_type=section&sections=",
        "",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(empty_section["message"]
        .as_str()
        .unwrap()
        .contains("Section '' is not valid"));

    // `user` historically resets the full document, ignores section and force
    // parameters, and reports the requested mode in its response.
    let (status, _, user_reset) = post_raw(
        gateway.clone(),
        "/api/v1/config/reset?confirm=true&reset_type=user&sections=unknown&force_update=true",
        "",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{user_reset}");
    assert_eq!(user_reset["data"]["reset_type"], "user");
    assert!(
        user_reset["data"]["sections_reset"]
            .as_array()
            .unwrap()
            .len()
            > 1
    );
    let (status, _, current) = get(gateway, "/api/v1/config").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(current["values"]["vision"]["camera"]["fps"], 30);
}

#[tokio::test]
async fn repeated_full_reset_is_idempotent_and_keeps_revision_stable() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    for _ in 0..2 {
        let (status, _, reset) =
            post_raw(gateway.clone(), "/api/v1/config/reset?confirm=true", "").await;
        assert_eq!(status, StatusCode::OK, "{reset}");
    }
    let (status, _, configuration) = get(gateway, "/api/v1/config").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        configuration["values"]["metadata"]["configuration_revision"],
        1
    );
}

#[tokio::test]
async fn section_reset_cannot_bypass_warnings_in_untouched_sections() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));
    let (status, _, update) = put_raw(
        gateway.clone(),
        "/api/v1/config?force_update=true",
        r#"{"vision":{"camera":{"fps":90}}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{update}");

    let (status, headers, reset) = post_raw(
        gateway,
        "/api/v1/config/reset?confirm=true&reset_type=section&sections=metadata&force_update=true",
        "",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(reset["message"]
        .as_str()
        .unwrap()
        .contains("force_update=true"));
    assert_eq!(headers["x-request-id"], "configuration-reset-test");
    assert_eq!(reset["request_id"], "configuration-reset-test");
    assert_eq!(reset["path"], "/api/v1/config/reset");
    assert_eq!(reset["method"], "POST");
}

#[tokio::test]
async fn full_reset_restores_trusted_executable_defaults_and_cannot_repair_a_corrupt_store() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("config.json");
    let mut nondefault: Value =
        serde_json::from_str(include_str!("../contracts/configuration/v1/defaults.json")).unwrap();
    nondefault["perception"]["rust_live"]["executable_path"] =
        Value::String("/deployment/managed/live-binary".to_owned());
    std::fs::write(&path, serde_json::to_vec(&nondefault).unwrap()).unwrap();
    let gateway = gateway_with_config_path(Some(path.clone()));

    let (status, _, reset) =
        post_raw(gateway.clone(), "/api/v1/config/reset?confirm=true", "").await;
    assert_eq!(status, StatusCode::OK, "{reset}");
    let (status, _, current) = get(gateway, "/api/v1/config").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        current["values"]["perception"]["rust_live"]["executable_path"],
        ""
    );

    let corrupt_path = directory.path().join("corrupt.json");
    let corrupt_contents = b"not JSON";
    std::fs::write(&corrupt_path, corrupt_contents).unwrap();
    let corrupt_gateway = gateway_with_config_path(Some(corrupt_path.clone()));
    let (status, _, error) =
        post_raw(corrupt_gateway, "/api/v1/config/reset?confirm=true", "").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(error["error"], "SYS_002");
    assert_eq!(std::fs::read(corrupt_path).unwrap(), corrupt_contents);
}

#[tokio::test]
async fn configuration_failures_are_sanitized_and_echo_request_ids() {
    let missing_authority = gateway_with_config_path(None);
    let (status, headers, unavailable) = get(missing_authority, "/api/v1/config").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(unavailable["error"], "SYS_002");
    assert_eq!(unavailable["message"], "Service unavailable");
    assert!(headers.contains_key("x-request-id"));

    let missing_authority = gateway_with_config_path(None);
    let (status, _, unavailable) = put_raw(
        missing_authority,
        "/api/v1/config",
        r#"{"metadata":{"version":"1.0.1"}}"#,
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(unavailable["error"], "SYS_002");
    assert_eq!(unavailable["message"], "Service unavailable");

    let missing_authority = gateway_with_config_path(None);
    let (status, _, unavailable) =
        post_raw(missing_authority, "/api/v1/config/reset?confirm=true", "").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(unavailable["error"], "SYS_002");
    assert_eq!(unavailable["message"], "Service unavailable");

    let directory = tempdir().unwrap();
    let path = directory.path().join("config.json");
    std::fs::write(&path, b"not JSON").unwrap();
    let corrupt_store = gateway_with_config_path(Some(path));
    let (status, _, failed) = get(corrupt_store, "/api/v1/config").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(failed["message"], "Service unavailable");
    assert!(failed["details"].is_null());
}

#[tokio::test]
async fn slow_updates_share_the_bounded_configuration_capacity_with_reads() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("config.json");
    let lock_path = directory.path().join(".config.json.lock");
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .unwrap();
    lock_file.lock_exclusive().unwrap();

    let config = ServerConfig {
        configuration_path: Some(path),
        maximum_configuration_requests: 1,
        ..ServerConfig::default()
    };
    let gateway = Gateway::new(config).unwrap();
    let first_gateway = gateway.clone();
    let first = tokio::spawn(async move {
        put_raw(
            first_gateway,
            "/api/v1/config",
            r#"{"metadata":{"version":"1.0.1"}}"#,
        )
        .await
    });

    // Let the update acquire its sole configuration slot and block behind the
    // deliberately held store lock before submitting a read.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let (status, _, overloaded) = get(gateway, "/api/v1/config").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(overloaded["error"], "SYS_002");

    FileExt::unlock(&lock_file).unwrap();
    let (first_status, _, _) = first.await.unwrap();
    assert_eq!(first_status, StatusCode::OK);
}

#[tokio::test]
async fn slow_resets_share_the_bounded_configuration_capacity_with_reads() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("config.json");
    let lock_path = directory.path().join(".config.json.lock");
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .unwrap();
    lock_file.lock_exclusive().unwrap();

    let config = ServerConfig {
        configuration_path: Some(path),
        maximum_configuration_requests: 1,
        ..ServerConfig::default()
    };
    let gateway = Gateway::new(config).unwrap();
    let first_gateway = gateway.clone();
    let first = tokio::spawn(async move {
        post_raw(first_gateway, "/api/v1/config/reset?confirm=true", "").await
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let (status, _, overloaded) = get(gateway, "/api/v1/config").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(overloaded["error"], "SYS_002");

    FileExt::unlock(&lock_file).unwrap();
    let (first_status, _, _) = first.await.unwrap();
    assert_eq!(first_status, StatusCode::OK);
}

#[test]
fn gateway_bounds_filesystem_backed_configuration_work() {
    let config = ServerConfig {
        maximum_configuration_requests: 0,
        ..ServerConfig::default()
    };
    assert!(config
        .validate()
        .unwrap_err()
        .contains("maximum configuration requests"));

    let config = ServerConfig {
        maximum_configuration_requests: 257,
        ..ServerConfig::default()
    };
    assert!(config
        .validate()
        .unwrap_err()
        .contains("maximum configuration requests"));
}

#[tokio::test]
async fn json_import_preserves_validation_only_and_import_response_contracts() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("config.json");
    let original = br#"{"metadata":{"application":"billiards-trainer","environment":"production","version":"1.0.0"}}"#;
    std::fs::write(&path, original).unwrap();
    let gateway = gateway_with_config_path(Some(path.clone()));
    let upload = br#"{"_metadata":{"exported_by":"api"},"metadata":{"version":"1.0.1"}}"#;

    let (status, _, validation) = post_multipart(
        gateway.clone(),
        "/api/v1/config/import?validate_only=true",
        "game.json",
        upload,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{validation}");
    assert_eq!(validation["success"], true);
    assert_eq!(validation["data"]["is_valid"], true);
    assert_eq!(
        validation["data"]["sections_found"],
        serde_json::json!(["metadata"])
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);

    let (status, headers, imported) = post_multipart(
        gateway.clone(),
        "/api/v1/config/import",
        "game.json",
        upload,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{imported}");
    assert!(headers.contains_key("x-request-id"));
    assert_eq!(imported["data"]["filename"], "game.json");
    assert_eq!(imported["data"]["format"], ".json");
    assert_eq!(imported["data"]["merge_strategy"], "replace");
    assert_eq!(
        imported["data"]["sections_imported"],
        serde_json::json!(["metadata"])
    );
    let (_, _, configuration) = get(gateway, "/api/v1/config").await;
    assert_eq!(configuration["values"]["metadata"]["version"], "1.0.1");
    assert!(configuration["values"].get("_metadata").is_none());
}

#[tokio::test]
async fn yaml_import_and_merge_strategy_are_supported() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));
    let yaml = b"vision:\n  camera:\n    fps: 30\n";

    let (status, _, result) = post_multipart(
        gateway.clone(),
        "/api/v1/config/import?merge_strategy=merge",
        "camera.YAML",
        yaml,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{result}");
    assert_eq!(result["data"]["format"], ".yaml");
    assert_eq!(result["data"]["merge_strategy"], "merge");
    let (_, _, configuration) = get(gateway, "/api/v1/config").await;
    assert_eq!(configuration["values"]["vision"]["camera"]["fps"], 30);
}

#[tokio::test]
async fn import_rejects_invalid_extensions_documents_and_multipart_contracts() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    let (status, _, unsupported) =
        post_multipart(gateway.clone(), "/api/v1/config/import", "game.txt", b"{}").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(unsupported["error"], "VAL_001");

    let (status, _, malformed) = post_multipart(
        gateway.clone(),
        "/api/v1/config/import",
        "game.json",
        b"{broken",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(malformed["message"]
        .as_str()
        .unwrap()
        .contains("File parse error"));

    let (status, _, non_object) =
        post_multipart(gateway.clone(), "/api/v1/config/import", "game.json", b"[]").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(non_object["message"]
        .as_str()
        .unwrap()
        .contains("must be a JSON/YAML object"));

    let boundary = "configuration-audit-boundary";
    let response = gateway
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/config/import")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(format!("--{boundary}--\r\n")))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn import_requires_force_for_warnings_and_never_changes_deployment_executables() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));
    let warning = br#"{"vision":{"camera":{"fps":90}}}"#;

    let (status, _, warning_result) = post_multipart(
        gateway.clone(),
        "/api/v1/config/import?merge_strategy=merge",
        "game.json",
        warning,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(warning_result["message"]
        .as_str()
        .unwrap()
        .contains("force_update=true"));

    let (status, _, forced) = post_multipart(
        gateway.clone(),
        "/api/v1/config/import?merge_strategy=merge&force_update=true",
        "game.json",
        warning,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{forced}");
    assert_eq!(forced["data"]["warnings"].as_array().unwrap().len(), 1);

    let executable = br#"{"perception":{"rust_live":{"executable_path":"/tmp/untrusted"}}}"#;
    let (status, _, rejected) = post_multipart(
        gateway,
        "/api/v1/config/import?force_update=true",
        "game.json",
        executable,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(rejected["message"]
        .as_str()
        .unwrap()
        .contains("deployment-managed"));
}

#[tokio::test]
async fn exports_report_exact_serialized_size_and_checksum_for_json_and_yaml() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));

    let (status, _, json_export) = get(
        gateway.clone(),
        "/api/v1/config/export?sections=vision&include_metadata=false",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json_export}");
    assert_eq!(json_export["format"], "json");
    assert_eq!(json_export["data"].as_object().unwrap().len(), 1);
    let json_serialized = serde_json::to_string_pretty(&json_export["data"]).unwrap();
    assert_eq!(json_export["size"], json_serialized.len());
    assert_eq!(
        json_export["checksum"],
        format!("{:x}", Sha256::digest(json_serialized.as_bytes()))
    );

    let (status, _, yaml_export) = get(
        gateway.clone(),
        "/api/v1/config/export?format=yaml&sections=vision&sections=metadata",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{yaml_export}");
    assert_eq!(yaml_export["format"], "yaml");
    assert_eq!(yaml_export["data"]["_metadata"]["exported_by"], "api");
    assert_eq!(
        yaml_export["data"]["_metadata"]["sections"],
        serde_json::json!(["metadata", "vision"])
    );
    let yaml_serialized = serde_yaml_ng::to_string(&yaml_export["data"]).unwrap();
    assert_eq!(yaml_export["size"], yaml_serialized.len());
    assert_eq!(
        yaml_export["checksum"],
        format!("{:x}", Sha256::digest(yaml_serialized.as_bytes()))
    );

    let (status, _, unknown) = get(gateway, "/api/v1/config/export?sections=missing").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(unknown["error"], "VAL_001");
}

#[tokio::test]
async fn download_returns_bounded_serialized_attachment_without_a_temporary_file() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));
    let (status, headers, bytes) = get_raw(
        gateway,
        "/api/v1/config/export/download?format=json&sections=metadata",
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["content-type"], "application/octet-stream");
    assert_eq!(headers["content-length"], bytes.len().to_string());
    let disposition = headers["content-disposition"].to_str().unwrap();
    assert!(disposition.starts_with("attachment; filename=\"billiards_config_"));
    assert!(disposition.ends_with(".json\""));
    let data: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(data["metadata"].is_object());
    assert_eq!(data["_metadata"]["exported_by"], "api");
}

#[tokio::test]
async fn multipart_uploads_are_limited_by_the_configuration_authority_body_limit() {
    let directory = tempdir().unwrap();
    let gateway = gateway_with_config_path(Some(directory.path().join("config.json")));
    let oversized = vec![b' '; 1_100_000];
    let (status, _, response) =
        post_multipart(gateway, "/api/v1/config/import", "large.json", &oversized).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
}

#[tokio::test]
async fn slow_imports_share_the_bounded_configuration_capacity_with_reads() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("config.json");
    let lock_path = directory.path().join(".config.json.lock");
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .unwrap();
    lock_file.lock_exclusive().unwrap();

    let config = ServerConfig {
        configuration_path: Some(path),
        maximum_configuration_requests: 1,
        ..ServerConfig::default()
    };
    let gateway = Gateway::new(config).unwrap();
    let import_gateway = gateway.clone();
    let import = tokio::spawn(async move {
        post_multipart(
            import_gateway,
            "/api/v1/config/import",
            "game.json",
            br#"{"metadata":{"version":"1.0.1"}}"#,
        )
        .await
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let (status, _, overloaded) = get(gateway, "/api/v1/config").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(overloaded["error"], "SYS_002");

    FileExt::unlock(&lock_file).unwrap();
    let (import_status, _, result) = import.await.unwrap();
    assert_eq!(import_status, StatusCode::OK, "{result}");
}
