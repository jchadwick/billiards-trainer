//! Rust gateway routes for session metadata alongside the authoritative scene.
//!
//! Session persistence is optional metadata. Reads degrade to the current scene
//! when the store is unavailable, while mutations fail closed. Blocking file
//! locks and durable writes always run on Tokio's blocking pool and consume a
//! bounded shared gateway permit.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{SecondsFormat, Utc};
use serde_json::{json, Value};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::game_session::{
    GameSessionSnapshotV1, GameSessionStore, GameSessionStoreError, GameType,
};

type SceneReader = Arc<dyn Fn() -> Option<Value> + Send + Sync + 'static>;

#[derive(Clone)]
pub(crate) struct GameSessionHttpState {
    store: Option<GameSessionStore>,
    scene_reader: SceneReader,
    request_slots: Arc<Semaphore>,
}

impl GameSessionHttpState {
    pub(crate) fn new(
        store: Option<GameSessionStore>,
        scene_reader: impl Fn() -> Option<Value> + Send + Sync + 'static,
        request_slots: Arc<Semaphore>,
    ) -> Self {
        Self {
            store,
            scene_reader: Arc::new(scene_reader),
            request_slots,
        }
    }

    pub(crate) fn router(self) -> Router {
        Router::new()
            .route("/api/v1/game/state", get(get_game_state))
            .route("/api/v1/game/reset", post(reset_game_state))
            .with_state(self)
    }
}

async fn get_game_state(State(state): State<GameSessionHttpState>) -> Response {
    let Some(mut scene) = (state.scene_reader)() else {
        return not_ready_response();
    };
    let Some(store) = state.store.clone() else {
        return Json(scene).into_response();
    };
    let Ok(_permit) = Arc::clone(&state.request_slots).try_acquire_owned() else {
        return Json(scene).into_response();
    };

    // Localization is the primary read path. Corrupt, missing or contended
    // optional session metadata cannot suppress a valid ball snapshot.
    if let Ok(Ok(snapshot)) = run_blocking_with_permit(_permit, move || store.load()).await {
        overlay_session(&mut scene, &snapshot);
    }
    Json(scene).into_response()
}

async fn reset_game_state(
    State(state): State<GameSessionHttpState>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let game_type = match parse_game_type(query.get("game_type")) {
        Ok(game_type) => game_type,
        Err(message) => return query_error(message),
    };
    match parse_boolean_query(&query, "preserve_table", true) {
        Ok(true) => {}
        Ok(false) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "detail": "Rust game-session reset does not alter table calibration; use the calibration controls for table changes"
                })),
            )
                .into_response()
        }
        Err(message) => return query_error(message),
    }
    if let Err(message) = parse_boolean_query(&query, "create_backup", true) {
        return query_error(message);
    }

    // Validate that an actual measured state exists before mutating session
    // metadata. Never create a rack or clear the scene as a reset side effect.
    if (state.scene_reader)().is_none() {
        return reset_scene_unavailable(false);
    }
    let Some(store) = state.store.clone() else {
        return service_unavailable("Rust game-session authority is unavailable");
    };
    let Ok(_permit) = Arc::clone(&state.request_slots).try_acquire_owned() else {
        return service_unavailable("Rust game-session capacity is unavailable");
    };

    let reset_result = run_blocking_with_permit(_permit, move || {
        let current = store.load()?;
        store.reset(current.revision, game_type)
    })
    .await;
    let session = match reset_result {
        Ok(Ok(session)) => session,
        Ok(Err(error)) => return store_error_response(error),
        Err(_) => return service_unavailable("Rust game-session authority is unavailable"),
    };

    // Re-read after the bounded filesystem operation so reset returns the
    // newest scene, not the location snapshot observed before the write.
    let Some(mut new_state) = (state.scene_reader)() else {
        return reset_scene_unavailable(true);
    };
    overlay_session(&mut new_state, &session);
    Json(json!({
        "success": true,
        "new_state": new_state,
        "backup_created": false,
        "reset_at": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, false),
    }))
    .into_response()
}

async fn run_blocking_with_permit<T: Send + 'static>(
    permit: OwnedSemaphorePermit,
    operation: impl FnOnce() -> T + Send + 'static,
) -> Result<T, tokio::task::JoinError> {
    tokio::task::spawn_blocking(move || {
        // Keep the request-capacity slot for the actual blocking operation,
        // even if its async HTTP caller disconnects after it has started.
        let _permit = permit;
        operation()
    })
    .await
}

fn overlay_session(scene: &mut Value, snapshot: &GameSessionSnapshotV1) {
    if let Some(object) = scene.as_object_mut() {
        object.insert("game_type".into(), json!(snapshot.game_type.as_str()));
        object.insert(
            "session".into(),
            serde_json::to_value(snapshot).unwrap_or(Value::Null),
        );
    }
}

fn parse_game_type(value: Option<&String>) -> Result<GameType, String> {
    match value.map(String::as_str).unwrap_or("practice") {
        "practice" => Ok(GameType::Practice),
        "8ball" => Ok(GameType::EightBall),
        "9ball" => Ok(GameType::NineBall),
        "straight" => Ok(GameType::Straight),
        _ => Err("game_type must be one of practice, 8ball, 9ball, straight".into()),
    }
}

fn parse_boolean_query(
    query: &HashMap<String, String>,
    key: &str,
    default: bool,
) -> Result<bool, String> {
    let Some(value) = query.get(key) else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "t" | "y" => Ok(true),
        "0" | "false" | "no" | "off" | "f" | "n" => Ok(false),
        _ => Err(format!("{key} must be a boolean")),
    }
}

fn query_error(detail: String) -> Response {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({"detail": detail})),
    )
        .into_response()
}

fn service_unavailable(detail: &'static str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({"detail": detail})),
    )
        .into_response()
}

fn store_error_response(error: GameSessionStoreError) -> Response {
    match error {
        GameSessionStoreError::StaleExpectedRevision { .. } => (
            StatusCode::CONFLICT,
            Json(json!({
                "detail": "Game session changed during reset; read current state and retry"
            })),
        )
            .into_response(),
        _ => service_unavailable("Rust game-session authority is unavailable"),
    }
}

fn not_ready_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "status": "not_ready",
            "error": "localization_unavailable",
            "message": "No localized scene is available yet.",
            "perception_source": "rust_scene",
        })),
    )
        .into_response()
}

fn reset_scene_unavailable(during_reset: bool) -> Response {
    let detail = if during_reset {
        "Current Rust scene state became unavailable during reset"
    } else {
        "Current Rust scene state is unavailable for reset response"
    };
    service_unavailable(detail)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::{json, Value};
    use tempfile::TempDir;
    use tokio::sync::Semaphore;
    use tower::ServiceExt;

    use crate::game_session::{GameSessionStore, GameSessionStoreError, GameType};

    use super::GameSessionHttpState;

    fn scene_value() -> Value {
        json!({
            "perception_source": "rust_scene",
            "frame_number": 7,
            "balls": [{"id": "track-1", "position": {"x": 10.0, "y": 20.0}}],
            "game_type": null,
            "session": null,
        })
    }

    fn scene() -> Option<Value> {
        Some(scene_value())
    }

    fn app(
        store: Option<GameSessionStore>,
        scene: impl Fn() -> Option<Value> + Send + Sync + 'static,
    ) -> axum::Router {
        GameSessionHttpState::new(store, scene, Arc::new(Semaphore::new(2))).router()
    }

    async fn request(app: axum::Router, method: &str, uri: &str) -> (StatusCode, Value) {
        let response = app
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    fn store(directory: &TempDir) -> GameSessionStore {
        GameSessionStore::new(
            directory.path().join("session.json"),
            std::time::Duration::from_millis(100),
            4096,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn state_overlays_rust_session_without_changing_latest_localization() {
        let directory = TempDir::new().unwrap();
        let state = app(Some(store(&directory)), scene);
        let (status, response) = request(state, "GET", "/api/v1/game/state").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(response["game_type"], "practice");
        assert_eq!(response["session"]["revision"], 0);
        assert_eq!(response["balls"][0]["position"]["x"], 10.0);
        assert_eq!(response["frame_number"], 7);
    }

    #[tokio::test]
    async fn state_keeps_scene_when_session_store_is_missing_or_corrupt() {
        let no_store = app(None, scene);
        let (status, response) = request(no_store, "GET", "/api/v1/game/state").await;
        assert_eq!(status, StatusCode::OK);
        assert!(response["session"].is_null());
        assert_eq!(response["balls"][0]["position"]["x"], 10.0);

        let directory = TempDir::new().unwrap();
        let store = store(&directory);
        std::fs::write(directory.path().join("session.json"), b"invalid").unwrap();
        let (status, response) =
            request(app(Some(store), scene), "GET", "/api/v1/game/state").await;
        assert_eq!(status, StatusCode::OK);
        assert!(response["session"].is_null());
        assert_eq!(response["frame_number"], 7);
    }

    #[tokio::test]
    async fn state_requires_an_actual_localized_scene() {
        let state = app(None, || None);
        let (status, response) = request(state, "GET", "/api/v1/game/state").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response["error"], "localization_unavailable");
    }

    #[tokio::test]
    async fn reset_commits_session_metadata_and_returns_the_latest_scene() {
        let directory = TempDir::new().unwrap();
        let store = store(&directory);
        let scenes = Arc::new(Mutex::new(0_u64));
        let read_scenes = Arc::clone(&scenes);
        let reader = move || {
            let mut sequence = read_scenes.lock().unwrap();
            *sequence += 1;
            let mut current = scene_value();
            current["frame_number"] = json!(*sequence);
            Some(current)
        };
        let (status, response) = request(
            app(Some(store.clone()), reader),
            "POST",
            "/api/v1/game/reset?game_type=9ball&create_backup=true",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(response["success"], true);
        assert_eq!(response["backup_created"], false);
        assert_eq!(response["new_state"]["frame_number"], 2);
        assert_eq!(response["new_state"]["session"]["game_type"], "9ball");
        assert_eq!(response["new_state"]["session"]["revision"], 1);
        assert_eq!(store.load().unwrap().game_type, GameType::NineBall);
    }

    #[tokio::test]
    async fn reset_rejects_table_mutation_and_invalid_game_types_before_writing() {
        let directory = TempDir::new().unwrap();
        let store = store(&directory);
        let (status, response) = request(
            app(Some(store.clone()), scene),
            "POST",
            "/api/v1/game/reset?preserve_table=false",
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(response["detail"].as_str().unwrap().contains("calibration"));
        assert!(!directory.path().join("session.json").exists());

        let (status, _) = request(
            app(Some(store.clone()), scene),
            "POST",
            "/api/v1/game/reset?game_type=pool",
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(store.load().unwrap().revision, 0);
    }

    #[tokio::test]
    async fn reset_fails_closed_without_scene_or_store() {
        let (status, response) = request(app(None, scene), "POST", "/api/v1/game/reset").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(response["detail"].as_str().unwrap().contains("authority"));

        let (status, response) = request(app(None, || None), "POST", "/api/v1/game/reset").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response["detail"],
            "Current Rust scene state is unavailable for reset response"
        );
    }

    #[tokio::test]
    async fn reset_reports_its_own_error_if_scene_disappears_after_commit() {
        let directory = TempDir::new().unwrap();
        let store = store(&directory);
        let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let scene_reads = Arc::clone(&reads);
        let reader = move || {
            if scene_reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                scene()
            } else {
                None
            }
        };
        let (status, response) = request(
            app(Some(store.clone()), reader),
            "POST",
            "/api/v1/game/reset",
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response["detail"],
            "Current Rust scene state became unavailable during reset"
        );
        assert_eq!(store.load().unwrap().revision, 1);
    }

    #[tokio::test]
    async fn reset_reads_current_revision_before_applying_the_requested_game_type() {
        let directory = TempDir::new().unwrap();
        let store = store(&directory);
        store.reset(0, GameType::Practice).unwrap();
        let concurrent_store = store.clone();
        let scenes = Arc::new(Mutex::new(0_u8));
        let scene_reader = move || {
            let mut calls = scenes.lock().unwrap();
            *calls += 1;
            if *calls == 1 {
                concurrent_store.reset(1, GameType::EightBall).unwrap();
            }
            scene()
        };
        // The external update is injected while checking scene availability,
        // before the bounded store operation reads its expected revision.
        let (status, response) = request(
            app(Some(store.clone()), scene_reader),
            "POST",
            "/api/v1/game/reset?game_type=straight",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(response["new_state"]["session"]["revision"], 3);
        assert_eq!(store.load().unwrap().game_type, GameType::Straight);
    }

    #[tokio::test]
    async fn stale_store_revision_maps_to_http_conflict() {
        let response = super::store_error_response(GameSessionStoreError::StaleExpectedRevision {
            expected: 2,
            actual: 3,
        });
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(body["detail"]
            .as_str()
            .unwrap()
            .contains("changed during reset"));
    }

    #[tokio::test]
    async fn reset_capacity_is_bounded_and_state_remains_best_effort() {
        let directory = TempDir::new().unwrap();
        let slots = Arc::new(Semaphore::new(1));
        let permit = Arc::clone(&slots).acquire_owned().await.unwrap();
        let state =
            GameSessionHttpState::new(Some(store(&directory)), scene, Arc::clone(&slots)).router();
        let (status, response) = request(state.clone(), "POST", "/api/v1/game/reset").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(response["detail"].as_str().unwrap().contains("capacity"));

        let (status, response) = request(state, "GET", "/api/v1/game/state").await;
        assert_eq!(status, StatusCode::OK);
        assert!(response["session"].is_null());
        drop(permit);
    }

    #[tokio::test]
    async fn cancelled_request_keeps_capacity_until_its_blocking_work_finishes() {
        let slots = Arc::new(Semaphore::new(1));
        let permit = Arc::clone(&slots).try_acquire_owned().unwrap();
        let (started, observed_started) = tokio::sync::oneshot::channel();
        let (release, wait_for_release) = std::sync::mpsc::channel();
        let operation = tokio::spawn(super::run_blocking_with_permit(permit, move || {
            started.send(()).unwrap();
            wait_for_release.recv().unwrap();
        }));
        observed_started.await.unwrap();
        operation.abort();
        assert_eq!(slots.available_permits(), 0);
        release.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while slots.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(slots.available_permits(), 1);
    }
}
