//! Process contract for the retained game-session authority command.

use std::io::Write;
use std::process::{Command, Output, Stdio};

use serde_json::{json, Value};
use tempfile::tempdir;

fn call(store: &std::path::Path, request: Value) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_billiards-perception-game-session"))
        .arg("--store")
        .arg(store)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("game-session authority process starts");
    let request = serde_json::to_vec(&request).expect("request serializes");
    child
        .stdin
        .take()
        .expect("child stdin is piped")
        .write_all(&request)
        .expect("request reaches the child");
    child.wait_with_output().expect("child process exits")
}

fn response(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).expect("child stdout is one JSON response")
}

#[test]
fn command_snapshots_resets_persists_and_rejects_a_stale_revision() {
    let directory = tempdir().expect("temporary directory");
    let store = directory.path().join("game-session.json");

    let initial = call(
        &store,
        json!({"protocol_version": 1, "operation": "snapshot"}),
    );
    assert!(initial.status.success());
    let initial = response(&initial);
    assert_eq!(initial["ok"], true);
    assert_eq!(initial["session"]["revision"], 0);
    assert_eq!(initial["session"]["game_type"], "practice");
    assert_eq!(initial["session"]["scores"], json!([0, 0]));

    let reset = call(
        &store,
        json!({
            "protocol_version": 1,
            "operation": "reset",
            "expected_revision": 0,
            "game_type": "9ball"
        }),
    );
    assert!(reset.status.success());
    let reset = response(&reset);
    assert_eq!(reset["operation"], "reset");
    assert_eq!(reset["session"]["revision"], 1);
    assert_eq!(reset["session"]["game_type"], "9ball");

    let stale_reset = call(
        &store,
        json!({
            "protocol_version": 1,
            "operation": "reset",
            "expected_revision": 0,
            "game_type": "8ball"
        }),
    );
    assert!(!stale_reset.status.success());
    let stale_reset = response(&stale_reset);
    assert_eq!(stale_reset["error"]["code"], "stale_revision");
    assert_eq!(stale_reset["error"]["current_revision"], 1);

    let current = call(
        &store,
        json!({"protocol_version": 1, "operation": "snapshot"}),
    );
    assert!(current.status.success());
    let current = response(&current);
    assert_eq!(current["session"]["revision"], 1);
    assert_eq!(current["session"]["game_type"], "9ball");
    assert!(current["session"].get("balls").is_none());
}
