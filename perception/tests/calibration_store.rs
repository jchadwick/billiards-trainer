use std::fs::{self, OpenOptions};
use std::process::Command;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use billiards_perception::calibration::{
    CalibrationError, ImagePointPx, TableCalibrationSolverInput,
};
use billiards_perception::calibration_store::{
    CalibrationCaptureContextV1, CalibrationStore, CalibrationStoreError, CalibrationStoreSnapshot,
};
use billiards_perception::table_visibility::{
    FeatureLayout, NormalizedPoint, RgbFeature, TableRegion, TableVisibilityConfig,
    TableVisibilityReference,
};
use billiards_perception::ImageSpace;
use fs2::FileExt;

const LOCK_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_STORE_FILE_BYTES: usize = 64 * 1024;

fn fixture_input() -> TableCalibrationSolverInput {
    TableCalibrationSolverInput {
        // The store must replace this caller-provided value with its own
        // compare-and-swap revision before solving.
        calibration_revision: 987,
        camera_profile_id: "test-webcam-1280x720".to_owned(),
        rectification_revision: 4,
        image_space: ImageSpace::Rectified,
        source_width_px: 1280,
        source_height_px: 720,
        table_width_mm: 2000.0,
        table_height_mm: 1000.0,
        corners_px: [
            ImagePointPx {
                x_px: 160.0,
                y_px: 90.0,
            },
            ImagePointPx {
                x_px: 1120.0,
                y_px: 90.0,
            },
            ImagePointPx {
                x_px: 1120.0,
                y_px: 630.0,
            },
            ImagePointPx {
                x_px: 160.0,
                y_px: 630.0,
            },
        ],
    }
}

fn store(path: &std::path::Path) -> CalibrationStore {
    CalibrationStore::new(path, LOCK_TIMEOUT, MAX_STORE_FILE_BYTES).expect("valid file path")
}

fn write_store_json(path: &std::path::Path, contents: &str) {
    fs::write(path, contents).expect("test store fixture should be writable");
}

fn visibility_reference(width: u32, height: u32) -> TableVisibilityReference {
    let config = TableVisibilityConfig::default();
    let cloth_count = usize::from(config.cloth_grid_width) * usize::from(config.cloth_grid_height);
    let rail_count = 4
        * usize::from(config.rail_profiles_per_side)
        * usize::from(config.rail_profile_depth_samples);
    TableVisibilityReference {
        schema_version: 1,
        source_width: width,
        source_height: height,
        table_region: TableRegion::new([
            NormalizedPoint { x: 0.1, y: 0.1 },
            NormalizedPoint { x: 0.9, y: 0.1 },
            NormalizedPoint { x: 0.9, y: 0.9 },
            NormalizedPoint { x: 0.1, y: 0.9 },
        ]),
        feature_layout: FeatureLayout {
            cloth_grid_width: config.cloth_grid_width,
            cloth_grid_height: config.cloth_grid_height,
            cloth_inset_fraction: config.cloth_inset_fraction,
            rail_profiles_per_side: config.rail_profiles_per_side,
            rail_profile_depth_samples: config.rail_profile_depth_samples,
            rail_profile_half_width_fraction: config.rail_profile_half_width_fraction,
        },
        cloth_features: vec![
            RgbFeature {
                red: 42,
                green: 132,
                blue: 54,
            };
            cloth_count
        ],
        rail_features: vec![
            RgbFeature {
                red: 106,
                green: 72,
                blue: 42,
            };
            rail_count
        ],
    }
}

#[test]
fn cross_process_lock_holder_test_entry() {
    let Ok(lock_path) = std::env::var("BILLIARDS_TEST_CALIBRATION_LOCK_PATH") else {
        return;
    };
    let ready_path = std::env::var("BILLIARDS_TEST_CALIBRATION_LOCK_READY")
        .expect("parent sets the lock-ready path");
    let lock_file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .expect("child lock file should open");
    lock_file
        .try_lock_exclusive()
        .expect("child process should acquire the lock");
    fs::write(ready_path, b"ready").expect("child should signal lock readiness");
    thread::sleep(Duration::from_millis(300));
}

#[test]
fn missing_path_has_no_revision_and_first_save_round_trips_with_revision_one() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("table-calibration.json");
    let store = store(&path);

    assert_eq!(
        store.load().unwrap(),
        CalibrationStoreSnapshot {
            revision: None,
            calibration: None,
            binding_context: None,
            visibility_reference: None,
        }
    );

    let saved = store.save(None, fixture_input()).unwrap();
    assert_eq!(saved.revision, Some(1));
    let calibration = saved.calibration.as_ref().unwrap();
    assert_eq!(calibration.calibration_revision, 1);
    assert_ne!(calibration.calibration_revision, 987);
    assert_eq!(store.load().unwrap(), saved);
}

#[test]
fn capture_binding_is_persisted_and_must_match_the_solved_calibration() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("table-calibration.json");
    let store = store(&path);
    let context = CalibrationCaptureContextV1 {
        engine_instance_id: "engine-a".to_owned(),
        capture_session_id: "session-a".to_owned(),
        source_frame_id: "frame-a".to_owned(),
        configuration_revision: 7,
        camera_profile_id: "test-webcam-1280x720".to_owned(),
        image_space: ImageSpace::Rectified,
        source_width_px: 1280,
        source_height_px: 720,
        calibration_revision: 0,
        rectification_revision: 4,
    };
    let reference = visibility_reference(context.source_width_px, context.source_height_px);
    let saved = store
        .save_with_context_and_visibility(
            None,
            fixture_input(),
            Some(context.clone()),
            Some(reference.clone()),
        )
        .unwrap();
    assert_eq!(saved.binding_context, Some(context));
    assert_eq!(saved.visibility_reference, Some(reference.clone()));
    let serialized: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(serialized["schema_version"], 3);
    assert_eq!(serialized["visibility_reference"]["schema_version"], 1);
    assert_eq!(store.load().unwrap(), saved);

    let mut mismatched = CalibrationCaptureContextV1 {
        camera_profile_id: "other-camera".to_owned(),
        ..saved.binding_context.clone().unwrap()
    };
    mismatched.source_frame_id = "frame-b".to_owned();
    let before = fs::read(&path).unwrap();
    assert!(matches!(
        store.save_with_context(saved.revision, fixture_input(), Some(mismatched)),
        Err(CalibrationStoreError::InvalidCaptureContext)
    ));
    assert_eq!(fs::read(&path).unwrap(), before);

    let mut malformed_reference = reference;
    malformed_reference.source_width += 1;
    assert!(matches!(
        store.save_with_context_and_visibility(
            saved.revision,
            fixture_input(),
            Some(saved.binding_context.clone().unwrap()),
            Some(malformed_reference),
        ),
        Err(CalibrationStoreError::InvalidVisibilityReference)
    ));
    assert_eq!(store.load().unwrap(), saved);
}

#[test]
fn read_limit_is_supplied_by_the_caller_and_zero_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("table-calibration.json");

    assert!(matches!(
        CalibrationStore::new(&path, LOCK_TIMEOUT, 0),
        Err(CalibrationStoreError::InvalidMaximumFileBytes)
    ));

    fs::write(&path, b"12345").unwrap();
    let limited = CalibrationStore::new(&path, LOCK_TIMEOUT, 4).unwrap();
    assert!(matches!(
        limited.load(),
        Err(CalibrationStoreError::StoreFileLimitExceeded {
            maximum: 4,
            actual: 5
        })
    ));

    let unwritten_path = directory.path().join("unwritten-calibration.json");
    let too_small = CalibrationStore::new(&unwritten_path, LOCK_TIMEOUT, 4).unwrap();
    assert!(matches!(
        too_small.save(None, fixture_input()),
        Err(CalibrationStoreError::StoreFileLimitExceeded { maximum: 4, .. })
    ));
    assert!(!unwritten_path.exists());
}

#[test]
fn clear_writes_tombstone_and_save_and_repeated_clear_keep_revisions_monotonic() {
    let directory = tempfile::tempdir().unwrap();
    let store = store(&directory.path().join("table-calibration.json"));

    let first = store.save(None, fixture_input()).unwrap();
    let cleared = store.clear(first.revision).unwrap();
    assert_eq!(cleared.revision, Some(2));
    assert!(cleared.calibration.is_none());
    assert_eq!(store.load().unwrap(), cleared);

    let restored = store.save(cleared.revision, fixture_input()).unwrap();
    assert_eq!(restored.revision, Some(3));
    assert_eq!(
        restored.calibration.as_ref().unwrap().calibration_revision,
        3
    );

    let cleared_again = store.clear(restored.revision).unwrap();
    assert_eq!(cleared_again.revision, Some(4));
    assert!(cleared_again.calibration.is_none());
}

#[test]
fn stale_expected_revision_is_rejected_without_changing_stored_state() {
    let directory = tempfile::tempdir().unwrap();
    let store = store(&directory.path().join("table-calibration.json"));
    let saved = store.save(None, fixture_input()).unwrap();
    let original_bytes = fs::read(directory.path().join("table-calibration.json")).unwrap();

    assert!(matches!(
        store.save(None, fixture_input()),
        Err(CalibrationStoreError::StaleExpectedRevision {
            expected: None,
            actual: Some(1)
        })
    ));
    assert_eq!(store.load().unwrap(), saved);
    assert_eq!(
        fs::read(directory.path().join("table-calibration.json")).unwrap(),
        original_bytes
    );
}

#[test]
fn simultaneous_first_writers_have_one_winner() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("table-calibration.json");
    let barrier = Arc::new(Barrier::new(3));
    let writers = (0..2)
        .map(|_| {
            let path = path.clone();
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let store = store(&path);
                barrier.wait();
                store.save(None, fixture_input())
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();

    let results = writers
        .into_iter()
        .map(|writer| writer.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(
                result,
                Err(CalibrationStoreError::StaleExpectedRevision {
                    expected: None,
                    actual: Some(1)
                })
            ))
            .count(),
        1
    );
    assert_eq!(store(&path).load().unwrap().revision, Some(1));
}

#[test]
fn lock_timeout_is_distinct_and_does_not_change_existing_record() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("table-calibration.json");
    let calibration_store = store(&path);
    let saved = calibration_store.save(None, fixture_input()).unwrap();
    let original_bytes = fs::read(&path).unwrap();

    let lock_path = directory.path().join("table-calibration.json.lock");
    let lock_file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .unwrap();
    lock_file.try_lock_exclusive().unwrap();

    let impatient_store =
        CalibrationStore::new(&path, Duration::from_millis(20), MAX_STORE_FILE_BYTES).unwrap();
    assert!(matches!(
        impatient_store.load(),
        Err(CalibrationStoreError::LockTimeout)
    ));
    assert_eq!(fs::read(&path).unwrap(), original_bytes);
    FileExt::unlock(&lock_file).unwrap();
    assert_eq!(calibration_store.load().unwrap(), saved);
}

#[test]
fn advisory_lock_serializes_independent_processes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("table-calibration.json");
    let lock_path = directory.path().join("table-calibration.json.lock");
    let ready_path = directory.path().join("lock-ready");
    let child = std::env::current_exe().unwrap();
    let mut holder = Command::new(child)
        .arg("--exact")
        .arg("cross_process_lock_holder_test_entry")
        .arg("--nocapture")
        .env("BILLIARDS_TEST_CALIBRATION_LOCK_PATH", &lock_path)
        .env("BILLIARDS_TEST_CALIBRATION_LOCK_READY", &ready_path)
        .spawn()
        .expect("lock holder process should start");

    let deadline = Instant::now() + Duration::from_secs(2);
    while !ready_path.exists() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        ready_path.exists(),
        "child did not acquire the advisory lock"
    );

    let impatient_store =
        CalibrationStore::new(&path, Duration::from_millis(30), MAX_STORE_FILE_BYTES).unwrap();
    assert!(matches!(
        impatient_store.load(),
        Err(CalibrationStoreError::LockTimeout)
    ));
    assert!(holder.wait().unwrap().success());
    assert_eq!(store(&path).load().unwrap().revision, None);
}

#[test]
fn malformed_and_unsupported_store_files_are_explicit_errors() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("table-calibration.json");
    let store = store(&path);

    write_store_json(&path, "{ definitely not json");
    assert!(matches!(
        store.load(),
        Err(CalibrationStoreError::MalformedStore(_))
    ));

    write_store_json(
        &path,
        r#"{"schema_version":99,"store_revision":1,"calibration":null}"#,
    );
    assert!(matches!(
        store.load(),
        Err(CalibrationStoreError::UnsupportedStoreSchemaVersion(99))
    ));

    write_store_json(&path, r#"{"schema_version":1,"store_revision":1}"#);
    assert!(matches!(
        store.load(),
        Err(CalibrationStoreError::MalformedStore(_))
    ));
}

#[test]
fn loaded_bundle_validation_and_revision_agreement_are_enforced() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("table-calibration.json");
    let store = store(&path);
    store.save(None, fixture_input()).unwrap();
    let mut envelope: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();

    envelope["calibration"]["calibration_revision"] = serde_json::json!(2);
    write_store_json(&path, &envelope.to_string());
    assert!(matches!(
        store.load(),
        Err(CalibrationStoreError::CalibrationRevisionMismatch {
            store_revision: 1,
            calibration_revision: 2
        })
    ));

    envelope["calibration"]["calibration_revision"] = serde_json::json!(1);
    envelope["calibration"]["pixel_to_table_mm"] =
        serde_json::json!([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    write_store_json(&path, &envelope.to_string());
    assert!(matches!(
        store.load(),
        Err(CalibrationStoreError::InvalidCalibration(
            CalibrationError::InvalidHomography
        ))
    ));
}

#[test]
fn invalid_solve_does_not_replace_the_last_valid_record() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("table-calibration.json");
    let store = store(&path);
    let saved = store.save(None, fixture_input()).unwrap();
    let original_bytes = fs::read(&path).unwrap();
    let mut invalid_input = fixture_input();
    invalid_input.corners_px[3] = invalid_input.corners_px[2];

    assert!(matches!(
        store.save(saved.revision, invalid_input),
        Err(CalibrationStoreError::InvalidCalibration(
            CalibrationError::InvalidCalibrationCorners
        ))
    ));
    assert_eq!(fs::read(&path).unwrap(), original_bytes);
    assert_eq!(store.load().unwrap(), saved);
}
