//! Hardware-free process-level coverage for the reviewed-label CLI workflow.
//!
//! This intentionally launches both shipped executables instead of calling
//! their library functions. It verifies argument parsing, file handoff,
//! artifact schemas, exit status, and the evaluator's actual report output.

#[cfg(unix)]
mod unix_cli_workflow {
    use serde_json::{json, Value};
    use sha2::Digest;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::process::{Command, Output};

    const RUST_TRACKED_SCENE: &str =
        include_str!("../contracts/v1/scene-rust-tracked-success.json");

    // Mirrors the compact CVAT 1.1 interpolation-task fixture used by the
    // importer unit tests: two reviewed frames, one removed ball, one ball
    // entering the sequence, and a cue that becomes occluded.
    const CVAT_XML: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><size>3</size><mode>interpolation</mode><original_size>
    <width>100</width><height>80</height>
  </original_size><labels><label><name>ball</name></label>
    <label><name>cue</name></label></labels></task></meta>
  <track id="ball-a" label="ball">
    <box frame="0" outside="0" occluded="0" keyframe="1" xtl="10" ytl="20" xbr="30" ybr="40"/>
    <box frame="2" outside="1" occluded="0" keyframe="1" xtl="10" ytl="20" xbr="30" ybr="40"/>
  </track>
  <track id="ball-b" label="ball">
    <box frame="2" outside="0" occluded="0" keyframe="1" xtl="40" ytl="30" xbr="60" ybr="50"/>
  </track>
  <track id="cue-a" label="cue">
    <polyline frame="0" outside="0" occluded="0" keyframe="1" points="4,5;80,50"/>
    <polyline frame="2" outside="0" occluded="1" keyframe="1"/>
  </track>
</annotations>"#;

    const FFPROBE_JSON: &str = r#"{"streams":[{"codec_type":"video","width":100,"height":80}],"frames":[{"best_effort_timestamp_time":"12.000"},{"best_effort_timestamp_time":"12.040"},{"best_effort_timestamp_time":"12.110"}]}"#;

    fn write_executable(path: &Path, contents: &str) {
        fs::write(path, contents).expect("write executable fixture");
        let mut permissions = fs::metadata(path)
            .expect("read fixture permissions")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).expect("make fixture executable");
    }

    fn successful_output(output: &Output, cli: &str) -> String {
        assert!(
            output.status.success(),
            "{cli} failed with {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        String::from_utf8(output.stdout.clone()).expect("CLI stdout is UTF-8")
    }

    fn scene_for_frame(
        source_frame_index: u64,
        transport_sequence: u64,
        ball_center: (f64, f64),
        cue_observed: bool,
    ) -> Value {
        let mut scene: Value = serde_json::from_str(RUST_TRACKED_SCENE).unwrap();
        let frame_id = format!("frame-{:04}", source_frame_index + 1);
        scene["source"]["frame_id"] = json!(frame_id);
        scene["source"]["width_px"] = json!(100);
        scene["source"]["height_px"] = json!(80);
        scene["transport_sequence"] = json!(transport_sequence);
        scene["snapshot_revision"] = json!(transport_sequence);

        let ball = &mut scene["balls"][0];
        ball["track_id"] = json!(format!("track-{source_frame_index}"));
        ball["last_observation"]["frame_id"] = json!(frame_id);
        ball["measurement"]["image_center_px"]["x"] = json!(ball_center.0);
        ball["measurement"]["image_center_px"]["y"] = json!(ball_center.1);
        ball["measurement"]["image_width_px"] = json!(100);
        ball["measurement"]["image_height_px"] = json!(80);

        let cue = &mut scene["cue"];
        cue["last_observation"]["frame_id"] = json!(frame_id);
        cue["image_width_px"] = json!(100);
        cue["image_height_px"] = json!(80);
        if cue_observed {
            cue["status"] = json!("observed");
            cue["image_center_px"] = json!({"x": 42.0, "y": 27.5});
        } else {
            cue["status"] = json!("not_observed");
            cue["image_center_px"] = Value::Null;
            cue["confidence"] = Value::Null;
        }
        scene
    }

    #[test]
    fn importer_cli_output_runs_through_evaluator_cli() {
        let directory = tempfile::tempdir().expect("temporary workflow directory");
        let root: &Path = directory.path();
        let video_path = root.join("capture.mp4");
        let xml_path = root.join("reviewed-cvat.xml");
        let reviewed_frames_path = root.join("reviewed-frames.json");
        let ffprobe_path = root.join("ffprobe-fixture");
        let annotations_path = root.join("temporal-annotations.json");
        let scene_path = root.join("rust-scenes.jsonl");
        let cue_geometry_path = root.join("cue-geometry.jsonl");
        let report_path = root.join("evaluation-report.json");

        // The CLIs bind annotations to the exact input bytes. The video is a
        // bounded synthetic identity fixture; ffprobe is replaced with a
        // deterministic executable so this test does not need a codec install.
        fs::write(&video_path, b"deterministic synthetic video identity")
            .expect("write synthetic video bytes");
        fs::write(&xml_path, CVAT_XML).expect("write CVAT task fixture");
        fs::write(&reviewed_frames_path, "[0,2]").expect("write reviewed frames");
        write_executable(
            &ffprobe_path,
            &format!("#!/bin/sh\nprintf '%s\\n' '{}'\n", FFPROBE_JSON),
        );

        let video_hash = format!("{:x}", sha2::Sha256::digest(fs::read(&video_path).unwrap()));
        let importer_output = Command::new(env!("CARGO_BIN_EXE_billiards-temporal-import-cvat"))
            .arg("--video")
            .arg(&video_path)
            .arg("--expected-video-sha256")
            .arg(&video_hash)
            .arg("--cvat-xml")
            .arg(&xml_path)
            .arg("--reviewed-frames")
            .arg(&reviewed_frames_path)
            .arg("--ffprobe")
            .arg(&ffprobe_path)
            .arg("--output")
            .arg(&annotations_path)
            .output()
            .expect("start temporal annotation importer CLI");
        let importer_summary: Value = serde_json::from_str(&successful_output(
            &importer_output,
            "billiards-temporal-import-cvat",
        ))
        .expect("importer prints a JSON summary");
        assert_eq!(importer_summary["video_sha256"], video_hash);
        assert_eq!(importer_summary["video_geometry"]["width"], 100);
        assert_eq!(importer_summary["video_geometry"]["height"], 80);
        assert_eq!(importer_summary["reviewed_frames"], 2);
        assert_eq!(
            importer_summary["interpolated_states_used_as_ground_truth"],
            false
        );

        let annotations: Value =
            serde_json::from_slice(&fs::read(&annotations_path).expect("imported labels exist"))
                .expect("importer wrote JSON annotations");
        assert_eq!(annotations["schema_version"], 1);
        assert_eq!(annotations["sequence"]["video_sha256"], video_hash);
        assert_eq!(annotations["frames"].as_array().unwrap().len(), 2);
        assert_eq!(annotations["frames"][0]["source_frame_index"], 0);
        assert_eq!(annotations["frames"][1]["source_frame_index"], 2);
        for (frame_index, expected_center) in [(0, (20.0, 30.0)), (1, (50.0, 40.0))] {
            let visible_ball = annotations["frames"][frame_index]["balls"]
                .as_array()
                .unwrap()
                .iter()
                .find(|ball| ball["status"] == "visible")
                .expect("each reviewed sample has a visible ball");
            assert_eq!(visible_ball["center"]["x"], expected_center.0);
            assert_eq!(visible_ball["center"]["y"], expected_center.1);
        }

        let scenes = [
            scene_for_frame(0, 1, (20.0, 30.0), true),
            scene_for_frame(2, 2, (50.0, 40.0), false),
        ];
        let scene_jsonl = scenes
            .iter()
            .map(serde_json::to_string)
            .collect::<Result<Vec<_>, _>>()
            .expect("serialize Rust scene fixtures")
            .join("\n");
        fs::write(&scene_path, format!("{scene_jsonl}\n")).expect("write scene JSONL");
        let cue_axis_angle = (45.0_f64).atan2(76.0).to_degrees();
        fs::write(
            &cue_geometry_path,
            format!(
                "{}\n",
                json!({
                    "source_video_sha256": video_hash.clone(),
                    "source_frame_index": 0,
                    "source_presentation_time_seconds": 0.0,
                    "cue_present": true,
                    "shaft_start_px": [4.0, 5.0],
                    "shaft_end_px": [80.0, 50.0],
                    "axis_angle_degrees": cue_axis_angle,
                    "confidence": 0.9,
                    "signed_direction": null
                })
            ),
        )
        .expect("write scheduled cue-geometry trace");

        let evaluator_output = Command::new(env!("CARGO_BIN_EXE_billiards-temporal-evaluate"))
            .arg("--video")
            .arg(&video_path)
            .arg("--annotations")
            .arg(&annotations_path)
            .arg("--scene-snapshots")
            .arg(&scene_path)
            .arg("--cue-geometry-jsonl")
            .arg(&cue_geometry_path)
            .arg("--ffprobe-executable")
            .arg(&ffprobe_path)
            .arg("--output")
            .arg(&report_path)
            .output()
            .expect("start temporal evaluator CLI");
        let report_stdout = successful_output(&evaluator_output, "billiards-temporal-evaluate");
        let report_from_stdout: Value =
            serde_json::from_str(&report_stdout).expect("evaluator stdout is a JSON report");
        let report_from_file: Value =
            serde_json::from_slice(&fs::read(&report_path).expect("evaluator report file exists"))
                .expect("evaluator wrote a JSON report");
        assert_eq!(report_from_stdout, report_from_file);
        assert_eq!(
            report_from_file["benchmark"],
            "rust-scene-temporal-localization"
        );
        assert_eq!(report_from_file["metadata"]["video_sha256"], video_hash);
        assert_eq!(report_from_file["metadata"]["video_geometry"]["width"], 100);
        assert_eq!(report_from_file["metadata"]["scored_engine_frames"], 2);
        assert_eq!(
            report_from_file["temporal_localization"]["ball_localization"]
                ["visible_ground_truth_balls"],
            2
        );
        assert_eq!(
            report_from_file["temporal_localization"]["ball_localization"]["matched_balls"],
            2
        );
        assert_eq!(
            report_from_file["temporal_localization"]["ball_localization"]["false_positives"],
            0
        );
        assert_eq!(
            report_from_file["temporal_localization"]["ball_localization"]["false_negatives"],
            0
        );
        assert_eq!(
            report_from_file["metadata"]["cue_geometry_trace_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        let cue_axis_error = report_from_file["temporal_localization"]["cue"]
            ["axis_angle_error_degrees"]["error_degrees_mean"]
            .as_f64()
            .unwrap();
        assert!(cue_axis_error < 1e-9);
    }
}
