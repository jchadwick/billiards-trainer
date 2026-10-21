use std::process::Command;

use serde_json::Value;

fn executable() -> Command {
    Command::new(env!("CARGO_BIN_EXE_billiards-capture-transport-bench"))
}

fn workload() -> Command {
    let mut command = executable();
    command.args([
        "--width-px",
        "8",
        "--height-px",
        "6",
        "--warmup-frames",
        "2",
        "--measured-frames",
        "11",
        "--repetitions",
        "2",
        "--sampling-limit",
        "4",
        "--seed",
        "772",
        "--max-payload-bytes",
        "4096",
        "--max-total-frames",
        "100",
        "--max-total-samples",
        "16",
    ]);
    command
}

#[test]
fn both_arms_produce_identical_payloads_with_bounded_samples_and_metadata() {
    let output = workload().output().unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();

    assert_eq!(report["schema_version"], 3);
    assert_eq!(report["scope"], "isolated transport microbenchmark; not a camera, detector, CaptureSource mailbox, or client-delivery benchmark");
    assert!(matches!(
        report["metadata"]["build_profile"].as_str(),
        Some("debug" | "release")
    ));
    assert_eq!(report["workload"]["payload_bytes_per_frame"], 144);
    assert_eq!(
        report["workload"]["application_frame_buffer_count_per_arm"],
        2
    );
    assert_eq!(
        report["workload"]["application_frame_buffer_bytes_per_arm"],
        288
    );
    assert_eq!(
        report["workload"]["payload_preparation_and_sha256"],
        "generated and SHA-256 checked during arm setup/readiness; excluded from measured frame loops"
    );
    assert_eq!(
        report["workload"]["payload_content_model"],
        "one deterministic RGB8 payload template repeated for every frame record in each arm"
    );
    assert_eq!(
        report["workload"]["timed_operations"],
        "in-process Vec ownership handoff versus helper frame serialization and parent pipe reads into a reusable buffer; payload generation and SHA-256 are excluded"
    );
    assert!(report["workload"]["producer_consumer_model"]
        .as_str()
        .unwrap()
        .contains("capacity_one_sync_channel"));
    assert_eq!(report["repetitions"].as_array().unwrap().len(), 2);
    assert_eq!(
        report["measurements_not_taken"]["camera_source_to_parent_latency"]["status"],
        "unavailable"
    );
    assert_eq!(
        report["measurements_not_taken"]["resident_memory"]["status"],
        "unavailable"
    );
    assert_eq!(
        report["measurements_not_taken"]["operating_system_pipe_capacity"]["status"],
        "unavailable"
    );

    for repetition in report["repetitions"].as_array().unwrap() {
        assert_eq!(
            repetition["in_process"]["frame_payload_sha256"],
            repetition["supervised_stdout_pipe"]["frame_payload_sha256"]
        );
        assert_eq!(
            repetition["in_process"]["sampled_frame_completion"]["samples"][0]
                ["frame_payload_sha256"],
            repetition["in_process"]["frame_payload_sha256"]
        );
        assert_eq!(repetition["in_process"]["measured_frames"], 11);
        assert_eq!(repetition["supervised_stdout_pipe"]["measured_frames"], 11);
        assert!(repetition["in_process"]["data_movement_model"]
            .as_str()
            .unwrap()
            .contains("not copied by application code"));
        assert!(repetition["supervised_stdout_pipe"]["data_movement_model"]
            .as_str()
            .unwrap()
            .contains("OS stdout pipe"));
        assert_eq!(
            repetition["in_process"]["sampled_frame_completion"]["sampled_frames"],
            4
        );
        assert_eq!(
            repetition["supervised_stdout_pipe"]["sampled_frame_completion"]["sampled_frames"],
            4
        );
        assert_eq!(
            repetition["in_process"]["startup_to_ready_ns"]["status"],
            "available"
        );
        assert!(
            repetition["in_process"]["startup_to_ready_ns"]["value"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(
            repetition["supervised_stdout_pipe"]["startup_to_ready_ns"]["status"],
            "available"
        );
        assert!(
            repetition["supervised_stdout_pipe"]["startup_to_ready_ns"]["value"]
                .as_u64()
                .unwrap()
                > 0
        );
    }
}

#[test]
fn invalid_payload_bound_fails_without_emitting_a_report() {
    let command = workload();
    let mut args = command
        .get_args()
        .map(|arg| arg.to_owned())
        .collect::<Vec<_>>();
    let max_bytes_index = args
        .iter()
        .position(|argument| argument == "--max-payload-bytes")
        .unwrap();
    args[max_bytes_index + 1] = "100".into();
    let output = executable().args(args).output().unwrap();

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("above --max-payload-bytes"));
}

#[test]
fn help_explains_the_opt_in_experiment_and_requires_explicit_limits() {
    let output = executable().arg("--help").output().unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("Run with --release"));
    assert!(help.contains("--sampling-limit"));
    assert!(help.contains("--max-payload-bytes"));
    assert!(help.contains("--max-total-frames"));
    assert!(help.contains("--max-total-samples"));
    assert!(help.contains("does not measure a camera, detector"));
}
