use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{bail, ensure, Context, Result};
use billiards_perception::ball_center_calibration::{
    solve_ball_center_calibration, BallCenterCalibrationSolverInput,
    BallCenterCalibrationSolverPolicy, BallCenterCorrespondence,
    BALL_CENTER_CALIBRATION_SCHEMA_VERSION, MAX_BALL_CENTER_CORRESPONDENCES,
};
use billiards_perception::ball_localization::REGULATION_BALL_DIAMETER_MM;
use billiards_perception::ball_localization_profile::{
    PLAYING_AREA_HEIGHT_MM, PLAYING_AREA_WIDTH_MM,
};
use billiards_perception::ImageSpace;
use clap::Parser;
use serde::Deserialize;

const DEFAULT_MAXIMUM_INPUT_BYTES: usize = 1_048_576;
const MAXIMUM_INPUT_BYTES: usize = 16_777_216;

#[derive(Debug, Parser)]
#[command(
    name = "billiards-perception-ball-center-calibrate",
    about = "Fit a versioned 2D table mapping from independently measured ball centers"
)]
struct Arguments {
    /// Versioned JSON containing verified image/table ball-center pairs.
    #[arg(long)]
    input: PathBuf,

    /// New output path. Existing files are never overwritten.
    #[arg(long)]
    output: PathBuf,

    /// Maximum number of bytes read from the input JSON.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_INPUT_BYTES)]
    maximum_input_bytes: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputV1 {
    schema_version: u8,
    calibration_revision: u64,
    camera_profile_id: String,
    optics_profile_id: String,
    optics_revision: u64,
    refiner_revision: u64,
    table_revision: u64,
    rectification_revision: u64,
    image_space: ImageSpace,
    source_width_px: u32,
    source_height_px: u32,
    table_width_mm: f64,
    table_height_mm: f64,
    ball_diameter_mm: f64,
    provenance: String,
    #[serde(default)]
    solver_policy: BallCenterCalibrationSolverPolicy,
    correspondences: Vec<BallCenterCorrespondence>,
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    ensure!(
        (1024..=MAXIMUM_INPUT_BYTES).contains(&arguments.maximum_input_bytes),
        "maximum input bytes must be between 1024 and {MAXIMUM_INPUT_BYTES}"
    );

    let input = read_bounded_input(&arguments.input, arguments.maximum_input_bytes)?;
    ensure!(
        input.schema_version == BALL_CENTER_CALIBRATION_SCHEMA_VERSION,
        "unsupported ball-center correspondence input version {}",
        input.schema_version
    );
    ensure!(
        input.table_width_mm == PLAYING_AREA_WIDTH_MM
            && input.table_height_mm == PLAYING_AREA_HEIGHT_MM,
        "table dimensions must match the verified playing area ({PLAYING_AREA_WIDTH_MM} × {PLAYING_AREA_HEIGHT_MM} mm)"
    );
    ensure!(
        input.ball_diameter_mm == REGULATION_BALL_DIAMETER_MM,
        "ball diameter must match the guaranteed 57.15 mm regulation ball"
    );
    ensure!(
        input.correspondences.len() <= MAX_BALL_CENTER_CORRESPONDENCES,
        "correspondence count exceeds the solver's absolute bound of {MAX_BALL_CENTER_CORRESPONDENCES}"
    );

    let calibration = solve_ball_center_calibration(BallCenterCalibrationSolverInput {
        calibration_revision: input.calibration_revision,
        camera_profile_id: input.camera_profile_id,
        optics_profile_id: input.optics_profile_id,
        optics_revision: input.optics_revision,
        refiner_revision: input.refiner_revision,
        table_revision: input.table_revision,
        rectification_revision: input.rectification_revision,
        image_space: input.image_space,
        source_width_px: input.source_width_px,
        source_height_px: input.source_height_px,
        table_width_mm: input.table_width_mm,
        table_height_mm: input.table_height_mm,
        ball_diameter_mm: input.ball_diameter_mm,
        provenance: input.provenance,
        solver_policy: input.solver_policy,
        correspondences: input.correspondences,
    })
    .context("ball-center calibration fit failed validation")?;

    let output = serde_json::to_vec_pretty(&calibration)
        .context("could not serialize the fitted calibration asset")?;
    write_new_file(&arguments.output, &output)?;
    // The asset embeds every residual. Print the same report separately so an
    // operator can review fit quality before configuring the runtime to load it.
    println!(
        "{}",
        serde_json::to_string_pretty(&calibration.fit_report)
            .context("could not serialize calibration residual report")?
    );
    eprintln!(
        "wrote {} correspondences to {} (RMS {:.3} mm, max {:.3} mm); this is fit error, not independent accuracy",
        calibration.correspondences.len(),
        arguments.output.display(),
        calibration.fit_report.rms_mm,
        calibration.fit_report.maximum_residual_mm,
    );
    Ok(())
}

fn read_bounded_input(path: &PathBuf, maximum_bytes: usize) -> Result<InputV1> {
    let limit = u64::try_from(maximum_bytes)
        .context("maximum input bytes exceed this platform's file limit")?
        .checked_add(1)
        .context("maximum input byte limit overflow")?;
    let file = File::open(path)
        .with_context(|| format!("could not open correspondence input {}", path.display()))?;
    let mut bytes = Vec::with_capacity(maximum_bytes.min(64 * 1024));
    file.take(limit)
        .read_to_end(&mut bytes)
        .with_context(|| format!("could not read correspondence input {}", path.display()))?;
    ensure!(
        bytes.len() <= maximum_bytes,
        "correspondence input exceeds the configured {maximum_bytes}-byte limit"
    );
    serde_json::from_slice(&bytes).context("correspondence input is not valid versioned JSON")
}

fn write_new_file(path: &PathBuf, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "could not create calibration output {} without overwriting",
                path.display()
            )
        })?;
    if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
        drop(file);
        let cleanup = fs::remove_file(path);
        if let Err(cleanup_error) = cleanup {
            bail!(
                "failed to write calibration output {} ({error}); partial-file cleanup also failed ({cleanup_error})",
                path.display()
            );
        }
        return Err(error).with_context(|| {
            format!(
                "could not finish writing calibration output {}",
                path.display()
            )
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{write_new_file, InputV1};
    use billiards_perception::ball_center_calibration::solve_ball_center_calibration;
    use std::fs;

    #[test]
    fn output_creation_is_bounded_to_a_new_path_and_never_overwrites() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("ball-center-calibration.json");
        write_new_file(&path, b"asset").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"asset");
        assert!(write_new_file(&path, b"replacement").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"asset");
    }

    #[test]
    fn versioned_correspondences_fit_and_report_each_independent_residual() {
        let mut correspondences = Vec::new();
        for (row, y_mm) in [-400.0, 0.0, 400.0].into_iter().enumerate() {
            for (column, x_mm) in [-900.0, -300.0, 300.0, 900.0].into_iter().enumerate() {
                correspondences.push(serde_json::json!({
                    "image_point_px": {
                        "x_px": 100.0 + column as f64 * 300.0,
                        "y_px": 100.0 + row as f64 * 200.0
                    },
                    "center_table_mm": { "x_mm": x_mm, "y_mm": y_mm },
                    "provenance": "synthetic CLI contract test; not physical calibration"
                }));
            }
        }
        let input: InputV1 = serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "calibration_revision": 1,
            "camera_profile_id": "camera-test",
            "optics_profile_id": "optics-test",
            "optics_revision": 2,
            "refiner_revision": 1,
            "table_revision": 1,
            "rectification_revision": 0,
            "image_space": "raw",
            "source_width_px": 1280,
            "source_height_px": 720,
            "table_width_mm": 2336.8,
            "table_height_mm": 1155.7,
            "ball_diameter_mm": 57.15,
            "provenance": "synthetic command contract test; not physical calibration",
            "correspondences": correspondences
        }))
        .unwrap();
        assert_eq!(input.correspondences.len(), 12);
        let calibration = solve_ball_center_calibration(
            billiards_perception::ball_center_calibration::BallCenterCalibrationSolverInput {
                calibration_revision: input.calibration_revision,
                camera_profile_id: input.camera_profile_id,
                optics_profile_id: input.optics_profile_id,
                optics_revision: input.optics_revision,
                refiner_revision: input.refiner_revision,
                table_revision: input.table_revision,
                rectification_revision: input.rectification_revision,
                image_space: input.image_space,
                source_width_px: input.source_width_px,
                source_height_px: input.source_height_px,
                table_width_mm: input.table_width_mm,
                table_height_mm: input.table_height_mm,
                ball_diameter_mm: input.ball_diameter_mm,
                provenance: input.provenance,
                solver_policy: input.solver_policy,
                correspondences: input.correspondences,
            },
        )
        .unwrap();
        assert_eq!(calibration.fit_report.residuals.len(), 12);
        assert!(calibration.fit_report.rms_mm < 1e-6);
    }
}
