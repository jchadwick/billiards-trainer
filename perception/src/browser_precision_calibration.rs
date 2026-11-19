//! Bounded fits over authenticated, operator-requested calibration stills.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::ball_localization::{
    refine_ball_batch, BallProposal, CameraIntrinsics, RefinementStatus,
};
use crate::ball_localization_profile::{
    BallLocalizationSettings, CameraOpticsProfileV1, BALL_REFINER_REVISION,
    CAMERA_OPTICS_PROFILE_VERSION,
};
use crate::browser_ball_calibration::{self, BrowserBallCalibrationSample};
use crate::capture_settings_wire::ResolvedCaptureSettings;
use crate::frame_preview::FramePreviewDescriptorV1;

// Match the still preview protocol ceiling plus bounded JSON metadata.
pub(crate) const MAXIMUM_STILL_REQUEST_BYTES: usize =
    (16 * 1024 * 1024_usize).div_ceil(3) * 4 + 64 * 1024;
pub(crate) const MAXIMUM_FIT_REQUEST_BYTES: usize = 128 * 1024 * 1024;
pub(crate) const MINIMUM_OPTICS_VIEWS: usize = 8;
const MAXIMUM_OPTICS_VIEWS: usize = 16;
const FIT_TIMEOUT: Duration = Duration::from_secs(90);
const MAXIMUM_FIT_OUTPUT_BYTES: u64 = 16_384;

#[derive(Deserialize, Serialize)]
pub(crate) struct Still {
    pub(crate) descriptor: FramePreviewDescriptorV1,
    pub(crate) jpeg_base64: String,
    capture_token: String,
}

impl Still {
    pub(crate) fn clear_token(&mut self) {
        self.capture_token.clear();
    }
}

#[derive(Deserialize)]
pub(crate) struct OpticsRequest {
    #[serde(default)]
    frames: Vec<Still>,
    slots: Option<Vec<usize>>,
    columns: u32,
    rows: u32,
    square_mm: f64,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct BallSample {
    #[serde(flatten)]
    pub(crate) still: Still,
    pub(crate) image_center_px: [f64; 2],
    pub(crate) radius_px: f64,
    pub(crate) table_center_mm: [f64; 2],
}

#[derive(Deserialize)]
pub(crate) struct BallRequest {
    #[serde(default)]
    samples: Vec<BallSample>,
    slots: Option<Vec<usize>>,
}

#[derive(Deserialize)]
pub(crate) struct BallCandidateRequest {
    #[serde(flatten)]
    still: Still,
    table_center_mm: [f64; 2],
}

pub(crate) fn fingerprint(descriptor: &FramePreviewDescriptorV1, jpeg: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(serde_json::to_vec(descriptor).expect("frame descriptor serializes"));
    hash.update(jpeg);
    hash.finalize().into()
}

pub(crate) fn asset_paths(root: &Path, camera_profile: &str) -> (PathBuf, PathBuf) {
    let key = format!("{:x}", Sha256::digest(camera_profile.as_bytes()));
    let directory = root
        .join("backend/calibration_data/browser-precision")
        .join(key);
    (
        directory.join("optics.json"),
        directory.join("ball-centers.json"),
    )
}

pub(crate) fn select_assets(
    settings: &mut BallLocalizationSettings,
    root: &Path,
    camera_profile: &str,
) {
    let (optics, centers) = asset_paths(root, camera_profile);
    if optics.is_file() {
        settings.optics_profile_path = Some(optics);
        settings.ball_center_calibration_path = None;
    }
    if centers.is_file() {
        settings.ball_center_calibration_path = Some(centers);
    }
    settings.resolve_paths(root);
}

pub(crate) struct FitContext<'a> {
    pub(crate) root: &'a Path,
    pub(crate) capture: &'a ResolvedCaptureSettings,
    pub(crate) settings: BallLocalizationSettings,
    pub(crate) tokens: &'a std::collections::VecDeque<(String, [u8; 32])>,
    pub(crate) maximum_jpeg_bytes: usize,
}

impl FitContext<'_> {
    pub(crate) fn decode(&self, still: &Still, archived: bool) -> Result<Vec<u8>> {
        let expected = &self.capture.expected;
        let geometry = expected
            .output_frame_geometry()
            .context("Capture geometry is unavailable.")?;
        let frame = &still.descriptor;
        ensure!(
            frame.camera_profile_id == expected.camera_profile_id()
                && (archived
                    || (frame.configuration_revision == expected.configuration_revision()
                        && frame.calibration_revision == expected.calibration_revision()))
                && frame.rectification_revision == expected.rectification_revision()
                && (frame.width_px, frame.height_px) == (geometry.width_px(), geometry.height_px())
                && expected.image_space() == crate::ImageSpace::Raw
                && frame.image_space == crate::detection_batch::ImageSpace::Raw,
            "A capture belongs to a different camera or calibration revision. Capture it again."
        );
        ensure!(
            still.jpeg_base64.len() <= self.maximum_jpeg_bytes.div_ceil(3) * 4,
            "Calibration still exceeds the JPEG size limit."
        );
        let bytes = base64::engine::general_purpose::STANDARD.decode(&still.jpeg_base64)?;
        ensure!(
            bytes.len() <= self.maximum_jpeg_bytes,
            "Calibration still exceeds the JPEG size limit."
        );
        if !archived {
            let digest = fingerprint(frame, &bytes);
            ensure!(
                self.tokens
                    .iter()
                    .any(|(token, stored)| token == &still.capture_token && stored == &digest),
                "Calibration capture expired or was modified. Capture it again."
            );
        }
        Ok(bytes)
    }

    pub(crate) fn fit_optics(&self, mut request: OpticsRequest) -> Result<Value> {
        use crate::browser_capture_archive::CaptureMode;
        let archived = request.slots.is_some();
        if let Some(slots) = request.slots.take() {
            ensure!(
                request.frames.is_empty(),
                "Choose either saved slots or new frames."
            );
            request.frames = self.archive(CaptureMode::Optics)?.load_slots(&slots)?;
        }
        ensure!(
            (MINIMUM_OPTICS_VIEWS..=MAXIMUM_OPTICS_VIEWS).contains(&request.frames.len()),
            "Capture {MINIMUM_OPTICS_VIEWS} to {MAXIMUM_OPTICS_VIEWS} checkerboard views."
        );
        ensure!(
            (4..=24).contains(&request.columns)
                && (4..=24).contains(&request.rows)
                && request.square_mm.is_finite()
                && (1.0..=200.0).contains(&request.square_mm),
            "Enter 4–24 inner corners per side and a measured square size of 1–200 mm."
        );
        let temporary = tempfile::tempdir()?;
        let stdout = tempfile::tempfile()?;
        let stderr = tempfile::tempfile()?;
        let python =
            std::env::var_os("BILLIARDS_CALIBRATION_PYTHON").unwrap_or_else(|| "python3".into());
        let mut command = std::process::Command::new(python);
        command
            .arg("-c")
            .arg(include_str!("browser_optics_fit.py"))
            .arg(request.columns.to_string())
            .arg(request.rows.to_string())
            .arg(request.square_mm.to_string())
            .stdin(Stdio::null())
            .stdout(stdout.try_clone()?)
            .stderr(stderr.try_clone()?);
        let mut unique = std::collections::HashSet::new();
        for (index, still) in request.frames.iter().enumerate() {
            let bytes = self.decode(still, archived)?;
            ensure!(
                unique.insert(Sha256::digest(&bytes)),
                "Capture distinct checkerboard views."
            );
            let path = temporary.path().join(format!("{index}.jpg"));
            std::fs::write(&path, bytes)?;
            command.arg(path);
        }
        let mut child = command
            .spawn()
            .context("Could not start the calibration Python interpreter.")?;
        let started = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if started.elapsed() < FIT_TIMEOUT => {
                    std::thread::sleep(Duration::from_millis(50))
                }
                result => {
                    let _ = child.kill();
                    let _ = child.wait();
                    result?;
                    anyhow::bail!("Checkerboard fit exceeded its time limit. Try fewer captures.");
                }
            }
        };
        if !status.success() {
            anyhow::bail!("{}", bounded_output(stderr)?.trim());
        }
        let report: Value = serde_json::from_str(&bounded_output(stdout)?)?;
        let intrinsics: CameraIntrinsics = serde_json::from_value(report["intrinsics"].clone())?;
        let expected = &self.capture.expected;
        let optics = CameraOpticsProfileV1 {
            schema_version: CAMERA_OPTICS_PROFILE_VERSION,
            optics_profile_id: uuid::Uuid::new_v4().to_string(),
            camera_profile_id: expected.camera_profile_id().to_owned(),
            optics_revision: 1,
            refiner_revision: BALL_REFINER_REVISION,
            image_space: expected.image_space(),
            intrinsics,
            provenance: format!(
                "Browser checkerboard fit, {} views, {} × {} inner corners, {} mm squares",
                request.frames.len(),
                request.columns,
                request.rows,
                request.square_mm
            ),
        };
        let geometry = expected
            .output_frame_geometry()
            .context("Capture geometry is unavailable.")?;
        optics.validate_frame(
            expected.camera_profile_id(),
            expected.image_space(),
            geometry.width_px(),
            geometry.height_px(),
        )?;
        save_asset(
            &asset_paths(self.root, expected.camera_profile_id()).0,
            &optics,
        )?;
        Ok(report)
    }

    pub(crate) fn fit_ball_centers(&self, mut request: BallRequest) -> Result<Value> {
        use crate::browser_capture_archive::CaptureMode;
        let slots = request.slots.take();
        if let Some(slots) = &slots {
            ensure!(
                request.samples.is_empty(),
                "Choose either saved slots or new samples."
            );
            request.samples = self.archive(CaptureMode::BallCenters)?.load_slots(slots)?;
        }
        ensure!(
            (browser_ball_calibration::MINIMUM_BROWSER_SAMPLES
                ..=browser_ball_calibration::MAXIMUM_BROWSER_SAMPLES)
                .contains(&request.samples.len()),
            "Capture 12 to 24 independently measured ball placements."
        );
        let expected = &self.capture.expected;
        let table = expected
            .calibration()
            .context("Calibrate table geometry first.")?;
        let optics = self.load_optics()?;
        let mut samples = Vec::with_capacity(request.samples.len());
        let mut unique = std::collections::HashSet::new();
        for (index, sample) in request.samples.into_iter().enumerate() {
            let bytes = self.decode(&sample.still, slots.is_some())?;
            let placement = slots.as_ref().map_or(index, |slots| slots[index]) + 1;
            let hash = Sha256::digest(&bytes);
            ensure!(
                unique.insert(hash),
                "Capture each ball placement on a distinct still."
            );
            let image =
                image::load_from_memory_with_format(&bytes, image::ImageFormat::Jpeg)?.to_rgb8();
            self.validate_ball_sample(&sample)?;
            let [x, y] = sample.image_center_px;
            let radius = sample.radius_px;
            let parameters = &self.settings.parameters;
            let proposals = [BallProposal {
                detection_index: 0,
                bbox: [x - radius, y - radius, x + radius, y + radius],
            }];
            let fitted =
                refine_ball_batch(image.as_raw(), &optics.intrinsics, &proposals, parameters)?;
            let center = fitted.first().filter(|value| value.status == RefinementStatus::Accepted)
                .and_then(|value| value.center_image_px)
                .with_context(|| format!("The ball outline was rejected in placement {placement}. Use an isolated, fully visible ball and mark its center and edge again."))?;
            samples.push(BrowserBallCalibrationSample {
                image_center_px: center,
                table_center_mm: sample.table_center_mm,
                frame_id: format!(
                    "{}:{}",
                    sample.still.descriptor.capture_session_id,
                    sample.still.descriptor.source_frame_id
                ),
                frame_sha256: format!("{hash:x}"),
            });
        }
        let asset =
            browser_ball_calibration::fit(samples, table, &optics, self.settings.table_revision)?;
        // A browser fit can build on manually configured optics. The
        // selected ball-center file is scoped to that camera and validated
        // against the optics identity at load time.
        save_asset(
            &asset_paths(self.root, expected.camera_profile_id()).1,
            &asset,
        )?;
        Ok(json!({"fit_report": asset.fit_report}))
    }
    pub(crate) fn ball_candidate(&self, request: BallCandidateRequest) -> Result<Value> {
        let bytes = self.decode(&request.still, false)?;
        let table = self
            .capture
            .expected
            .calibration()
            .context("Calibrate table geometry first.")?;
        let optics = self.load_optics()?;
        let intrinsics = &optics.intrinsics;
        let [x_mm, y_mm] = request.table_center_mm;
        ensure!(
            table.contains_playing_surface_point(crate::calibration::TablePointMm { x_mm, y_mm }),
            "Enter the measured ball center inside the playing surface."
        );
        let target = crate::browser_ball_placement::BallSearchTarget::project(
            table,
            intrinsics,
            &self.settings.parameters,
            request.table_center_mm,
        )?;
        let image =
            image::load_from_memory_with_format(&bytes, image::ImageFormat::Jpeg)?.to_rgb8();
        let candidate = target.refine(image.as_raw(), intrinsics, &self.settings.parameters)?;
        Ok(json!({"candidate": candidate}))
    }

    pub(crate) fn load_optics(&self) -> Result<CameraOpticsProfileV1> {
        let expected = &self.capture.expected;
        let geometry = expected
            .output_frame_geometry()
            .context("Capture geometry is unavailable.")?;
        let path = self
            .settings
            .optics_profile_path
            .as_ref()
            .context("Fit camera optics first.")?;
        let file = std::fs::File::open(path).context("Could not open the optics profile.")?;
        let optics: CameraOpticsProfileV1 =
            serde_json::from_reader(file.take(self.settings.maximum_asset_bytes as u64))?;
        optics.validate_frame(
            expected.camera_profile_id(),
            expected.image_space(),
            geometry.width_px(),
            geometry.height_px(),
        )?;
        Ok(optics)
    }

    pub(crate) fn validate_ball_sample(&self, sample: &BallSample) -> Result<()> {
        let [x, y] = sample.image_center_px;
        let frame = &sample.still.descriptor;
        let parameters = &self.settings.parameters;
        let scale = f64::from(frame.width_px) / f64::from(parameters.reference_width_px);
        ensure!(
            (0.0..f64::from(frame.width_px)).contains(&x)
                && (0.0..f64::from(frame.height_px)).contains(&y)
                && (parameters.minimum_radius_px * scale..=parameters.maximum_radius_px * scale)
                    .contains(&sample.radius_px),
            "Mark the center and outer edge of the ball inside the image."
        );
        let table = self
            .capture
            .expected
            .calibration()
            .context("Calibrate table geometry first.")?;
        let [x_mm, y_mm] = sample.table_center_mm;
        ensure!(
            table.contains_playing_surface_point(crate::calibration::TablePointMm { x_mm, y_mm }),
            "Enter the measured ball center inside the playing surface."
        );
        Ok(())
    }
}

fn bounded_output(mut file: std::fs::File) -> Result<String> {
    file.seek(SeekFrom::Start(0))?;
    let mut output = String::new();
    file.take(MAXIMUM_FIT_OUTPUT_BYTES)
        .read_to_string(&mut output)?;
    Ok(output)
}

pub(crate) fn save_asset(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    let parent = path
        .parent()
        .context("Calibration asset path needs a parent directory.")?;
    std::fs::create_dir_all(parent)?;
    let file = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(file.as_file(), value)?;
    file.as_file().sync_all()?;
    file.persist(path)?;
    std::fs::File::open(parent)?.sync_all()?;
    Ok(())
}
