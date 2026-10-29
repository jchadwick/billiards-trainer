//! Bounded, revision-bound settings and camera intrinsics for ball localization.
//!
//! The camera's image geometry is an input to the local viewing transform.
//! It is deliberately stored separately from the cloth and ball-center maps:
//! none of those maps may be silently reused for a different capture profile.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::ball_center_calibration::{
    BallCenterCalibrationContext, BallCenterCalibrationV1, ValidatedBallCenterCalibrationV1,
};
use crate::ball_localization::{
    BallLocalizationConfig, CameraIntrinsics, REGULATION_BALL_DIAMETER_MM,
};
use crate::calibration::CalibrationContext;
use crate::ImageSpace;

/// The current on-disk optics-profile contract version.
pub const CAMERA_OPTICS_PROFILE_VERSION: u8 = 1;
const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_PROFILE_ID_CHARACTERS: usize = 128;
const MAX_PROVENANCE_CHARACTERS: usize = 2048;
pub const PLAYING_AREA_WIDTH_MM: f64 = 2336.8;
pub const PLAYING_AREA_HEIGHT_MM: f64 = 1155.7;
pub const BALL_REFINER_REVISION: u64 = 5;

/// Runtime settings. Missing assets indicate unavailable precision, rather than
/// selecting a different localization algorithm or manufacturing coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BallLocalizationSettings {
    pub parameters: BallLocalizationConfig,
    /// Measured regulation-ball diameter. This physical value never changes
    /// with camera resolution or apparent perspective.
    pub physical_diameter_mm: f64,
    pub table_width_mm: f64,
    pub table_height_mm: f64,
    pub table_revision: u64,
    pub optics_profile_path: Option<PathBuf>,
    pub ball_center_calibration_path: Option<PathBuf>,
    pub maximum_asset_bytes: usize,
}

impl Default for BallLocalizationSettings {
    fn default() -> Self {
        Self {
            parameters: BallLocalizationConfig::default(),
            physical_diameter_mm: 57.15,
            table_width_mm: PLAYING_AREA_WIDTH_MM,
            table_height_mm: PLAYING_AREA_HEIGHT_MM,
            table_revision: 1,
            optics_profile_path: None,
            ball_center_calibration_path: None,
            maximum_asset_bytes: 4_194_304,
        }
    }
}

impl BallLocalizationSettings {
    pub fn validate(&self) -> Result<()> {
        self.parameters
            .validate()
            .map_err(|error| anyhow::anyhow!("invalid ball localization parameters: {error}"))?;
        ensure!(
            self.physical_diameter_mm == REGULATION_BALL_DIAMETER_MM,
            "physical diameter must match the guaranteed 57.15 mm regulation ball"
        );
        ensure!(
            self.table_width_mm == PLAYING_AREA_WIDTH_MM
                && self.table_height_mm == PLAYING_AREA_HEIGHT_MM
                && self.table_revision > 0
                && self.table_revision <= MAX_SAFE_JSON_INTEGER,
            "ball localization table dimensions or revision do not match the verified playing area"
        );
        ensure!(
            (1024..=16_777_216).contains(&self.maximum_asset_bytes),
            "ball localization asset limit must be between 1024 and 16777216 bytes"
        );
        for (name, path) in [
            ("optics_profile_path", self.optics_profile_path.as_ref()),
            (
                "ball_center_calibration_path",
                self.ball_center_calibration_path.as_ref(),
            ),
        ] {
            if let Some(path) = path {
                ensure!(!path.as_os_str().is_empty(), "{name} must not be empty");
            }
        }
        ensure!(
            self.ball_center_calibration_path.is_none() || self.optics_profile_path.is_some(),
            "a ball-center calibration requires an optics profile"
        );
        Ok(())
    }

    /// Resolve configured paths exactly once at the supervisor boundary.
    /// Absolute paths remain absolute; paths are not inferred from frame data.
    pub fn resolve_paths(&mut self, project_root: &Path) {
        for path in [
            &mut self.optics_profile_path,
            &mut self.ball_center_calibration_path,
        ]
        .into_iter()
        .flatten()
        {
            if path.is_relative() {
                *path = project_root.join(&*path);
            }
        }
    }
}

/// Intrinsics calibrated for one explicit raw source geometry and camera.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraOpticsProfileV1 {
    pub schema_version: u8,
    pub optics_profile_id: String,
    pub camera_profile_id: String,
    pub optics_revision: u64,
    pub refiner_revision: u64,
    pub image_space: ImageSpace,
    pub intrinsics: CameraIntrinsics,
    pub provenance: String,
}

impl CameraOpticsProfileV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == CAMERA_OPTICS_PROFILE_VERSION,
            "unsupported camera optics profile version"
        );
        ensure!(
            !self.camera_profile_id.is_empty()
                && self.camera_profile_id.chars().count() <= MAX_PROFILE_ID_CHARACTERS
                && !self.camera_profile_id.chars().any(char::is_control),
            "invalid optics camera profile identity"
        );
        ensure!(
            !self.optics_profile_id.is_empty()
                && self.optics_profile_id.chars().count() <= MAX_PROFILE_ID_CHARACTERS
                && !self.optics_profile_id.chars().any(char::is_control),
            "invalid optics profile identity"
        );
        ensure!(
            self.optics_revision <= MAX_SAFE_JSON_INTEGER
                && self.refiner_revision <= MAX_SAFE_JSON_INTEGER,
            "optics or refiner revision exceeds the safe JSON integer range"
        );
        ensure!(
            self.image_space == ImageSpace::Raw,
            "the local-view estimator requires raw image coordinates"
        );
        ensure!(
            !self.provenance.trim().is_empty()
                && self.provenance.chars().count() <= MAX_PROVENANCE_CHARACTERS
                && !self.provenance.chars().any(char::is_control),
            "optics provenance must be non-empty and bounded"
        );
        self.intrinsics
            .validate()
            .map_err(|error| anyhow::anyhow!("invalid camera intrinsics: {error}"))?;
        Ok(())
    }

    /// Require an exact match rather than guessing how to scale intrinsics.
    pub fn validate_frame(
        &self,
        camera_profile_id: &str,
        image_space: ImageSpace,
        width_px: u32,
        height_px: u32,
    ) -> Result<()> {
        self.validate()?;
        ensure!(
            self.camera_profile_id == camera_profile_id,
            "optics camera profile mismatch"
        );
        ensure!(
            self.image_space == image_space,
            "optics image space mismatch"
        );
        ensure!(
            self.intrinsics.source_width_px == width_px
                && self.intrinsics.source_height_px == height_px,
            "optics source dimensions mismatch"
        );
        Ok(())
    }
}

/// Assets that have been bounded and validated for one exact capture context.
/// The validated homography is immutable, so each ball projection performs
/// only context checks and matrix arithmetic rather than revalidating a file.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedBallLocalizationProfile {
    pub settings: BallLocalizationSettings,
    pub optics: Option<CameraOpticsProfileV1>,
    pub calibration: Option<ValidatedBallCenterCalibrationV1>,
    pub calibration_error: Option<String>,
}

impl LoadedBallLocalizationProfile {
    pub fn load_for_frame(
        settings: BallLocalizationSettings,
        camera_profile_id: &str,
        image_space: ImageSpace,
        width_px: u32,
        height_px: u32,
        rectification_revision: u64,
    ) -> Result<Self> {
        settings.validate()?;
        let Some(optics_path) = settings.optics_profile_path.as_deref() else {
            return Ok(Self {
                settings,
                optics: None,
                calibration: None,
                calibration_error: None,
            });
        };
        let optics: CameraOpticsProfileV1 =
            read_bounded_json(optics_path, settings.maximum_asset_bytes)?;
        optics.validate_frame(camera_profile_id, image_space, width_px, height_px)?;
        ensure!(
            optics.refiner_revision == BALL_REFINER_REVISION,
            "optics profile uses unsupported ball refiner revision {} (runtime supports {BALL_REFINER_REVISION})",
            optics.refiner_revision
        );

        let calibration_result = settings.ball_center_calibration_path.as_deref().map(|path| {
                let asset: BallCenterCalibrationV1 =
                    read_bounded_json(path, settings.maximum_asset_bytes)?;
                ensure!(
                    asset.ball_diameter_mm == settings.physical_diameter_mm
                        && asset.table_width_mm == settings.table_width_mm
                        && asset.table_height_mm == settings.table_height_mm
                        && asset.table_revision == settings.table_revision,
                    "ball-center calibration physical table or ball dimensions differ from configured values"
                );
                let validated = asset
                    .into_validated()
                    .map_err(|error| anyhow::anyhow!("invalid ball-center calibration: {error}"))?;
                validated.validate_context(BallCenterCalibrationContext {
                    frame: CalibrationContext {
                        calibration_revision: validated.as_asset().calibration_revision,
                        camera_profile_id,
                        rectification_revision,
                        image_space,
                        source_width_px: width_px,
                        source_height_px: height_px,
                    },
                    optics_profile_id: &optics.optics_profile_id,
                    optics_revision: optics.optics_revision,
                    refiner_revision: optics.refiner_revision,
                    table_revision: settings.table_revision,
                })
                .map_err(|error| anyhow::anyhow!("ball-center calibration context mismatch: {error}"))?;
                Ok(validated)
            });
        let (calibration, calibration_error) = match calibration_result {
            Some(Ok(calibration)) => (Some(calibration), None),
            Some(Err(error)) => (None, Some(format!("{error:#}"))),
            None => (None, None),
        };

        Ok(Self {
            settings,
            optics: Some(optics),
            calibration,
            calibration_error,
        })
    }

    pub fn calibration_context<'a>(
        &'a self,
        camera_profile_id: &'a str,
    ) -> Option<BallCenterCalibrationContext<'a>> {
        let optics = self.optics.as_ref()?;
        let calibration = self.calibration.as_ref()?;
        let asset = calibration.as_asset();
        Some(BallCenterCalibrationContext {
            frame: CalibrationContext {
                calibration_revision: asset.calibration_revision,
                camera_profile_id,
                rectification_revision: asset.rectification_revision,
                image_space: asset.image_space,
                source_width_px: asset.source_width_px,
                source_height_px: asset.source_height_px,
            },
            optics_profile_id: &optics.optics_profile_id,
            optics_revision: optics.optics_revision,
            refiner_revision: optics.refiner_revision,
            table_revision: self.settings.table_revision,
        })
    }
}

/// Read an asset with an explicit allocation bound, including non-regular
/// inputs whose metadata cannot safely establish their eventual length.
pub fn read_bounded_json<T: DeserializeOwned>(path: &Path, maximum_bytes: usize) -> Result<T> {
    ensure!(maximum_bytes > 0, "asset byte limit must be positive");
    let bound = u64::try_from(maximum_bytes)
        .context("asset byte limit exceeds supported platform range")?
        .checked_add(1)
        .context("asset byte limit overflow")?;
    let file =
        File::open(path).with_context(|| format!("could not open asset {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(bound)
        .read_to_end(&mut bytes)
        .with_context(|| format!("could not read asset {}", path.display()))?;
    if bytes.len() > maximum_bytes {
        bail!("asset {} exceeds its byte limit", path.display());
    }
    serde_json::from_slice(&bytes).with_context(|| format!("invalid asset {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn optics() -> CameraOpticsProfileV1 {
        CameraOpticsProfileV1 {
            schema_version: CAMERA_OPTICS_PROFILE_VERSION,
            optics_profile_id: "optics-a".to_owned(),
            camera_profile_id: "camera-a".to_owned(),
            optics_revision: 4,
            refiner_revision: BALL_REFINER_REVISION,
            image_space: ImageSpace::Raw,
            intrinsics: CameraIntrinsics {
                source_width_px: 3840,
                source_height_px: 2160,
                fx: 2100.0,
                fy: 2100.0,
                cx: 1920.0,
                cy: 1080.0,
            },
            provenance: "independent checkerboard acquisition".to_owned(),
        }
    }

    #[test]
    fn optics_require_exact_frame_context_and_valid_intrinsics() {
        let profile = optics();
        profile
            .validate_frame("camera-a", ImageSpace::Raw, 3840, 2160)
            .unwrap();
        assert!(profile
            .validate_frame("camera-b", ImageSpace::Raw, 3840, 2160)
            .is_err());
        assert!(profile
            .validate_frame("camera-a", ImageSpace::Raw, 1280, 720)
            .is_err());
        assert!(profile
            .validate_frame("camera-a", ImageSpace::Rectified, 3840, 2160)
            .is_err());
        let mut invalid = profile;
        invalid.intrinsics.fx = f64::NAN;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn optics_profile_must_match_the_current_refiner_revision() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("optics.json");
        let mut profile = optics();
        assert_eq!(profile.refiner_revision, BALL_REFINER_REVISION);
        profile.refiner_revision -= 1;
        std::fs::write(&path, serde_json::to_vec(&profile).unwrap()).unwrap();
        let settings = BallLocalizationSettings {
            optics_profile_path: Some(path),
            ..BallLocalizationSettings::default()
        };
        assert!(LoadedBallLocalizationProfile::load_for_frame(
            settings,
            "camera-a",
            ImageSpace::Raw,
            3840,
            2160,
            0,
        )
        .is_err());
    }

    #[test]
    fn bounded_asset_reads_reject_oversized_and_malformed_data() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("profile.json");
        std::fs::write(&path, serde_json::to_vec(&optics()).unwrap()).unwrap();
        let loaded: CameraOpticsProfileV1 = read_bounded_json(&path, 4096).unwrap();
        assert_eq!(loaded, optics());
        assert!(read_bounded_json::<CameraOpticsProfileV1>(&path, 3).is_err());
        std::fs::write(&path, b"not-json").unwrap();
        assert!(read_bounded_json::<CameraOpticsProfileV1>(&path, 4096).is_err());
    }

    #[test]
    fn paths_are_resolved_once_and_calibration_requires_optics() {
        let mut settings = BallLocalizationSettings {
            optics_profile_path: Some(PathBuf::from("profiles/camera.json")),
            ball_center_calibration_path: Some(PathBuf::from("profiles/balls.json")),
            ..BallLocalizationSettings::default()
        };
        settings.validate().unwrap();
        settings.resolve_paths(Path::new("/project"));
        settings.resolve_paths(Path::new("/another"));
        assert_eq!(
            settings.optics_profile_path.as_deref(),
            Some(Path::new("/project/profiles/camera.json"))
        );
        settings.optics_profile_path = None;
        assert!(settings.validate().is_err());
    }
}
