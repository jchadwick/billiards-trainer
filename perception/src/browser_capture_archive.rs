//! Persistent, bounded slots for operator-selected stills, independent of
//! producer sessions and unrelated configuration revisions.

use std::collections::HashSet;
use std::io::Read;
use std::path::PathBuf;

use anyhow::{ensure, Context, Result};
use base64::Engine;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::browser_ball_calibration::MAXIMUM_BROWSER_SAMPLES;
use crate::browser_precision_calibration::{
    asset_paths, save_asset, BallSample, FitContext, Still, MINIMUM_OPTICS_VIEWS,
};

const THUMBNAIL_SIDE_PX: u32 = 320;
// Bound metadata, including the small thumbnail, independently from the source JPEG.
const MAXIMUM_METADATA_BYTES: u64 = 512 * 1024;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum CaptureMode {
    Optics,
    BallCenters,
}

impl CaptureMode {
    fn capacity(self) -> usize {
        match self {
            Self::Optics => MINIMUM_OPTICS_VIEWS,
            Self::BallCenters => MAXIMUM_BROWSER_SAMPLES,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Entry<T> {
    capture: T,
    thumbnail_base64: String,
}

pub(crate) struct CaptureArchive {
    directory: PathBuf,
    mode: CaptureMode,
    maximum_file_bytes: u64,
    table_size_mm: Option<(f64, f64)>,
}

impl FitContext<'_> {
    pub(crate) fn archive(&self, mode: CaptureMode) -> Result<CaptureArchive> {
        let expected = &self.capture.expected;
        let geometry = expected
            .output_frame_geometry()
            .context("Capture geometry is unavailable.")?;
        ensure!(
            expected.image_space() == crate::ImageSpace::Raw,
            "Calibration needs raw camera frames."
        );
        let camera_asset = asset_paths(self.root, expected.camera_profile_id()).0;
        let camera_directory = camera_asset
            .parent()
            .context("Camera archive has no parent.")?;
        let archive = camera_directory.join("captures-v1").join(format!(
            "{}x{}-r{}",
            geometry.width_px(),
            geometry.height_px(),
            expected.rectification_revision()
        ));
        let (directory, table_size_mm) = match mode {
            CaptureMode::Optics => (archive.join("optics"), None),
            CaptureMode::BallCenters => {
                let table = expected
                    .calibration()
                    .context("Calibrate table geometry first.")?;
                let key = Sha256::digest(serde_json::to_vec(&table.capture_geometry_key())?);
                (
                    archive.join(format!("ball-centers-{key:x}")),
                    Some(table.playing_surface_dimensions_mm()),
                )
            }
        };
        Ok(CaptureArchive {
            directory,
            mode,
            maximum_file_bytes: (self.maximum_jpeg_bytes.div_ceil(3) * 4) as u64
                + MAXIMUM_METADATA_BYTES,
            table_size_mm,
        })
    }

    pub(crate) fn archive_capture(
        &self,
        mode: CaptureMode,
        slot: usize,
        value: Value,
    ) -> Result<Value> {
        let archive = self.archive(mode)?;
        archive.path(slot)?;
        match mode {
            CaptureMode::Optics => {
                let mut still: Still = serde_json::from_value(value)?;
                let thumbnail = self.thumbnail(&still)?;
                archive.check_duplicate(slot, &still, None)?;
                still.clear_token();
                archive.save(slot, still, thumbnail)
            }
            CaptureMode::BallCenters => {
                let mut sample: BallSample = serde_json::from_value(value)?;
                self.validate_ball_sample(&sample)?;
                let thumbnail = self.thumbnail(&sample.still)?;
                archive.check_duplicate(slot, &sample.still, Some(sample.table_center_mm))?;
                sample.still.clear_token();
                archive.save(slot, sample, thumbnail)
            }
        }
    }

    fn thumbnail(&self, still: &Still) -> Result<String> {
        let bytes = self.decode(still, false)?;
        let image = image::load_from_memory_with_format(&bytes, image::ImageFormat::Jpeg)?;
        ensure!(
            (image.width(), image.height())
                == (still.descriptor.width_px, still.descriptor.height_px),
            "Calibration JPEG dimensions do not match the captured frame."
        );
        let thumbnail = image.thumbnail(THUMBNAIL_SIDE_PX, THUMBNAIL_SIDE_PX);
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut jpeg).encode_image(&thumbnail)?;
        Ok(base64::engine::general_purpose::STANDARD.encode(jpeg))
    }
}

trait CaptureMetadata {
    fn metadata(&self, slot: usize, thumbnail: String) -> Value;
}

impl CaptureMetadata for Still {
    fn metadata(&self, slot: usize, thumbnail: String) -> Value {
        json!({"slot": slot, "thumbnail_base64": thumbnail})
    }
}

impl CaptureMetadata for BallSample {
    fn metadata(&self, slot: usize, thumbnail: String) -> Value {
        json!({
            "slot": slot,
            "thumbnail_base64": thumbnail,
            "table_center_mm": self.table_center_mm,
            "image_center_px": self.image_center_px,
            "radius_px": self.radius_px,
        })
    }
}

impl CaptureArchive {
    fn path(&self, slot: usize) -> Result<PathBuf> {
        ensure!(slot < self.mode.capacity(), "Capture slot is out of range.");
        Ok(self.directory.join(format!("{slot}.json")))
    }

    fn read<T: DeserializeOwned>(&self, slot: usize) -> Result<Option<Entry<T>>> {
        let file = match std::fs::File::open(self.path(slot)?) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        ensure!(
            file.metadata()?.len() <= self.maximum_file_bytes,
            "Saved capture exceeds its size limit."
        );
        Ok(Some(serde_json::from_reader(std::io::BufReader::new(
            file.take(self.maximum_file_bytes),
        ))?))
    }

    pub(crate) fn load_slots<T: DeserializeOwned>(&self, slots: &[usize]) -> Result<Vec<T>> {
        ensure!(
            slots.len() <= self.mode.capacity(),
            "Too many capture slots."
        );
        let mut unique = HashSet::new();
        slots
            .iter()
            .map(|&slot| {
                ensure!(unique.insert(slot), "Choose each capture slot only once.");
                Ok(self
                    .read(slot)?
                    .with_context(|| format!("Capture slot {} is empty.", slot + 1))?
                    .capture)
            })
            .collect()
    }

    fn save<T: Serialize + CaptureMetadata>(
        &self,
        slot: usize,
        capture: T,
        thumbnail_base64: String,
    ) -> Result<Value> {
        let path = self.path(slot)?;
        let entry = Entry {
            capture,
            thumbnail_base64,
        };
        save_asset(&path, &entry)?;
        Ok(entry.capture.metadata(slot, entry.thumbnail_base64))
    }

    fn check_duplicate(
        &self,
        target: usize,
        still: &Still,
        measured: Option<[f64; 2]>,
    ) -> Result<()> {
        for slot in (0..self.mode.capacity()).filter(|slot| *slot != target) {
            let existing = match self.mode {
                CaptureMode::Optics => self.read::<Still>(slot)?.map(|entry| (entry.capture, None)),
                CaptureMode::BallCenters => self
                    .read::<BallSample>(slot)?
                    .map(|entry| (entry.capture.still, Some(entry.capture.table_center_mm))),
            };
            if let Some((existing, placement)) = existing {
                ensure!(
                    existing.jpeg_base64 != still.jpeg_base64
                        && (existing.descriptor.capture_session_id
                            != still.descriptor.capture_session_id
                            || existing.descriptor.source_frame_id
                                != still.descriptor.source_frame_id),
                    "This frame is already saved in another slot."
                );
                ensure!(
                    measured.is_none() || measured != placement,
                    "Choose a different measured ball position."
                );
            }
        }
        Ok(())
    }

    pub(crate) fn list(&self) -> Result<Value> {
        let mut captures = Vec::new();
        for slot in 0..self.mode.capacity() {
            let record = match self.mode {
                CaptureMode::Optics => self
                    .read::<Still>(slot)?
                    .map(|entry| entry.capture.metadata(slot, entry.thumbnail_base64)),
                CaptureMode::BallCenters => self
                    .read::<BallSample>(slot)?
                    .map(|entry| entry.capture.metadata(slot, entry.thumbnail_base64)),
            };
            captures.extend(record);
        }
        Ok(json!({"captures": captures, "table_size_mm": self.table_size_mm}))
    }

    pub(crate) fn remove(&self, slot: Option<usize>) -> Result<Value> {
        match slot {
            Some(slot) => match std::fs::remove_file(self.path(slot)?) {
                Ok(()) => std::fs::File::open(&self.directory)?.sync_all()?,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
                Err(error) => return Err(error.into()),
            },
            None => {
                // Rename is the commit point, so readers never see a partly
                // cleared set even if process shutdown interrupts cleanup.
                if let Some(parent) = self.directory.parent().filter(|path| path.is_dir()) {
                    let discarded = tempfile::tempdir_in(parent)?;
                    match std::fs::rename(&self.directory, discarded.path().join("captures")) {
                        Ok(()) => std::fs::File::open(parent)?.sync_all()?,
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
                        Err(error) => return Err(error.into()),
                    }
                }
            }
        }
        Ok(json!({"removed": true}))
    }
}
