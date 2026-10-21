//! All-or-nothing calibrated measurements for complete detector batches.
//!
//! This adapter preserves frame and detection identity while adding table
//! coordinates. It deliberately does not assign track IDs; temporal association
//! belongs to the next Stage 3 layer. Applied camera and revision metadata is
//! supplied by the control/runtime boundary because detection-batch v1 records
//! inference provenance but does not claim which calibration was applied.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::calibration::{
    CalibrationContext, CalibrationError, ImagePointPx, TablePointMm, ValidatedTableCalibrationV1,
};
use crate::detection_batch::{
    BatchOutcome, BatchValidationError, DetectionBatchV1, ImageSpace as BatchImageSpace,
    Point as BatchPoint, SourceFrame as BatchSourceFrame,
};
use crate::ImageSpace;

/// Runtime-applied metadata that must be acknowledged by the calibration
/// control boundary, rather than inferred from the calibration asset itself.
#[derive(Debug, Clone, Copy)]
pub struct AppliedCalibrationContext<'a> {
    pub calibration_revision: u64,
    pub camera_profile_id: &'a str,
    pub rectification_revision: u64,
}

/// A detector observation with a source-image measurement and calibrated point.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedDetection {
    pub detection_index: u64,
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f64,
    pub center_image_px: BatchPoint,
    pub center_table_mm: TablePointMm,
}

/// Complete transformed output for one successful source-frame batch.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedDetectionBatch {
    pub engine_instance_id: String,
    pub batch_sequence: u64,
    pub source: BatchSourceFrame,
    pub image_space: BatchImageSpace,
    /// Camera profile acknowledged by the runtime when this result was made.
    /// Keeping it here lets downstream consumers reject cross-camera results.
    pub camera_profile_id: String,
    pub calibration_revision: u64,
    pub rectification_revision: u64,
    pub detections: Vec<CalibratedDetection>,
}

/// Why an inference batch could not produce a complete calibrated result.
#[derive(Debug, Clone, PartialEq)]
pub enum GeometryError {
    InvalidDetectionBatch(BatchValidationError),
    DetectionBatchNotSuccessful(BatchOutcome),
    IncompatibleFrameCalibration(CalibrationError),
    DetectionProjection {
        detection_index: u64,
        source: CalibrationError,
    },
}

impl Display for GeometryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "calibrated detection error: {self:?}")
    }
}

impl Error for GeometryError {}

/// Apply a validated calibration to every detection in one successful batch.
///
/// The operation is all-or-nothing: if a frame context or any point is
/// incompatible, no partial calibrated batch is returned. Detector failures
/// and skipped batches remain explicit errors and cannot become empty scenes.
/// The returned detections keep their source order and batch-local indices;
/// those indices are not stable ball identities or track IDs.
pub fn calibrate_detection_batch(
    batch: &DetectionBatchV1,
    calibration: &ValidatedTableCalibrationV1,
    applied: AppliedCalibrationContext<'_>,
) -> Result<CalibratedDetectionBatch, GeometryError> {
    batch
        .validate()
        .map_err(GeometryError::InvalidDetectionBatch)?;
    if batch.outcome != BatchOutcome::Success {
        return Err(GeometryError::DetectionBatchNotSuccessful(batch.outcome));
    }

    let context = CalibrationContext {
        calibration_revision: applied.calibration_revision,
        camera_profile_id: applied.camera_profile_id,
        rectification_revision: applied.rectification_revision,
        image_space: match batch.image_space {
            BatchImageSpace::Raw => ImageSpace::Raw,
            BatchImageSpace::Rectified => ImageSpace::Rectified,
        },
        source_width_px: batch.source.width_px,
        source_height_px: batch.source.height_px,
    };
    calibration
        .validate_frame_context(context)
        .map_err(GeometryError::IncompatibleFrameCalibration)?;

    let mut detections = Vec::with_capacity(batch.detections.len());
    for detection in &batch.detections {
        let center_table_mm = calibration
            .project(
                ImagePointPx {
                    x_px: detection.center_image_px.x,
                    y_px: detection.center_image_px.y,
                },
                context,
            )
            .map_err(|source| GeometryError::DetectionProjection {
                detection_index: detection.detection_index,
                source,
            })?;
        detections.push(CalibratedDetection {
            detection_index: detection.detection_index,
            class_id: detection.class_id,
            class_name: detection.class_name.clone(),
            confidence: detection.confidence,
            center_image_px: detection.center_image_px,
            center_table_mm,
        });
    }

    Ok(CalibratedDetectionBatch {
        engine_instance_id: batch.engine_instance_id.clone(),
        batch_sequence: batch.batch_sequence,
        source: batch.source.clone(),
        image_space: batch.image_space,
        camera_profile_id: applied.camera_profile_id.to_owned(),
        calibration_revision: applied.calibration_revision,
        rectification_revision: applied.rectification_revision,
        detections,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use crate::calibration::{CalibrationError, TableCalibrationV1, ValidatedTableCalibrationV1};
    use crate::detection_batch::{decode_detection_batch_v1, DetectionBatchV1};

    use super::{calibrate_detection_batch, AppliedCalibrationContext, GeometryError};

    const CALIBRATION_FIXTURE: &str =
        include_str!("../contracts/calibration/v1/table-calibration-center-origin.json");
    const SUCCESS_BATCH_FIXTURE: &str =
        include_str!("../contracts/detection-batch/v1/detection-batch-success.json");
    const EMPTY_BATCH_FIXTURE: &str =
        include_str!("../contracts/detection-batch/v1/detection-batch-empty-success.json");
    const FAILURE_BATCH_FIXTURE: &str =
        include_str!("../contracts/detection-batch/v1/detection-batch-failure.json");

    fn calibration() -> ValidatedTableCalibrationV1 {
        serde_json::from_str::<TableCalibrationV1>(CALIBRATION_FIXTURE)
            .unwrap()
            .into_validated()
            .unwrap()
    }

    fn batch_from_fixture(source: &str) -> DetectionBatchV1 {
        let mut value: Value = serde_json::from_str(source).unwrap();
        value["source"]["width_px"] = json!(1000);
        value["source"]["height_px"] = json!(500);
        value["image_space"] = json!("rectified");
        if let Some(detection) = value["detections"]
            .as_array_mut()
            .and_then(|items| items.first_mut())
        {
            detection["bbox_image_px"] = json!({
                "left": 400.0,
                "top": 200.0,
                "right": 600.0,
                "bottom": 300.0
            });
            detection["center_image_px"] = json!({"x": 500.0, "y": 250.0});
        }
        decode_detection_batch_v1(&serde_json::to_vec(&value).unwrap()).unwrap()
    }

    fn applied_context(calibration_revision: u64) -> AppliedCalibrationContext<'static> {
        AppliedCalibrationContext {
            calibration_revision,
            camera_profile_id: "test-camera-profile",
            rectification_revision: 2,
        }
    }

    #[test]
    fn calibrated_batch_preserves_frame_identity_and_source_measurements() {
        let batch = batch_from_fixture(SUCCESS_BATCH_FIXTURE);
        let result = calibrate_detection_batch(&batch, &calibration(), applied_context(3)).unwrap();

        assert_eq!(result.engine_instance_id, batch.engine_instance_id);
        assert_eq!(result.batch_sequence, batch.batch_sequence);
        assert_eq!(result.source, batch.source);
        assert_eq!(result.detections.len(), 1);
        let calibrated = &result.detections[0];
        assert_eq!(
            calibrated.detection_index,
            batch.detections[0].detection_index
        );
        assert_eq!(
            calibrated.center_image_px,
            batch.detections[0].center_image_px
        );
        assert_eq!(calibrated.class_name, batch.detections[0].class_name);
        assert_eq!(calibrated.center_table_mm.x_mm, 0.0);
        assert_eq!(calibrated.center_table_mm.y_mm, 0.0);
    }

    #[test]
    fn failed_batch_is_never_reinterpreted_as_an_empty_calibrated_scene() {
        let batch = batch_from_fixture(FAILURE_BATCH_FIXTURE);
        let error = calibrate_detection_batch(&batch, &calibration(), applied_context(3))
            .expect_err("failed inference must not produce calibrated success");

        assert_eq!(
            error,
            GeometryError::DetectionBatchNotSuccessful(batch.outcome)
        );
    }

    #[test]
    fn empty_success_still_requires_the_applied_frame_calibration() {
        let batch = batch_from_fixture(EMPTY_BATCH_FIXTURE);
        let error = calibrate_detection_batch(&batch, &calibration(), applied_context(4))
            .expect_err("empty observations also carry calibration provenance");

        assert_eq!(
            error,
            GeometryError::IncompatibleFrameCalibration(
                CalibrationError::CalibrationRevisionMismatch
            )
        );
    }

    #[test]
    fn one_unprojectable_detection_rejects_the_whole_calibrated_batch() {
        let mut batch_value: Value = serde_json::from_str(SUCCESS_BATCH_FIXTURE).unwrap();
        batch_value["source"]["width_px"] = json!(1000);
        batch_value["source"]["height_px"] = json!(500);
        batch_value["image_space"] = json!("rectified");
        batch_value["detections"][0]["bbox_image_px"] = json!({
            "left": 100.0,
            "top": 200.0,
            "right": 500.0,
            "bottom": 300.0
        });
        batch_value["detections"][0]["center_image_px"] = json!({"x": 300.0, "y": 250.0});
        let mut second_detection = batch_value["detections"][0].clone();
        second_detection["detection_index"] = json!(1);
        second_detection["bbox_image_px"] = json!({
            "left": 450.0,
            "top": 200.0,
            "right": 550.0,
            "bottom": 300.0
        });
        second_detection["center_image_px"] = json!({"x": 500.0, "y": 250.0});
        batch_value["detections"]
            .as_array_mut()
            .unwrap()
            .push(second_detection);
        let batch = decode_detection_batch_v1(&serde_json::to_vec(&batch_value).unwrap()).unwrap();

        let mut calibration =
            serde_json::from_str::<TableCalibrationV1>(CALIBRATION_FIXTURE).unwrap();
        calibration.pixel_to_table_mm = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -500.0]];
        let error = calibrate_detection_batch(
            &batch,
            &calibration.into_validated().unwrap(),
            applied_context(3),
        )
        .expect_err("no partial transformed batch may escape");

        assert_eq!(
            error,
            GeometryError::DetectionProjection {
                detection_index: 1,
                source: CalibrationError::ProjectiveSingularity,
            }
        );
    }
}
