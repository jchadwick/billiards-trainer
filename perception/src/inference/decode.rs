use anyhow::{ensure, Result};
use serde::Serialize;

use super::preprocess::LetterboxGeometry;

/// A generic class detection; the current model supports cue and ball only.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Detection {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    pub bbox_xyxy_px: [f32; 4],
    pub center_xy_px: [f32; 2],
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    class_id: usize,
    confidence: f32,
    /// Model-space box used by Ultralytics NMS before clipping to the source.
    bbox_model: [f32; 4],
    /// Original-image box clipped after suppression, as in Ultralytics scaling.
    bbox_source: [f32; 4],
}

/// Decode the fixed YOLOv8 `[1, 4 + classes, anchors]` output and run
/// class-aware greedy NMS. Coordinates are converted from model letterbox
/// space into the original image's pixel space.
pub fn decode_yolo_v8_output(
    shape: &[usize],
    values: &[f32],
    class_names: &[String],
    geometry: LetterboxGeometry,
    confidence_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
) -> Result<Vec<Detection>> {
    ensure!(!class_names.is_empty(), "model class list is empty");
    ensure!(
        confidence_threshold.is_finite() && (0.0..=1.0).contains(&confidence_threshold),
        "confidence threshold must be finite and between 0 and 1"
    );
    ensure!(
        iou_threshold.is_finite() && (0.0..=1.0).contains(&iou_threshold),
        "IoU threshold must be finite and between 0 and 1"
    );
    ensure!(max_detections > 0, "max detections must be positive");
    ensure!(
        shape.len() == 3,
        "YOLO output must have three dimensions, got {shape:?}"
    );
    ensure!(
        shape[0] == 1,
        "only single-image model output is supported, got {shape:?}"
    );
    ensure!(
        shape[1] == 4 + class_names.len(),
        "output channel count does not match xywh plus the class map: {shape:?}"
    );
    ensure!(shape[2] > 0, "YOLO output contains no anchor rows");
    let required = shape[0]
        .checked_mul(shape[1])
        .and_then(|count| count.checked_mul(shape[2]))
        .ok_or_else(|| anyhow::anyhow!("YOLO output dimensions overflow addressable memory"))?;
    ensure!(
        values.len() == required,
        "YOLO output data length does not match shape"
    );
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "YOLO output contains NaN or infinity"
    );

    let channels = shape[1];
    let anchors = shape[2];
    let mut candidates = Vec::new();
    for anchor in 0..anchors {
        let mut best_class = 0;
        let mut best_score = f32::NEG_INFINITY;
        for class_id in 0..class_names.len() {
            let score = values[(4 + class_id) * anchors + anchor];
            if score > best_score {
                best_score = score;
                best_class = class_id;
            }
        }
        // Ultralytics uses a strict `score > confidence_threshold` check.
        if best_score <= confidence_threshold {
            continue;
        }

        let center_x = values[anchor];
        let center_y = values[anchors + anchor];
        let width = values[2 * anchors + anchor];
        let height = values[3 * anchors + anchor];
        ensure!(
            width >= 0.0 && height >= 0.0,
            "YOLO output contains a negative box size at anchor {anchor}"
        );
        let scale = geometry.scale;
        ensure!(
            scale.is_finite() && scale > 0.0,
            "letterbox scale is invalid"
        );
        let model_x1 = center_x - width / 2.0;
        let model_y1 = center_y - height / 2.0;
        let model_x2 = center_x + width / 2.0;
        let model_y2 = center_y + height / 2.0;
        let x1 = ((model_x1 - geometry.pad_left as f32) / scale)
            .clamp(0.0, geometry.source_width as f32);
        let y1 = ((model_y1 - geometry.pad_top as f32) / scale)
            .clamp(0.0, geometry.source_height as f32);
        let x2 = ((model_x2 - geometry.pad_left as f32) / scale)
            .clamp(0.0, geometry.source_width as f32);
        let y2 = ((model_y2 - geometry.pad_top as f32) / scale)
            .clamp(0.0, geometry.source_height as f32);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        candidates.push(Candidate {
            class_id: best_class,
            confidence: best_score,
            bbox_model: [model_x1, model_y1, model_x2, model_y2],
            bbox_source: [x1, y1, x2, y2],
        });
    }

    // Stable tie-breaking by anchor keeps repeated runs deterministic.
    candidates.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    let mut selected: Vec<Candidate> = Vec::with_capacity(max_detections.min(candidates.len()));
    for candidate in candidates {
        if selected.iter().any(|previous| {
            previous.class_id == candidate.class_id
                && intersection_over_union(previous.bbox_model, candidate.bbox_model)
                    > iou_threshold
        }) {
            continue;
        }
        selected.push(candidate);
        if selected.len() == max_detections {
            break;
        }
    }

    let mut detections: Vec<_> = selected
        .into_iter()
        .map(|candidate| {
            let [x1, y1, x2, y2] = candidate.bbox_source;
            Detection {
                class_id: candidate.class_id,
                class_name: class_names[candidate.class_id].clone(),
                confidence: candidate.confidence,
                bbox_xyxy_px: candidate.bbox_source,
                center_xy_px: [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            }
        })
        .collect();
    detections.sort_by(|left, right| {
        left.class_id
            .cmp(&right.class_id)
            .then_with(|| left.center_xy_px[0].total_cmp(&right.center_xy_px[0]))
            .then_with(|| left.center_xy_px[1].total_cmp(&right.center_xy_px[1]))
            .then_with(|| right.confidence.total_cmp(&left.confidence))
    });
    debug_assert_eq!(channels, 4 + class_names.len());
    Ok(detections)
}

fn intersection_over_union(left: [f32; 4], right: [f32; 4]) -> f32 {
    let intersection_width = (left[2].min(right[2]) - left[0].max(right[0])).max(0.0);
    let intersection_height = (left[3].min(right[3]) - left[1].max(right[1])).max(0.0);
    let intersection = intersection_width * intersection_height;
    let left_area = (left[2] - left[0]).max(0.0) * (left[3] - left[1]).max(0.0);
    let right_area = (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    intersection / (left_area + right_area - intersection).max(f32::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::decode_yolo_v8_output;
    use crate::inference::LetterboxGeometry;

    fn geometry() -> LetterboxGeometry {
        LetterboxGeometry {
            source_width: 100,
            source_height: 80,
            resized_width: 64,
            resized_height: 51,
            pad_left: 0,
            pad_top: 6,
            scale: 0.64,
            model_size: 64,
        }
    }

    fn names() -> Vec<String> {
        vec!["cue".to_owned(), "ball".to_owned()]
    }

    #[test]
    fn decodes_xywh_to_original_xyxy_and_center_coordinates() {
        // [cx, cy, w, h, cue score, ball score] at one anchor.
        let output = [32.0, 32.0, 12.8, 12.8, 0.1, 0.9];
        let detections =
            decode_yolo_v8_output(&[1, 6, 1], &output, &names(), geometry(), 0.4, 0.45, 300)
                .unwrap();

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].class_name, "ball");
        assert!((detections[0].center_xy_px[0] - 50.0).abs() < 0.001);
        assert!((detections[0].center_xy_px[1] - 40.625).abs() < 0.001);
    }

    #[test]
    fn confidence_filter_and_class_aware_nms_keep_different_classes() {
        let anchors = 3;
        let output = [
            32.0, 32.5, 32.0, // x centers
            32.0, 32.0, 32.0, // y centers
            16.0, 16.0, 16.0, // widths
            16.0, 16.0, 16.0, // heights
            0.95, 0.0, 0.1, // cue scores
            0.0, 0.90, 0.80, // ball scores
        ];
        let detections = decode_yolo_v8_output(
            &[1, 6, anchors],
            &output,
            &names(),
            geometry(),
            0.4,
            0.45,
            300,
        )
        .unwrap();

        // The overlapping cue candidate suppresses the other cue; overlapping
        // balls remain because NMS is class-aware. The 0.1 candidate is filtered.
        assert_eq!(detections.len(), 2);
        assert_eq!(
            detections
                .iter()
                .filter(|item| item.class_name == "cue")
                .count(),
            1
        );
        assert_eq!(
            detections
                .iter()
                .filter(|item| item.class_name == "ball")
                .count(),
            1
        );
    }

    #[test]
    fn confidence_equal_to_threshold_is_excluded() {
        let output = [32.0, 32.0, 16.0, 16.0, 0.0, 0.4];
        let detections =
            decode_yolo_v8_output(&[1, 6, 1], &output, &names(), geometry(), 0.4, 0.45, 300)
                .unwrap();

        assert!(detections.is_empty());
    }

    #[test]
    fn nms_uses_model_boxes_before_source_image_clipping() {
        // In model space these boxes overlap by 1/3, below the 0.4 NMS limit.
        // Clipping the first box to the source image raises its IoU with the
        // second to 1/2. Ultralytics suppresses before clipping, so both remain.
        let output = [
            0.0, 10.0, // center x
            25.2, 25.2, // center y
            20.0, 20.0, // width
            12.8, 12.8, // height
            0.0, 0.0, // cue scores
            0.9, 0.8, // ball scores
        ];
        let detections =
            decode_yolo_v8_output(&[1, 6, 2], &output, &names(), geometry(), 0.4, 0.4, 300)
                .unwrap();

        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].class_name, "ball");
        assert!((detections[0].bbox_xyxy_px[0] - 0.0).abs() < 0.001);
        assert!((detections[0].bbox_xyxy_px[2] - 15.625).abs() < 0.001);
        assert!((detections[1].bbox_xyxy_px[2] - 31.25).abs() < 0.001);
    }

    #[test]
    fn rejects_wrong_shapes_lengths_and_non_finite_values() {
        let error =
            decode_yolo_v8_output(&[1, 5, 1], &[0.0; 6], &names(), geometry(), 0.4, 0.45, 300);
        assert!(error.is_err());
        let error =
            decode_yolo_v8_output(&[1, 6, 1], &[0.0; 5], &names(), geometry(), 0.4, 0.45, 300);
        assert!(error.is_err());
        let error = decode_yolo_v8_output(
            &[1, 6, 1],
            &[0.0, 0.0, 0.0, f32::NAN, 0.0, 0.0],
            &names(),
            geometry(),
            0.4,
            0.45,
            300,
        );
        assert!(error.is_err());
    }
}
