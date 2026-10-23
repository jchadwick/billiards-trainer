//! Pure validation for ffprobe video-stream identity and presentation timing.
//!
//! This module intentionally does not launch ffprobe or ffmpeg. Callers own
//! subprocess timeouts and output collection, then pass the bounded ffprobe
//! JSON here. Stream selection is checked again against the index included
//! in the JSON so a command-line selector mistake cannot silently bind a
//! timeline to the wrong stream. Timestamps remain media presentation times;
//! callers must keep them separate from local monotonic runtime timestamps.

use anyhow::{bail, Context, Result};
use serde_json::Value;

/// Default upper bound for ffprobe JSON retained by an audit process (512 MiB).
pub const DEFAULT_MAXIMUM_PROBE_JSON_BYTES: usize = 512 * 1024 * 1024;
/// Default upper bound for frame timestamps retained by an audit process.
pub const DEFAULT_MAXIMUM_FRAME_COUNT: usize = 2_000_000;
/// Default upper bound on decoded image geometry.
pub const DEFAULT_MAXIMUM_PIXEL_COUNT: u64 = 64 * 1024 * 1024;

/// Resource bounds applied before allocating timestamp storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoTimelineLimits {
    pub maximum_probe_json_bytes: usize,
    pub maximum_frame_count: usize,
    pub maximum_pixel_count: u64,
}

impl Default for VideoTimelineLimits {
    fn default() -> Self {
        Self {
            maximum_probe_json_bytes: DEFAULT_MAXIMUM_PROBE_JSON_BYTES,
            maximum_frame_count: DEFAULT_MAXIMUM_FRAME_COUNT,
            maximum_pixel_count: DEFAULT_MAXIMUM_PIXEL_COUNT,
        }
    }
}

/// Validated geometry and normalized presentation times for one video stream.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoTimeline {
    /// ffprobe's global stream index. Callers should select this exact stream
    /// when decoding rather than assuming the stream is the first video track.
    pub stream_index: u32,
    pub width: u32,
    pub height: u32,
    /// First source PTS before normalization, in seconds.
    pub origin_seconds: f64,
    /// Strictly increasing source PTS values relative to origin_seconds.
    pub presentation_times_seconds: Vec<f64>,
}

impl VideoTimeline {
    pub fn frame_count(&self) -> usize {
        self.presentation_times_seconds.len()
    }
}

/// Parse and validate ffprobe JSON emitted for exactly one selected stream.
///
/// The JSON must include the selected stream's global index, codec_type,
/// width, and height, plus each frame's best_effort_timestamp_time.
/// Rotation metadata is accepted only when it explicitly describes zero
/// degrees; nonzero and unrecognized display-matrix transforms are rejected
/// because this parser does not transform decoded frame pixels.
pub fn parse_ffprobe_timeline(
    bytes: &[u8],
    selected_stream_index: u32,
    limits: VideoTimelineLimits,
) -> Result<VideoTimeline> {
    if bytes.len() > limits.maximum_probe_json_bytes {
        bail!(
            "ffprobe JSON is {} bytes, exceeding configured limit {}",
            bytes.len(),
            limits.maximum_probe_json_bytes
        );
    }
    if limits.maximum_frame_count == 0 {
        bail!("maximum ffprobe frame count must be greater than zero");
    }
    if limits.maximum_pixel_count == 0 {
        bail!("maximum video pixel count must be greater than zero");
    }

    let root: Value = serde_json::from_slice(bytes).context("ffprobe returned invalid JSON")?;
    let root = root
        .as_object()
        .context("ffprobe output root must be an object")?;
    let streams = root
        .get("streams")
        .and_then(Value::as_array)
        .context("ffprobe streams must be an array")?;
    let frames = root
        .get("frames")
        .and_then(Value::as_array)
        .context("ffprobe frames must be an array")?;

    if streams.len() != 1 {
        bail!("ffprobe output must contain exactly one selected video stream");
    }
    let stream = streams[0]
        .as_object()
        .context("ffprobe selected stream must be an object")?;
    let stream_index = required_u32(
        stream.get("index"),
        "video stream index is missing or invalid",
    )?;
    if stream_index != selected_stream_index {
        bail!(
            "ffprobe selected stream index {stream_index} does not match requested index {selected_stream_index}"
        );
    }
    if stream.get("codec_type").and_then(Value::as_str) != Some("video") {
        bail!("ffprobe selected stream is not a video stream");
    }
    validate_rotation(stream, "video stream")?;

    let width = positive_u32(
        stream.get("width"),
        "video stream width is missing or invalid",
    )?;
    let height = positive_u32(
        stream.get("height"),
        "video stream height is missing or invalid",
    )?;
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .context("video stream pixel count overflowed")?;
    if pixels > limits.maximum_pixel_count {
        bail!(
            "video stream geometry {width}x{height} exceeds configured pixel limit {}",
            limits.maximum_pixel_count
        );
    }
    if frames.is_empty() {
        bail!("video contains no decodable video frames");
    }
    if frames.len() > limits.maximum_frame_count {
        bail!(
            "video frame count {} exceeds configured limit {}",
            frames.len(),
            limits.maximum_frame_count
        );
    }

    let mut source_times = Vec::with_capacity(frames.len());
    for (frame_index, frame) in frames.iter().enumerate() {
        let object = frame
            .as_object()
            .with_context(|| format!("ffprobe frame {frame_index} must be an object"))?;
        validate_rotation(object, &format!("video frame {frame_index}"))?;
        let raw = object.get("best_effort_timestamp_time").with_context(|| {
            format!("ffprobe frame {frame_index} has no presentation timestamp")
        })?;
        let timestamp = parse_seconds(raw).with_context(|| {
            format!("ffprobe frame {frame_index} has an invalid presentation timestamp")
        })?;
        if source_times
            .last()
            .is_some_and(|previous| timestamp <= *previous)
        {
            bail!("video presentation timestamps must be finite and strictly increasing");
        }
        source_times.push(timestamp);
    }

    let origin_seconds = source_times[0];
    for timestamp in &mut source_times {
        *timestamp -= origin_seconds;
    }
    if source_times.iter().any(|timestamp| !timestamp.is_finite()) {
        bail!("normalized video presentation timestamps must be finite");
    }
    if source_times.windows(2).any(|pair| pair[1] <= pair[0]) {
        bail!("normalized video presentation timestamps must be strictly increasing");
    }

    Ok(VideoTimeline {
        stream_index,
        width,
        height,
        origin_seconds,
        presentation_times_seconds: source_times,
    })
}

fn validate_rotation(object: &serde_json::Map<String, Value>, label: &str) -> Result<()> {
    if let Some(tags) = object.get("tags") {
        let tags = tags
            .as_object()
            .with_context(|| format!("{label} tags must be an object"))?;
        if let Some(rotation) = tags.get("rotate") {
            ensure_zero_rotation(rotation, &format!("{label} rotate tag"))?;
        }
    }
    if let Some(side_data) = object.get("side_data_list") {
        let side_data = side_data
            .as_array()
            .with_context(|| format!("{label} side_data_list must be an array"))?;
        for (index, item) in side_data.iter().enumerate() {
            let item = item
                .as_object()
                .with_context(|| format!("{label} side-data entry {index} must be an object"))?;
            if let Some(rotation) = item.get("rotation") {
                ensure_zero_rotation(rotation, &format!("{label} side-data rotation"))?;
            } else if item.contains_key("displaymatrix") {
                bail!("{label} has a display matrix without a supported rotation value");
            }
        }
    }
    Ok(())
}

fn ensure_zero_rotation(value: &Value, label: &str) -> Result<()> {
    let rotation = parse_seconds(value).with_context(|| {
        format!("{label} is malformed; only explicit zero-degree rotation is supported")
    })?;
    if rotation != 0.0 {
        bail!("{label} is {rotation} degrees; rotated video is not supported");
    }
    Ok(())
}

fn parse_seconds(value: &Value) -> Option<f64> {
    let number = value
        .as_str()
        .and_then(|string| string.parse::<f64>().ok())
        .or_else(|| value.as_f64())?;
    number.is_finite().then_some(number)
}

fn required_u32(value: Option<&Value>, message: &str) -> Result<u32> {
    value
        .and_then(Value::as_u64)
        .and_then(|number| u32::try_from(number).ok())
        .with_context(|| message.to_owned())
}

fn positive_u32(value: Option<&Value>, message: &str) -> Result<u32> {
    required_u32(value, message).and_then(|number| {
        if number == 0 {
            bail!("{message}");
        }
        Ok(number)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn timeline_json() -> Value {
        json!({
            "streams": [{
                "index": 3,
                "codec_type": "video",
                "width": 1920,
                "height": 1080
            }],
            "frames": [
                {"best_effort_timestamp_time": "12.000"},
                {"best_effort_timestamp_time": "12.040"},
                {"best_effort_timestamp_time": "12.110"}
            ]
        })
    }

    fn parse(value: &Value, limits: VideoTimelineLimits) -> Result<VideoTimeline> {
        parse_ffprobe_timeline(&serde_json::to_vec(value)?, 3, limits)
    }

    #[test]
    fn validates_selected_stream_and_normalizes_strictly_increasing_pts() {
        let timeline = parse(&timeline_json(), VideoTimelineLimits::default()).unwrap();
        assert_eq!(timeline.stream_index, 3);
        assert_eq!((timeline.width, timeline.height), (1920, 1080));
        assert_eq!(timeline.origin_seconds, 12.0);
        assert_eq!(timeline.frame_count(), 3);
        assert_eq!(timeline.presentation_times_seconds[0], 0.0);
        assert!((timeline.presentation_times_seconds[1] - 0.04).abs() < 1e-12);
        assert!((timeline.presentation_times_seconds[2] - 0.11).abs() < 1e-12);
    }

    #[test]
    fn rejects_wrong_missing_or_ambiguous_selected_stream_identity() {
        let mut wrong = timeline_json();
        wrong["streams"][0]["index"] = json!(2);
        assert!(parse(&wrong, VideoTimelineLimits::default())
            .unwrap_err()
            .to_string()
            .contains("does not match"));

        let mut missing = timeline_json();
        missing["streams"][0]
            .as_object_mut()
            .unwrap()
            .remove("index");
        assert!(parse(&missing, VideoTimelineLimits::default()).is_err());

        let mut ambiguous = timeline_json();
        ambiguous["streams"].as_array_mut().unwrap().push(json!({
            "index": 4, "codec_type": "video", "width": 1, "height": 1
        }));
        assert!(parse(&ambiguous, VideoTimelineLimits::default()).is_err());
    }

    #[test]
    fn rejects_bad_timestamps_and_empty_or_oversized_frame_lists() {
        for frames in [
            json!([]),
            json!([{"best_effort_timestamp_time": "NaN"}]),
            json!([{"best_effort_timestamp_time": "inf"}]),
            json!([
                {"best_effort_timestamp_time": "1.0"},
                {"best_effort_timestamp_time": "1.0"}
            ]),
            json!([
                {"best_effort_timestamp_time": "2.0"},
                {"best_effort_timestamp_time": "1.0"}
            ]),
        ] {
            let mut value = timeline_json();
            value["frames"] = frames;
            assert!(parse(&value, VideoTimelineLimits::default()).is_err());
        }

        let mut oversized = timeline_json();
        oversized["frames"].as_array_mut().unwrap().push(json!({
            "best_effort_timestamp_time": "12.120"
        }));
        let limits = VideoTimelineLimits {
            maximum_frame_count: 3,
            ..VideoTimelineLimits::default()
        };
        assert!(parse(&oversized, limits)
            .unwrap_err()
            .to_string()
            .contains("frame count"));
    }

    #[test]
    fn rejects_invalid_geometry_and_pixel_limit_excess() {
        for geometry in [(0, 1080), (1920, 0), (u32::MAX, u32::MAX)] {
            let mut value = timeline_json();
            value["streams"][0]["width"] = json!(geometry.0);
            value["streams"][0]["height"] = json!(geometry.1);
            assert!(parse(&value, VideoTimelineLimits::default()).is_err());
        }

        let limits = VideoTimelineLimits {
            maximum_pixel_count: 100,
            ..VideoTimelineLimits::default()
        };
        assert!(parse(&timeline_json(), limits)
            .unwrap_err()
            .to_string()
            .contains("pixel limit"));
    }

    #[test]
    fn rejects_nonzero_and_uninterpretable_rotation_metadata() {
        let mut tagged = timeline_json();
        tagged["streams"][0]["tags"] = json!({"rotate": "90"});
        assert!(parse(&tagged, VideoTimelineLimits::default())
            .unwrap_err()
            .to_string()
            .contains("not supported"));

        let mut malformed_rotation = timeline_json();
        malformed_rotation["streams"][0]["tags"] = json!({"rotate": "upright"});
        assert!(parse(&malformed_rotation, VideoTimelineLimits::default())
            .unwrap_err()
            .to_string()
            .contains("malformed"));

        let mut malformed = timeline_json();
        malformed["streams"][0]["side_data_list"] = json!([{
            "side_data_type": "Display Matrix",
            "displaymatrix": "unparsed"
        }]);
        assert!(parse(&malformed, VideoTimelineLimits::default())
            .unwrap_err()
            .to_string()
            .contains("display matrix"));

        let mut zero = timeline_json();
        zero["streams"][0]["side_data_list"] = json!([{
            "side_data_type": "Display Matrix",
            "rotation": 0
        }]);
        assert!(parse(&zero, VideoTimelineLimits::default()).is_ok());
    }

    #[test]
    fn bounds_probe_json_and_rejects_zero_limits() {
        let bytes = serde_json::to_vec(&timeline_json()).unwrap();
        let limits = VideoTimelineLimits {
            maximum_probe_json_bytes: bytes.len() - 1,
            ..VideoTimelineLimits::default()
        };
        assert!(parse_ffprobe_timeline(&bytes, 3, limits)
            .unwrap_err()
            .to_string()
            .contains("exceeding configured limit"));

        let limits = VideoTimelineLimits {
            maximum_frame_count: 0,
            ..VideoTimelineLimits::default()
        };
        assert!(parse(&timeline_json(), limits).is_err());
    }
}
