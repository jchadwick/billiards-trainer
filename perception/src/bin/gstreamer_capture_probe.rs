//! Opt-in GStreamer capture helper process.
//!
//! This executable runs as a supervised capture helper rather than owning the
//! live session itself. Synthetic `videotestsrc` mode is the default. The
//! separately compiled physical mode binds one exact discovery selector,
//! verifies camera input caps independently from emitted RGB8 caps, and writes
//! no camera controls. The live executable starts physical mode only with its
//! explicit feature-gated source choice; its normal source remains unavailable.

use std::io::{self, Write};
use std::process;
use std::thread;
use std::time::Instant;

use anyhow::{anyhow, ensure, Context, Result};
use billiards_perception::capture::{CaptureFrameBudget, CaptureHelperTimingNs, RgbFrameEnvelope};
#[cfg(feature = "camera-gstreamer-source")]
use billiards_perception::capture_helper_protocol::CaptureHelperInputProfile;
use billiards_perception::capture_helper_protocol::{write_ready_record, CaptureHelperReady};
use billiards_perception::capture_settings_wire::parse_capture_settings_json;
use billiards_perception::live_application::read_capture_settings;
use clap::{Parser, ValueEnum};
use gst::prelude::*;
use gstreamer as gst;
use gstreamer_app::AppSink;
use gstreamer_video::{VideoFormat, VideoFrameExt, VideoFrameRef, VideoInfo};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Run a bounded synthetic or opt-in physical GStreamer capture helper")]
struct Args {
    /// Explicit helper mode. Physical capture is compiled only by the opt-in
    /// camera-gstreamer-source feature and is never selected by live_main.
    #[arg(long, value_enum, default_value_t = SourceMode::Synthetic)]
    source_mode: SourceMode,
    /// URL used only by the explicit HTTP-MJPEG source mode.
    #[arg(long)]
    http_mjpeg_url: Option<String>,
    /// Required session identity supplied by the supervising process.
    #[arg(long)]
    session_id: String,
    /// Maximum bytes accepted for the exact capture-settings request on stdin.
    #[arg(long)]
    capture_settings_max_bytes: usize,
    /// Maximum JSON bytes in the helper-ready record written before frame 0.
    #[arg(long)]
    ready_record_max_bytes: usize,
    /// Finite number of frames to emit before exiting. Required for synthetic
    /// mode; physical mode may instead run until its supervising process stops it.
    #[arg(long, conflicts_with = "until_stopped")]
    frames: Option<u64>,
    /// Keep physical capture running until the supervisor terminates this
    /// helper. This is intentionally unavailable for synthetic runs.
    #[arg(long, conflicts_with = "frames")]
    until_stopped: bool,
    /// Delay before each pull to exercise the one-buffer leaky queue.
    #[arg(long)]
    slow_reader_ms: u64,
    /// Maximum time to wait for one synthetic sample.
    #[arg(long)]
    pull_timeout_ms: u64,
    /// Collect scalar capture-stage durations for the opt-in live audit.
    #[arg(long, default_value_t = false)]
    capture_audit_timing: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SourceMode {
    Synthetic,
    Physical,
    HttpMjpeg,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
struct FrameHeader<'a> {
    record_schema_version: u8,
    session_id: &'a str,
    frame_id: String,
    configuration_revision: u64,
    camera_profile_id: &'a str,
    capture_settings_sha256: &'a str,
    width_px: u32,
    height_px: u32,
    pixel_format: &'static str,
    row_stride_bytes: usize,
    /// Kept opaque: PTS is pipeline presentation time, not verified source time.
    unverified_pipeline_pts_ticks: Option<String>,
    /// No synthetic source timestamp is presented as camera exposure time.
    source_timestamp_ticks: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    host_read_completed_elapsed_ns: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    capture_timings_ns: Option<CaptureHelperTimingNs>,
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("GStreamer capture helper failed: {error:#}");
        process::exit(2);
    }
}

fn run(args: Args) -> Result<()> {
    validate_args(&args)?;
    if args.source_mode == SourceMode::Physical {
        #[cfg(feature = "camera-gstreamer-source")]
        {
            return run_physical(args);
        }
        #[cfg(not(feature = "camera-gstreamer-source"))]
        anyhow::bail!(
            "physical capture helper was not compiled; enable camera-gstreamer-source explicitly"
        );
    }
    if args.source_mode == SourceMode::HttpMjpeg {
        #[cfg(feature = "camera-gstreamer-source")]
        {
            return run_http_mjpeg(args);
        }
        #[cfg(not(feature = "camera-gstreamer-source"))]
        anyhow::bail!(
            "HTTP-MJPEG capture helper was not compiled; enable camera-gstreamer-source explicitly"
        );
    }
    let mut stdin = io::stdin().lock();
    let exact_capture_settings =
        read_capture_settings(&mut stdin, args.capture_settings_max_bytes)?;
    let capture_settings = parse_capture_settings_json(&exact_capture_settings)
        .context("invalid exact capture-settings request")?;
    ensure!(
        capture_settings.requested_backend == "auto"
            || capture_settings.requested_backend == "gstreamer",
        "synthetic GStreamer probe cannot satisfy requested backend {:?}",
        capture_settings.requested_backend
    );

    let expected = &capture_settings.expected;
    let profile = expected.expected_negotiated_profile();
    let (width, height) = expected
        .output_frame_geometry()
        .map(|geometry| (geometry.width_px(), geometry.height_px()))
        .unwrap_or((profile.width_px(), profile.height_px()));
    let frame_rate = profile.frame_rate();
    let fps_numerator = frame_rate.frames_numerator();
    let fps_denominator = frame_rate.seconds_denominator();
    let frame_limit = expected.frame_limit();
    let capture_settings_sha256 = capture_settings.request_sha256.as_str();
    ensure!(
        profile.pixel_format().identifier() == "RGB8",
        "synthetic GStreamer probe supports only requested RGB8 output, got {:?}",
        profile.pixel_format().identifier()
    );
    gst::init().context("initialize GStreamer")?;

    let frame_budget = CaptureFrameBudget::new(expected.aggregate_frame_limit());
    let expected_len = RgbFrameEnvelope::required_rgb8_len(width, height, frame_limit)?;

    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field(
            "width",
            i32::try_from(width).context("width exceeds GStreamer range")?,
        )
        .field(
            "height",
            i32::try_from(height).context("height exceeds GStreamer range")?,
        )
        .field(
            "framerate",
            gst::Fraction::new(
                i32::try_from(fps_numerator)
                    .context("frame-rate numerator exceeds GStreamer range")?,
                i32::try_from(fps_denominator)
                    .context("frame-rate denominator exceeds GStreamer range")?,
            ),
        )
        .build();
    let pipeline_description = format!(
        "videotestsrc is-live=true ! videoconvert ! video/x-raw,format=RGB,width={},height={},framerate={}/{} ! appsink name=probe_sink max-buffers=1 drop=true wait-on-eos=false sync=false",
        width, height, fps_numerator, fps_denominator
    );
    let pipeline = gst::parse::launch(&pipeline_description)
        .context("build synthetic videotestsrc pipeline")?
        .downcast::<gst::Pipeline>()
        .map_err(|_| anyhow!("synthetic pipeline parser did not return a pipeline"))?;
    let sink = pipeline
        .by_name("probe_sink")
        .ok_or_else(|| anyhow!("synthetic pipeline has no named appsink"))?
        .downcast::<AppSink>()
        .map_err(|_| anyhow!("synthetic pipeline sink is not an appsink"))?;
    // Set the same typed caps used in the textual pipeline and verify the
    // negotiated sample caps again after every pull.
    sink.set_caps(Some(&caps));
    pipeline
        .set_state(gst::State::Playing)
        .context("start synthetic GStreamer pipeline")?;
    let _pipeline_guard = PipelineStateGuard(&pipeline);

    let legacy_helper_start = (!args.capture_audit_timing).then(Instant::now);
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());

    // Do not announce readiness until GStreamer has produced a real sample and
    // its actual negotiated caps match the request. The same sample then
    // becomes frame 0, so startup validation does not discard an observation.
    let first_sample_pull_started = args.capture_audit_timing.then(Instant::now);
    let first_sample = sink
        .try_pull_sample(gst::ClockTime::from_mseconds(args.pull_timeout_ms))
        .ok_or_else(|| anyhow!("timed out waiting for first synthetic sample"))?;
    let first_sample_pull_wait_ns = elapsed_nanos(first_sample_pull_started);
    let first_caps = first_sample
        .caps()
        .ok_or_else(|| anyhow!("first synthetic sample has no negotiated caps"))?;
    let first_video_info = VideoInfo::from_caps(first_caps)
        .context("parse first negotiated sample video information")?;
    let (actual_fps_numerator, actual_fps_denominator) = validate_video_info(
        &first_video_info,
        width,
        height,
        fps_numerator,
        fps_denominator,
    )?;
    let ready = CaptureHelperReady {
        record_schema_version: 1,
        session_id: args.session_id.clone(),
        configuration_revision: capture_settings.configuration_revision,
        camera_profile_id: expected.camera_profile_id().to_owned(),
        capture_settings_sha256: capture_settings_sha256.to_owned(),
        width_px: first_video_info.width(),
        height_px: first_video_info.height(),
        frame_rate_numerator: actual_fps_numerator,
        frame_rate_seconds_denominator: actual_fps_denominator,
        // GStreamer caps use `RGB`; the capture-settings contract names the
        // equivalent packed eight-bit output format `RGB8`.
        pixel_format: profile.pixel_format().identifier().to_owned(),
        source_device_id: None,
        source_identifier: None,
        source_identifier_source: None,
        input_profile: None,
    };
    write_ready_record(&mut output, &ready, args.ready_record_max_bytes)
        .context("write bounded helper-ready record to stdout")?;
    output.flush().context("flush helper-ready record")?;

    let mut first_sample = Some(first_sample);
    let mut sequence = 0_u64;
    loop {
        if args.frames.is_some_and(|limit| sequence >= limit) {
            break;
        }
        if sequence > 0 && args.slow_reader_ms > 0 {
            thread::sleep(std::time::Duration::from_millis(args.slow_reader_ms));
        }
        let (sample, appsink_pull_wait_ns) = if sequence == 0 {
            (
                first_sample
                    .take()
                    .expect("the validated first sample is available for frame 0"),
                first_sample_pull_wait_ns,
            )
        } else {
            let pull_started = args.capture_audit_timing.then(Instant::now);
            let sample = sink
                .try_pull_sample(gst::ClockTime::from_mseconds(args.pull_timeout_ms))
                .ok_or_else(|| anyhow!("timed out waiting for synthetic sample {sequence}"))?;
            (sample, elapsed_nanos(pull_started))
        };
        let sample_caps = sample
            .caps()
            .ok_or_else(|| anyhow!("synthetic sample has no negotiated caps"))?;
        let video_info = VideoInfo::from_caps(sample_caps)
            .context("parse negotiated sample video information")?;
        validate_video_info(&video_info, width, height, fps_numerator, fps_denominator)?;

        let buffer = sample
            .buffer()
            .ok_or_else(|| anyhow!("synthetic sample has no frame buffer"))?;
        let allocation_started = args.capture_audit_timing.then(Instant::now);
        let mut owned_pixels = frame_budget
            .allocate_rgb8_buffer(width, height, frame_limit)
            .context("allocate budgeted owned RGB frame")?;
        let rgb_buffer_allocation_ns = elapsed_nanos(allocation_started);
        let copy_started = args.capture_audit_timing.then(Instant::now);
        let stride = copy_mapped_rgb_frame(
            buffer,
            &video_info,
            width,
            height,
            owned_pixels.as_mut_slice(),
        )?;
        let rgb_mapping_copy_ns = elapsed_nanos(copy_started);
        ensure!(
            owned_pixels.len() == expected_len,
            "RGB output length changed"
        );

        let header = FrameHeader {
            record_schema_version: if args.capture_audit_timing { 3 } else { 2 },
            session_id: &args.session_id,
            frame_id: sequence.to_string(),
            configuration_revision: capture_settings.configuration_revision,
            camera_profile_id: expected.camera_profile_id(),
            capture_settings_sha256,
            width_px: width,
            height_px: height,
            pixel_format: "rgb8",
            row_stride_bytes: stride,
            unverified_pipeline_pts_ticks: buffer.pts().map(|pts| pts.nseconds().to_string()),
            source_timestamp_ticks: None,
            host_read_completed_elapsed_ns: legacy_helper_start
                .map(|started| started.elapsed().as_nanos().to_string()),
            capture_timings_ns: args.capture_audit_timing.then(|| CaptureHelperTimingNs {
                appsink_pull_wait_ns: appsink_pull_wait_ns
                    .expect("timed mode measures appsink wait"),
                rgb_buffer_allocation_ns: rgb_buffer_allocation_ns
                    .expect("timed mode measures RGB allocation"),
                rgb_mapping_copy_ns: rgb_mapping_copy_ns
                    .expect("timed mode measures RGB mapping and copy"),
            }),
        };
        write_record(&mut output, &header, owned_pixels.as_slice())
            .context("write bounded frame record to helper stdout")?;
        output.flush().context("flush helper frame record")?;
        sequence = sequence
            .checked_add(1)
            .context("synthetic frame sequence overflowed")?;
    }
    Ok(())
}

/// Open one exact DeviceMonitor result by its hashed, versioned selector and
/// stream bounded records until the configured frame limit or supervisor stop.
/// No camera-control property is inspected or written on this path.
#[cfg(feature = "camera-gstreamer-source")]
fn run_physical(args: Args) -> Result<()> {
    use billiards_perception::camera_discovery::select_capture_selector_index;
    use billiards_perception::capture_settings_wire::ResolvedCameraControlsPolicy;
    use std::sync::{Arc, Mutex};

    let mut stdin = io::stdin().lock();
    let exact_capture_settings =
        read_capture_settings(&mut stdin, args.capture_settings_max_bytes)?;
    let capture_settings = parse_capture_settings_json(&exact_capture_settings)
        .context("invalid exact capture-settings request")?;
    ensure!(
        capture_settings.schema_version == 3,
        "physical helper requires capture-settings v3"
    );
    ensure!(
        matches!(
            &capture_settings.camera_controls,
            ResolvedCameraControlsPolicy::SourceDefaults
        ),
        "physical helper supports source_defaults only and never writes camera controls"
    );
    ensure!(
        capture_settings.requested_backend == "auto"
            || capture_settings.requested_backend == "gstreamer",
        "physical GStreamer helper cannot satisfy requested backend {:?}",
        capture_settings.requested_backend
    );
    ensure!(capture_settings.expected.image_space() == billiards_perception::ImageSpace::Raw);
    ensure!(capture_settings.expected.rectification_revision() == 0);

    let selector = capture_settings.expected.device_id();
    ensure!(
        billiards_perception::camera_discovery::is_capture_selector(selector),
        "physical helper requires a gstreamer-device-v1 selector from camera discovery"
    );
    let input_profile = capture_settings.expected.expected_negotiated_profile();
    let output_geometry = capture_settings
        .expected
        .output_frame_geometry()
        .context("physical helper requires capture-settings v3 output geometry")?;

    let (input_caps, output_caps) = physical_capture_caps(
        input_profile,
        output_geometry.width_px(),
        output_geometry.height_px(),
    )?;

    let monitor = gst::DeviceMonitor::new();
    ensure!(
        monitor.add_filter(Some("Video/Source"), None).is_some(),
        "GStreamer has no Video/Source device provider"
    );
    monitor
        .start()
        .context("start GStreamer camera discovery")?;
    let _monitor_guard = DeviceMonitorStopGuard(monitor.clone());
    let devices = monitor.devices();
    let camera_devices = devices
        .iter()
        .map(camera_device_metadata)
        .collect::<Vec<_>>();
    let selected_index = select_capture_selector_index(selector, &camera_devices)
        .context("resolve exact stable camera selector")?;
    let selected = devices
        .iter()
        .nth(selected_index)
        .context("selected camera disappeared from DeviceMonitor inventory")?;
    let selected_metadata = camera_devices
        .get(selected_index)
        .context("selected camera metadata disappeared from DeviceMonitor inventory")?;
    let source_identifier = selected_metadata
        .identifier_value
        .as_ref()
        .context("selected camera has no persistent provider identifier")?;
    let source_identifier_source = selected_metadata
        .identifier_source
        .as_ref()
        .context("selected camera has no persistent identifier property key")?;
    let source = selected
        .create_element(Some("selected_camera_source"))
        .context("create source element for the selected DeviceMonitor camera")?;

    let input_filter = gst::ElementFactory::make("capsfilter")
        .name("camera_input_profile")
        .build()
        .context("create camera input caps filter")?;
    input_filter.set_property("caps", &input_caps);
    let decoder = gst::ElementFactory::make("decodebin")
        .name("camera_decoder")
        .build()
        .context("create camera decoder")?;
    let converter = gst::ElementFactory::make("videoconvert")
        .name("camera_rgb_converter")
        .build()
        .context("create RGB converter")?;
    let scaler = gst::ElementFactory::make("videoscale")
        .name("camera_output_scaler")
        .build()
        .context("create output scaler")?;
    let output_filter = gst::ElementFactory::make("capsfilter")
        .name("camera_rgb8_output_profile")
        .build()
        .context("create RGB output caps filter")?;
    output_filter.set_property("caps", &output_caps);
    let sink_element = gst::ElementFactory::make("appsink")
        .name("physical_camera_sink")
        .build()
        .context("create physical camera appsink")?;
    let sink = sink_element
        .clone()
        .downcast::<AppSink>()
        .map_err(|_| anyhow!("physical helper sink is not an appsink"))?;
    sink.set_caps(Some(&output_caps));
    sink.set_max_buffers(1);
    sink.set_drop(true);
    sink.set_wait_on_eos(false);
    sink.set_sync(false);

    let pipeline = gst::Pipeline::new();
    let elements: [&gst::Element; 7] = [
        &source,
        &input_filter,
        &decoder,
        &converter,
        &scaler,
        &output_filter,
        sink.upcast_ref(),
    ];
    pipeline
        .add_many(elements)
        .context("add physical camera pipeline elements")?;
    source
        .link(&input_filter)
        .context("link selected camera to the input caps filter")?;
    input_filter
        .link(&decoder)
        .context("link camera input caps to decoder")?;
    converter
        .link(&scaler)
        .context("link RGB converter to output scaler")?;
    scaler
        .link(&output_filter)
        .context("link output scaler to RGB caps filter")?;
    output_filter
        .link(&sink)
        .context("link RGB caps filter to bounded appsink")?;

    let converter_sink = converter
        .static_pad("sink")
        .context("RGB converter has no sink pad")?;
    let dynamic_link_error = Arc::new(Mutex::new(None::<String>));
    let link_error_for_callback = Arc::clone(&dynamic_link_error);
    decoder.connect_pad_added(move |_decoder, source_pad| {
        let caps = source_pad
            .current_caps()
            .unwrap_or_else(|| source_pad.query_caps(None));
        let is_video = caps
            .structure(0)
            .is_some_and(|structure| structure.name().starts_with("video/"));
        if !is_video {
            return;
        }
        if let Err(error) = source_pad.link(&converter_sink) {
            *link_error_for_callback
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error.to_string());
        }
    });

    pipeline
        .set_state(gst::State::Playing)
        .context("start physical camera pipeline")?;
    let _pipeline_guard = PipelineStateGuard(&pipeline);
    let legacy_helper_start = (!args.capture_audit_timing).then(Instant::now);
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    let first_sample_pull_started = args.capture_audit_timing.then(Instant::now);
    let first_sample = sink
        .try_pull_sample(gst::ClockTime::from_mseconds(args.pull_timeout_ms))
        .ok_or_else(|| anyhow!("timed out waiting for first physical camera sample"))?;
    let first_sample_pull_wait_ns = elapsed_nanos(first_sample_pull_started);

    if let Some(error) = dynamic_link_error
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
    {
        return Err(anyhow!("could not link decoded camera video: {error}"));
    }
    let actual_input_caps = input_filter
        .static_pad("sink")
        .and_then(|pad| pad.current_caps())
        .context("camera source did not negotiate input caps")?;
    let actual_input_profile = parse_input_profile(&actual_input_caps)?;
    validate_input_profile(&actual_input_profile, input_profile)?;

    let first_caps = first_sample
        .caps()
        .context("first physical sample has no negotiated RGB caps")?;
    let first_video_info = VideoInfo::from_caps(first_caps)
        .context("parse first physical output sample video information")?;
    let rate = input_profile.frame_rate();
    validate_video_info(
        &first_video_info,
        output_geometry.width_px(),
        output_geometry.height_px(),
        rate.frames_numerator(),
        rate.seconds_denominator(),
    )?;

    let ready = CaptureHelperReady {
        record_schema_version: 2,
        session_id: args.session_id.clone(),
        configuration_revision: capture_settings.configuration_revision,
        camera_profile_id: capture_settings.expected.camera_profile_id().to_owned(),
        capture_settings_sha256: capture_settings.request_sha256.clone(),
        width_px: first_video_info.width(),
        height_px: first_video_info.height(),
        frame_rate_numerator: rate.frames_numerator(),
        frame_rate_seconds_denominator: rate.seconds_denominator(),
        pixel_format: "RGB8".to_owned(),
        source_device_id: Some(selector.to_owned()),
        source_identifier: Some(source_identifier.clone()),
        source_identifier_source: Some(source_identifier_source.clone()),
        input_profile: Some(actual_input_profile),
    };
    write_ready_record(&mut output, &ready, args.ready_record_max_bytes)
        .context("write bounded physical helper-ready record")?;
    output
        .flush()
        .context("flush physical helper-ready record")?;

    let frame_budget = CaptureFrameBudget::new(capture_settings.expected.aggregate_frame_limit());
    let frame_limit = capture_settings.expected.frame_limit();
    let expected_len = RgbFrameEnvelope::required_rgb8_len(
        output_geometry.width_px(),
        output_geometry.height_px(),
        frame_limit,
    )?;
    let mut first_sample = Some(first_sample);
    let mut sequence = 0_u64;
    loop {
        if args.frames.is_some_and(|limit| sequence >= limit) {
            break;
        }
        if sequence > 0 && args.slow_reader_ms > 0 {
            thread::sleep(std::time::Duration::from_millis(args.slow_reader_ms));
        }
        let (sample, appsink_pull_wait_ns) = if sequence == 0 {
            (
                first_sample
                    .take()
                    .expect("the validated first physical sample is frame zero"),
                first_sample_pull_wait_ns,
            )
        } else {
            let pull_started = args.capture_audit_timing.then(Instant::now);
            let sample = sink
                .try_pull_sample(gst::ClockTime::from_mseconds(args.pull_timeout_ms))
                .ok_or_else(|| anyhow!("timed out waiting for physical sample {sequence}"))?;
            (sample, elapsed_nanos(pull_started))
        };
        let caps = sample
            .caps()
            .context("physical sample has no negotiated RGB caps")?;
        let info = VideoInfo::from_caps(caps).context("parse physical RGB sample information")?;
        validate_video_info(
            &info,
            output_geometry.width_px(),
            output_geometry.height_px(),
            rate.frames_numerator(),
            rate.seconds_denominator(),
        )?;
        let buffer = sample
            .buffer()
            .context("physical sample has no frame buffer")?;
        let allocation_started = args.capture_audit_timing.then(Instant::now);
        let mut owned_pixels = frame_budget
            .allocate_rgb8_buffer(
                output_geometry.width_px(),
                output_geometry.height_px(),
                frame_limit,
            )
            .context("allocate budgeted physical RGB frame")?;
        let rgb_buffer_allocation_ns = elapsed_nanos(allocation_started);
        let copy_started = args.capture_audit_timing.then(Instant::now);
        let stride = copy_mapped_rgb_frame(
            buffer,
            &info,
            output_geometry.width_px(),
            output_geometry.height_px(),
            owned_pixels.as_mut_slice(),
        )?;
        let rgb_mapping_copy_ns = elapsed_nanos(copy_started);
        ensure!(
            owned_pixels.len() == expected_len,
            "physical RGB output length changed"
        );

        let header = FrameHeader {
            record_schema_version: if args.capture_audit_timing { 3 } else { 2 },
            session_id: &args.session_id,
            frame_id: sequence.to_string(),
            configuration_revision: capture_settings.configuration_revision,
            camera_profile_id: capture_settings.expected.camera_profile_id(),
            capture_settings_sha256: &capture_settings.request_sha256,
            width_px: output_geometry.width_px(),
            height_px: output_geometry.height_px(),
            pixel_format: "rgb8",
            row_stride_bytes: stride,
            // This is GStreamer pipeline PTS only; it is not source exposure.
            unverified_pipeline_pts_ticks: buffer.pts().map(|pts| pts.nseconds().to_string()),
            source_timestamp_ticks: None,
            host_read_completed_elapsed_ns: legacy_helper_start
                .map(|started| started.elapsed().as_nanos().to_string()),
            capture_timings_ns: args.capture_audit_timing.then(|| CaptureHelperTimingNs {
                appsink_pull_wait_ns: appsink_pull_wait_ns
                    .expect("timed mode measures appsink wait"),
                rgb_buffer_allocation_ns: rgb_buffer_allocation_ns
                    .expect("timed mode measures RGB allocation"),
                rgb_mapping_copy_ns: rgb_mapping_copy_ns
                    .expect("timed mode measures RGB mapping and copy"),
            }),
        };
        write_record(&mut output, &header, owned_pixels.as_slice())
            .context("write bounded physical frame record")?;
        output.flush().context("flush physical frame record")?;
        sequence = sequence
            .checked_add(1)
            .context("physical frame sequence overflowed")?;
    }
    Ok(())
}

/// Consume a remote multipart MJPEG stream through a leaky, bounded GStreamer
/// path. This is an explicit development source; it does not report camera
/// controls or source exposure timestamps.
#[cfg(feature = "camera-gstreamer-source")]
fn run_http_mjpeg(args: Args) -> Result<()> {
    use billiards_perception::capture_settings_wire::ResolvedCameraControlsPolicy;
    use std::sync::{Arc, Mutex};

    let url = args
        .http_mjpeg_url
        .as_deref()
        .context("HTTP-MJPEG URL was not supplied")?;
    let mut stdin = io::stdin().lock();
    let exact_capture_settings =
        read_capture_settings(&mut stdin, args.capture_settings_max_bytes)?;
    let settings = parse_capture_settings_json(&exact_capture_settings)
        .context("invalid exact Rust capture-settings request")?;
    ensure!(
        settings.schema_version == 3,
        "HTTP-MJPEG source requires capture-settings v3"
    );
    ensure!(
        settings.expected.device_id()
            == billiards_perception::gstreamer_capture_source::HTTP_MJPEG_STREAM_DEVICE_ID,
        "HTTP-MJPEG source settings have the wrong source identity"
    );
    ensure!(
        settings.requested_backend == "auto" || settings.requested_backend == "gstreamer",
        "GStreamer HTTP-MJPEG helper cannot satisfy requested backend {:?}",
        settings.requested_backend
    );
    ensure!(
        matches!(
            &settings.camera_controls,
            ResolvedCameraControlsPolicy::SourceDefaults
        ),
        "HTTP-MJPEG stream does not support requested camera controls"
    );
    ensure!(settings.expected.image_space() == billiards_perception::ImageSpace::Raw);
    ensure!(settings.expected.rectification_revision() == 0);

    let output_profile = settings.expected.expected_negotiated_profile();
    ensure!(
        output_profile.pixel_format().identifier() == "RGB8",
        "HTTP-MJPEG source emits RGB8 only"
    );
    let output_geometry = settings
        .expected
        .output_frame_geometry()
        .context("HTTP-MJPEG source requires capture-settings v3 output geometry")?;
    let rate = output_profile.frame_rate();
    let (fps_num, fps_den) = (rate.frames_numerator(), rate.seconds_denominator());
    let frame_limit = settings.expected.frame_limit();
    let expected_len = RgbFrameEnvelope::required_rgb8_len(
        output_geometry.width_px(),
        output_geometry.height_px(),
        frame_limit,
    )?;

    gst::init().context("initialize GStreamer")?;
    let source = gst::ElementFactory::make("souphttpsrc")
        .name("remote_mjpeg_source")
        .build()
        .context("create GStreamer souphttpsrc; install the GStreamer Soup plugin")?;
    source.set_property("location", url);
    source.set_property("is-live", true);
    // souphttpsrc provides a byte stream, not camera exposure timestamps. Give
    // decoded frames local arrival timestamps so videorate can enforce the
    // configured maximum cadence. These timestamps remain pipeline timing only
    // and are never exposed as source capture times.
    source.set_property("do-timestamp", true);
    let timeout_seconds = u32::try_from(args.pull_timeout_ms.saturating_add(999) / 1000)
        .context("HTTP-MJPEG timeout exceeds GStreamer source range")?
        .max(1);
    source.set_property("timeout", timeout_seconds);

    let demux = gst::ElementFactory::make("multipartdemux")
        .name("remote_mjpeg_demux")
        .build()
        .context("create GStreamer multipartdemux; install the GStreamer multipart plugin")?;
    let decoder = gst::ElementFactory::make("jpegdec")
        .name("remote_mjpeg_decoder")
        .build()
        .context("create GStreamer jpegdec; install the GStreamer JPEG plugin")?;
    let rate_filter = gst::ElementFactory::make("videorate")
        .name("remote_mjpeg_rate_limit")
        .build()
        .context("create GStreamer videorate")?;
    // Permit frame dropping to meet the requested maximum cadence, but never
    // manufacture duplicate observations when the remote stream is slower.
    rate_filter.set_property("drop-only", true);
    let converter = gst::ElementFactory::make("videoconvert")
        .name("remote_mjpeg_rgb_converter")
        .build()
        .context("create GStreamer RGB converter")?;
    let scaler = gst::ElementFactory::make("videoscale")
        .name("remote_mjpeg_output_scaler")
        .build()
        .context("create GStreamer output scaler")?;
    let output_filter = gst::ElementFactory::make("capsfilter")
        .name("remote_mjpeg_rgb8_output_profile")
        .build()
        .context("create GStreamer output caps filter")?;
    let output_caps = rgb_output_caps(
        output_geometry.width_px(),
        output_geometry.height_px(),
        fps_num,
        fps_den,
    )?;
    output_filter.set_property("caps", &output_caps);
    let sink = gst::ElementFactory::make("appsink")
        .name("remote_mjpeg_latest_frame_sink")
        .build()
        .context("create bounded GStreamer appsink")?
        .downcast::<AppSink>()
        .map_err(|_| anyhow!("remote HTTP-MJPEG sink is not an appsink"))?;
    sink.set_caps(Some(&output_caps));
    sink.set_max_buffers(1);
    sink.set_drop(true);
    sink.set_wait_on_eos(false);
    sink.set_sync(false);

    let pipeline = gst::Pipeline::new();
    pipeline
        .add_many([
            &source,
            &demux,
            &decoder,
            &rate_filter,
            &converter,
            &scaler,
            &output_filter,
            sink.upcast_ref(),
        ])
        .context("add HTTP-MJPEG pipeline elements")?;
    source
        .link(&demux)
        .context("link HTTP source to multipart demuxer")?;
    decoder
        .link(&rate_filter)
        .context("link JPEG decoder to latest-rate limiter")?;
    rate_filter
        .link(&converter)
        .context("link rate limiter to RGB converter")?;
    converter
        .link(&scaler)
        .context("link RGB converter to output scaler")?;
    scaler
        .link(&output_filter)
        .context("link output scaler to RGB8 caps filter")?;
    output_filter
        .link(&sink)
        .context("link RGB8 caps filter to bounded appsink")?;

    let decoder_sink = decoder
        .static_pad("sink")
        .context("JPEG decoder has no sink pad")?;
    let dynamic_link_error = Arc::new(Mutex::new(None::<String>));
    let link_error_for_callback = Arc::clone(&dynamic_link_error);
    demux.connect_pad_added(move |_demux, source_pad| {
        let caps = source_pad
            .current_caps()
            .unwrap_or_else(|| source_pad.query_caps(None));
        if !caps
            .structure(0)
            .is_some_and(|structure| structure.name() == "image/jpeg")
        {
            return;
        }
        if let Err(error) = source_pad.link(&decoder_sink) {
            *link_error_for_callback
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error.to_string());
        }
    });

    pipeline
        .set_state(gst::State::Playing)
        .context("start remote HTTP-MJPEG pipeline")?;
    let _pipeline_guard = PipelineStateGuard(&pipeline);
    let legacy_helper_start = (!args.capture_audit_timing).then(Instant::now);
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    let first_sample_pull_started = args.capture_audit_timing.then(Instant::now);
    // A brief source gap is not a failed capture session. Keep waiting for the
    // next live frame, but surface actual GStreamer errors and EOS below. This
    // is important for best-effort latest-state consumers: no old frame is
    // replayed while the source catches up.
    let first_sample = loop {
        if let Some(sample) = pull_http_mjpeg_sample(
            &sink,
            &pipeline,
            gst::ClockTime::from_mseconds(args.pull_timeout_ms),
        )? {
            break sample;
        }
    };
    let first_sample_pull_wait_ns = elapsed_nanos(first_sample_pull_started);
    if let Some(error) = dynamic_link_error
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
    {
        return Err(anyhow!("could not link multipart JPEG stream: {error}"));
    }
    let first_caps = first_sample
        .caps()
        .context("first remote MJPEG sample has no negotiated RGB caps")?;
    let first_info =
        VideoInfo::from_caps(first_caps).context("parse first remote MJPEG output sample")?;
    validate_video_info(
        &first_info,
        output_geometry.width_px(),
        output_geometry.height_px(),
        fps_num,
        fps_den,
    )?;
    let ready = CaptureHelperReady {
        record_schema_version: 1,
        session_id: args.session_id.clone(),
        configuration_revision: settings.configuration_revision,
        camera_profile_id: settings.expected.camera_profile_id().to_owned(),
        capture_settings_sha256: settings.request_sha256.clone(),
        width_px: first_info.width(),
        height_px: first_info.height(),
        frame_rate_numerator: fps_num,
        frame_rate_seconds_denominator: fps_den,
        pixel_format: "RGB8".to_owned(),
        source_device_id: None,
        source_identifier: None,
        source_identifier_source: None,
        input_profile: None,
    };
    write_ready_record(&mut output, &ready, args.ready_record_max_bytes)
        .context("write bounded HTTP-MJPEG helper-ready record")?;
    output
        .flush()
        .context("flush HTTP-MJPEG helper-ready record")?;

    let frame_budget = CaptureFrameBudget::new(settings.expected.aggregate_frame_limit());
    let mut first_sample = Some(first_sample);
    let mut sequence = 0_u64;
    loop {
        if args.frames.is_some_and(|limit| sequence >= limit) {
            break;
        }
        if sequence > 0 && args.slow_reader_ms > 0 {
            thread::sleep(std::time::Duration::from_millis(args.slow_reader_ms));
        }
        let (sample, appsink_pull_wait_ns) = if sequence == 0 {
            (
                first_sample
                    .take()
                    .expect("validated first remote frame is available as frame zero"),
                first_sample_pull_wait_ns,
            )
        } else {
            let pull_started = args.capture_audit_timing.then(Instant::now);
            let sample = loop {
                if let Some(sample) = pull_http_mjpeg_sample(
                    &sink,
                    &pipeline,
                    gst::ClockTime::from_mseconds(args.pull_timeout_ms),
                )? {
                    break sample;
                }
            };
            (sample, elapsed_nanos(pull_started))
        };
        let caps = sample
            .caps()
            .context("remote MJPEG output sample has no negotiated RGB caps")?;
        let info = VideoInfo::from_caps(caps).context("parse remote MJPEG RGB sample")?;
        validate_video_info(
            &info,
            output_geometry.width_px(),
            output_geometry.height_px(),
            fps_num,
            fps_den,
        )?;
        let buffer = sample
            .buffer()
            .context("remote MJPEG output sample has no frame buffer")?;
        let allocation_started = args.capture_audit_timing.then(Instant::now);
        let mut owned_pixels = frame_budget
            .allocate_rgb8_buffer(
                output_geometry.width_px(),
                output_geometry.height_px(),
                frame_limit,
            )
            .context("allocate budgeted remote RGB frame")?;
        let rgb_buffer_allocation_ns = elapsed_nanos(allocation_started);
        let copy_started = args.capture_audit_timing.then(Instant::now);
        let stride = copy_mapped_rgb_frame(
            buffer,
            &info,
            output_geometry.width_px(),
            output_geometry.height_px(),
            owned_pixels.as_mut_slice(),
        )?;
        let rgb_mapping_copy_ns = elapsed_nanos(copy_started);
        ensure!(
            owned_pixels.len() == expected_len,
            "remote RGB output length changed"
        );
        let header = FrameHeader {
            record_schema_version: if args.capture_audit_timing { 3 } else { 2 },
            session_id: &args.session_id,
            frame_id: sequence.to_string(),
            configuration_revision: settings.configuration_revision,
            camera_profile_id: settings.expected.camera_profile_id(),
            capture_settings_sha256: &settings.request_sha256,
            width_px: output_geometry.width_px(),
            height_px: output_geometry.height_px(),
            pixel_format: "rgb8",
            row_stride_bytes: stride,
            // GStreamer PTS is pipeline time. It is not camera exposure time.
            unverified_pipeline_pts_ticks: buffer.pts().map(|pts| pts.nseconds().to_string()),
            source_timestamp_ticks: None,
            host_read_completed_elapsed_ns: legacy_helper_start
                .map(|started| started.elapsed().as_nanos().to_string()),
            capture_timings_ns: args.capture_audit_timing.then(|| CaptureHelperTimingNs {
                appsink_pull_wait_ns: appsink_pull_wait_ns.expect("timed mode measures pull wait"),
                rgb_buffer_allocation_ns: rgb_buffer_allocation_ns
                    .expect("timed mode measures RGB allocation"),
                rgb_mapping_copy_ns: rgb_mapping_copy_ns
                    .expect("timed mode measures RGB mapping and copy"),
            }),
        };
        write_record(&mut output, &header, owned_pixels.as_slice())
            .context("write bounded remote MJPEG frame record")?;
        output.flush().context("flush remote MJPEG frame record")?;
        sequence = sequence
            .checked_add(1)
            .context("remote MJPEG frame sequence overflowed")?;
    }
    Ok(())
}

/// Pull one sample from the live HTTP-MJPEG pipeline. `None` means that the
/// timeout elapsed without a frame; it is a transient gap, not a reason to
/// terminate capture. A real GStreamer error or EOS remains terminal so the
/// supervising runtime can report or recover from the source failure.
#[cfg(feature = "camera-gstreamer-source")]
fn pull_http_mjpeg_sample(
    sink: &AppSink,
    pipeline: &gst::Pipeline,
    timeout: gst::ClockTime,
) -> Result<Option<gst::Sample>> {
    if let Some(sample) = sink.try_pull_sample(timeout) {
        return Ok(Some(sample));
    }

    if let Some(message) = pipeline
        .bus()
        .context("remote HTTP-MJPEG pipeline has no message bus")?
        .pop_filtered(&[gst::MessageType::Error, gst::MessageType::Eos])
    {
        match message.view() {
            gst::MessageView::Error(error) => {
                return Err(anyhow!(
                    "remote HTTP-MJPEG source pipeline failed: {}",
                    error.error()
                ));
            }
            gst::MessageView::Eos(..) => {
                return Err(anyhow!("remote HTTP-MJPEG source reached end of stream"));
            }
            _ => {}
        }
    }

    if sink.is_eos() {
        return Err(anyhow!("remote HTTP-MJPEG appsink reached end of stream"));
    }
    Ok(None)
}

fn elapsed_nanos(started: Option<Instant>) -> Option<u64> {
    started.map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX))
}

#[cfg(feature = "camera-gstreamer-source")]
fn physical_capture_caps(
    input_profile: &billiards_perception::capture::CaptureVideoProfile,
    output_width_px: u32,
    output_height_px: u32,
) -> Result<(gst::Caps, gst::Caps)> {
    // Caps construction uses GStreamer's initialized type system (for
    // fractions and value serialization), so keep initialization inside the
    // operation that creates the camera and output profiles.
    gst::init().context("initialize GStreamer")?;
    let input_caps = camera_input_caps(input_profile)?;
    let output_caps = rgb_output_caps(
        output_width_px,
        output_height_px,
        input_profile.frame_rate().frames_numerator(),
        input_profile.frame_rate().seconds_denominator(),
    )?;
    Ok((input_caps, output_caps))
}

#[cfg(feature = "camera-gstreamer-source")]
fn camera_device_metadata(
    device: &gst::Device,
) -> billiards_perception::camera_discovery::CameraDevice {
    use billiards_perception::camera_discovery::{
        properties_from_gstreamer, CameraDevice, DeviceMetadata,
    };

    let properties = device
        .properties()
        .map(|properties| properties_from_gstreamer(properties.as_ref()));
    CameraDevice::from_metadata(
        std::env::consts::OS,
        DeviceMetadata {
            display_name: optional_string(device.display_name().as_str()),
            device_class: optional_string(device.device_class().as_str()),
            provider_properties: properties,
            advertised_caps: device.caps().map(|caps| caps.to_string()),
        },
    )
}

#[cfg(feature = "camera-gstreamer-source")]
fn optional_string(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_owned())
}

#[cfg(feature = "camera-gstreamer-source")]
fn camera_input_caps(
    profile: &billiards_perception::capture::CaptureVideoProfile,
) -> Result<gst::Caps> {
    let width =
        i32::try_from(profile.width_px()).context("camera width exceeds GStreamer range")?;
    let height =
        i32::try_from(profile.height_px()).context("camera height exceeds GStreamer range")?;
    let rate = profile.frame_rate();
    let fps = gst::Fraction::new(
        i32::try_from(rate.frames_numerator())
            .context("frame-rate numerator exceeds GStreamer range")?,
        i32::try_from(rate.seconds_denominator())
            .context("frame-rate denominator exceeds GStreamer range")?,
    );
    if let Some(format) = gst_raw_format_for_camera_id(profile.pixel_format().identifier()) {
        Ok(gst::Caps::builder("video/x-raw")
            .field("format", format)
            .field("width", width)
            .field("height", height)
            .field("framerate", fps)
            .build())
    } else if profile.pixel_format().identifier() == "MJPG" {
        Ok(gst::Caps::builder("image/jpeg")
            .field("width", width)
            .field("height", height)
            .field("framerate", fps)
            .build())
    } else {
        anyhow::bail!(
            "physical helper does not support camera input format {:?}",
            profile.pixel_format().identifier()
        )
    }
}

#[cfg(feature = "camera-gstreamer-source")]
fn rgb_output_caps(width: u32, height: u32, fps_num: u32, fps_den: u32) -> Result<gst::Caps> {
    Ok(gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field(
            "width",
            i32::try_from(width).context("output width exceeds GStreamer range")?,
        )
        .field(
            "height",
            i32::try_from(height).context("output height exceeds GStreamer range")?,
        )
        .field(
            "framerate",
            gst::Fraction::new(
                i32::try_from(fps_num)
                    .context("output frame-rate numerator exceeds GStreamer range")?,
                i32::try_from(fps_den)
                    .context("output frame-rate denominator exceeds GStreamer range")?,
            ),
        )
        .build())
}

#[cfg(feature = "camera-gstreamer-source")]
fn gst_raw_format_for_camera_id(identifier: &str) -> Option<&'static str> {
    match identifier {
        "RGB8" => Some("RGB"),
        "BGR8" => Some("BGR"),
        "YUYV" => Some("YUY2"),
        "UYVY" => Some("UYVY"),
        "NV12" => Some("NV12"),
        _ => None,
    }
}

#[cfg(feature = "camera-gstreamer-source")]
fn camera_id_for_gst_caps(caps: &gst::CapsRef) -> Result<String> {
    let structure = caps
        .structure(0)
        .context("camera input caps have no media structure")?;
    if structure.name() == "image/jpeg" {
        return Ok("MJPG".to_owned());
    }
    ensure!(
        structure.name() == "video/x-raw",
        "unsupported negotiated camera media type {:?}",
        structure.name()
    );
    let info = VideoInfo::from_caps(caps).context("parse negotiated raw camera input caps")?;
    let identifier = match info.format().to_str().as_str() {
        "RGB" => "RGB8",
        "BGR" => "BGR8",
        "YUY2" => "YUYV",
        "UYVY" => "UYVY",
        "NV12" => "NV12",
        actual => anyhow::bail!("unsupported negotiated GStreamer camera format {actual:?}"),
    };
    Ok(identifier.to_owned())
}

#[cfg(feature = "camera-gstreamer-source")]
fn parse_input_profile(caps: &gst::CapsRef) -> Result<CaptureHelperInputProfile> {
    let structure = caps
        .structure(0)
        .context("camera input caps have no media structure")?;
    let width = structure
        .get::<i32>("width")
        .context("camera input caps omitted width")?;
    let height = structure
        .get::<i32>("height")
        .context("camera input caps omitted height")?;
    let frame_rate = structure
        .get::<gst::Fraction>("framerate")
        .context("camera input caps omitted frame rate")?;
    let width_px = u32::try_from(width).context("camera input width is not positive")?;
    let height_px = u32::try_from(height).context("camera input height is not positive")?;
    let frame_rate_numerator = u32::try_from(frame_rate.numer())
        .context("camera input frame-rate numerator is not positive")?;
    let frame_rate_seconds_denominator = u32::try_from(frame_rate.denom())
        .context("camera input frame-rate denominator is not positive")?;
    ensure!(
        width_px > 0 && height_px > 0,
        "camera input geometry must be positive"
    );
    ensure!(
        frame_rate_numerator > 0 && frame_rate_seconds_denominator > 0,
        "camera input rate must be positive"
    );
    Ok(CaptureHelperInputProfile {
        width_px,
        height_px,
        frame_rate_numerator,
        frame_rate_seconds_denominator,
        pixel_format: camera_id_for_gst_caps(caps)?,
    })
}

#[cfg(feature = "camera-gstreamer-source")]
fn validate_input_profile(
    actual: &CaptureHelperInputProfile,
    expected: &billiards_perception::capture::CaptureVideoProfile,
) -> Result<()> {
    ensure!(
        actual.width_px == expected.width_px()
            && actual.height_px == expected.height_px()
            && actual.frame_rate_numerator == expected.frame_rate().frames_numerator()
            && actual.frame_rate_seconds_denominator == expected.frame_rate().seconds_denominator()
            && actual.pixel_format == expected.pixel_format().identifier(),
        "camera negotiated input profile {actual:?} does not match requested profile {expected:?}"
    );
    Ok(())
}

#[cfg(feature = "camera-gstreamer-source")]
struct DeviceMonitorStopGuard(gst::DeviceMonitor);

#[cfg(feature = "camera-gstreamer-source")]
impl Drop for DeviceMonitorStopGuard {
    fn drop(&mut self) {
        self.0.stop();
    }
}

fn validate_args(args: &Args) -> Result<()> {
    ensure!(
        !args.session_id.is_empty(),
        "session identity must not be empty"
    );
    ensure!(args.session_id.len() <= 128, "session identity is too long");
    match (args.frames, args.until_stopped, args.source_mode) {
        (Some(frames), false, _) => ensure!(frames > 0, "frame count must be positive"),
        (None, true, SourceMode::Physical | SourceMode::HttpMjpeg) => {}
        (None, true, SourceMode::Synthetic) => {
            anyhow::bail!("run-until-stopped mode is unavailable for synthetic capture")
        }
        (None, false, _) => anyhow::bail!("provide --frames or --until-stopped"),
        (Some(_), true, _) => anyhow::bail!("--frames conflicts with --until-stopped"),
    }
    ensure!(
        args.capture_settings_max_bytes > 0,
        "capture-settings byte limit must be positive"
    );
    ensure!(
        args.ready_record_max_bytes > 0,
        "helper-ready JSON byte limit must be positive"
    );
    ensure!(args.pull_timeout_ms > 0, "pull timeout must be positive");
    match (args.source_mode, args.http_mjpeg_url.as_deref()) {
        (SourceMode::HttpMjpeg, Some(url)) => ensure!(
            billiards_perception::configuration_authority::valid_http_uri(url),
            "HTTP-MJPEG URL must be a valid credential-free HTTP(S) URI"
        ),
        (SourceMode::HttpMjpeg, None) => {
            anyhow::bail!("--http-mjpeg-url is required for HTTP-MJPEG source mode")
        }
        (_, Some(_)) => {
            anyhow::bail!("--http-mjpeg-url is valid only for HTTP-MJPEG source mode")
        }
        (_, None) => {}
    }
    ensure!(
        args.slow_reader_ms < args.pull_timeout_ms,
        "slow-reader delay must be shorter than the pull timeout"
    );
    Ok(())
}

#[cfg(test)]
mod argument_tests {
    use super::{validate_args, Args, SourceMode};
    use clap::Parser;

    fn args(source_mode: SourceMode, frames: Option<u64>, until_stopped: bool) -> Args {
        Args {
            source_mode,
            http_mjpeg_url: None,
            session_id: "capture-test-session".to_owned(),
            capture_settings_max_bytes: 16_384,
            ready_record_max_bytes: 4_096,
            frames,
            until_stopped,
            slow_reader_ms: 0,
            pull_timeout_ms: 1_000,
            capture_audit_timing: false,
        }
    }

    #[test]
    fn synthetic_capture_requires_a_positive_finite_frame_count() {
        assert!(validate_args(&args(SourceMode::Synthetic, Some(1), false)).is_ok());
        assert!(validate_args(&args(SourceMode::Synthetic, Some(0), false)).is_err());
        assert!(validate_args(&args(SourceMode::Synthetic, None, false)).is_err());
        assert!(validate_args(&args(SourceMode::Synthetic, None, true)).is_err());
    }

    #[test]
    fn physical_capture_can_run_until_the_supervisor_stops_it() {
        assert!(validate_args(&args(SourceMode::Physical, Some(1), false)).is_ok());
        assert!(validate_args(&args(SourceMode::Physical, None, true)).is_ok());
        assert!(validate_args(&args(SourceMode::Physical, None, false)).is_err());
        assert!(validate_args(&args(SourceMode::Physical, Some(1), true)).is_err());
    }

    #[test]
    fn remote_mjpeg_requires_a_valid_url_and_can_run_continuously() {
        let mut source = args(SourceMode::HttpMjpeg, None, true);
        assert!(validate_args(&source).is_err());
        source.http_mjpeg_url = Some("http://127.0.0.1:8001/stream.mjpg".to_owned());
        assert!(validate_args(&source).is_ok());
        source.http_mjpeg_url = Some("http://user:pass@example.test/stream".to_owned());
        assert!(validate_args(&source).is_err());
    }

    #[test]
    fn command_line_requires_one_run_duration_mode() {
        let common = [
            "capture-helper",
            "--session-id",
            "capture-test-session",
            "--capture-settings-max-bytes",
            "16384",
            "--ready-record-max-bytes",
            "4096",
            "--slow-reader-ms",
            "0",
            "--pull-timeout-ms",
            "1000",
        ];
        let mut continuous = common.to_vec();
        continuous.extend(["--source-mode", "physical", "--until-stopped"]);
        assert!(Args::try_parse_from(continuous.clone()).is_ok());

        let mut conflicting = continuous;
        conflicting.extend(["--frames", "1"]);
        assert!(Args::try_parse_from(conflicting).is_err());
    }
}

#[cfg(all(test, feature = "camera-gstreamer-source"))]
mod http_mjpeg_sample_tests {
    use super::pull_http_mjpeg_sample;
    use gst::prelude::*;
    use gstreamer as gst;
    use gstreamer_app::{AppSink, AppSrc};

    #[test]
    fn transient_empty_pull_waits_for_a_later_frame_and_eos_is_terminal() {
        gst::init().unwrap();
        let pipeline = gst::Pipeline::new();
        let source = gst::ElementFactory::make("appsrc")
            .build()
            .unwrap()
            .downcast::<AppSrc>()
            .unwrap();
        let sink = gst::ElementFactory::make("appsink")
            .build()
            .unwrap()
            .downcast::<AppSink>()
            .unwrap();
        source.set_caps(Some(
            &gst::Caps::builder("application/octet-stream").build(),
        ));
        sink.set_wait_on_eos(false);
        pipeline
            .add_many([
                source.upcast_ref::<gst::Element>(),
                sink.upcast_ref::<gst::Element>(),
            ])
            .unwrap();
        source.link(&sink).unwrap();
        pipeline.set_state(gst::State::Playing).unwrap();

        let timeout = gst::ClockTime::from_mseconds(10);
        assert!(pull_http_mjpeg_sample(&sink, &pipeline, timeout)
            .unwrap()
            .is_none());

        source
            .push_buffer(gst::Buffer::from_mut_slice(vec![1_u8, 2, 3]))
            .unwrap();
        let sample = pull_http_mjpeg_sample(&sink, &pipeline, timeout)
            .unwrap()
            .expect("the sample after a transient gap must be delivered");
        assert_eq!(sample.buffer().unwrap().size(), 3);

        source.end_of_stream().unwrap();
        assert!(pull_http_mjpeg_sample(&sink, &pipeline, timeout).is_err());
        pipeline.set_state(gst::State::Null).unwrap();
    }
}

/// Verify the actual caps on a sample against the request and return the
/// negotiated frame rate in its original rational representation.
fn validate_video_info(
    video_info: &VideoInfo,
    expected_width: u32,
    expected_height: u32,
    expected_fps_numerator: u32,
    expected_fps_denominator: u32,
) -> Result<(u32, u32)> {
    ensure!(
        video_info.format() == VideoFormat::Rgb,
        "appsink emitted {:?}, expected RGB",
        video_info.format()
    );
    ensure!(
        video_info.width() == expected_width && video_info.height() == expected_height,
        "appsink emitted {}x{}, expected {}x{}",
        video_info.width(),
        video_info.height(),
        expected_width,
        expected_height
    );

    let actual_fps = video_info.fps();
    ensure!(
        actual_fps.numer() > 0 && actual_fps.denom() > 0,
        "appsink emitted a non-positive frame rate {}/{}",
        actual_fps.numer(),
        actual_fps.denom()
    );
    let actual_numerator = u32::try_from(actual_fps.numer())
        .context("negotiated frame-rate numerator is outside the supported range")?;
    let actual_denominator = u32::try_from(actual_fps.denom())
        .context("negotiated frame-rate denominator is outside the supported range")?;
    ensure!(
        i128::from(actual_numerator) * i128::from(expected_fps_denominator)
            == i128::from(expected_fps_numerator) * i128::from(actual_denominator),
        "appsink negotiated frame rate {actual_numerator}/{actual_denominator}, expected {expected_fps_numerator}/{expected_fps_denominator}"
    );
    Ok((actual_numerator, actual_denominator))
}

/// Copy only visible RGB bytes from a potentially padded GStreamer plane.
fn copy_rgb_rows(
    source: &[u8],
    source_stride: usize,
    width_px: u32,
    height_px: u32,
    destination: &mut [u8],
) -> Result<()> {
    let row_bytes = usize::try_from(width_px)
        .context("frame width exceeds addressable memory")?
        .checked_mul(3)
        .ok_or_else(|| anyhow!("RGB row byte count overflow"))?;
    let height = usize::try_from(height_px).context("frame height exceeds addressable memory")?;
    ensure!(
        source_stride >= row_bytes,
        "source row stride is shorter than RGB row"
    );
    let required_source = source_stride
        .checked_mul(height.saturating_sub(1))
        .and_then(|preceding_rows| preceding_rows.checked_add(row_bytes))
        .ok_or_else(|| anyhow!("source plane length overflow"))?;
    let required_destination = row_bytes
        .checked_mul(height)
        .ok_or_else(|| anyhow!("destination plane length overflow"))?;
    ensure!(source.len() >= required_source, "source plane is truncated");
    ensure!(
        destination.len() == required_destination,
        "destination length does not match tightly packed RGB geometry"
    );

    for row in 0..height {
        let source_start = row * source_stride;
        let destination_start = row * row_bytes;
        destination[destination_start..destination_start + row_bytes]
            .copy_from_slice(&source[source_start..source_start + row_bytes]);
    }
    Ok(())
}

/// Map the sample and copy using its mapped plane stride. `GstVideoMeta` may
/// override the caps-derived `VideoInfo` stride for an individual buffer, so
/// read the stride from the mapped frame that owns the pixel plane.
fn copy_mapped_rgb_frame(
    buffer: &gst::BufferRef,
    video_info: &VideoInfo,
    width_px: u32,
    height_px: u32,
    destination: &mut [u8],
) -> Result<usize> {
    let frame = VideoFrameRef::from_buffer_ref_readable(buffer, video_info)
        .context("map negotiated RGB frame")?;
    let mapped_stride = frame
        .plane_stride()
        .first()
        .copied()
        .context("mapped RGB frame has no first-plane stride")?;
    let stride = usize::try_from(mapped_stride)
        .context("GStreamer reported a negative mapped RGB row stride")?;
    let source_pixels = frame.plane_data(0).context("map RGB frame plane")?;
    copy_rgb_rows(source_pixels, stride, width_px, height_px, destination)?;
    Ok(stride)
}

fn write_record<W: Write, T: Serialize>(
    writer: &mut W,
    header: &T,
    pixels: &[u8],
) -> io::Result<()> {
    let header = serde_json::to_vec(header).map_err(io::Error::other)?;
    let header_len = u32::try_from(header.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "header exceeds framing limit"))?;
    let pixel_len = u32::try_from(pixels.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "pixels exceed framing limit"))?;
    writer.write_all(&header_len.to_be_bytes())?;
    writer.write_all(&header)?;
    writer.write_all(&pixel_len.to_be_bytes())?;
    writer.write_all(pixels)
}

struct PipelineStateGuard<'a>(&'a gst::Pipeline);

impl Drop for PipelineStateGuard<'_> {
    fn drop(&mut self) {
        // A helper supervisor can terminate the process if a native state
        // transition blocks here. This best-effort cleanup is not the hard
        // cancellation guarantee; the process boundary is.
        let _ = self.0.set_state(gst::State::Null);
    }
}

#[cfg(test)]
mod tests {
    use super::{copy_mapped_rgb_frame, copy_rgb_rows};

    #[cfg(feature = "camera-gstreamer-source")]
    #[test]
    fn physical_caps_initialization_precedes_gstreamer_caps_construction() {
        use super::physical_capture_caps;
        use billiards_perception::capture::{
            CaptureFrameRate, CapturePixelFormat, CaptureVideoProfile,
        };

        let profile = CaptureVideoProfile::new(
            16,
            16,
            CaptureFrameRate::new(30, 1).unwrap(),
            CapturePixelFormat::new("NV12").unwrap(),
        )
        .unwrap();
        let (input_caps, output_caps) = physical_capture_caps(&profile, 16, 16).unwrap();

        assert!(input_caps.to_string().contains("NV12"));
        assert!(output_caps.to_string().contains("format=(string)RGB"));
    }

    #[test]
    fn copies_visible_rgb_bytes_and_discards_source_padding() {
        let source = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 99, 10, 11, 12, 13, 14, 15, 16, 17, 18, 98,
        ];
        let mut destination = [0; 18];

        copy_rgb_rows(&source, 10, 3, 2, &mut destination).unwrap();

        assert_eq!(
            destination,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,]
        );
    }

    #[test]
    fn rejects_truncated_planes_and_short_strides() {
        let mut destination = [0; 18];
        assert!(copy_rgb_rows(&[0; 17], 10, 3, 2, &mut destination).is_err());
        assert!(copy_rgb_rows(&[0; 18], 8, 3, 2, &mut destination).is_err());
        assert!(copy_rgb_rows(&[0; 20], 10, 3, 2, &mut destination[..17]).is_err());
    }

    #[test]
    fn mapped_video_meta_stride_controls_rgb_row_copy() {
        use gstreamer as gst;
        use gstreamer_video::{VideoFormat, VideoFrameFlags, VideoInfo, VideoMeta};

        gst::init().unwrap();
        let mut source = vec![0; 32];
        source[..9].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        source[16..25].copy_from_slice(&[10, 11, 12, 13, 14, 15, 16, 17, 18]);
        let mut buffer = gst::Buffer::from_mut_slice(source);
        VideoMeta::add_full(
            buffer.get_mut().unwrap(),
            VideoFrameFlags::empty(),
            VideoFormat::Rgb,
            3,
            2,
            &[0],
            &[16],
        )
        .unwrap();

        let caps = gst::Caps::builder("video/x-raw")
            .field("format", "RGB")
            .field("width", 3i32)
            .field("height", 2i32)
            .field("framerate", gst::Fraction::new(30, 1))
            .build();
        let video_info = VideoInfo::from_caps(&caps).unwrap();
        assert_ne!(video_info.stride()[0], 16);

        let mut destination = [0; 18];
        let mapped_stride =
            copy_mapped_rgb_frame(buffer.as_ref(), &video_info, 3, 2, &mut destination).unwrap();

        assert_eq!(mapped_stride, 16);
        assert_eq!(
            destination,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        );
    }

    #[cfg(feature = "camera-gstreamer-source")]
    #[test]
    fn physical_camera_format_mapping_is_explicit_and_round_trips_supported_caps() {
        use super::{camera_id_for_gst_caps, gst_raw_format_for_camera_id};
        use gstreamer as gst;

        gst::init().unwrap();
        for (camera_id, gst_format) in [
            ("RGB8", "RGB"),
            ("BGR8", "BGR"),
            ("YUYV", "YUY2"),
            ("UYVY", "UYVY"),
            ("NV12", "NV12"),
        ] {
            assert_eq!(gst_raw_format_for_camera_id(camera_id), Some(gst_format));
            let caps = gst::Caps::builder("video/x-raw")
                .field("format", gst_format)
                .field("width", 640i32)
                .field("height", 480i32)
                .field("framerate", gst::Fraction::new(30, 1))
                .build();
            assert_eq!(camera_id_for_gst_caps(&caps).unwrap(), camera_id);
        }
        let jpeg_caps = gst::Caps::builder("image/jpeg")
            .field("width", 640i32)
            .field("height", 480i32)
            .field("framerate", gst::Fraction::new(30, 1))
            .build();
        assert_eq!(camera_id_for_gst_caps(&jpeg_caps).unwrap(), "MJPG");
        assert_eq!(gst_raw_format_for_camera_id("MJPG"), None);
        assert_eq!(gst_raw_format_for_camera_id("unknown"), None);
    }
}
