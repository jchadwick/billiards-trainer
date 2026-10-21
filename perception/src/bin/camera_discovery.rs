//! Opt-in read-only camera inventory using GStreamer's DeviceMonitor.
//!
//! This binary filters for `Video/Source` devices and reads provider metadata.
//! It never creates a camera source element or capture pipeline.

use std::io::{self, Write};
use std::process;
use std::process::Command;

use anyhow::{ensure, Context, Result};
use billiards_perception::camera_discovery::{
    properties_from_gstreamer, CameraDevice, CameraDiscoveryReport, DeviceMetadata, HostPlatform,
    ProviderStatus,
};
use clap::Parser;
use gst::prelude::*;
use gstreamer as gst;

const MIN_OUTPUT_BYTES: usize = 512;
const MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_OUTPUT_BYTES: usize = 1024 * 1024;

#[derive(Debug, Parser)]
#[command(about = "Enumerate camera metadata without opening a camera")]
struct Args {
    /// Maximum number of bytes written to stdout, including the final newline.
    #[arg(long, default_value_t = DEFAULT_OUTPUT_BYTES)]
    max_output_bytes: usize,
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("camera discovery failed: {error:#}");
        process::exit(2);
    }
}

fn run(args: Args) -> Result<()> {
    ensure!(
        (MIN_OUTPUT_BYTES..=MAX_OUTPUT_BYTES).contains(&args.max_output_bytes),
        "--max-output-bytes must be between {MIN_OUTPUT_BYTES} and {MAX_OUTPUT_BYTES}"
    );

    gst::init().context("initialize GStreamer")?;
    let monitor = gst::DeviceMonitor::new();
    let (provider_status, devices) = if monitor.add_filter(Some("Video/Source"), None).is_some() {
        monitor.start().context("start GStreamer device monitor")?;
        let _monitor_guard = MonitorStopGuard(monitor.clone());

        // DeviceMonitor reports metadata only. In particular, do not call
        // Device::create_element or create/set state on a source pipeline here.
        let devices = monitor
            .devices()
            .iter()
            .map(|device| {
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
            })
            .collect();
        (ProviderStatus::Started, devices)
    } else {
        (ProviderStatus::NoMatchingProvider, Vec::new())
    };

    let report = CameraDiscoveryReport::new(
        HostPlatform {
            os: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            version: platform_version(),
        },
        gst::version_string().to_string(),
        provider_status,
        devices,
    );
    let json = report.to_bounded_json(args.max_output_bytes)?;
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    output
        .write_all(&json)
        .context("write camera discovery JSON")?;
    output.flush().context("flush camera discovery JSON")?;
    Ok(())
}

fn optional_string(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_owned())
}

#[cfg(target_os = "macos")]
fn platform_version() -> Option<String> {
    Command::new("/usr/bin/sw_vers")
        .arg("-productVersion")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|version| !version.is_empty())
}

#[cfg(target_os = "linux")]
fn platform_version() -> Option<String> {
    Command::new("/usr/bin/uname")
        .arg("-r")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|version| !version.is_empty())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn platform_version() -> Option<String> {
    None
}

struct MonitorStopGuard(gst::DeviceMonitor);

impl Drop for MonitorStopGuard {
    fn drop(&mut self) {
        self.0.stop();
    }
}
