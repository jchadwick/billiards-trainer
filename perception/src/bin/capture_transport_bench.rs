use std::ffi::OsStr;
use std::io;
use std::process;

use anyhow::{Context, Result};
use billiards_perception::capture_transport_benchmark::{
    run_benchmark, run_helper_child, BenchmarkConfig,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "billiards-capture-transport-bench",
    about = "Opt-in deterministic RGB8 in-process versus supervised-pipe microbenchmark",
    long_about = "Compares a bounded in-process handoff with the same setup-generated RGB8 payload transferred over a supervised child process stdout pipe. Payload generation and SHA-256 verification happen before measured frame loops. This isolated experiment does not measure a camera, detector, CaptureSource mailbox, or client delivery. Run with --release for performance measurements. Every workload limit is explicit.",
    version
)]
struct Arguments {
    /// RGB8 frame width in pixels.
    #[arg(long)]
    width_px: u32,

    /// RGB8 frame height in pixels.
    #[arg(long)]
    height_px: u32,

    /// Number of unreported frames per arm and repetition before measurement.
    #[arg(long)]
    warmup_frames: u64,

    /// Number of frames measured per arm and repetition.
    #[arg(long)]
    measured_frames: u64,

    /// Number of paired in-process and child-pipe repetitions.
    #[arg(long)]
    repetitions: u32,

    /// Maximum sampled per-frame completion records retained for each arm and repetition.
    #[arg(long)]
    sampling_limit: usize,

    /// Seed for deterministic RGB8 frame generation.
    #[arg(long)]
    seed: u64,

    /// Required upper bound on one RGB8 frame allocation.
    #[arg(long)]
    max_payload_bytes: usize,

    /// Required upper bound on transmitted frame records per arm across all repetitions.
    #[arg(long)]
    max_total_frames: u64,

    /// Required upper bound on retained timing/hash samples across both arms and all repetitions.
    #[arg(long)]
    max_total_samples: usize,
}

#[derive(Debug, Parser)]
#[command(name = "capture-transport-pipe-helper")]
struct HelperArguments {
    #[arg(long)]
    width_px: u32,
    #[arg(long)]
    height_px: u32,
    #[arg(long)]
    warmup_frames: u64,
    #[arg(long)]
    measured_frames: u64,
    #[arg(long)]
    repetitions: u32,
    #[arg(long)]
    sampling_limit: usize,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    max_payload_bytes: usize,
    #[arg(long)]
    max_total_frames: u64,
    #[arg(long)]
    max_total_samples: usize,
}

impl From<Arguments> for BenchmarkConfig {
    fn from(arguments: Arguments) -> Self {
        Self {
            width_px: arguments.width_px,
            height_px: arguments.height_px,
            warmup_frames: arguments.warmup_frames,
            measured_frames: arguments.measured_frames,
            repetitions: arguments.repetitions,
            sampling_limit: arguments.sampling_limit,
            seed: arguments.seed,
            max_payload_bytes: arguments.max_payload_bytes,
            max_total_frames: arguments.max_total_frames,
            max_total_samples: arguments.max_total_samples,
        }
    }
}

impl From<HelperArguments> for BenchmarkConfig {
    fn from(arguments: HelperArguments) -> Self {
        Self {
            width_px: arguments.width_px,
            height_px: arguments.height_px,
            warmup_frames: arguments.warmup_frames,
            measured_frames: arguments.measured_frames,
            repetitions: arguments.repetitions,
            sampling_limit: arguments.sampling_limit,
            seed: arguments.seed,
            max_payload_bytes: arguments.max_payload_bytes,
            max_total_frames: arguments.max_total_frames,
            max_total_samples: arguments.max_total_samples,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("capture transport benchmark failed: {error:#}");
        process::exit(2);
    }
}

fn run() -> Result<()> {
    let mut arguments = std::env::args_os();
    let program = arguments
        .next()
        .unwrap_or_else(|| "capture-transport-bench".into());
    let first_argument = arguments.next();
    if first_argument.as_deref() == Some(OsStr::new("--internal-child")) {
        let helper = HelperArguments::try_parse_from(std::iter::once(program).chain(arguments))?;
        return run_helper_child(helper.into());
    }

    let parsed = match Arguments::try_parse_from(
        std::iter::once(program)
            .chain(first_argument)
            .chain(arguments),
    ) {
        Ok(parsed) => parsed,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            print!("{error}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let report = run_benchmark(parsed.into())?;
    let mut output = io::BufWriter::new(io::stdout().lock());
    serde_json::to_writer_pretty(&mut output, &report)
        .context("could not serialize benchmark JSON report")?;
    use std::io::Write;
    output
        .write_all(b"\n")
        .context("could not finish benchmark JSON report")?;
    output
        .flush()
        .context("could not flush benchmark JSON report")?;
    Ok(())
}
