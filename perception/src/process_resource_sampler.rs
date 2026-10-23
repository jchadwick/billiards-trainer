//! Bounded, off-thread sampling of one process during a performance audit.
//!
//! Sampling is opt-in and deliberately runs on a dedicated worker thread so
//! process inspection cannot block the WebSocket state receiver. Process CPU
//! percentage is reported using `sysinfo`'s per-core percentage semantics;
//! accumulated CPU milliseconds and resident memory bytes are retained as
//! separate measurements. The sampler binds itself to the process start time
//! observed at startup so a reused PID cannot silently join two runs.

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use sysinfo::{Pid, Process, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

pub const MAXIMUM_RESOURCE_SAMPLES: usize = 1_000_000;

#[derive(Debug, Clone)]
pub struct ProcessResourceSamplerConfig {
    pub process_id: u32,
    pub sample_interval: Duration,
    pub maximum_samples: usize,
}

impl ProcessResourceSamplerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.process_id == 0 {
            bail!("resource sample process ID must be positive");
        }
        if self.sample_interval.is_zero() {
            bail!("resource sample interval must be positive");
        }
        if self.sample_interval < sysinfo::MINIMUM_CPU_UPDATE_INTERVAL {
            bail!(
                "resource sample interval must be at least {} ms for valid process CPU samples",
                duration_millis(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL)
            );
        }
        if !(1..=MAXIMUM_RESOURCE_SAMPLES).contains(&self.maximum_samples) {
            bail!("maximum resource samples must be between 1 and {MAXIMUM_RESOURCE_SAMPLES}");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProcessResourceSample {
    /// Milliseconds since this sampler started, from its own monotonic clock.
    pub elapsed_ms: f64,
    pub status: &'static str,
    pub unavailable_reason: Option<String>,
    /// Per-core process percentage as reported by `sysinfo`; it may exceed
    /// 100% when the process uses multiple cores.
    pub cpu_percent_per_core: Option<f64>,
    /// Cumulative process CPU time, in milliseconds.
    pub accumulated_cpu_time_ms: Option<u64>,
    pub resident_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PercentileSummary {
    pub count: usize,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
    pub maximum: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProcessResourceSummary {
    pub status: &'static str,
    pub unavailable_reason: Option<String>,
    pub process_id: u32,
    pub process_start_time_unix_seconds: Option<u64>,
    pub sample_interval_ms: u64,
    pub sample_count: u64,
    pub available_sample_count: u64,
    pub unavailable_sample_count: u64,
    pub retained_sample_count: usize,
    pub retention_limit: usize,
    pub samples_truncated: bool,
    pub cpu_percent_per_core: PercentileSummary,
    pub accumulated_cpu_time_ms: ResourceDelta,
    pub resident_memory_bytes: ResourceRange,
    pub retained_samples: Vec<ProcessResourceSample>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ResourceDelta {
    pub first: Option<u64>,
    pub last: Option<u64>,
    pub delta: Option<i64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ResourceRange {
    pub first: Option<u64>,
    pub last: Option<u64>,
    pub minimum: Option<u64>,
    pub maximum: Option<u64>,
    pub delta: Option<i64>,
}

/// A sampler handle is cheap to snapshot. Call `stop` before the measured
/// child exits, then consume `summary` to capture final worker output.
pub struct ProcessResourceSampler {
    config: ProcessResourceSamplerConfig,
    state: Arc<Mutex<Accumulator>>,
    stop_sender: Option<Sender<()>>,
    worker: Option<JoinHandle<()>>,
}

impl ProcessResourceSampler {
    pub fn start(config: ProcessResourceSamplerConfig) -> Result<Self> {
        config.validate()?;
        let state = Arc::new(Mutex::new(Accumulator::new(config.maximum_samples)));
        let (stop_sender, stop_receiver) = mpsc::channel();
        let worker_state = Arc::clone(&state);
        let worker_config = config.clone();
        let worker = thread::Builder::new()
            .name(format!("performance-audit-resources-{}", config.process_id))
            .spawn(move || sample_loop(worker_config, worker_state, stop_receiver))
            .context("could not start process resource sampler")?;
        Ok(Self {
            config,
            state,
            stop_sender: Some(stop_sender),
            worker: Some(worker),
        })
    }

    /// Stop and join the worker. Repeated calls are safe.
    pub fn stop(&mut self) -> Result<()> {
        if let Some(sender) = self.stop_sender.take() {
            let _ = sender.send(());
        }
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("process resource sampler worker panicked"))?;
        }
        Ok(())
    }

    pub fn summary(&self) -> ProcessResourceSummary {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.summary(&self.config)
    }
}

impl Drop for ProcessResourceSampler {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

struct Accumulator {
    samples: VecDeque<ProcessResourceSample>,
    sample_count: u64,
    available_sample_count: u64,
    unavailable_sample_count: u64,
    process_start_time_unix_seconds: Option<u64>,
    last_unavailable_reason: Option<String>,
}

impl Accumulator {
    fn new(maximum_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(maximum_samples.min(1024)),
            sample_count: 0,
            available_sample_count: 0,
            unavailable_sample_count: 0,
            process_start_time_unix_seconds: None,
            last_unavailable_reason: None,
        }
    }

    fn push(&mut self, mut sample: ProcessResourceSample, maximum_samples: usize) {
        self.sample_count = self.sample_count.saturating_add(1);
        if sample.status == "available" {
            self.available_sample_count = self.available_sample_count.saturating_add(1);
        } else {
            self.unavailable_sample_count = self.unavailable_sample_count.saturating_add(1);
            self.last_unavailable_reason = sample.unavailable_reason.clone();
        }
        if self.samples.len() == maximum_samples {
            self.samples.pop_front();
        }
        // A sample arrives from the worker after elapsed time has been
        // recorded. Keep this value as given and never infer another clock.
        sample.elapsed_ms = sample.elapsed_ms.max(0.0);
        self.samples.push_back(sample);
    }

    fn summary(&self, config: &ProcessResourceSamplerConfig) -> ProcessResourceSummary {
        let retained_samples = self.samples.iter().cloned().collect::<Vec<_>>();
        let cpu = retained_samples
            .iter()
            .filter_map(|sample| sample.cpu_percent_per_core)
            .collect::<Vec<_>>();
        let cpu_time = retained_samples
            .iter()
            .filter_map(|sample| sample.accumulated_cpu_time_ms)
            .collect::<Vec<_>>();
        let memory = retained_samples
            .iter()
            .filter_map(|sample| sample.resident_memory_bytes)
            .collect::<Vec<_>>();
        let status = if self.available_sample_count == 0 {
            "unavailable"
        } else if self.unavailable_sample_count > 0 {
            "partial"
        } else {
            "available"
        };
        let unavailable_reason = if self.sample_count == 0 {
            Some("no resource samples were collected".to_owned())
        } else {
            self.last_unavailable_reason.clone()
        };
        ProcessResourceSummary {
            status,
            unavailable_reason,
            process_id: config.process_id,
            process_start_time_unix_seconds: self.process_start_time_unix_seconds,
            sample_interval_ms: duration_millis(config.sample_interval),
            sample_count: self.sample_count,
            available_sample_count: self.available_sample_count,
            unavailable_sample_count: self.unavailable_sample_count,
            retained_sample_count: retained_samples.len(),
            retention_limit: config.maximum_samples,
            samples_truncated: self.sample_count > retained_samples.len() as u64,
            cpu_percent_per_core: percentile_summary(cpu),
            accumulated_cpu_time_ms: resource_delta(cpu_time),
            resident_memory_bytes: resource_range(memory),
            retained_samples,
        }
    }
}

fn sample_loop(
    config: ProcessResourceSamplerConfig,
    state: Arc<Mutex<Accumulator>>,
    stop_receiver: Receiver<()>,
) {
    let pid = Pid::from_u32(config.process_id);
    let mut system = System::new_with_specifics(
        RefreshKind::nothing()
            .with_processes(ProcessRefreshKind::nothing().with_cpu().with_memory()),
    );
    let refresh_kind = ProcessRefreshKind::nothing().with_cpu().with_memory();
    system.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), true, refresh_kind);
    let expected_start_time = system.process(pid).map(Process::start_time);
    if let Some(start_time) = expected_start_time {
        let mut state_guard = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state_guard.process_start_time_unix_seconds = Some(start_time);
    }
    let started = Instant::now();

    loop {
        match stop_receiver.recv_timeout(config.sample_interval) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => return,
            Err(RecvTimeoutError::Timeout) => {}
        }
        system.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), true, refresh_kind);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let sample = match system.process(pid) {
            Some(process) if Some(process.start_time()) == expected_start_time => {
                available_sample(elapsed_ms, process)
            }
            Some(_) => unavailable_sample(
                elapsed_ms,
                "process identity could not be bound at sampler startup or the process ID was reused".to_owned(),
            ),
            None => unavailable_sample(
                elapsed_ms,
                "target process is no longer visible to the resource sampler".to_owned(),
            ),
        };
        state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(sample, config.maximum_samples);
    }
}

fn available_sample(elapsed_ms: f64, process: &Process) -> ProcessResourceSample {
    let cpu_percent = f64::from(process.cpu_usage());
    ProcessResourceSample {
        elapsed_ms,
        status: "available",
        unavailable_reason: None,
        cpu_percent_per_core: (cpu_percent.is_finite() && cpu_percent >= 0.0)
            .then_some(cpu_percent),
        accumulated_cpu_time_ms: Some(process.accumulated_cpu_time()),
        resident_memory_bytes: Some(process.memory()),
    }
}

fn unavailable_sample(elapsed_ms: f64, reason: String) -> ProcessResourceSample {
    ProcessResourceSample {
        elapsed_ms,
        status: "unavailable",
        unavailable_reason: Some(reason),
        cpu_percent_per_core: None,
        accumulated_cpu_time_ms: None,
        resident_memory_bytes: None,
    }
}

fn percentile_summary(mut values: Vec<f64>) -> PercentileSummary {
    values.retain(|value| value.is_finite());
    values.sort_by(f64::total_cmp);
    PercentileSummary {
        count: values.len(),
        p50: percentile(&values, 0.50),
        p95: percentile(&values, 0.95),
        p99: percentile(&values, 0.99),
        maximum: values.last().copied(),
    }
}

fn percentile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
    values.get(index).copied()
}

fn resource_delta(values: Vec<u64>) -> ResourceDelta {
    let first = values.first().copied();
    let last = values.last().copied();
    ResourceDelta {
        first,
        last,
        delta: first.zip(last).map(|(start, end)| signed_delta(start, end)),
    }
}

fn resource_range(values: Vec<u64>) -> ResourceRange {
    let first = values.first().copied();
    let last = values.last().copied();
    ResourceRange {
        first,
        last,
        minimum: values.iter().min().copied(),
        maximum: values.iter().max().copied(),
        delta: first.zip(last).map(|(start, end)| signed_delta(start, end)),
    }
}

fn signed_delta(start: u64, end: u64) -> i64 {
    if end >= start {
        end.saturating_sub(start).min(i64::MAX as u64) as i64
    } else {
        -(start.saturating_sub(end).min(i64::MAX as u64) as i64)
    }
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(maximum_samples: usize) -> ProcessResourceSamplerConfig {
        ProcessResourceSamplerConfig {
            process_id: 1,
            sample_interval: sysinfo::MINIMUM_CPU_UPDATE_INTERVAL,
            maximum_samples,
        }
    }

    fn sample(
        elapsed_ms: f64,
        cpu: Option<f64>,
        cpu_time: Option<u64>,
        memory: Option<u64>,
    ) -> ProcessResourceSample {
        ProcessResourceSample {
            elapsed_ms,
            status: if cpu_time.is_some() || memory.is_some() {
                "available"
            } else {
                "unavailable"
            },
            unavailable_reason: None,
            cpu_percent_per_core: cpu,
            accumulated_cpu_time_ms: cpu_time,
            resident_memory_bytes: memory,
        }
    }

    #[test]
    fn config_rejects_zero_pid_interval_and_retention() {
        let mut value = config(10);
        value.process_id = 0;
        assert!(value.validate().is_err());
        value = config(10);
        value.sample_interval = Duration::ZERO;
        assert!(value.validate().is_err());
        value.sample_interval = Duration::from_millis(1);
        assert!(value.validate().is_err());
        value = config(0);
        assert!(value.validate().is_err());
    }

    #[test]
    fn summaries_are_bounded_and_disclose_retention_and_unavailable_samples() {
        let mut accumulator = Accumulator::new(3);
        accumulator.push(sample(1.0, Some(10.0), Some(20), Some(200)), 3);
        accumulator.push(sample(2.0, Some(30.0), Some(50), Some(150)), 3);
        accumulator.push(unavailable_sample(3.0, "process disappeared".to_owned()), 3);
        accumulator.push(sample(4.0, Some(60.0), Some(80), Some(300)), 3);
        let summary = accumulator.summary(&config(3));
        assert_eq!(summary.status, "partial");
        assert_eq!(summary.sample_count, 4);
        assert_eq!(summary.available_sample_count, 3);
        assert_eq!(summary.unavailable_sample_count, 1);
        assert_eq!(summary.retained_sample_count, 3);
        assert!(summary.samples_truncated);
        assert_eq!(summary.retained_samples[0].elapsed_ms, 2.0);
        assert_eq!(summary.cpu_percent_per_core.p50, Some(60.0));
        assert_eq!(summary.accumulated_cpu_time_ms.delta, Some(30));
        assert_eq!(summary.resident_memory_bytes.minimum, Some(150));
        assert_eq!(
            summary.unavailable_reason.as_deref(),
            Some("process disappeared")
        );
    }

    #[test]
    fn empty_and_unavailable_summaries_are_explicit() {
        let empty = Accumulator::new(4).summary(&config(4));
        assert_eq!(empty.status, "unavailable");
        assert_eq!(empty.cpu_percent_per_core.count, 0);
        assert_eq!(
            empty.unavailable_reason.as_deref(),
            Some("no resource samples were collected")
        );

        let mut accumulator = Accumulator::new(4);
        accumulator.push(unavailable_sample(1.0, "denied".to_owned()), 4);
        let summary = accumulator.summary(&config(4));
        assert_eq!(summary.status, "unavailable");
        assert_eq!(summary.unavailable_sample_count, 1);
        assert_eq!(summary.resident_memory_bytes.maximum, None);
    }

    #[test]
    fn sampler_can_measure_the_current_process_without_blocking_the_caller() {
        let mut sampler = ProcessResourceSampler::start(ProcessResourceSamplerConfig {
            process_id: std::process::id(),
            sample_interval: sysinfo::MINIMUM_CPU_UPDATE_INTERVAL,
            maximum_samples: 4,
        })
        .expect("sampler should start");
        thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL * 2 + Duration::from_millis(50));
        sampler.stop().expect("sampler should stop cleanly");
        let summary = sampler.summary();
        assert_eq!(summary.process_id, std::process::id());
        assert!(summary.process_start_time_unix_seconds.is_some());
        assert!(summary.sample_count >= 1);
        assert_eq!(summary.retained_sample_count, summary.sample_count as usize);
        assert!(summary.resident_memory_bytes.maximum.is_some());
    }
}
