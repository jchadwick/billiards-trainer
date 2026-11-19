//! Deterministic phase-aware scheduling for lightweight and full perception.
//!
//! This module owns no threads and reads no clocks. The capture coordinator
//! supplies monotonic elapsed time and evidence, which makes cadence and
//! settlement behavior reproducible in both live processing and video replay.

use std::fmt;
use std::time::Duration;

/// Current operating phase of table perception.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// A layout is being reacquired or verified after motion/uncertainty.
    Recovering,
    /// A stationary, trusted layout is present and no current cue is visible.
    PreShot,
    /// A cue is visible on a trusted stationary layout.
    Aiming,
    /// Localized ball movement was observed.
    Shot,
}

/// Lightweight motion classification supplied by the image monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionEvidence {
    Quiet,
    Moving,
    OccludedOrUncertain,
}

/// Independent cue-presence evidence. `Unknown` means no usable cue update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueEvidence {
    Visible,
    NotVisible,
    Unknown,
}

/// Image-plane point used only for comparing consecutive full-layout results.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImagePoint {
    pub x: f64,
    pub y: f64,
}

/// A completed full-table observation. Failed/missing calls never add to the
/// successful settlement consensus.
#[derive(Debug, Clone, PartialEq)]
pub enum LayoutObservation {
    Success(Vec<ImagePoint>),
    Failed,
}

/// Per-phase processing rates in frames/work items per second. A zero ball
/// rate requests the lowest configured nonzero ball cadence (falling back to
/// the default pre-shot rate) so phase scheduling cannot stop observations.
/// Zero motion and cue rates still disable those optional jobs for the phase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseCadence {
    pub motion_hz: f64,
    pub ball_hz: f64,
    pub cue_hz: f64,
}

/// Rates and evidence thresholds, normally loaded from Rust live configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerConfig {
    pub recovering: PhaseCadence,
    pub pre_shot: PhaseCadence,
    pub aiming: PhaseCadence,
    pub shot: PhaseCadence,
    pub quiet_dwell: Duration,
    pub layout_consensus_count: usize,
    pub layout_position_tolerance_px: f64,
    pub motion_evidence_expiry: Duration,
    pub cue_evidence_expiry: Duration,
    pub layout_evidence_expiry: Duration,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            recovering: PhaseCadence {
                motion_hz: 15.0,
                ball_hz: 5.0,
                cue_hz: 5.0,
            },
            pre_shot: PhaseCadence {
                motion_hz: 15.0,
                ball_hz: 1.0,
                cue_hz: 5.0,
            },
            aiming: PhaseCadence {
                motion_hz: 15.0,
                ball_hz: 1.0,
                cue_hz: 5.0,
            },
            shot: PhaseCadence {
                motion_hz: 15.0,
                ball_hz: 0.0,
                cue_hz: 0.0,
            },
            quiet_dwell: Duration::from_millis(800),
            // Retained for compatibility with older configuration files. A
            // single successful clean full-table scan now replaces the layout.
            layout_consensus_count: 1,
            layout_position_tolerance_px: 4.0,
            motion_evidence_expiry: Duration::from_millis(200),
            cue_evidence_expiry: Duration::from_millis(250),
            layout_evidence_expiry: Duration::from_millis(1500),
        }
    }
}

impl SchedulerConfig {
    /// Reject impossible rates and evidence thresholds before the live loop starts.
    pub fn validate(&self) -> Result<(), SchedulerError> {
        for (phase, cadence) in [
            ("recovering", self.recovering),
            ("pre_shot", self.pre_shot),
            ("aiming", self.aiming),
            ("shot", self.shot),
        ] {
            for (job, rate) in [
                ("motion", cadence.motion_hz),
                ("ball", cadence.ball_hz),
                ("cue", cadence.cue_hz),
            ] {
                if !rate.is_finite() || rate < 0.0 {
                    return Err(SchedulerError::InvalidConfig(format!(
                        "{phase} {job} rate must be finite and nonnegative"
                    )));
                }
            }
        }
        if self.quiet_dwell.is_zero()
            || self.motion_evidence_expiry.is_zero()
            || self.cue_evidence_expiry.is_zero()
            || self.layout_evidence_expiry.is_zero()
        {
            return Err(SchedulerError::InvalidConfig(
                "dwell and evidence-expiry durations must be positive".to_owned(),
            ));
        }
        if self.layout_consensus_count == 0 {
            return Err(SchedulerError::InvalidConfig(
                "layout consensus count must be positive".to_owned(),
            ));
        }
        if !self.layout_position_tolerance_px.is_finite()
            || self.layout_position_tolerance_px <= 0.0
        {
            return Err(SchedulerError::InvalidConfig(
                "layout position tolerance must be finite and positive".to_owned(),
            ));
        }
        Ok(())
    }

    fn cadence_for_phase(&self, phase: Phase) -> PhaseCadence {
        let mut cadence = match phase {
            Phase::Recovering => self.recovering,
            Phase::PreShot => self.pre_shot,
            Phase::Aiming => self.aiming,
            Phase::Shot => self.shot,
        };
        if cadence.ball_hz == 0.0 {
            cadence.ball_hz = self.fallback_ball_cadence();
        }
        cadence
    }

    fn fallback_ball_cadence(&self) -> f64 {
        [
            self.recovering.ball_hz,
            self.pre_shot.ball_hz,
            self.aiming.ball_hz,
            self.shot.ball_hz,
        ]
        .into_iter()
        .filter(|rate| *rate > 0.0)
        .reduce(f64::min)
        .unwrap_or_else(|| Self::default().pre_shot.ball_hz)
    }
}

/// Whether an otherwise due job can start at the current tick. Unavailable
/// work consumes its deadline, so a later tick never creates a catch-up burst.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkAvailability {
    pub motion: bool,
    pub ball: bool,
    pub cue: bool,
}

impl Default for WorkAvailability {
    fn default() -> Self {
        Self {
            motion: true,
            ball: true,
            cue: true,
        }
    }
}

/// Self-contained decision for one coordinator tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleDecision {
    pub phase: Phase,
    pub map_trusted: bool,
    pub guidance_valid: bool,
    pub motion_generation: u64,
    pub layout_revision: u64,
    pub run_motion: bool,
    pub run_ball: bool,
    pub run_cue: bool,
}

/// Latest scheduler evidence used by live diagnostics and delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerEvidence {
    pub phase: Phase,
    pub map_trusted: bool,
    /// A complete empty layout has passed quiet-dwell and layout consensus.
    /// This is deliberately separate from raw detector output: a single
    /// successful empty detection is only a candidate, not proof the table is
    /// empty.
    pub empty_layout_verified: bool,
    pub guidance_valid: bool,
    pub trust_reason: TrustReason,
    pub motion_generation: u64,
    pub layout_revision: u64,
    pub motion_at: Option<Duration>,
    pub cue_at: Option<Duration>,
    pub layout_at: Option<Duration>,
    pub motion: Option<MotionEvidence>,
    pub cue: CueEvidence,
}

/// Why the current layout is or is not eligible for downstream guidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustReason {
    Trusted,
    AwaitingSettlement,
    MotionDetected,
    OccludedOrUncertain,
    MotionEvidenceExpired,
    LayoutVerificationExpired,
    ContextChanged,
}

/// Scheduler configuration, time, and layout validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    InvalidConfig(String),
    TimeWentBackwards,
    MotionGenerationOverflow,
    LayoutRevisionOverflow,
    InvalidLayout,
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid scheduler configuration: {message}")
            }
            Self::TimeWentBackwards => formatter.write_str("scheduler clock moved backwards"),
            Self::MotionGenerationOverflow => formatter.write_str("motion generation overflow"),
            Self::LayoutRevisionOverflow => formatter.write_str("layout revision overflow"),
            Self::InvalidLayout => formatter.write_str("layout contains a non-finite coordinate"),
        }
    }
}

impl std::error::Error for SchedulerError {}

/// Coordinator-local scheduler. Missed work is skipped, never queued.
pub struct PhaseScheduler {
    config: SchedulerConfig,
    phase: Phase,
    trusted: bool,
    trust_reason: TrustReason,
    motion_generation: u64,
    layout_revision: u64,
    last_now: Option<Duration>,
    motion_at: Option<Duration>,
    motion: Option<MotionEvidence>,
    quiet_since: Option<Duration>,
    cue_at: Option<Duration>,
    cue: CueEvidence,
    layout_at: Option<Duration>,
    candidate_layout: Option<Vec<ImagePoint>>,
    consistent_layouts: usize,
    deadlines: [Option<Duration>; 3],
}

impl PhaseScheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self, SchedulerError> {
        config.validate()?;
        Ok(Self {
            config,
            phase: Phase::Recovering,
            trusted: false,
            trust_reason: TrustReason::AwaitingSettlement,
            motion_generation: 0,
            layout_revision: 0,
            last_now: None,
            motion_at: None,
            motion: None,
            quiet_since: None,
            cue_at: None,
            cue: CueEvidence::Unknown,
            layout_at: None,
            candidate_layout: None,
            consistent_layouts: 0,
            deadlines: [None; 3],
        })
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn motion_generation(&self) -> u64 {
        self.motion_generation
    }

    /// The configured capture-age bound shared by cue freshness and silence.
    pub fn cue_evidence_expiry(&self) -> Duration {
        self.config.cue_evidence_expiry
    }

    /// Whether a full scan captured after settlement completed can replace the
    /// whole-table layout now. This prevents slow work captured before the
    /// quiet dwell from becoming trusted merely because it finished later.
    pub fn clean_layout_capture_is_current(&self, captured_at: Duration, now: Duration) -> bool {
        self.motion == Some(MotionEvidence::Quiet)
            && self
                .motion_at
                .is_some_and(|at| now.saturating_sub(at) <= self.config.motion_evidence_expiry)
            && self.quiet_since.is_some_and(|quiet_since| {
                quiet_since
                    .checked_add(self.config.quiet_dwell)
                    .is_some_and(|settled_at| captured_at >= settled_at)
            })
    }

    pub fn map_trusted(&self) -> bool {
        self.trusted
    }

    pub fn evidence(&self, now: Duration) -> Result<SchedulerEvidence, SchedulerError> {
        if self.last_now.is_some_and(|previous| now < previous) {
            return Err(SchedulerError::TimeWentBackwards);
        }
        Ok(self.evidence_unchecked(now))
    }

    pub fn observe_motion(
        &mut self,
        now: Duration,
        evidence: MotionEvidence,
    ) -> Result<(), SchedulerError> {
        self.check_time(now)?;
        let previous_motion = self.motion;
        self.motion_at = Some(now);
        self.motion = Some(evidence);
        match evidence {
            MotionEvidence::Quiet => {
                if previous_motion != Some(MotionEvidence::Quiet) || self.quiet_since.is_none() {
                    self.quiet_since = Some(now);
                }
                self.refresh_phase(now);
            }
            MotionEvidence::Moving => {
                let onset = self.trusted || previous_motion == Some(MotionEvidence::Quiet);
                self.invalidate_layout(TrustReason::MotionDetected);
                self.quiet_since = None;
                self.cue_at = None;
                self.cue = CueEvidence::Unknown;
                if onset {
                    self.bump_motion_generation()?;
                }
                self.transition(Phase::Shot, now);
            }
            MotionEvidence::OccludedOrUncertain => {
                let onset = self.trusted || previous_motion == Some(MotionEvidence::Quiet);
                self.invalidate_layout(TrustReason::OccludedOrUncertain);
                self.quiet_since = None;
                self.cue_at = None;
                self.cue = CueEvidence::Unknown;
                if onset {
                    self.bump_motion_generation()?;
                }
                self.transition(Phase::Recovering, now);
            }
        }
        Ok(())
    }

    /// Mark the whole-table layout stale without pretending the camera saw
    /// motion. Appearance evidence can invalidate an accepted layout while
    /// current whole-table motion still provides useful scan cadence and
    /// quiet-dwell information.
    pub fn mark_layout_untrusted(
        &mut self,
        now: Duration,
        reason: TrustReason,
    ) -> Result<(), SchedulerError> {
        self.check_time(now)?;
        // A visibility transition after a trusted layout invalidates work that
        // began under the old view, while leaving uninterrupted quiet motion
        // evidence and its dwell intact for subsequent detections.
        if self.trusted {
            self.bump_motion_generation()?;
        }
        self.invalidate_layout(reason);
        self.refresh_phase(now);
        Ok(())
    }

    pub fn observe_cue(
        &mut self,
        now: Duration,
        evidence: CueEvidence,
    ) -> Result<(), SchedulerError> {
        self.observe_cue_at(now, now, evidence).map(|_| ())
    }

    /// Record cue evidence produced asynchronously from a frame observed at
    /// `observed_at`. Scheduler time advances at `now`, while the evidence age
    /// remains anchored to the source observation rather than being refreshed
    /// when a worker happens to finish.
    pub fn observe_cue_at(
        &mut self,
        now: Duration,
        observed_at: Duration,
        evidence: CueEvidence,
    ) -> Result<bool, SchedulerError> {
        self.check_time(now)?;
        if observed_at > now {
            return Err(SchedulerError::TimeWentBackwards);
        }
        if self.cue_at.is_some_and(|previous| observed_at < previous) {
            self.refresh_phase(now);
            return Ok(false);
        }
        self.cue_at = Some(observed_at);
        self.cue = evidence;
        self.refresh_phase(now);
        Ok(true)
    }

    pub fn observe_layout(
        &mut self,
        now: Duration,
        observation: LayoutObservation,
    ) -> Result<(), SchedulerError> {
        self.observe_layout_at(now, now, observation)
    }

    /// Record when the detector measured the layout separately from when the
    /// coordinator made its trust decision after asynchronous processing.
    pub fn observe_layout_at(
        &mut self,
        now: Duration,
        observed_at: Duration,
        observation: LayoutObservation,
    ) -> Result<(), SchedulerError> {
        self.check_time(now)?;
        if observed_at > now {
            return Err(SchedulerError::TimeWentBackwards);
        }
        if let LayoutObservation::Success(points) = &observation {
            if points
                .iter()
                .any(|point| !point.x.is_finite() || !point.y.is_finite())
            {
                return Err(SchedulerError::InvalidLayout);
            }
        }
        if !self.quiet_dwell_complete(now) {
            return Ok(());
        }
        match observation {
            LayoutObservation::Failed => {}
            LayoutObservation::Success(points) => {
                self.layout_at = Some(observed_at);
                // Every successful clean full scan is authoritative. Consensus
                // settings remain deserializable for old configs, but no longer
                // delay replacement or require agreement before trust.
                self.candidate_layout = Some(points);
                self.consistent_layouts = 1;
                if !self.trusted {
                    self.layout_revision = self
                        .layout_revision
                        .checked_add(1)
                        .ok_or(SchedulerError::LayoutRevisionOverflow)?;
                    self.trusted = true;
                    self.trust_reason = TrustReason::Trusted;
                    self.refresh_phase(now);
                }
            }
        }
        Ok(())
    }

    /// Invalidate all evidence after a source, calibration, or configuration
    /// context change. The generation prevents an older detector result from
    /// restoring trust after the change.
    pub fn invalidate_context(&mut self, now: Duration) -> Result<(), SchedulerError> {
        self.check_time(now)?;
        self.invalidate_layout(TrustReason::ContextChanged);
        self.motion_at = None;
        self.motion = None;
        self.quiet_since = None;
        self.cue_at = None;
        self.cue = CueEvidence::Unknown;
        self.bump_motion_generation()?;
        self.transition(Phase::Recovering, now);
        Ok(())
    }

    /// Advance evidence expiry and phase transitions without consuming cadence
    /// slots. Use this when time passes without a source frame; the next frame
    /// can then receive any work that is due at its observation time.
    pub fn advance_time(&mut self, now: Duration) -> Result<SchedulerEvidence, SchedulerError> {
        self.check_time(now)?;
        self.expire_evidence(now)?;
        self.refresh_phase(now);
        Ok(self.evidence_unchecked(now))
    }

    /// Advance deadlines and return scheduled jobs due now. An unavailable due
    /// job loses this slot and must wait for its next interval. Call
    /// [`advance_time`](Self::advance_time) instead when no source frame is
    /// available, so timeouts cannot consume otherwise due work.
    pub fn tick(
        &mut self,
        now: Duration,
        available: WorkAvailability,
    ) -> Result<ScheduleDecision, SchedulerError> {
        let evidence = self.advance_time(now)?;
        let cadence = self.config.cadence_for_phase(self.phase);
        let rates = [cadence.motion_hz, cadence.ball_hz, cadence.cue_hz];
        let available = [available.motion, available.ball, available.cue];
        let mut due = [false; 3];
        for index in 0..3 {
            let rate = rates[index];
            if rate == 0.0 {
                self.deadlines[index] = None;
                continue;
            }
            let deadline = self.deadlines[index].get_or_insert(now);
            if now >= *deadline {
                due[index] = available[index];
                // Schedule relative to the current tick. A late caller loses
                // missed slots instead of creating a burst of catch-up work.
                self.deadlines[index] = Some(now.saturating_add(period_for_hz(rate)));
            }
        }
        Ok(ScheduleDecision {
            phase: evidence.phase,
            map_trusted: evidence.map_trusted,
            guidance_valid: evidence.guidance_valid,
            motion_generation: evidence.motion_generation,
            layout_revision: evidence.layout_revision,
            run_motion: due[0],
            run_ball: due[1],
            run_cue: due[2],
        })
    }

    fn evidence_unchecked(&self, now: Duration) -> SchedulerEvidence {
        let motion_evidence_fresh = self
            .motion_at
            .is_some_and(|at| now.saturating_sub(at) <= self.config.motion_evidence_expiry);
        let motion_fresh_quiet =
            self.motion == Some(MotionEvidence::Quiet) && motion_evidence_fresh;
        // Measurement age is diagnostic only. A quiet stationary layout stays
        // trusted while current whole-table motion evidence remains fresh.
        let layout_fresh = self.layout_at.is_some();
        let cue_fresh = self
            .cue_at
            .is_some_and(|at| now.saturating_sub(at) <= self.config.cue_evidence_expiry)
            && self.cue != CueEvidence::Unknown;
        let map_trusted = self.trusted && motion_fresh_quiet && layout_fresh;
        let phase = if self.phase == Phase::Shot && self.motion == Some(MotionEvidence::Moving) {
            Phase::Shot
        } else if !map_trusted {
            Phase::Recovering
        } else if self.cue == CueEvidence::Visible && cue_fresh {
            Phase::Aiming
        } else {
            Phase::PreShot
        };
        let trust_reason = if map_trusted {
            TrustReason::Trusted
        } else if !motion_evidence_fresh {
            TrustReason::MotionEvidenceExpired
        } else if self.motion != Some(MotionEvidence::Quiet) {
            self.trust_reason
        } else if !layout_fresh {
            TrustReason::LayoutVerificationExpired
        } else {
            self.trust_reason
        };
        SchedulerEvidence {
            phase,
            map_trusted: self.trusted && motion_fresh_quiet && layout_fresh,
            empty_layout_verified: self.trusted
                && motion_fresh_quiet
                && layout_fresh
                && self.candidate_layout.as_ref().is_some_and(Vec::is_empty),
            guidance_valid: self.trusted
                && motion_fresh_quiet
                && layout_fresh
                && cue_fresh
                && self.cue == CueEvidence::Visible,
            trust_reason,
            motion_generation: self.motion_generation,
            layout_revision: self.layout_revision,
            motion_at: self.motion_at,
            cue_at: self.cue_at,
            layout_at: self.layout_at,
            motion: self.motion,
            cue: self.cue,
        }
    }

    fn check_time(&mut self, now: Duration) -> Result<(), SchedulerError> {
        if self.last_now.is_some_and(|previous| now < previous) {
            return Err(SchedulerError::TimeWentBackwards);
        }
        self.last_now = Some(now);
        Ok(())
    }

    fn expire_evidence(&mut self, now: Duration) -> Result<(), SchedulerError> {
        let motion_fresh = self
            .motion_at
            .is_some_and(|at| now.saturating_sub(at) <= self.config.motion_evidence_expiry);
        if !motion_fresh && self.motion_at.is_some() {
            let should_bump = self.trusted || self.motion == Some(MotionEvidence::Quiet);
            self.invalidate_layout(TrustReason::MotionEvidenceExpired);
            self.quiet_since = None;
            self.motion = None;
            self.motion_at = None;
            if should_bump {
                self.bump_motion_generation()?;
            }
            self.transition(Phase::Recovering, now);
        }
        if self
            .cue_at
            .is_some_and(|at| now.saturating_sub(at) > self.config.cue_evidence_expiry)
        {
            self.cue = CueEvidence::Unknown;
        }
        Ok(())
    }

    fn quiet_dwell_complete(&self, now: Duration) -> bool {
        self.motion == Some(MotionEvidence::Quiet)
            && self
                .motion_at
                .is_some_and(|at| now.saturating_sub(at) <= self.config.motion_evidence_expiry)
            && self
                .quiet_since
                .is_some_and(|since| now.saturating_sub(since) >= self.config.quiet_dwell)
    }

    fn refresh_phase(&mut self, now: Duration) {
        let motion_fresh_quiet = self.motion == Some(MotionEvidence::Quiet)
            && self
                .motion_at
                .is_some_and(|at| now.saturating_sub(at) <= self.config.motion_evidence_expiry);
        let layout_fresh = self.layout_at.is_some();
        if !self.trusted || !motion_fresh_quiet || !layout_fresh {
            if self.phase != Phase::Shot || self.motion != Some(MotionEvidence::Moving) {
                self.transition(Phase::Recovering, now);
            }
            return;
        }
        if self.cue == CueEvidence::Visible
            && self
                .cue_at
                .is_some_and(|at| now.saturating_sub(at) <= self.config.cue_evidence_expiry)
        {
            self.transition(Phase::Aiming, now);
        } else {
            // Losing or aging out cue evidence does not invalidate verified
            // balls; it simply removes aiming eligibility.
            self.transition(Phase::PreShot, now);
        }
    }

    fn invalidate_layout(&mut self, reason: TrustReason) {
        self.trusted = false;
        self.trust_reason = reason;
        self.layout_at = None;
        self.candidate_layout = None;
        self.consistent_layouts = 0;
    }

    fn bump_motion_generation(&mut self) -> Result<(), SchedulerError> {
        self.motion_generation = self
            .motion_generation
            .checked_add(1)
            .ok_or(SchedulerError::MotionGenerationOverflow)?;
        Ok(())
    }

    fn transition(&mut self, phase: Phase, now: Duration) {
        if self.phase != phase {
            self.phase = phase;
            self.deadlines = [Some(now); 3];
        }
    }
}

#[allow(dead_code)]
fn layouts_match(left: &[ImagePoint], right: &[ImagePoint], tolerance: f64) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut used = vec![false; right.len()];
    for point in left {
        let Some((index, squared_distance)) = right
            .iter()
            .enumerate()
            .filter(|(index, _)| !used[*index])
            .map(|(index, other)| {
                let dx = point.x - other.x;
                let dy = point.y - other.y;
                (index, dx.mul_add(dx, dy * dy))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
        else {
            return false;
        };
        if squared_distance > tolerance * tolerance {
            return false;
        }
        used[index] = true;
    }
    true
}

fn period_for_hz(hz: f64) -> Duration {
    let seconds = 1.0 / hz;
    if !seconds.is_finite() || seconds >= u64::MAX as f64 {
        return Duration::MAX;
    }
    if seconds < 1e-9 {
        Duration::from_nanos(1)
    } else {
        Duration::from_secs_f64(seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    fn scheduler() -> PhaseScheduler {
        PhaseScheduler::new(SchedulerConfig::default()).unwrap()
    }

    fn layout(x: f64) -> LayoutObservation {
        LayoutObservation::Success(vec![ImagePoint { x, y: 10.0 }])
    }

    fn settle(scheduler: &mut PhaseScheduler, start: u64) {
        for time in [
            start,
            start + 150,
            start + 300,
            start + 450,
            start + 600,
            start + 750,
            start + 800,
        ] {
            scheduler
                .observe_motion(ms(time), MotionEvidence::Quiet)
                .unwrap();
        }
        for time in [start + 800, start + 850, start + 900] {
            scheduler.observe_layout(ms(time), layout(10.0)).unwrap();
        }
    }

    #[test]
    fn one_successful_scan_after_quiet_dwell_establishes_trust() {
        let mut s = scheduler();
        s.observe_motion(ms(0), MotionEvidence::Quiet).unwrap();
        s.observe_layout(ms(799), layout(10.0)).unwrap();
        assert!(!s.map_trusted());
        s.observe_motion(ms(800), MotionEvidence::Quiet).unwrap();
        s.observe_layout(ms(800), LayoutObservation::Failed)
            .unwrap();
        s.observe_layout(ms(810), layout(10.0)).unwrap();
        assert!(s.map_trusted());
    }

    #[test]
    fn visibility_conflict_stales_layout_without_erasing_quiet_motion_dwell() {
        let mut s = scheduler();
        settle(&mut s, 0);
        let before = s.evidence(ms(900)).unwrap();
        assert!(before.map_trusted);
        s.mark_layout_untrusted(ms(900), TrustReason::OccludedOrUncertain)
            .unwrap();

        let after = s.evidence(ms(900)).unwrap();
        assert!(!after.map_trusted);
        assert_eq!(after.motion, Some(MotionEvidence::Quiet));
        assert_eq!(after.motion_generation, before.motion_generation + 1);
        assert!(s.clean_layout_capture_is_current(ms(850), ms(900)));
    }

    #[test]
    fn slow_layout_decision_keeps_the_detector_capture_timestamp() {
        let config = SchedulerConfig {
            motion_evidence_expiry: ms(1_000),
            layout_evidence_expiry: ms(5_000),
            ..SchedulerConfig::default()
        };
        let mut s = PhaseScheduler::new(config).unwrap();
        for time in [0, 800, 1_600, 2_400, 3_000] {
            s.observe_motion(ms(time), MotionEvidence::Quiet).unwrap();
        }
        s.observe_layout_at(ms(3_000), ms(1_000), layout(10.0))
            .unwrap();

        let evidence = s.evidence(ms(3_000)).unwrap();
        assert!(
            evidence.map_trusted,
            "current quiet continuity controls trust"
        );
        assert_eq!(evidence.layout_at, Some(ms(1_000)));
        assert_eq!(
            evidence.layout_at.map(|at| ms(3_000).saturating_sub(at)),
            Some(ms(2_000)),
            "decision time must not rejuvenate a slow detector measurement"
        );
    }

    #[test]
    fn empty_layout_requires_quiet_dwell_and_one_successful_scan() {
        let mut s = scheduler();
        s.observe_motion(ms(0), MotionEvidence::Quiet).unwrap();
        s.observe_layout(ms(799), LayoutObservation::Success(vec![]))
            .unwrap();
        assert!(!s.evidence(ms(799)).unwrap().empty_layout_verified);

        s.observe_motion(ms(800), MotionEvidence::Quiet).unwrap();
        s.observe_layout(ms(800), LayoutObservation::Success(vec![]))
            .unwrap();
        assert!(s.evidence(ms(800)).unwrap().empty_layout_verified);
    }

    #[test]
    fn failed_or_nonempty_observations_and_invalidation_clear_empty_consensus() {
        let mut s = scheduler();
        s.observe_motion(ms(0), MotionEvidence::Quiet).unwrap();
        s.observe_motion(ms(800), MotionEvidence::Quiet).unwrap();
        s.observe_layout(ms(800), LayoutObservation::Success(vec![]))
            .unwrap();
        assert!(s.evidence(ms(900)).unwrap().empty_layout_verified);

        s.observe_layout(ms(910), LayoutObservation::Failed)
            .unwrap();
        assert!(s.evidence(ms(910)).unwrap().empty_layout_verified);
        s.observe_layout(ms(920), layout(10.0)).unwrap();
        assert!(!s.evidence(ms(920)).unwrap().empty_layout_verified);

        s.observe_motion(ms(930), MotionEvidence::Moving).unwrap();
        assert!(!s.evidence(ms(930)).unwrap().empty_layout_verified);
        s.observe_motion(ms(1_000), MotionEvidence::Quiet).unwrap();
        s.observe_motion(ms(1_800), MotionEvidence::Quiet).unwrap();
        s.observe_layout(ms(1_800), LayoutObservation::Success(vec![]))
            .unwrap();
        assert!(s.evidence(ms(1_800)).unwrap().empty_layout_verified);
        s.invalidate_context(ms(1_810)).unwrap();
        assert!(!s.evidence(ms(1_810)).unwrap().empty_layout_verified);
    }

    #[test]
    fn only_localized_motion_declares_a_shot_and_invalidates_immediately() {
        let mut s = scheduler();
        settle(&mut s, 0);
        let generation = s.motion_generation();
        s.observe_motion(ms(950), MotionEvidence::Moving).unwrap();
        assert_eq!(s.phase(), Phase::Shot);
        assert!(!s.map_trusted());
        assert_eq!(s.motion_generation(), generation + 1);
        assert_eq!(
            s.evidence(ms(950)).unwrap().trust_reason,
            TrustReason::MotionDetected
        );
        s.observe_motion(ms(1_000), MotionEvidence::Moving).unwrap();
        assert_eq!(s.motion_generation(), generation + 1);
        s.observe_motion(ms(1_050), MotionEvidence::OccludedOrUncertain)
            .unwrap();
        assert_eq!(s.phase(), Phase::Recovering);
        assert_eq!(
            s.evidence(ms(1_050)).unwrap().trust_reason,
            TrustReason::OccludedOrUncertain
        );
    }

    #[test]
    fn zero_shot_ball_rate_uses_a_low_configured_scan_without_restoring_trust() {
        let config = SchedulerConfig {
            recovering: PhaseCadence {
                motion_hz: 15.0,
                ball_hz: 5.0,
                cue_hz: 5.0,
            },
            pre_shot: PhaseCadence {
                motion_hz: 15.0,
                ball_hz: 1.0,
                cue_hz: 5.0,
            },
            aiming: PhaseCadence {
                motion_hz: 15.0,
                ball_hz: 1.0,
                cue_hz: 5.0,
            },
            shot: PhaseCadence {
                motion_hz: 15.0,
                ball_hz: 0.0,
                cue_hz: 0.0,
            },
            motion_evidence_expiry: ms(2_000),
            ..SchedulerConfig::default()
        };
        let mut s = PhaseScheduler::new(config).unwrap();
        settle(&mut s, 0);
        s.observe_motion(ms(950), MotionEvidence::Moving).unwrap();
        assert_eq!(s.phase(), Phase::Shot);

        let first = s.tick(ms(950), WorkAvailability::default()).unwrap();
        assert_eq!(first.phase, Phase::Shot);
        assert!(first.run_ball);
        assert!(!first.map_trusted);

        // A zero-rate shot preference becomes the slowest configured nonzero
        // ball cadence; it stays sparse while still allowing observations.
        let early = s.tick(ms(1_949), WorkAvailability::default()).unwrap();
        assert!(!early.run_ball);
        let next = s.tick(ms(1_950), WorkAvailability::default()).unwrap();
        assert!(next.run_ball);
        assert!(!next.map_trusted);
    }

    #[test]
    fn all_zero_ball_rates_fall_back_to_the_default_pre_shot_cadence() {
        let disabled = PhaseCadence {
            motion_hz: 0.0,
            ball_hz: 0.0,
            cue_hz: 0.0,
        };
        let config = SchedulerConfig {
            recovering: disabled,
            pre_shot: disabled,
            aiming: disabled,
            shot: disabled,
            ..SchedulerConfig::default()
        };
        let fallback = SchedulerConfig::default().pre_shot.ball_hz;
        assert!(fallback > 0.0);
        assert_eq!(config.cadence_for_phase(Phase::Shot).ball_hz, fallback);
    }

    #[test]
    fn zero_ball_rate_uses_the_lowest_positive_cadence_from_every_phase() {
        let disabled = PhaseCadence {
            motion_hz: 0.0,
            ball_hz: 0.0,
            cue_hz: 0.0,
        };
        let config = SchedulerConfig {
            recovering: disabled,
            pre_shot: disabled,
            aiming: disabled,
            shot: PhaseCadence {
                ball_hz: 3.0,
                ..disabled
            },
            ..SchedulerConfig::default()
        };

        assert_eq!(config.cadence_for_phase(Phase::Recovering).ball_hz, 3.0);
    }

    #[test]
    fn quiet_after_shot_returns_to_recovery_and_restores_ball_work() {
        let mut s = scheduler();
        settle(&mut s, 0);
        s.observe_motion(ms(950), MotionEvidence::Moving).unwrap();
        s.observe_motion(ms(1_000), MotionEvidence::Quiet).unwrap();
        assert_eq!(s.phase(), Phase::Recovering);
        let decision = s.tick(ms(1_000), WorkAvailability::default()).unwrap();
        assert!(decision.run_ball);
        assert!(decision.run_cue);
    }

    #[test]
    fn cue_loss_returns_to_preshot_without_invalidating_layout() {
        let mut s = scheduler();
        settle(&mut s, 0);
        s.observe_cue(ms(910), CueEvidence::Visible).unwrap();
        assert_eq!(s.phase(), Phase::Aiming);
        s.observe_cue(ms(920), CueEvidence::NotVisible).unwrap();
        assert_eq!(s.phase(), Phase::PreShot);
        assert!(s.map_trusted());
    }

    #[test]
    fn asynchronous_cue_completion_keeps_frame_observation_age_and_rejects_older_result() {
        let mut s = scheduler();
        settle(&mut s, 0);

        assert!(s
            .observe_cue_at(ms(920), ms(900), CueEvidence::Visible)
            .unwrap());
        assert_eq!(s.evidence(ms(920)).unwrap().cue_at, Some(ms(900)));

        assert!(!s
            .observe_cue_at(ms(930), ms(890), CueEvidence::NotVisible)
            .unwrap());
        assert_eq!(s.evidence(ms(930)).unwrap().cue_at, Some(ms(900)));
        assert_eq!(s.evidence(ms(930)).unwrap().cue, CueEvidence::Visible);
    }

    #[test]
    fn deadlines_skip_missed_slots_without_catchup() {
        let mut s = scheduler();
        let first = s.tick(ms(0), WorkAvailability::default()).unwrap();
        assert!(first.run_motion && first.run_ball && first.run_cue);
        let late = s.tick(ms(1_000), WorkAvailability::default()).unwrap();
        assert!(late.run_motion && late.run_ball && late.run_cue);
        let immediate = s.tick(ms(1_001), WorkAvailability::default()).unwrap();
        assert!(!immediate.run_motion && !immediate.run_ball && !immediate.run_cue);
    }

    #[test]
    fn advancing_time_without_a_frame_does_not_consume_due_work() {
        let mut s = scheduler();
        let first = s.tick(ms(0), WorkAvailability::default()).unwrap();
        assert!(first.run_motion && first.run_ball && first.run_cue);

        // A receive timeout advances expiry state between frames. It must not
        // make the due motion, ball, or cue jobs unavailable to the next frame.
        s.advance_time(ms(200)).unwrap();
        let next_frame = s.tick(ms(201), WorkAvailability::default()).unwrap();
        assert!(next_frame.run_motion && next_frame.run_ball && next_frame.run_cue);
    }

    #[test]
    fn asynchronous_phase_transition_does_not_consume_the_next_frame_deadline() {
        // Use a 5 Hz motion cadence so the frame boundary and deadline match
        // the live capture case that exposed a second, same-frame scheduler
        // tick consuming motion work just before it was deliberately skipped.
        let cadence = PhaseCadence {
            motion_hz: 5.0,
            ball_hz: 5.0,
            cue_hz: 5.0,
        };
        let config = SchedulerConfig {
            recovering: cadence,
            pre_shot: cadence,
            aiming: cadence,
            shot: cadence,
            ..SchedulerConfig::default()
        };
        let mut s = PhaseScheduler::new(config).unwrap();
        settle(&mut s, 0);
        s.observe_motion(ms(900), MotionEvidence::Quiet).unwrap();

        let first = s.tick(ms(900), WorkAvailability::default()).unwrap();
        assert!(first.run_motion && first.run_ball && first.run_cue);

        // This source frame arrives a fraction before the 200 ms deadline.
        let early = s.tick(ms(1_099), WorkAvailability::default()).unwrap();
        assert!(!early.run_motion && !early.run_ball && !early.run_cue);
        s.observe_motion(ms(1_099), MotionEvidence::Quiet).unwrap();

        // The cue worker completes just after that frame and moves the phase
        // to Aiming. Runtime evidence refreshes must not call tick here.
        assert!(s
            .observe_cue_at(ms(1_100), ms(1_099), CueEvidence::Visible)
            .unwrap());
        assert_eq!(s.phase(), Phase::Aiming);
        s.advance_time(ms(1_100)).unwrap();

        // The next frame is still entitled to the due slot. A tick at 1,100
        // would have advanced its deadline and incorrectly hidden this work.
        let next = s.tick(ms(1_299), WorkAvailability::default()).unwrap();
        assert!(next.run_motion && next.run_ball && next.run_cue);
        assert_eq!(next.phase, Phase::Aiming);
    }

    #[test]
    fn stale_motion_and_layout_remove_trust_and_guidance() {
        let mut s = scheduler();
        settle(&mut s, 0);
        s.observe_cue(ms(910), CueEvidence::Visible).unwrap();
        assert!(s.map_trusted());
        let stale_motion = s.tick(ms(1_151), WorkAvailability::default()).unwrap();
        assert!(!stale_motion.map_trusted);
        assert!(!s.evidence(ms(1_151)).unwrap().empty_layout_verified);
        assert_eq!(stale_motion.phase, Phase::Recovering);
    }

    #[test]
    fn an_old_full_scan_stays_trusted_while_current_motion_evidence_is_fresh() {
        let config = SchedulerConfig {
            motion_evidence_expiry: ms(3_000),
            layout_evidence_expiry: ms(100),
            ..SchedulerConfig::default()
        };
        let mut s = PhaseScheduler::new(config).unwrap();
        s.observe_motion(ms(0), MotionEvidence::Quiet).unwrap();
        s.observe_motion(ms(800), MotionEvidence::Quiet).unwrap();
        let generation = s.motion_generation();

        // The source scan is already older than the legacy layout-expiry
        // setting here. Its original age must not be reset by later quiet
        // frames; current motion evidence is the trust lifetime.
        s.observe_layout(ms(800), layout(10.0)).unwrap();
        assert!(s.evidence(ms(2_300)).unwrap().map_trusted);
        assert!(s.evidence(ms(3_800)).unwrap().map_trusted);
        assert_eq!(s.evidence(ms(3_800)).unwrap().layout_at, Some(ms(800)));
        assert_eq!(s.motion_generation(), generation);
    }

    #[test]
    fn context_invalidation_rejects_old_generation() {
        let mut s = scheduler();
        settle(&mut s, 0);
        let before = s.motion_generation();
        s.invalidate_context(ms(950)).unwrap();
        assert_eq!(s.motion_generation(), before + 1);
        assert!(!s.map_trusted());
        assert_eq!(s.phase(), Phase::Recovering);
    }

    #[test]
    fn malformed_layout_and_time_regression_are_rejected() {
        let mut s = scheduler();
        s.observe_motion(ms(10), MotionEvidence::Quiet).unwrap();
        assert_eq!(
            s.observe_layout(
                ms(11),
                LayoutObservation::Success(vec![ImagePoint {
                    x: f64::NAN,
                    y: 0.0
                }])
            ),
            Err(SchedulerError::InvalidLayout)
        );
        assert_eq!(
            s.tick(ms(9), WorkAvailability::default()),
            Err(SchedulerError::TimeWentBackwards)
        );
    }
}
