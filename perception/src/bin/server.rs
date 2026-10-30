//! Rust application gateway for live localization and browser state delivery.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::time::Duration;

use billiards_perception::configuration_authority::{AuthorityLimits, ConfigurationAuthority};
use billiards_perception::live_audit::LiveAuditConfig;
use billiards_perception::server::{
    run_with_live, GatewayAuditConfig, LiveProcessLaunchOptions, ServerConfig,
    DEFAULT_MAX_CONFIGURATION_REQUESTS,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "billiards-server",
    about = "Serve Rust scene localization over local HTTP and WebSocket"
)]
struct ServerArgs {
    /// HTTP interface. Defaults to loopback for local development.
    #[arg(long, default_value = "127.0.0.1")]
    http_bind: IpAddr,
    /// HTTP port used by the local development gateway.
    #[arg(long, default_value_t = 8001)]
    http_port: u16,
    /// Scene ingress interface. Must be loopback.
    #[arg(long, default_value = "127.0.0.1")]
    scene_bind: IpAddr,
    /// Loopback scene ingress port for the live-v3 producer.
    #[arg(long, default_value_t = 49_321)]
    scene_port: u16,
    /// Maximum payload size for one framed scene message, excluding its header (hard limit: 32 MiB).
    #[arg(long, default_value_t = 262_144)]
    maximum_scene_bytes: usize,
    /// Active configuration file. Required for live capture and configuration HTTP routes.
    #[arg(long = "config")]
    configuration: Option<PathBuf>,
    /// Deployment project/data root. Relative state-store paths stay inside it; defaults to the service working directory.
    #[arg(long)]
    project_root: Option<PathBuf>,
    /// Optional prebuilt HTML/JS/CSS directory served by the Rust gateway.
    #[arg(long)]
    web_root: Option<PathBuf>,
    /// Deadline for reading each framed TCP header or body.
    #[arg(long, default_value_t = 2_000)]
    tcp_read_timeout_ms: u64,
    /// Override the configured maximum observation age used to mark locations stale.
    #[arg(long)]
    stale_after_ms: Option<u64>,
    /// Remove cached locations after this many milliseconds without localization.
    #[arg(long)]
    expire_after_ms: Option<u64>,
    /// Disconnect WebSocket clients whose writes exceed this deadline.
    #[arg(long, default_value_t = 250)]
    websocket_send_deadline_ms: u64,
    /// Bound simultaneous WebSocket consumers.
    #[arg(long, default_value_t = 64)]
    maximum_websocket_clients: usize,
    /// Bound concurrent filesystem-backed configuration requests.
    #[arg(long, default_value_t = DEFAULT_MAX_CONFIGURATION_REQUESTS)]
    maximum_configuration_requests: usize,
    /// Start and supervise Rust live capture/inference using the active config.
    #[arg(long)]
    start_live: bool,
    /// Override the colocated live detector executable.
    #[arg(long)]
    live_executable: Option<PathBuf>,
    /// Override the colocated GStreamer capture helper executable.
    #[arg(long)]
    capture_helper_executable: Option<PathBuf>,
    /// Optional loopback receiver for bounded live timing records.
    /// All --live-audit-* settings must be supplied together with --start-live.
    #[arg(long)]
    live_audit_tcp: Option<SocketAddr>,
    /// Maximum number of queued audit records (1..=4096).
    #[arg(long)]
    live_audit_queue_capacity: Option<usize>,
    /// Maximum audit record size in bytes (1024..=1048576).
    #[arg(long)]
    live_audit_max_record_bytes: Option<usize>,
    /// Audit receiver connect deadline in milliseconds.
    #[arg(long)]
    live_audit_connect_timeout_ms: Option<u64>,
    /// Audit socket write deadline in milliseconds.
    #[arg(long)]
    live_audit_write_timeout_ms: Option<u64>,
    /// Optional loopback receiver for gateway WebSocket timing records.
    /// This audit is independent of detector timing and does not require --start-live.
    #[arg(long)]
    gateway_audit_tcp: Option<SocketAddr>,
    /// Maximum number of queued gateway audit records (1..=4096).
    #[arg(long)]
    gateway_audit_queue_capacity: Option<usize>,
    /// Maximum gateway audit record size in bytes (1024..=1048576).
    #[arg(long)]
    gateway_audit_max_record_bytes: Option<usize>,
    /// Gateway audit receiver connect deadline in milliseconds.
    #[arg(long)]
    gateway_audit_connect_timeout_ms: Option<u64>,
    /// Gateway audit socket write deadline in milliseconds.
    #[arg(long)]
    gateway_audit_write_timeout_ms: Option<u64>,
    /// Optional caller-supplied gateway audit run identity.
    #[arg(long)]
    gateway_audit_run_id: Option<String>,
}

impl ServerArgs {
    fn live_options(&self) -> Result<LiveProcessLaunchOptions, String> {
        let audit_config = self.live_audit_config()?;
        let gateway_audit_config = self.gateway_audit_config()?;
        Ok(LiveProcessLaunchOptions {
            enabled: self.start_live,
            executable_path: self.live_executable.clone(),
            capture_helper_path: self.capture_helper_executable.clone(),
            audit_config,
            gateway_audit_config,
        })
    }

    fn gateway_audit_config(&self) -> Result<Option<GatewayAuditConfig>, String> {
        let supplied = [
            self.gateway_audit_tcp.is_some(),
            self.gateway_audit_queue_capacity.is_some(),
            self.gateway_audit_max_record_bytes.is_some(),
            self.gateway_audit_connect_timeout_ms.is_some(),
            self.gateway_audit_write_timeout_ms.is_some(),
        ];
        if supplied.iter().all(|value| !value) {
            if self.gateway_audit_run_id.is_some() {
                return Err("--gateway-audit-run-id requires --gateway-audit-* settings".to_owned());
            }
            return Ok(None);
        }
        if !supplied.iter().all(|value| *value) {
            return Err("all --gateway-audit-* settings must be supplied together".to_owned());
        }

        let exporter = LiveAuditConfig {
            address: self.gateway_audit_tcp.expect("validated above"),
            queue_capacity: self.gateway_audit_queue_capacity.expect("validated above"),
            maximum_record_bytes: self
                .gateway_audit_max_record_bytes
                .expect("validated above"),
            connect_timeout: Duration::from_millis(
                self.gateway_audit_connect_timeout_ms
                    .expect("validated above"),
            ),
            write_timeout: Duration::from_millis(
                self.gateway_audit_write_timeout_ms
                    .expect("validated above"),
            ),
        };
        exporter
            .validate()
            .map_err(|error| format!("invalid gateway audit settings: {error}"))?;
        let config = GatewayAuditConfig {
            exporter,
            run_id: self
                .gateway_audit_run_id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        };
        config.validate()?;
        Ok(Some(config))
    }

    fn live_audit_config(&self) -> Result<Option<LiveAuditConfig>, String> {
        let supplied = [
            self.live_audit_tcp.is_some(),
            self.live_audit_queue_capacity.is_some(),
            self.live_audit_max_record_bytes.is_some(),
            self.live_audit_connect_timeout_ms.is_some(),
            self.live_audit_write_timeout_ms.is_some(),
        ];
        if supplied.iter().all(|value| !value) {
            return Ok(None);
        }
        if !self.start_live {
            return Err("--live-audit-* settings require --start-live".to_owned());
        }
        if !supplied.iter().all(|value| *value) {
            return Err("all --live-audit-* settings must be supplied together".to_owned());
        }

        let config = LiveAuditConfig {
            address: self.live_audit_tcp.expect("validated above"),
            queue_capacity: self.live_audit_queue_capacity.expect("validated above"),
            maximum_record_bytes: self.live_audit_max_record_bytes.expect("validated above"),
            connect_timeout: Duration::from_millis(
                self.live_audit_connect_timeout_ms.expect("validated above"),
            ),
            write_timeout: Duration::from_millis(
                self.live_audit_write_timeout_ms.expect("validated above"),
            ),
        };
        config
            .validate()
            .map_err(|error| format!("invalid live audit settings: {error}"))?;
        Ok(Some(config))
    }

    fn config(self) -> Result<ServerConfig, String> {
        let stale_after = self.configured_stale_after()?;
        let expire_after = self.configured_expire_after()?;
        let config = ServerConfig {
            http_bind: self.http_bind,
            http_port: self.http_port,
            scene_bind: self.scene_bind,
            scene_port: self.scene_port,
            maximum_scene_bytes: self.maximum_scene_bytes,
            configuration_path: self.configuration,
            project_root: self.project_root,
            web_root: self.web_root,
            tcp_read_timeout: Duration::from_millis(self.tcp_read_timeout_ms),
            stale_after,
            expire_after,
            websocket_send_deadline: Duration::from_millis(self.websocket_send_deadline_ms),
            maximum_websocket_clients: self.maximum_websocket_clients,
            maximum_configuration_requests: self.maximum_configuration_requests,
        };
        config.validate()?;
        Ok(config)
    }

    fn configured_stale_after(&self) -> Result<Duration, String> {
        if let Some(milliseconds) = self.stale_after_ms {
            return Ok(Duration::from_millis(milliseconds));
        }

        let Some(effective) = self.effective_configuration()? else {
            return Ok(ServerConfig::default().stale_after);
        };
        let path = "/perception/freshness/maximum_observation_age_seconds";
        let seconds = effective
            .pointer(path)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| format!("configuration is missing {path}"))?;
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err(format!("{path} must be finite and positive"));
        }
        Duration::try_from_secs_f64(seconds)
            .map_err(|error| format!("{path} is out of range: {error}"))
    }

    fn configured_expire_after(&self) -> Result<Duration, String> {
        if let Some(milliseconds) = self.expire_after_ms {
            return Ok(Duration::from_millis(milliseconds));
        }

        let Some(effective) = self.effective_configuration()? else {
            return Ok(ServerConfig::default().expire_after);
        };
        let path = "/perception/freshness/expire_after_ms";
        let milliseconds = effective
            .pointer(path)
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("{path} must be a positive integer"))?;
        Ok(Duration::from_millis(milliseconds))
    }

    fn effective_configuration(&self) -> Result<Option<serde_json::Value>, String> {
        let Some(configuration_path) = self.configuration.as_ref() else {
            return Ok(None);
        };
        let authority =
            ConfigurationAuthority::new(configuration_path.clone(), AuthorityLimits::default())?;
        Ok(Some(authority.effective_configuration()?))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let arguments = ServerArgs::parse();
    let live_options = arguments.live_options().map_err(std::io::Error::other)?;
    let config = arguments.config().map_err(std::io::Error::other)?;
    run_with_live(config, live_options).await
}

#[cfg(test)]
mod tests {
    use super::ServerArgs;
    use clap::Parser;
    use serde_json::Value;
    use std::fs;
    use std::time::Duration;

    #[test]
    fn cli_exposes_local_safe_defaults_and_rejects_non_loopback_ingress() {
        let args = ServerArgs::try_parse_from(["billiards-server"]).unwrap();
        assert!(args.live_options().unwrap().audit_config.is_none());
        assert!(args.live_options().unwrap().gateway_audit_config.is_none());
        let config = args.config().unwrap();
        assert_eq!(config.http_port, 8001);
        assert_eq!(config.scene_port, 49_321);
        assert_eq!(config.maximum_scene_bytes, 262_144);
        assert!(config.configuration_path.is_none());
        assert!(config.project_root.is_none());
        assert!(config.web_root.is_none());
        assert_eq!(config.maximum_configuration_requests, 4);
        assert_eq!(config.stale_after, Duration::from_millis(100));

        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--start-live",
            "--live-executable",
            "/opt/billiards/bin/live",
            "--capture-helper-executable",
            "/opt/billiards/bin/camera-helper",
        ])
        .unwrap();
        let live = args.live_options().unwrap();
        assert!(live.enabled);
        assert_eq!(
            live.executable_path.as_deref(),
            Some(std::path::Path::new("/opt/billiards/bin/live"))
        );
        assert_eq!(
            live.capture_helper_path.as_deref(),
            Some(std::path::Path::new("/opt/billiards/bin/camera-helper"))
        );

        let args =
            ServerArgs::try_parse_from(["billiards-server", "--scene-bind", "0.0.0.0"]).unwrap();
        assert!(args.config().unwrap_err().contains("loopback"));

        let args =
            ServerArgs::try_parse_from(["billiards-server", "--project-root", "relative"]).unwrap();
        assert!(args
            .config()
            .unwrap_err()
            .contains("project root must be an absolute"));
    }

    #[test]
    fn live_audit_cli_is_all_or_none_opt_in_and_loopback_only() {
        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--start-live",
            "--live-audit-tcp",
            "127.0.0.1:49322",
            "--live-audit-queue-capacity",
            "37",
            "--live-audit-max-record-bytes",
            "8192",
            "--live-audit-connect-timeout-ms",
            "1250",
            "--live-audit-write-timeout-ms",
            "875",
        ])
        .unwrap();
        let audit = args.live_options().unwrap().audit_config.unwrap();
        assert_eq!(audit.address, "127.0.0.1:49322".parse().unwrap());
        assert_eq!(audit.queue_capacity, 37);
        assert_eq!(audit.maximum_record_bytes, 8_192);
        assert_eq!(audit.connect_timeout, Duration::from_millis(1_250));
        assert_eq!(audit.write_timeout, Duration::from_millis(875));

        let args =
            ServerArgs::try_parse_from(["billiards-server", "--live-audit-tcp", "127.0.0.1:49322"])
                .unwrap();
        assert!(args
            .live_options()
            .unwrap_err()
            .contains("require --start-live"));

        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--start-live",
            "--live-audit-tcp",
            "127.0.0.1:49322",
        ])
        .unwrap();
        assert!(args
            .live_options()
            .unwrap_err()
            .contains("must be supplied together"));

        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--start-live",
            "--live-audit-tcp",
            "0.0.0.0:49322",
            "--live-audit-queue-capacity",
            "37",
            "--live-audit-max-record-bytes",
            "8192",
            "--live-audit-connect-timeout-ms",
            "1250",
            "--live-audit-write-timeout-ms",
            "875",
        ])
        .unwrap();
        assert!(args
            .live_options()
            .unwrap_err()
            .contains("must be loopback"));

        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--start-live",
            "--live-audit-tcp",
            "127.0.0.1:49322",
            "--live-audit-queue-capacity",
            "0",
            "--live-audit-max-record-bytes",
            "8192",
            "--live-audit-connect-timeout-ms",
            "1250",
            "--live-audit-write-timeout-ms",
            "875",
        ])
        .unwrap();
        assert!(args
            .live_options()
            .unwrap_err()
            .contains("queue capacity must be between"));
    }

    #[test]
    fn gateway_audit_cli_is_independent_of_live_detector_and_detector_audit() {
        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--gateway-audit-tcp",
            "127.0.0.1:49323",
            "--gateway-audit-queue-capacity",
            "41",
            "--gateway-audit-max-record-bytes",
            "8192",
            "--gateway-audit-connect-timeout-ms",
            "1200",
            "--gateway-audit-write-timeout-ms",
            "900",
            "--gateway-audit-run-id",
            "test-run",
        ])
        .unwrap();
        let options = args.live_options().unwrap();
        assert!(!options.enabled);
        assert!(options.audit_config.is_none());
        let gateway_audit = options.gateway_audit_config.unwrap();
        assert_eq!(gateway_audit.run_id, "test-run");
        assert_eq!(
            gateway_audit.exporter.address,
            "127.0.0.1:49323".parse().unwrap()
        );
        assert_eq!(gateway_audit.exporter.queue_capacity, 41);
        assert_eq!(gateway_audit.exporter.maximum_record_bytes, 8192);
        assert_eq!(
            gateway_audit.exporter.connect_timeout,
            Duration::from_millis(1200)
        );
        assert_eq!(
            gateway_audit.exporter.write_timeout,
            Duration::from_millis(900)
        );

        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--gateway-audit-tcp",
            "127.0.0.1:49323",
        ])
        .unwrap();
        assert!(args
            .live_options()
            .unwrap_err()
            .contains("must be supplied together"));

        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--gateway-audit-run-id",
            "orphan-run-id",
        ])
        .unwrap();
        assert!(args
            .live_options()
            .unwrap_err()
            .contains("requires --gateway-audit-*"));
    }

    #[test]
    fn cli_accepts_explicit_runtime_policy_values() {
        let web_root = tempfile::tempdir().unwrap();
        let web_root_path = web_root.path().to_str().unwrap();
        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--http-port",
            "8123",
            "--scene-port",
            "49201",
            "--maximum-scene-bytes",
            "4096",
            "--config",
            "config.json",
            "--project-root",
            "/srv/billiards",
            "--web-root",
            web_root_path,
            "--tcp-read-timeout-ms",
            "500",
            "--stale-after-ms",
            "200",
            "--expire-after-ms",
            "1000",
            "--websocket-send-deadline-ms",
            "100",
            "--maximum-configuration-requests",
            "8",
        ])
        .unwrap();
        let config = args.config().unwrap();
        assert_eq!(config.http_port, 8123);
        assert_eq!(config.scene_port, 49_201);
        assert_eq!(
            config.project_root.as_deref(),
            Some(std::path::Path::new("/srv/billiards"))
        );
        assert_eq!(config.web_root.as_deref(), Some(web_root.path()));
        assert_eq!(config.stale_after.as_millis(), 200);
        assert_eq!(config.maximum_configuration_requests, 8);
    }

    #[test]
    fn configured_freshness_sets_gateway_stale_threshold() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("config.json");
        let mut document: Value = serde_json::from_str(include_str!(
            "../../contracts/configuration/v1/defaults.json"
        ))
        .unwrap();
        document["perception"]["freshness"]["maximum_observation_age_seconds"] =
            serde_json::json!(0.25);
        fs::write(&config_path, serde_json::to_vec(&document).unwrap()).unwrap();

        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--config",
            config_path.to_str().unwrap(),
        ])
        .unwrap();
        let config = args.config().unwrap();
        assert_eq!(config.stale_after, Duration::from_millis(250));
        assert_eq!(config.expire_after, Duration::from_millis(1_000));
    }

    #[test]
    fn generic_provider_uses_configured_freshness_expiry_and_validates_its_order() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("config.json");
        let mut document: Value = serde_json::from_str(include_str!(
            "../../contracts/configuration/v1/defaults.json"
        ))
        .unwrap();
        document["perception"]["rust_live"]["provider"] = serde_json::json!("openvino");
        document["perception"]["freshness"]["maximum_observation_age_seconds"] =
            serde_json::json!(2.0);
        document["perception"]["freshness"]["expire_after_ms"] = serde_json::json!(4_000);
        fs::write(&config_path, serde_json::to_vec(&document).unwrap()).unwrap();

        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--config",
            config_path.to_str().unwrap(),
        ])
        .unwrap();
        let config = args.config().unwrap();
        assert_eq!(config.stale_after, Duration::from_secs(2));
        assert_eq!(config.expire_after, Duration::from_secs(4));

        document["perception"]["freshness"]["expire_after_ms"] = serde_json::json!(1_000);
        fs::write(&config_path, serde_json::to_vec(&document).unwrap()).unwrap();
        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--config",
            config_path.to_str().unwrap(),
        ])
        .unwrap();
        let error = args.config().unwrap_err();
        assert!(
            error.contains("Configuration validation failed"),
            "unexpected freshness validation error: {error}"
        );
    }

    #[test]
    fn explicit_stale_threshold_overrides_configured_freshness() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("config.json");
        let mut document: Value = serde_json::from_str(include_str!(
            "../../contracts/configuration/v1/defaults.json"
        ))
        .unwrap();
        document["perception"]["freshness"]["maximum_observation_age_seconds"] =
            serde_json::json!(0.25);
        fs::write(&config_path, serde_json::to_vec(&document).unwrap()).unwrap();

        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--config",
            config_path.to_str().unwrap(),
            "--stale-after-ms",
            "200",
        ])
        .unwrap();
        assert_eq!(
            args.config().unwrap().stale_after,
            Duration::from_millis(200)
        );
    }
}
