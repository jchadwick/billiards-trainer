//! Separately launchable Rust localization gateway for side-by-side migration.

use std::net::IpAddr;
use std::path::PathBuf;
use std::time::Duration;

use billiards_perception::server::{
    run_with_live, LiveProcessLaunchOptions, ServerConfig, DEFAULT_MAX_CONFIGURATION_REQUESTS,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "billiards-server",
    about = "Serve Rust scene localization over local HTTP and WebSocket"
)]
struct ServerArgs {
    /// HTTP interface. Defaults to loopback for side-by-side development.
    #[arg(long, default_value = "127.0.0.1")]
    http_bind: IpAddr,
    /// HTTP port. Defaults to 8001 so the current gateway can remain on 8000.
    #[arg(long, default_value_t = 8001)]
    http_port: u16,
    /// Scene ingress interface. Must be loopback.
    #[arg(long, default_value = "127.0.0.1")]
    scene_bind: IpAddr,
    /// Loopback scene ingress port for replay-v2 and live-v3 producers.
    #[arg(long, default_value_t = 49_321)]
    scene_port: u16,
    /// Maximum payload size for one framed scene message, excluding its header (hard limit: 32 MiB).
    #[arg(long, default_value_t = 262_144)]
    maximum_scene_bytes: usize,
    /// Active configuration file. Required for live capture and configuration HTTP routes; replay-only mode may omit it.
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
    /// Mark locations stale after this many milliseconds without fresh localization.
    #[arg(long, default_value_t = 100)]
    stale_after_ms: u64,
    /// Remove cached locations after this many milliseconds without localization.
    #[arg(long, default_value_t = 1_000)]
    expire_after_ms: u64,
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
}

impl ServerArgs {
    fn live_options(&self) -> LiveProcessLaunchOptions {
        LiveProcessLaunchOptions {
            enabled: self.start_live,
            executable_path: self.live_executable.clone(),
            capture_helper_path: self.capture_helper_executable.clone(),
        }
    }

    fn config(self) -> Result<ServerConfig, String> {
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
            stale_after: Duration::from_millis(self.stale_after_ms),
            expire_after: Duration::from_millis(self.expire_after_ms),
            websocket_send_deadline: Duration::from_millis(self.websocket_send_deadline_ms),
            maximum_websocket_clients: self.maximum_websocket_clients,
            maximum_configuration_requests: self.maximum_configuration_requests,
        };
        config.validate()?;
        Ok(config)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let arguments = ServerArgs::parse();
    let live_options = arguments.live_options();
    let config = arguments.config().map_err(std::io::Error::other)?;
    run_with_live(config, live_options).await
}

#[cfg(test)]
mod tests {
    use super::ServerArgs;
    use clap::Parser;

    #[test]
    fn cli_exposes_local_safe_defaults_and_rejects_non_loopback_ingress() {
        let args = ServerArgs::try_parse_from(["billiards-server"]).unwrap();
        let config = args.config().unwrap();
        assert_eq!(config.http_port, 8001);
        assert_eq!(config.scene_port, 49_321);
        assert_eq!(config.maximum_scene_bytes, 262_144);
        assert!(config.configuration_path.is_none());
        assert!(config.project_root.is_none());
        assert!(config.web_root.is_none());
        assert_eq!(config.maximum_configuration_requests, 4);

        let args = ServerArgs::try_parse_from([
            "billiards-server",
            "--start-live",
            "--live-executable",
            "/opt/billiards/bin/live",
            "--capture-helper-executable",
            "/opt/billiards/bin/camera-helper",
        ])
        .unwrap();
        let live = args.live_options();
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
}
