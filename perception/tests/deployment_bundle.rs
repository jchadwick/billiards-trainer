//! Exercise the deployment launcher with the real server and a staged bundle.
//!
//! This deliberately disables live capture: packaging and process lifecycle
//! should be testable without a camera, model, GStreamer, or optional model-export tooling.

use serde_json::Value;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(20);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(8);
const CONNECT_TIMEOUT: Duration = Duration::from_millis(250);
const IO_TIMEOUT: Duration = Duration::from_secs(2);

struct StagedDeployment {
    _directory: TempDir,
    root: PathBuf,
    script: PathBuf,
    config: PathBuf,
    web_root: PathBuf,
    run_dir: PathBuf,
    pid_file: PathBuf,
    log_file: PathBuf,
}

impl StagedDeployment {
    fn create() -> Self {
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("perception crate has a repository parent")
            .to_path_buf();
        let directory = tempfile::tempdir().expect("create isolated deployment root");
        let root = directory.path().to_path_buf();
        let script = root.join("scripts/deploy/run.sh");
        fs::create_dir_all(script.parent().expect("script has parent"))
            .expect("create staged script directory");
        fs::copy(repository.join("scripts/deploy/run.sh"), &script)
            .expect("copy deployment launcher into staged root");

        let web_root = root.join("backend/static");
        copy_directory(&repository.join("backend/static"), &web_root)
            .expect("copy built browser assets into staged root");

        let config = root.join("config.json");
        let defaults = repository.join("perception/contracts/configuration/v1/defaults.json");
        let defaults_text = fs::read_to_string(defaults).expect("read versioned config defaults");
        let mut parsed: Value =
            serde_json::from_str(&defaults_text).expect("versioned config defaults are valid JSON");
        assert_eq!(
            parsed.pointer("/metadata/version").and_then(Value::as_str),
            Some("1.0.0"),
            "the staged config must be derived from the checked-in v1 defaults"
        );
        // The versioned defaults include a recorded-video sample setting, but
        // this live deployment test exercises the camera-backed server path.
        parsed["vision"]["camera"]["video_file_path"] = Value::String(String::new());
        let staged_config =
            serde_json::to_vec_pretty(&parsed).expect("serialize live-compatible config");
        fs::write(&config, staged_config).expect("stage valid live configuration");

        let run_dir = root.join("run");
        fs::create_dir_all(&run_dir).expect("create staged run directory");
        let pid_file = run_dir.join("billiards-server.pid");
        let log_file = root.join("deployment.log");

        Self {
            _directory: directory,
            root,
            script,
            config,
            web_root,
            run_dir,
            pid_file,
            log_file,
        }
    }
}

struct RunningDeployment {
    child: Child,
    pid_file: PathBuf,
    log_file: PathBuf,
}

impl RunningDeployment {
    fn launch(staged: &StagedDeployment, http_port: u16, scene_port: u16) -> Self {
        let log = File::create(&staged.log_file).expect("create deployment log");
        let stderr = log.try_clone().expect("clone log handle for stderr");
        let child = Command::new("bash")
            .arg(&staged.script)
            .current_dir(&staged.root)
            .env(
                "BILLIARDS_SERVER_EXECUTABLE",
                env!("CARGO_BIN_EXE_billiards-server"),
            )
            .env("BILLIARDS_CONFIG", &staged.config)
            .env("BILLIARDS_PROJECT_ROOT", &staged.root)
            .env("BILLIARDS_WEB_ROOT", &staged.web_root)
            .env("BILLIARDS_RUN_DIR", &staged.run_dir)
            .env("BILLIARDS_PID_FILE", &staged.pid_file)
            .env("BILLIARDS_HTTP_BIND", "127.0.0.1")
            .env("BILLIARDS_HTTP_PORT", http_port.to_string())
            .env("BILLIARDS_SCENE_BIND", "127.0.0.1")
            .env("BILLIARDS_SCENE_PORT", scene_port.to_string())
            .env("BILLIARDS_START_LIVE", "false")
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(stderr))
            .spawn()
            .expect("launch staged deployment script");
        Self {
            child,
            pid_file: staged.pid_file.clone(),
            log_file: staged.log_file.clone(),
        }
    }

    fn wait_until_http_ready(&mut self, address: SocketAddr) -> u32 {
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        let mut server_pid = None;
        while Instant::now() < deadline {
            if let Some(status) = self
                .child
                .try_wait()
                .expect("inspect deployment launcher status")
            {
                panic!(
                    "deployment launcher exited before HTTP readiness ({status}):\n{}",
                    self.log_contents()
                );
            }
            if let Ok(contents) = fs::read_to_string(&self.pid_file) {
                server_pid = contents.trim().parse::<u32>().ok();
            }
            if let Ok((status, _, _)) = http_get(address, "/api/v1/health/live") {
                if status == 200 {
                    if let Some(server_pid) = server_pid {
                        return server_pid;
                    }
                }
            }
            thread::sleep(Duration::from_millis(50));
        }
        panic!(
            "deployment did not become HTTP-ready within {STARTUP_TIMEOUT:?}; PID file: {:?}\n{}",
            fs::read_to_string(&self.pid_file).ok(),
            self.log_contents()
        );
    }

    fn terminate_and_wait(&mut self) -> ExitStatus {
        #[cfg(unix)]
        {
            let result = unsafe { libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM) };
            assert_eq!(result, 0, "send SIGTERM to deployment launcher");
        }

        wait_for_exit(&mut self.child, SHUTDOWN_TIMEOUT).unwrap_or_else(|| {
            panic!(
                "deployment launcher did not stop within {SHUTDOWN_TIMEOUT:?}:\n{}",
                self.log_contents()
            )
        })
    }

    fn log_contents(&self) -> String {
        let Ok(mut file) = File::open(&self.log_file) else {
            return "<deployment log unavailable>".to_owned();
        };
        let _ = file.seek(SeekFrom::Start(0));
        let mut output = String::new();
        let _ = file.read_to_string(&mut output);
        output
    }
}

impl Drop for RunningDeployment {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_some() {
            return;
        }

        #[cfg(unix)]
        unsafe {
            let _ = libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM);
        }
        if wait_for_exit(&mut self.child, Duration::from_secs(3)).is_some() {
            return;
        }

        #[cfg(unix)]
        unsafe {
            if let Ok(contents) = fs::read_to_string(&self.pid_file) {
                if let Ok(server_pid) = contents.trim().parse::<libc::pid_t>() {
                    let _ = libc::kill(server_pid, libc::SIGTERM);
                }
            }
            let _ = libc::kill(self.child.id() as libc::pid_t, libc::SIGKILL);
        }
        let _ = self.child.wait();
    }
}

#[test]
#[cfg(unix)]
fn deployment_bundle_serves_health_and_static_client_then_cleans_up_on_sigterm() {
    let staged = StagedDeployment::create();
    let (http_port, scene_port) = two_free_ports();
    let address = SocketAddr::from(([127, 0, 0, 1], http_port));
    let mut deployment = RunningDeployment::launch(&staged, http_port, scene_port);

    let server_pid = deployment.wait_until_http_ready(address);
    let pid_contents = fs::read_to_string(&staged.pid_file).expect("launcher PID file exists");
    assert_eq!(
        pid_contents.trim().parse::<u32>().unwrap(),
        server_pid,
        "PID file identifies the server process"
    );

    let (status, content_type, health_body) =
        http_get(address, "/api/v1/health/live").expect("health request reaches the staged server");
    assert_eq!(status, 200);
    assert!(content_type.starts_with("application/json"));
    let health: Value = serde_json::from_slice(&health_body).expect("health response is JSON");
    assert_eq!(health["status"], "alive");

    let (status, content_type, html) =
        http_get(address, "/").expect("root request reaches the staged static client");
    assert_eq!(status, 200);
    assert!(content_type.starts_with("text/html"));
    let expected_html =
        fs::read(staged.web_root.join("index.html")).expect("staged web root has its index");
    assert_eq!(
        html, expected_html,
        "the staged deployment serves its client"
    );

    let status = deployment.terminate_and_wait();
    assert_eq!(
        status.code(),
        Some(143),
        "run.sh reports its documented SIGTERM cleanup status"
    );
    assert!(
        !staged.pid_file.exists(),
        "launcher removes the PID file after supervised shutdown"
    );
    assert!(
        http_get(address, "/api/v1/health/live").is_err(),
        "server no longer accepts requests after launcher shutdown"
    );
}

fn two_free_ports() -> (u16, u16) {
    let first = TcpListener::bind(("127.0.0.1", 0))
        .expect("reserve ephemeral HTTP port")
        .local_addr()
        .expect("read HTTP port")
        .port();
    let second = loop {
        let candidate = TcpListener::bind(("127.0.0.1", 0))
            .expect("reserve ephemeral scene port")
            .local_addr()
            .expect("read scene port")
            .port();
        if candidate != first {
            break candidate;
        }
    };
    (first, second)
}

fn copy_directory(source: &Path, destination: &Path) -> io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let kind = entry.file_type()?;
        if kind.is_dir() {
            copy_directory(&source_path, &destination_path)?;
        } else if kind.is_file() {
            fs::copy(source_path, destination_path)?;
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported static asset type: {}", source_path.display()),
            ));
        }
    }
    Ok(())
}

fn http_get(address: SocketAddr, path: &str) -> io::Result<(u16, String, Vec<u8>)> {
    let mut stream = TcpStream::connect_timeout(&address, CONNECT_TIMEOUT)?;
    stream.set_read_timeout(Some(IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IO_TIMEOUT))?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n"
    )?;

    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    let split = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing HTTP header end"))?;
    let headers = std::str::from_utf8(&response[..split])
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut lines = headers.lines();
    let status = lines
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid HTTP status line"))?;
    let content_type = lines
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-type")
                .then(|| value.trim().to_owned())
        })
        .unwrap_or_default();
    Ok((status, content_type, response[split + 4..].to_vec()))
}

fn wait_for_exit(child: &mut Child, timeout: Duration) -> Option<ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return Some(status);
        }
        if Instant::now() >= deadline {
            return None;
        }
        thread::sleep(Duration::from_millis(25));
    }
}
