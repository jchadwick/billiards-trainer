//! Test-only saved-frame source for detector/viewer interference comparisons.
//!
//! Frames are loaded once and published through a Tokio `watch` channel. Each
//! subscriber therefore observes the newest frame after it is ready to read;
//! slow subscribers do not accumulate independent frame queues or make the
//! publisher wait for them.

use std::fs;
use std::future::IntoFuture;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{ensure, Context, Result};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use clap::Parser;
use futures_util::stream;
use tokio::net::TcpListener;
use tokio::sync::{watch, OwnedSemaphorePermit, Semaphore};
use tokio::time::{self, MissedTickBehavior};

const DEFAULT_BIND: IpAddr = IpAddr::V4(std::net::Ipv4Addr::LOCALHOST);
const DEFAULT_PORT: u16 = 8002;
const DEFAULT_FPS: f64 = 20.0;
const DEFAULT_MAX_CLIENTS: usize = 8;
const DEFAULT_MAX_FRAMES: usize = 10_000;
const MAX_FRAME_COUNT: usize = 1_000_000;
const DEFAULT_MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_MAX_TOTAL_BYTES: usize = 512 * 1024 * 1024;
const DEFAULT_SHUTDOWN_GRACE_SECONDS: u64 = 3;
const MULTIPART_BOUNDARY: &str = "frame";

#[derive(Debug, Parser)]
#[command(about = "Serve pre-encoded JPEG frames to multiple live-view clients")]
struct Args {
    /// Directory containing pre-encoded .jpg/.jpeg frames. Files are sorted by
    /// filename and played in a loop.
    #[arg(long)]
    frames_dir: PathBuf,

    /// Local address for the fixture HTTP server.
    #[arg(long, default_value_t = DEFAULT_BIND)]
    bind: IpAddr,

    /// HTTP port. Use 0 to ask the operating system for an available port.
    #[arg(long, default_value_t = DEFAULT_PORT)]
    port: u16,

    /// Frame publication cadence, in frames per second (0.1 through 120).
    #[arg(long, default_value_t = DEFAULT_FPS)]
    fps: f64,

    /// Maximum simultaneous MJPEG subscribers.
    #[arg(long, default_value_t = DEFAULT_MAX_CLIENTS)]
    max_clients: usize,

    /// Maximum number of preloaded frames, bounding per-frame metadata too.
    #[arg(long, default_value_t = DEFAULT_MAX_FRAMES)]
    max_frames: usize,

    /// Reject any source JPEG larger than this many bytes.
    #[arg(long, default_value_t = DEFAULT_MAX_FRAME_BYTES)]
    max_frame_bytes: usize,

    /// Reject the frame set if its total encoded size exceeds this limit.
    #[arg(long, default_value_t = DEFAULT_MAX_TOTAL_BYTES)]
    max_total_bytes: usize,

    /// Maximum time to wait for connected clients after shutdown starts.
    #[arg(long, default_value_t = DEFAULT_SHUTDOWN_GRACE_SECONDS)]
    shutdown_grace_seconds: u64,
    /// Fixture producer identity copied into every multipart frame header.
    #[arg(long, default_value = "fixture-engine")]
    engine_instance_id: String,
    /// Capture session identity copied into every multipart frame header.
    #[arg(long, default_value = "fixture-session")]
    source_session_id: String,
    /// Capture configuration revision applied to every fixture frame.
    #[arg(long, default_value_t = 0)]
    configuration_revision: u64,
    /// Calibration revision applied to every fixture frame.
    #[arg(long, default_value_t = 0)]
    calibration_revision: u64,
    /// Rectification revision applied to every fixture frame.
    #[arg(long, default_value_t = 0)]
    rectification_revision: u64,
}

#[derive(Clone, Debug)]
struct SourceFrame {
    index: usize,
    name: String,
    jpeg: Bytes,
}

#[derive(Debug)]
struct PublishedFrame {
    sequence: u64,
    source_index: usize,
    source_name: String,
    engine_instance_id: String,
    source_session_id: String,
    configuration_revision: u64,
    calibration_revision: u64,
    rectification_revision: u64,
    jpeg: Bytes,
}

#[derive(Debug, Clone)]
struct FrameStreamIdentity {
    engine_instance_id: String,
    source_session_id: String,
    configuration_revision: u64,
    calibration_revision: u64,
    rectification_revision: u64,
}

#[derive(Clone)]
struct AppState {
    latest: watch::Sender<Option<Arc<PublishedFrame>>>,
    client_slots: Arc<Semaphore>,
    shutdown: watch::Receiver<bool>,
}

#[derive(Clone, Copy)]
enum PartStage {
    Header,
    Image,
    Suffix,
}

struct ClientStream {
    latest: watch::Receiver<Option<Arc<PublishedFrame>>>,
    shutdown: watch::Receiver<bool>,
    current: Option<(Arc<PublishedFrame>, PartStage)>,
    has_observed: bool,
    _permit: OwnedSemaphorePermit,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let frames = load_frames(
        &args.frames_dir,
        args.max_frames,
        args.max_frame_bytes,
        args.max_total_bytes,
    )?;
    let frame_count = frames.len();
    let address = SocketAddr::new(args.bind, args.port);
    let listener = TcpListener::bind(address)
        .await
        .with_context(|| format!("bind MJPEG fixture to {address}"))?;
    let actual_address = listener.local_addr()?;
    let (latest, _) = watch::channel(None);
    let (shutdown, shutdown_receiver) = watch::channel(false);
    let state = AppState {
        latest: latest.clone(),
        client_slots: Arc::new(Semaphore::new(args.max_clients)),
        shutdown: shutdown_receiver.clone(),
    };
    let router = router(state);
    let publisher = tokio::spawn(publish_frames(
        latest,
        frames,
        Duration::from_secs_f64(1.0 / args.fps),
        FrameStreamIdentity {
            engine_instance_id: safe_header_value(&args.engine_instance_id),
            source_session_id: safe_header_value(&args.source_session_id),
            configuration_revision: args.configuration_revision,
            calibration_revision: args.calibration_revision,
            rectification_revision: args.rectification_revision,
        },
    ));

    eprintln!(
        "serving {} JPEG frames at {:.1} fps on http://{actual_address}/stream.mjpg (max clients: {})",
        frame_count,
        args.fps,
        args.max_clients
    );

    let graceful_shutdown = async move {
        let mut receiver = shutdown_receiver;
        if !*receiver.borrow() {
            let _ = receiver.changed().await;
        }
    };
    let mut server = Box::pin(
        axum::serve(listener, router)
            .with_graceful_shutdown(graceful_shutdown)
            .into_future(),
    );
    let shutdown_grace = Duration::from_secs(args.shutdown_grace_seconds);
    let result = tokio::select! {
        result = &mut server => result.context("serve MJPEG fixture"),
        _ = shutdown_signal() => {
            shutdown.send_replace(true);
            match time::timeout(shutdown_grace, &mut server).await {
                Ok(result) => result.context("serve MJPEG fixture"),
                Err(_) => {
                    eprintln!("MJPEG clients did not close within {} seconds; closing remaining connections", shutdown_grace.as_secs());
                    Ok(())
                }
            }
        }
    };
    publisher.abort();
    let _ = publisher.await;
    result?;
    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let mut terminate = signal(SignalKind::terminate())
            .expect("install SIGTERM listener for MJPEG fixture shutdown");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = terminate.recv() => {},
        }
    }

    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

fn validate_args(args: &Args) -> Result<()> {
    ensure!(
        args.fps.is_finite() && (0.1..=120.0).contains(&args.fps),
        "--fps must be between 0.1 and 120"
    );
    ensure!(
        args.max_clients > 0,
        "--max-clients must be greater than zero"
    );
    ensure!(
        args.max_clients <= Semaphore::MAX_PERMITS,
        "--max-clients exceeds the semaphore limit ({})",
        Semaphore::MAX_PERMITS
    );
    ensure!(
        (1..=MAX_FRAME_COUNT).contains(&args.max_frames),
        "--max-frames must be between 1 and {MAX_FRAME_COUNT}"
    );
    ensure!(
        args.max_frame_bytes > 0,
        "--max-frame-bytes must be greater than zero"
    );
    ensure!(
        args.max_total_bytes > 0,
        "--max-total-bytes must be greater than zero"
    );
    ensure!(
        (1..=60).contains(&args.shutdown_grace_seconds),
        "--shutdown-grace-seconds must be between 1 and 60"
    );
    for (name, value) in [
        ("--engine-instance-id", args.engine_instance_id.as_str()),
        ("--source-session-id", args.source_session_id.as_str()),
    ] {
        ensure!(
            !value.is_empty() && value.len() <= 128 && !value.chars().any(char::is_control),
            "{name} must contain between 1 and 128 non-control characters"
        );
    }
    ensure!(
        [
            args.configuration_revision,
            args.calibration_revision,
            args.rectification_revision
        ]
        .into_iter()
        .all(|revision| revision <= 9_007_199_254_740_991),
        "frame revisions must fit the JSON safe-integer range"
    );
    Ok(())
}

fn load_frames(
    directory: &Path,
    max_frames: usize,
    max_frame_bytes: usize,
    max_total_bytes: usize,
) -> Result<Vec<SourceFrame>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)
        .with_context(|| format!("read JPEG frame directory {}", directory.display()))?
    {
        let entry = entry.with_context(|| {
            format!(
                "read an entry in JPEG frame directory {}",
                directory.display()
            )
        })?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        let is_jpeg = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                extension.eq_ignore_ascii_case("jpg") || extension.eq_ignore_ascii_case("jpeg")
            });
        if is_jpeg {
            ensure!(
                paths.len() < max_frames,
                "frame set contains more than --max-frames ({max_frames}) files"
            );
            paths.push(path);
        }
    }
    paths.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
    ensure!(
        !paths.is_empty(),
        "no .jpg or .jpeg frames found in {}",
        directory.display()
    );
    ensure!(
        paths.len() <= max_frames,
        "frame set contains {} files, above --max-frames ({max_frames})",
        paths.len()
    );

    let mut total_bytes = 0_usize;
    let mut frames = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = fs::read(&path).with_context(|| format!("read frame {}", path.display()))?;
        ensure!(!bytes.is_empty(), "frame {} is empty", path.display());
        ensure!(
            bytes.len() <= max_frame_bytes,
            "frame {} is {} bytes, above --max-frame-bytes ({max_frame_bytes})",
            path.display(),
            bytes.len()
        );
        ensure!(
            bytes.starts_with(&[0xff, 0xd8]) && bytes.windows(2).any(|pair| pair == [0xff, 0xd9]),
            "frame {} does not contain JPEG start/end markers",
            path.display()
        );
        total_bytes = total_bytes
            .checked_add(bytes.len())
            .context("total JPEG frame size overflow")?;
        ensure!(
            total_bytes <= max_total_bytes,
            "frame set exceeds --max-total-bytes ({max_total_bytes})"
        );
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .context("frame filenames must be valid UTF-8")?;
        frames.push(SourceFrame {
            index: frames.len(),
            name: safe_header_value(file_name),
            jpeg: Bytes::from(bytes),
        });
    }
    Ok(frames)
}

fn safe_header_value(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

async fn publish_frames(
    latest: watch::Sender<Option<Arc<PublishedFrame>>>,
    frames: Vec<SourceFrame>,
    period: Duration,
    identity: FrameStreamIdentity,
) {
    let mut ticker = time::interval(period);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut sequence = 0_u64;
    loop {
        ticker.tick().await;
        let frame = &frames[(sequence as usize) % frames.len()];
        latest.send_replace(Some(Arc::new(PublishedFrame {
            sequence,
            source_index: frame.index,
            source_name: frame.name.clone(),
            engine_instance_id: identity.engine_instance_id.clone(),
            source_session_id: identity.source_session_id.clone(),
            configuration_revision: identity.configuration_revision,
            calibration_revision: identity.calibration_revision,
            rectification_revision: identity.rectification_revision,
            jpeg: frame.jpeg.clone(),
        })));
        sequence = sequence.wrapping_add(1);
    }
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/stream.mjpg", get(stream_mjpeg))
        .with_state(state)
}

async fn stream_mjpeg(State(state): State<AppState>) -> Response {
    let permit = match state.client_slots.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "subscriber limit reached").into_response();
        }
    };
    let client = ClientStream {
        latest: state.latest.subscribe(),
        shutdown: state.shutdown.clone(),
        current: None,
        has_observed: false,
        _permit: permit,
    };
    let body = Body::from_stream(stream::unfold(client, |mut client| async move {
        loop {
            if *client.shutdown.borrow_and_update() {
                return None;
            }

            if let Some((frame, stage)) = client.current.take() {
                let (part, next) = match stage {
                    PartStage::Header => {
                        let header = format!(
                            "--{MULTIPART_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\nX-Frame-Id: {}\r\nX-Engine-Instance-Id: {}\r\nX-Source-Session-Id: {}\r\nX-Configuration-Revision: {}\r\nX-Calibration-Revision: {}\r\nX-Rectification-Revision: {}\r\nX-Source-Frame-Index: {}\r\nX-Source-Frame-Name: {}\r\n\r\n",
                            frame.jpeg.len(),
                            frame.sequence,
                            frame.engine_instance_id,
                            frame.source_session_id,
                            frame.configuration_revision,
                            frame.calibration_revision,
                            frame.rectification_revision,
                            frame.source_index,
                            frame.source_name,
                        );
                        (Bytes::from(header), Some(PartStage::Image))
                    }
                    PartStage::Image => (frame.jpeg.clone(), Some(PartStage::Suffix)),
                    PartStage::Suffix => (Bytes::from_static(b"\r\n"), None),
                };
                if let Some(next) = next {
                    client.current = Some((frame, next));
                }
                return Some((Ok::<Bytes, std::convert::Infallible>(part), client));
            }

            if client.has_observed {
                tokio::select! {
                    biased;
                    changed = client.shutdown.changed() => {
                        if changed.is_err() || *client.shutdown.borrow_and_update() {
                            return None;
                        }
                    }
                    changed = client.latest.changed() => {
                        if changed.is_err() {
                            return None;
                        }
                    }
                }
            }
            client.has_observed = true;
            if let Some(frame) = client.latest.borrow_and_update().clone() {
                client.current = Some((frame, PartStage::Header));
            }
        }
    }));
    let mut response = body.into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("multipart/x-mixed-replace; boundary=frame"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store, no-cache, must-revalidate, max-age=0"),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io;
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpStream;
    use tokio::time::timeout;

    struct HttpClient {
        reader: BufReader<TcpStream>,
        chunk_remaining: usize,
    }

    struct ReceivedFrame {
        sequence: u64,
        source_index: usize,
        engine_instance_id: String,
        source_session_id: String,
        configuration_revision: u64,
        calibration_revision: u64,
        rectification_revision: u64,
        jpeg: Vec<u8>,
    }

    impl HttpClient {
        async fn connect(address: SocketAddr) -> io::Result<(Self, u16, HashMap<String, String>)> {
            let socket = TcpStream::connect(address).await?;
            let mut client = Self {
                reader: BufReader::new(socket),
                chunk_remaining: 0,
            };
            client
                .reader
                .get_mut()
                .write_all(b"GET /stream.mjpg HTTP/1.1\r\nHost: localhost\r\nAccept: multipart/x-mixed-replace\r\n\r\n")
                .await?;
            let mut status_line = String::new();
            client.reader.read_line(&mut status_line).await?;
            let status = status_line
                .split_whitespace()
                .nth(1)
                .and_then(|value| value.parse::<u16>().ok())
                .ok_or_else(|| {
                    io::Error::other(format!("invalid HTTP status line {status_line:?}"))
                })?;
            let mut headers = HashMap::new();
            loop {
                let mut line = String::new();
                client.reader.read_line(&mut line).await?;
                if line == "\r\n" || line.is_empty() {
                    break;
                }
                let (name, value) = line
                    .split_once(':')
                    .ok_or_else(|| io::Error::other(format!("invalid HTTP header {line:?}")))?;
                headers.insert(name.to_ascii_lowercase(), value.trim().to_owned());
            }
            Ok((client, status, headers))
        }

        async fn read_chunk_header(&mut self) -> io::Result<()> {
            let mut line = String::new();
            self.reader.read_line(&mut line).await?;
            if line.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "HTTP stream ended",
                ));
            }
            let size = line
                .trim()
                .split(';')
                .next()
                .and_then(|size| usize::from_str_radix(size, 16).ok())
                .ok_or_else(|| io::Error::other(format!("invalid HTTP chunk length {line:?}")))?;
            if size == 0 {
                loop {
                    let mut trailer = String::new();
                    self.reader.read_line(&mut trailer).await?;
                    if trailer == "\r\n" || trailer.is_empty() {
                        break;
                    }
                }
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "HTTP stream completed",
                ));
            }
            self.chunk_remaining = size;
            Ok(())
        }

        async fn read_body_exact(&mut self, output: &mut [u8]) -> io::Result<()> {
            let mut offset = 0;
            while offset < output.len() {
                if self.chunk_remaining == 0 {
                    self.read_chunk_header().await?;
                }
                let count = (output.len() - offset).min(self.chunk_remaining);
                self.reader
                    .read_exact(&mut output[offset..offset + count])
                    .await?;
                offset += count;
                self.chunk_remaining -= count;
                if self.chunk_remaining == 0 {
                    let mut separator = [0_u8; 2];
                    self.reader.read_exact(&mut separator).await?;
                    if separator != *b"\r\n" {
                        return Err(io::Error::other("invalid HTTP chunk separator"));
                    }
                }
            }
            Ok(())
        }

        async fn read_body_line(&mut self) -> io::Result<Vec<u8>> {
            let mut line = Vec::new();
            loop {
                let mut byte = [0_u8; 1];
                self.read_body_exact(&mut byte).await?;
                line.push(byte[0]);
                if line.ends_with(b"\r\n") {
                    line.truncate(line.len() - 2);
                    return Ok(line);
                }
            }
        }

        async fn read_frame(&mut self) -> io::Result<ReceivedFrame> {
            let boundary = self.read_body_line().await?;
            if boundary != format!("--{MULTIPART_BOUNDARY}").as_bytes() {
                return Err(io::Error::other(format!(
                    "unexpected multipart boundary {boundary:?}"
                )));
            }
            let mut content_length = None;
            let mut sequence = None;
            let mut source_index = None;
            let mut engine_instance_id = None;
            let mut source_session_id = None;
            let mut configuration_revision = None;
            let mut calibration_revision = None;
            let mut rectification_revision = None;
            loop {
                let line = self.read_body_line().await?;
                if line.is_empty() {
                    break;
                }
                let line = String::from_utf8(line).map_err(|error| {
                    io::Error::other(format!("multipart header is not UTF-8: {error}"))
                })?;
                let (name, value) = line.split_once(':').ok_or_else(|| {
                    io::Error::other(format!("invalid multipart header {line:?}"))
                })?;
                match name.to_ascii_lowercase().as_str() {
                    "content-length" => content_length = value.trim().parse::<usize>().ok(),
                    "x-frame-id" => sequence = value.trim().parse::<u64>().ok(),
                    "x-engine-instance-id" => engine_instance_id = Some(value.trim().to_owned()),
                    "x-source-session-id" => source_session_id = Some(value.trim().to_owned()),
                    "x-configuration-revision" => {
                        configuration_revision = value.trim().parse::<u64>().ok()
                    }
                    "x-calibration-revision" => {
                        calibration_revision = value.trim().parse::<u64>().ok()
                    }
                    "x-rectification-revision" => {
                        rectification_revision = value.trim().parse::<u64>().ok()
                    }
                    "x-source-frame-index" => source_index = value.trim().parse::<usize>().ok(),
                    _ => {}
                }
            }
            let content_length = content_length
                .ok_or_else(|| io::Error::other("missing multipart content length"))?;
            let mut jpeg = vec![0_u8; content_length];
            self.read_body_exact(&mut jpeg).await?;
            let mut suffix = [0_u8; 2];
            self.read_body_exact(&mut suffix).await?;
            if suffix != *b"\r\n" {
                return Err(io::Error::other("invalid multipart frame suffix"));
            }
            Ok(ReceivedFrame {
                sequence: sequence.ok_or_else(|| io::Error::other("missing frame sequence"))?,
                source_index: source_index
                    .ok_or_else(|| io::Error::other("missing source frame index"))?,
                engine_instance_id: engine_instance_id
                    .ok_or_else(|| io::Error::other("missing engine identity"))?,
                source_session_id: source_session_id
                    .ok_or_else(|| io::Error::other("missing source session identity"))?,
                configuration_revision: configuration_revision
                    .ok_or_else(|| io::Error::other("missing configuration revision"))?,
                calibration_revision: calibration_revision
                    .ok_or_else(|| io::Error::other("missing calibration revision"))?,
                rectification_revision: rectification_revision
                    .ok_or_else(|| io::Error::other("missing rectification revision"))?,
                jpeg,
            })
        }
    }

    fn test_state(
        max_clients: usize,
    ) -> (
        AppState,
        watch::Sender<Option<Arc<PublishedFrame>>>,
        watch::Sender<bool>,
    ) {
        let (latest, _) = watch::channel(None);
        let (shutdown, shutdown_receiver) = watch::channel(false);
        (
            AppState {
                latest: latest.clone(),
                client_slots: Arc::new(Semaphore::new(max_clients)),
                shutdown: shutdown_receiver,
            },
            latest,
            shutdown,
        )
    }

    fn publish(
        latest: &watch::Sender<Option<Arc<PublishedFrame>>>,
        sequence: u64,
        source_index: usize,
        jpeg: Bytes,
    ) {
        latest.send_replace(Some(Arc::new(PublishedFrame {
            sequence,
            source_index,
            source_name: format!("frame-{source_index:04}.jpg"),
            engine_instance_id: "engine-test".to_owned(),
            source_session_id: "session-test".to_owned(),
            configuration_revision: 7,
            calibration_revision: 2,
            rectification_revision: 3,
            jpeg,
        })));
    }

    async fn start_server(state: AppState) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let mut shutdown = state.shutdown.clone();
        let graceful_shutdown = async move {
            if !*shutdown.borrow() {
                let _ = shutdown.changed().await;
            }
        };
        let task = tokio::spawn(async move {
            axum::serve(listener, router(state))
                .with_graceful_shutdown(graceful_shutdown)
                .await
                .unwrap();
        });
        (address, task)
    }

    fn tiny_jpeg(id: u8) -> Bytes {
        Bytes::from(vec![0xff, 0xd8, id, 0xff, 0xd9])
    }

    #[test]
    fn shutdown_grace_configuration_is_bounded() {
        let mut args = Args {
            frames_dir: PathBuf::from("frames"),
            bind: DEFAULT_BIND,
            port: DEFAULT_PORT,
            fps: DEFAULT_FPS,
            max_clients: DEFAULT_MAX_CLIENTS,
            max_frames: DEFAULT_MAX_FRAMES,
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            max_total_bytes: DEFAULT_MAX_TOTAL_BYTES,
            shutdown_grace_seconds: DEFAULT_SHUTDOWN_GRACE_SECONDS,
            engine_instance_id: "fixture-engine".to_owned(),
            source_session_id: "fixture-session".to_owned(),
            configuration_revision: 0,
            calibration_revision: 0,
            rectification_revision: 0,
        };
        assert!(validate_args(&args).is_ok());
        args.max_frames = MAX_FRAME_COUNT + 1;
        assert!(validate_args(&args).is_err());
        args.max_frames = DEFAULT_MAX_FRAMES;
        args.shutdown_grace_seconds = 0;
        assert!(validate_args(&args).is_err());
        args.shutdown_grace_seconds = 61;
        assert!(validate_args(&args).is_err());
    }

    async fn connect_ok(address: SocketAddr) -> HttpClient {
        let (client, status, headers) = HttpClient::connect(address).await.unwrap();
        assert_eq!(status, StatusCode::OK.as_u16());
        assert_eq!(
            headers.get("content-type").unwrap(),
            "multipart/x-mixed-replace; boundary=frame"
        );
        assert_eq!(
            headers.get("transfer-encoding").map(String::as_str),
            Some("chunked")
        );
        client
    }

    async fn stop_server(task: tokio::task::JoinHandle<()>) {
        task.abort();
        let _ = task.await;
    }

    #[tokio::test]
    async fn two_real_http_clients_receive_ordered_latest_multipart_frames() {
        let (state, latest, _shutdown) = test_state(2);
        let (address, server) = start_server(state).await;
        let mut first = connect_ok(address).await;
        let mut second = connect_ok(address).await;

        let first_frame = tiny_jpeg(1);
        publish(&latest, 10, 0, first_frame.clone());
        let (left, right) = tokio::join!(first.read_frame(), second.read_frame());
        let left = left.unwrap();
        let right = right.unwrap();
        assert_eq!(left.sequence, 10);
        assert_eq!(right.sequence, 10);
        assert_eq!(left.engine_instance_id, "engine-test");
        assert_eq!(left.source_session_id, "session-test");
        assert_eq!(left.configuration_revision, 7);
        assert_eq!(left.calibration_revision, 2);
        assert_eq!(left.rectification_revision, 3);
        assert_eq!(left.jpeg, first_frame);
        assert_eq!(right.jpeg, left.jpeg);

        publish(&latest, 11, 1, tiny_jpeg(2));
        publish(&latest, 12, 2, tiny_jpeg(3));
        let (left, right) = tokio::join!(first.read_frame(), second.read_frame());
        let left = left.unwrap();
        let right = right.unwrap();
        assert_eq!((left.sequence, right.sequence), (12, 12));
        assert_eq!((left.source_index, right.source_index), (2, 2));
        stop_server(server).await;
    }

    #[tokio::test]
    async fn max_client_limit_is_enforced_and_released_when_a_client_disconnects() {
        let (state, _latest, _shutdown) = test_state(1);
        let (address, server) = start_server(state).await;
        let first = connect_ok(address).await;
        let (_rejected, status, _) = HttpClient::connect(address).await.unwrap();
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE.as_u16());
        drop(first);

        let replacement = timeout(Duration::from_secs(2), async {
            loop {
                let (client, status, _) = HttpClient::connect(address).await.unwrap();
                if status == StatusCode::OK.as_u16() {
                    break client;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("client permit should be released after disconnect");
        drop(replacement);
        stop_server(server).await;
    }

    #[tokio::test]
    async fn a_nonreading_client_does_not_block_a_fast_subscriber() {
        let (state, latest, _shutdown) = test_state(2);
        let (address, server) = start_server(state).await;
        let _slow = connect_ok(address).await;

        // A single large part fills the slow socket's receive window, making
        // this an actual backpressured HTTP subscriber rather than only a
        // delayed watch receiver.
        let large_jpeg = Bytes::from(vec![0x5a; 8 * 1024 * 1024]);
        publish(&latest, 1, 0, large_jpeg.clone());
        tokio::time::sleep(Duration::from_millis(100)).await;
        for sequence in 2..=20 {
            publish(&latest, sequence, sequence as usize, large_jpeg.clone());
        }

        let mut fast = connect_ok(address).await;
        let received = timeout(Duration::from_secs(3), fast.read_frame())
            .await
            .expect("fast client should remain responsive")
            .unwrap();
        assert_eq!(received.sequence, 20);
        assert_eq!(received.source_index, 20);
        assert_eq!(received.jpeg.len(), large_jpeg.len());
        stop_server(server).await;
    }

    #[tokio::test]
    async fn shutdown_ends_active_streams_and_releases_the_server() {
        let (state, latest, shutdown) = test_state(1);
        let (address, server) = start_server(state).await;
        let mut client = connect_ok(address).await;
        publish(&latest, 1, 0, tiny_jpeg(1));
        assert_eq!(client.read_frame().await.unwrap().sequence, 1);

        shutdown.send_replace(true);
        assert!(timeout(Duration::from_secs(2), client.read_frame())
            .await
            .expect("stream should stop on shutdown")
            .is_err());
        timeout(Duration::from_secs(2), server)
            .await
            .expect("server should complete graceful shutdown")
            .unwrap();
    }

    #[test]
    fn frame_order_is_filename_order_and_load_limits_are_enforced() {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join("frame-10.jpg"),
            [0xff, 0xd8, 10, 0xff, 0xd9],
        )
        .unwrap();
        fs::write(
            root.path().join("frame-02.jpeg"),
            [0xff, 0xd8, 2, 0xff, 0xd9],
        )
        .unwrap();
        fs::write(root.path().join("ignore.txt"), b"not a JPEG").unwrap();

        let frames = load_frames(root.path(), 10, 8, 16).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].name, "frame-02.jpeg");
        assert_eq!(frames[0].index, 0);
        assert_eq!(frames[1].name, "frame-10.jpg");
        assert!(load_frames(root.path(), 10, 4, 16).is_err());
        assert!(load_frames(root.path(), 10, 8, 8).is_err());
        assert!(load_frames(root.path(), 1, 8, 16).is_err());
    }

    #[test]
    fn response_header_filename_is_sanitized() {
        assert_eq!(
            safe_header_value("frame\r\nInjected: yes.jpg"),
            "frame__Injected__yes.jpg"
        );
    }
}
