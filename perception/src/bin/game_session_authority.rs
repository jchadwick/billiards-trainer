use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use billiards_perception::game_session::{
    handle_authority_request, GameSessionStore, DEFAULT_LOCK_TIMEOUT_MS, DEFAULT_MAX_INPUT_BYTES,
    DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_MAX_STORE_BYTES, MINIMUM_AUTHORITY_RESPONSE_BYTES,
};
use clap::Parser;

/// One-shot Rust game-session metadata authority. The store path is mandatory
/// so invocation cannot silently select a working-directory-dependent file.
#[derive(Debug, Parser)]
#[command(name = "billiards-perception-game-session")]
struct Arguments {
    /// Explicit persistent session metadata file.
    #[arg(long)]
    store: PathBuf,
    /// Maximum bytes accepted from stdin, including JSON whitespace.
    #[arg(long, default_value_t = DEFAULT_MAX_INPUT_BYTES)]
    max_input_bytes: usize,
    /// Maximum bytes emitted to stdout, excluding the final newline.
    #[arg(long, default_value_t = DEFAULT_MAX_OUTPUT_BYTES)]
    max_output_bytes: usize,
    /// Maximum bytes read from or written to the session store.
    #[arg(long, default_value_t = DEFAULT_MAX_STORE_BYTES)]
    max_store_bytes: usize,
    /// Bounded wait for another process holding the session lock.
    #[arg(long, default_value_t = DEFAULT_LOCK_TIMEOUT_MS)]
    lock_timeout_ms: u64,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    if !arguments.store.is_absolute()
        || !(1..=65_536).contains(&arguments.max_input_bytes)
        || !(MINIMUM_AUTHORITY_RESPONSE_BYTES..=65_536).contains(&arguments.max_output_bytes)
        || !(1..=65_536).contains(&arguments.max_store_bytes)
        || !(1..=60_000).contains(&arguments.lock_timeout_ms)
    {
        let _ = writeln!(
            io::stdout().lock(),
            "{{\"protocol_version\":1,\"ok\":false,\"operation\":\"unknown\",\"error\":{{\"code\":\"invalid_limits\",\"message\":\"store path and I/O limits are invalid\"}}}}"
        );
        return ExitCode::FAILURE;
    }

    let store = match GameSessionStore::new(
        arguments.store,
        Duration::from_millis(arguments.lock_timeout_ms),
        arguments.max_store_bytes,
    ) {
        Ok(store) => store,
        Err(error) => {
            let _ = writeln!(
                io::stdout().lock(),
                "{{\"protocol_version\":1,\"ok\":false,\"operation\":\"unknown\",\"error\":{{\"code\":\"invalid_limits\",\"message\":{}}}}}",
                serde_json::to_string(&error.to_string())
                    .unwrap_or_else(|_| "\"Invalid game-session authority configuration\"".to_owned())
            );
            return ExitCode::FAILURE;
        }
    };

    let mut input = Vec::new();
    if let Err(error) = io::stdin()
        .take((arguments.max_input_bytes as u64).saturating_add(1))
        .read_to_end(&mut input)
    {
        eprintln!("game-session request I/O failed: {error}");
        return ExitCode::FAILURE;
    }
    let response = match handle_authority_request(
        &store,
        &input,
        arguments.max_input_bytes,
        arguments.max_output_bytes,
    ) {
        Ok(response) => response,
        Err(error) => {
            eprintln!("game-session authority request failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let ok = serde_json::from_slice::<serde_json::Value>(&response)
        .ok()
        .and_then(|value| value.get("ok").and_then(serde_json::Value::as_bool))
        .unwrap_or(false);
    let mut output = io::stdout().lock();
    if output.write_all(&response).is_err() || output.write_all(b"\n").is_err() {
        eprintln!("game-session response could not be written");
        return ExitCode::FAILURE;
    }
    if output.flush().is_err() {
        return ExitCode::FAILURE;
    }
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
