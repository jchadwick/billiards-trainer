use std::io;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use billiards_perception::configuration_authority::{
    run_one_shot, AuthorityLimits, ConfigurationAuthority, DEFAULT_LOCK_TIMEOUT_MS,
    DEFAULT_MAX_CONFIG_BYTES, DEFAULT_MAX_INPUT_BYTES, DEFAULT_MAX_OUTPUT_BYTES,
};
use clap::Parser;

/// One-shot JSON configuration authority. The path is mandatory so invoking
/// this binary can never silently select or modify the repository config file.
#[derive(Debug, Parser)]
#[command(name = "billiards-perception-config")]
struct Arguments {
    /// Active unified configuration file selected by the service gateway.
    #[arg(long)]
    config: PathBuf,
    /// Maximum bytes accepted from stdin, including JSON whitespace.
    #[arg(long, default_value_t = DEFAULT_MAX_INPUT_BYTES)]
    max_input_bytes: usize,
    /// Maximum bytes emitted to stdout, including the final newline.
    #[arg(long, default_value_t = DEFAULT_MAX_OUTPUT_BYTES)]
    max_output_bytes: usize,
    /// Maximum bytes read from or written to the active configuration file.
    #[arg(long, default_value_t = DEFAULT_MAX_CONFIG_BYTES)]
    max_config_bytes: usize,
    /// Bounded wait for another process holding the configuration lock.
    #[arg(long, default_value_t = DEFAULT_LOCK_TIMEOUT_MS)]
    lock_timeout_ms: u64,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    let limits = AuthorityLimits {
        max_input_bytes: arguments.max_input_bytes,
        max_output_bytes: arguments.max_output_bytes,
        max_config_bytes: arguments.max_config_bytes,
        lock_timeout: Duration::from_millis(arguments.lock_timeout_ms),
    };
    let authority = match ConfigurationAuthority::new(arguments.config, limits) {
        Ok(authority) => authority,
        Err(error) => {
            let _ = writeln!(
                io::stdout().lock(),
                "{{\"protocol_version\":1,\"ok\":false,\"error\":{{\"code\":\"invalid_limits\",\"message\":{}}}}}",
                serde_json::to_string(&error).unwrap_or_else(|_| "\"Invalid authority configuration\"".to_owned())
            );
            return ExitCode::FAILURE;
        }
    };
    match run_one_shot(&authority, io::stdin().lock(), io::stdout().lock()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("configuration command I/O failed: {error}");
            ExitCode::FAILURE
        }
    }
}

use std::io::Write;
