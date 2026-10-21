//! Bounded, length-prefixed message framing for the proposed local TCP stream.
//!
//! This module intentionally depends only on `Read` and `Write`; socket
//! ownership, deadlines, connection supervision, and peer identity are
//! responsibilities of a later transport adapter.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::{self, Read, Write};

const HEADER_LENGTH: usize = std::mem::size_of::<u32>();

/// Errors produced by the length-prefixed wire framing helpers.
#[derive(Debug)]
pub enum FrameError {
    InvalidMaximum(usize),
    EmptyMessage,
    MessageTooLarge { length: usize, maximum: usize },
    AllocationFailed { length: usize },
    Io(io::Error),
}

impl Display for FrameError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMaximum(value) => write!(
                formatter,
                "maximum message size must be between 1 and {}, got {value}",
                u32::MAX
            ),
            Self::EmptyMessage => write!(formatter, "zero-length messages are not allowed"),
            Self::MessageTooLarge { length, maximum } => {
                write!(
                    formatter,
                    "message length {length} exceeds configured maximum {maximum}"
                )
            }
            Self::AllocationFailed { length } => {
                write!(formatter, "unable to reserve {length} bytes for a message")
            }
            Self::Io(error) => write!(formatter, "framed message I/O failed: {error}"),
        }
    }
}

impl Error for FrameError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for FrameError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Write one non-empty payload preceded by a four-byte unsigned big-endian length.
///
/// The configured maximum covers payload bytes and excludes the header. If a
/// write fails partway through, callers must discard the stream because its
/// framing state is no longer recoverable.
pub fn write_framed_message<W: Write>(
    writer: &mut W,
    payload: &[u8],
    maximum_message_bytes: usize,
) -> Result<(), FrameError> {
    validate_maximum(maximum_message_bytes)?;
    if payload.is_empty() {
        return Err(FrameError::EmptyMessage);
    }
    if payload.len() > maximum_message_bytes {
        return Err(FrameError::MessageTooLarge {
            length: payload.len(),
            maximum: maximum_message_bytes,
        });
    }
    let length = u32::try_from(payload.len()).map_err(|_| FrameError::MessageTooLarge {
        length: payload.len(),
        maximum: maximum_message_bytes,
    })?;
    writer.write_all(&length.to_be_bytes())?;
    writer.write_all(payload)?;
    Ok(())
}

/// Read one length-prefixed message without allocating beyond its configured limit.
pub fn read_framed_message<R: Read>(
    reader: &mut R,
    maximum_message_bytes: usize,
) -> Result<Vec<u8>, FrameError> {
    validate_maximum(maximum_message_bytes)?;
    let mut header = [0_u8; HEADER_LENGTH];
    reader.read_exact(&mut header)?;
    let length = u32::from_be_bytes(header) as usize;
    if length == 0 {
        return Err(FrameError::EmptyMessage);
    }
    if length > maximum_message_bytes {
        return Err(FrameError::MessageTooLarge {
            length,
            maximum: maximum_message_bytes,
        });
    }

    let mut payload = Vec::new();
    payload
        .try_reserve_exact(length)
        .map_err(|_| FrameError::AllocationFailed { length })?;
    payload.resize(length, 0);
    reader.read_exact(&mut payload)?;
    Ok(payload)
}

fn validate_maximum(maximum_message_bytes: usize) -> Result<(), FrameError> {
    if maximum_message_bytes == 0 || maximum_message_bytes > u32::MAX as usize {
        return Err(FrameError::InvalidMaximum(maximum_message_bytes));
    }
    Ok(())
}
