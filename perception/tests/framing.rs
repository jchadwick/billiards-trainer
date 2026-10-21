use std::io::{self, Cursor, Read, Write};

use billiards_perception::framing::{read_framed_message, write_framed_message, FrameError};

struct FragmentedReader<R> {
    inner: R,
    maximum_chunk: usize,
}

impl<R: Read> Read for FragmentedReader<R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let limit = output.len().min(self.maximum_chunk);
        self.inner.read(&mut output[..limit])
    }
}

struct FragmentedWriter {
    bytes: Vec<u8>,
    maximum_chunk: usize,
}

impl Write for FragmentedWriter {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        let count = input.len().min(self.maximum_chunk);
        self.bytes.extend_from_slice(&input[..count]);
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn writes_big_endian_length_and_reads_payload() {
    let mut bytes = Vec::new();
    write_framed_message(&mut bytes, b"scene", 32).unwrap();

    assert_eq!(&bytes[..4], &[0, 0, 0, 5]);
    assert_eq!(
        read_framed_message(&mut Cursor::new(bytes), 32).unwrap(),
        b"scene"
    );
}

#[test]
fn exact_size_limit_is_inclusive() {
    let mut bytes = Vec::new();
    write_framed_message(&mut bytes, b"1234", 4).unwrap();
    assert_eq!(
        read_framed_message(&mut Cursor::new(bytes), 4).unwrap(),
        b"1234"
    );
}

#[test]
fn write_and_read_handle_short_io_operations() {
    let mut writer = FragmentedWriter {
        bytes: Vec::new(),
        maximum_chunk: 1,
    };
    write_framed_message(&mut writer, b"partial-io", 32).unwrap();
    let mut reader = FragmentedReader {
        inner: Cursor::new(writer.bytes),
        maximum_chunk: 1,
    };

    assert_eq!(read_framed_message(&mut reader, 32).unwrap(), b"partial-io");
}

#[test]
fn can_read_consecutive_messages_without_consuming_the_next_frame() {
    let mut bytes = Vec::new();
    write_framed_message(&mut bytes, b"first", 32).unwrap();
    write_framed_message(&mut bytes, b"second", 32).unwrap();
    let mut reader = Cursor::new(bytes);

    assert_eq!(read_framed_message(&mut reader, 32).unwrap(), b"first");
    assert_eq!(read_framed_message(&mut reader, 32).unwrap(), b"second");
}

#[test]
fn oversize_header_is_rejected_before_reading_or_allocating_its_body() {
    let mut reader = Cursor::new(1024_u32.to_be_bytes());

    let error = read_framed_message(&mut reader, 128).unwrap_err();
    assert!(matches!(
        error,
        FrameError::MessageTooLarge {
            length: 1024,
            maximum: 128
        }
    ));
    assert_eq!(reader.position(), 4);
}

#[test]
fn zero_length_messages_are_rejected_on_both_paths() {
    assert!(matches!(
        write_framed_message(&mut Vec::new(), b"", 32),
        Err(FrameError::EmptyMessage)
    ));

    let mut reader = Cursor::new(0_u32.to_be_bytes());
    assert!(matches!(
        read_framed_message(&mut reader, 32),
        Err(FrameError::EmptyMessage)
    ));
}

#[test]
fn truncated_headers_and_bodies_return_io_errors() {
    let mut header = Cursor::new([0_u8, 0_u8]);
    assert!(matches!(
        read_framed_message(&mut header, 32),
        Err(FrameError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof
    ));

    let mut body = Cursor::new([0_u8, 0_u8, 0_u8, 4, b'a', b'b']);
    assert!(matches!(
        read_framed_message(&mut body, 32),
        Err(FrameError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof
    ));
}

#[test]
fn invalid_maximum_is_rejected_before_io() {
    let mut empty_reader = Cursor::new(Vec::<u8>::new());
    assert!(matches!(
        read_framed_message(&mut empty_reader, 0),
        Err(FrameError::InvalidMaximum(0))
    ));
    assert!(matches!(
        write_framed_message(&mut Vec::new(), b"x", 0),
        Err(FrameError::InvalidMaximum(0))
    ));
    if usize::BITS > u32::BITS {
        assert!(matches!(
            write_framed_message(&mut Vec::new(), b"x", u32::MAX as usize + 1),
            Err(FrameError::InvalidMaximum(_))
        ));
    }
}

#[test]
fn rejects_payload_larger_than_configured_limit() {
    let error = write_framed_message(&mut Vec::new(), b"larger", 4).unwrap_err();
    assert!(matches!(
        error,
        FrameError::MessageTooLarge {
            length: 6,
            maximum: 4
        }
    ));
}
