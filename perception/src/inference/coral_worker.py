"""Persistent, framed TFLite worker for the Rust Coral inference backend.

This process owns the interpreter and Edge TPU delegate for its lifetime. The
Rust side sends one contiguous float32 NHWC tensor per frame and receives the
float32 YOLO output. Control messages are JSON lines during startup; inference
messages use fixed little-endian lengths followed by raw tensor bytes.
"""

import json
import math
import struct
import sys
import traceback


def read_exact_into(stream, target):
    """Fill a reusable byte buffer without allocating a second tensor copy."""
    view = memoryview(target)
    offset = 0
    while offset < len(view):
        bytes_read = stream.readinto(view[offset:])
        if not bytes_read:
            return False
        offset += bytes_read
    return True


def write_startup_error(error):
    message = str(error)
    sys.stdout.write(json.dumps({"ok": False, "error": message}) + "\n")
    sys.stdout.flush()


def load_tflite_api():
    """Load TensorFlow Lite from the standalone runtime, LiteRT, or TensorFlow.

    Coral's standalone wheel is the preferred option. Some supported Python
    versions do not have a published Coral wheel, though. LiteRT is the
    supported successor to TensorFlow's bundled interpreter and exposes the
    same delegate-loading API. Full TensorFlow remains a final fallback for
    environments where neither standalone runtime is available.
    """
    try:
        import tflite_runtime.interpreter as tflite

        return tflite
    except ImportError as standalone_error:
        try:
            from ai_edge_litert.interpreter import Interpreter, load_delegate
        except ImportError:
            try:
                import tensorflow as tf
            except ImportError as tensorflow_error:
                raise RuntimeError(
                    "install tflite_runtime, ai-edge-litert, or TensorFlow with "
                    "the TFLite interpreter enabled"
                ) from tensorflow_error

            if not hasattr(tf.lite, "experimental") or not hasattr(
                tf.lite.experimental, "load_delegate"
            ):
                raise RuntimeError(
                    "the installed TensorFlow package does not expose the TFLite "
                    "delegate loader"
                ) from standalone_error

            class TensorFlowLiteApi:
                Interpreter = tf.lite.Interpreter
                load_delegate = staticmethod(tf.lite.experimental.load_delegate)

            return TensorFlowLiteApi

        return type(
            "LiteRtApi",
            (),
            {
                "Interpreter": Interpreter,
                "load_delegate": staticmethod(load_delegate),
            },
        )


def run():
    try:
        import numpy as np
        tflite = load_tflite_api()

        model_path = sys.argv[1]
        delegate_library = sys.argv[2]
        thread_setting = sys.argv[3] if len(sys.argv) > 3 else ""
        num_threads = int(thread_setting) if thread_setting else None

        delegate = tflite.load_delegate(delegate_library)
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[delegate],
            num_threads=num_threads,
        )
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        input_shape = [int(value) for value in input_detail["shape"]]
        output_shape = [int(value) for value in output_detail["shape"]]
        if np.dtype(input_detail["dtype"]).name != "float32":
            raise RuntimeError(
                "expected float32 Coral model input, got "
                + np.dtype(input_detail["dtype"]).name
            )
        if np.dtype(output_detail["dtype"]).name != "float32":
            raise RuntimeError(
                "expected float32 Coral model output, got "
                + np.dtype(output_detail["dtype"]).name
            )
        input_buffer = bytearray(
            math.prod(input_shape) * np.dtype(input_detail["dtype"]).itemsize
        )

        sys.stdout.write(
            json.dumps(
                {
                    "ok": True,
                    "input": {"shape": input_shape, "dtype": "float32"},
                    "output": {"shape": output_shape, "dtype": "float32"},
                }
            )
            + "\n"
        )
        sys.stdout.flush()
    except Exception as error:
        write_startup_error(error)
        return 1

    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer
    length_buffer = bytearray(8)
    while True:
        if not read_exact_into(input_stream, length_buffer):
            return 0
        input_length = struct.unpack("<Q", length_buffer)[0]
        if input_length == 0:
            return 0

        if input_length != len(input_buffer):
            message = (
                f"input tensor frame has {input_length} bytes; "
                f"expected {len(input_buffer)}"
            ).encode("utf-8")
            output_stream.write(b"\x01")
            output_stream.write(struct.pack("<I", len(message)))
            output_stream.write(message)
            output_stream.flush()
            return 1

        try:
            if not read_exact_into(input_stream, input_buffer):
                raise RuntimeError("input tensor ended before its declared length")
            tensor = np.frombuffer(input_buffer, dtype=np.float32).reshape(input_shape)
            interpreter.set_tensor(input_detail["index"], tensor)
            interpreter.invoke()
            output = np.asarray(
                interpreter.get_tensor(output_detail["index"]), dtype=np.float32
            )
            if list(output.shape) != output_shape:
                raise RuntimeError(
                    "output shape changed during inference: " + str(list(output.shape))
                )
            output_stream.write(b"\x00")
            output_stream.write(output.tobytes(order="C"))
            output_stream.flush()
        except Exception as error:
            message = str(error).encode("utf-8", errors="replace")[:16384]
            output_stream.write(b"\x01")
            output_stream.write(struct.pack("<I", len(message)))
            output_stream.write(message)
            output_stream.flush()


if __name__ == "__main__":
    sys.exit(run())
