# Historical ONNX Runtime OpenVINO provider application smoke

This is a short application integration smoke for the earlier ONNX Runtime
OpenVINO Execution Provider implementation. It does not qualify the direct
OpenVINO Runtime backend selected by ADR-0045. The direct backend has only a
separate one-frame CLI smoke inside the API container; that test bypassed the
API supervisor. The native backend has not yet been qualified end to end
through the API and frontend.

## Deployment and backend identity

The API and production frontend images were built on the Apple M4 development
Mac for linux/amd64, transferred to 192.168.1.31, and started as two
containers. The frontend is available at http://192.168.1.31:8081, with the
API on port 8000. Both containers reported healthy; the frontend's
/api/v1/health proxy returned HTTP 200.

The backend container used the 2026.4.1 OpenVINO libraries from a temporary
runtime test image through a read-only mount. However, the Rust executable in
this application run was the legacy ONNX Runtime build: its recorded SHA-256
is 318821996b543bcf7f945210355e6d309825b6672f05997d32ded6400d71080b, and its
runtime metadata identifies the OpenVINOExecutionProvider. The configuration
selected provider=openvino, but that value alone does not identify which
implementation ran. The direct native C API backend was not present in this
application binary. No authentication or token was used for the WebSocket
handshake or state subscription.

## State delivery

Both provider runs used the same 30 consecutive 3840×2160 frames, model and
thresholds, three warm-up inferences, and a requested 5 fps replay. A headless
subscriber connected through the frontend's /api/v1/websocket/ws proxy and
subscribed to state:

| Run | State messages received | Latest observed balls |
|---|---:|---:|
| CPU | 18 | 12 |
| ONNX Runtime OpenVINO EP | 22 | 13 |

The OpenVINO EP run wrote 22 detection batches. Its last snapshot identified
frame-000030; after the finite replay ended, the service correctly reported
source silence. Both application containers remained healthy. Message counts
are not a performance comparison because the two subscriber connections
began at different points in their separate app startups.

These observations show that the legacy ORT provider could publish state
through the supervised API and retained frontend. They do not verify native
OpenVINO loading, inference, or supervisor environment forwarding.

## Limits and follow-up

This smoke does not qualify camera capture, camera exposure timestamps, a
full-length replay, source-to-client latency, browser rendering, calibration,
numbered-ball identity, temporal accuracy, or trajectory predictions. The
input contains no temporal ground truth. The target host-level OpenVINO
installer was not used; this app used matching libraries from the target-host
2026.4.1 container image. The existing ustreamer process on port 8080 was left
untouched.

The next native-backend integration run must build and deploy the ADR-0045
binary, record its binary hash and actual binding metadata, and verify runtime
loading through the API supervisor and WebSocket receipt through the frontend.
The native backend still requires Apple Silicon runtime-backed validation on
the Mac.

Machine-readable details are in run-metadata.json.
