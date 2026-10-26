# Detection batch v1

This contract carries the complete detector output for one source frame. It
does not claim tracking or physical-ball identity. The tracked scene contract is
[`../../v1/scene-snapshot.schema.json`](../../v1/scene-snapshot.schema.json)
and is produced after cross-frame association.

`engine_instance_id` distinguishes engine restarts. `source.session_id` and
`source.frame_id` identify the exact input observation. `batch_sequence` orders
all published outcomes inside one engine instance. A source timestamp is null
when the input has no trustworthy source-clock time, as with unpaced replay of
still images; this must not be replaced with inference or publication time.
When present, the source stage says whether the value came from exposure,
camera-read completion, shared-memory publication, or frame arrival. A shared
memory timestamp is publication time, not camera exposure.

Inference start, completion, and batch publication use nanosecond ticks from one
process-local monotonic clock domain. In the current synchronous JSONL audit
export, `published_at` is captured immediately before the batch is encoded and
written; it measures the publication attempt, not durable disk completion or
remote receipt. Receivers may calculate inference and publication durations
from those fields. They may compare those ticks with another process only after
a platform test proves the clock domains comparable. Otherwise, report
source-to-inference and receiver durations as separate segments.

`detection_index` is the zero-based array position in this immutable batch. It is
unique only within the source frame and remains unchanged if this same batch is
retransmitted. Its observation key is the combination of engine instance,
source session, source frame, and detection index. The index is not a stable
track ID and must not be used to infer that two detections from different frames
are the same physical ball. Class IDs and names are specific to the model class
map. The current model distinguishes generic `cue` and `ball`; it does not
recognize numbered balls.

All box and center coordinates use the declared `image_space` and the original
source frame's pixel dimensions. They are never model-input pixels. The current
inference decoder reports the clipped box center. A successful empty batch
represents a completed observation with no detections. Failed and skipped
batches carry a reason and no detections, so they cannot clear a previously
valid state as if the table were observed empty.

The JSON Schema defines structural constraints. Rust validation also checks
source/image bounds, center-to-box consistency, model class references,
batch-local index order, limits, and comparable inference clock ordering. The
shared fixtures are exercised by Rust contract tests.
